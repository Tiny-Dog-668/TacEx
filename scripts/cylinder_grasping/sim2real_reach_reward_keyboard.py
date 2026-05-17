from __future__ import annotations

import argparse
import traceback
from collections import deque

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Keyboard teleoperation scene for validating the sim2real reach reward."
)
parser.add_argument(
    "--translation_step",
    type=float,
    default=0.004,
    help="End-effector translation per control step in meters.",
)
parser.add_argument(
    "--print_interval_steps",
    type=int,
    default=6,
    help="How many sim steps between reward printouts.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=20,
    help="Number of zero-action settle steps after reset.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if getattr(args_cli, "headless", False):
    raise RuntimeError("Keyboard teleoperation requires GUI mode. Remove --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import carb.input as carb_input
import isaaclab.utils.math as math_utils
import omni.appwindow
import torch

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.sim2real_grasp_env import Sim2RealGraspEnv, Sim2RealGraspEnvCfg


def _format_vec3(vec: torch.Tensor) -> str:
    return f"[{vec[0].item():.4f}, {vec[1].item():.4f}, {vec[2].item():.4f}]"


def _compute_reach_metrics(env: Sim2RealGraspEnv) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    cylinder_pos = env._cylinder.data.root_pos_w[:1]
    hand_pos = env._robot.data.body_link_pos_w[:1, env._body_idx]
    hand_quat = env._robot.data.body_link_quat_w[:1, env._body_idx]
    ee_pos, _ = math_utils.combine_frame_transforms(hand_pos, hand_quat, env._offset_pos[:1], env._offset_rot[:1])
    sigma = max(float(env.cfg.reach_sigma), 1e-6)
    reach_distance = torch.norm(cylinder_pos - ee_pos, dim=-1)
    reach_reward = 1.0 - torch.tanh(reach_distance / sigma)
    return (
        float(reach_reward[0].item()),
        float(reach_distance[0].item()),
        ee_pos[0].detach().clone(),
        cylinder_pos[0].detach().clone(),
    )


def _print_reach_metrics(env: Sim2RealGraspEnv, prefix: str = "[reach]") -> None:
    reach_reward, reach_distance, ee_pos, cylinder_pos = _compute_reach_metrics(env)
    print(
        f"{prefix} reward={reach_reward:.4f}  distance={reach_distance:.4f} m  "
        f"ee={_format_vec3(ee_pos)}  cylinder={_format_vec3(cylinder_pos)}"
    )


def _step_env(env: Sim2RealGraspEnv, actions: torch.Tensor) -> None:
    env._pre_physics_step(actions)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env._sync_cap_visual()
    env.sim.render()


def _reset_env(env: Sim2RealGraspEnv, warmup_steps: int) -> None:
    env.reset()
    zero_actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    for _ in range(max(int(warmup_steps), 0)):
        _step_env(env, zero_actions)
    _print_reach_metrics(env, prefix="[reset]")


def main() -> None:
    env_cfg = Sim2RealGraspEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.reward_print_interval = 0
    env_cfg.wrist_visual_randomization_enabled = False
    env_cfg.light_randomization_enabled = False
    env_cfg.ground_color_randomization_enabled = False
    env_cfg.plate_color_randomization_enabled = False

    env = Sim2RealGraspEnv(env_cfg)
    _reset_env(env, args_cli.warmup_steps)

    translation_step_units = float(args_cli.translation_step) / max(float(env.cfg.action_scale), 1e-6)
    translation_step_units = max(min(translation_step_units, 1.0), 0.0)
    print_interval_steps = max(int(args_cli.print_interval_steps), 1)

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window is not None else None
    input_iface = carb_input.acquire_input_interface() if keyboard is not None else None
    kb_sub = None

    pressed = {
        "up": False,
        "down": False,
        "left": False,
        "right": False,
        "forward": False,
        "backward": False,
        "open": False,
        "close": False,
    }
    command_queue: deque[str] = deque()

    def _on_kb_event(event, *args, **kwargs):
        if event.type not in (
            carb_input.KeyboardEventType.KEY_PRESS,
            carb_input.KeyboardEventType.KEY_REPEAT,
            carb_input.KeyboardEventType.KEY_RELEASE,
        ):
            return True

        is_down = event.type in (
            carb_input.KeyboardEventType.KEY_PRESS,
            carb_input.KeyboardEventType.KEY_REPEAT,
        )

        if event.input in (carb_input.KeyboardInput.UP, carb_input.KeyboardInput.W):
            pressed["up"] = is_down
        elif event.input in (carb_input.KeyboardInput.DOWN, carb_input.KeyboardInput.S):
            pressed["down"] = is_down
        elif event.input in (carb_input.KeyboardInput.LEFT, carb_input.KeyboardInput.A):
            pressed["left"] = is_down
        elif event.input in (carb_input.KeyboardInput.RIGHT, carb_input.KeyboardInput.D):
            pressed["right"] = is_down
        elif event.input == carb_input.KeyboardInput.E:
            pressed["forward"] = is_down
        elif event.input == carb_input.KeyboardInput.Q:
            pressed["backward"] = is_down
        elif event.input == carb_input.KeyboardInput.J:
            pressed["open"] = is_down
        elif event.input == carb_input.KeyboardInput.K:
            pressed["close"] = is_down
        elif event.input == carb_input.KeyboardInput.R and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append("reset")
        elif event.input == carb_input.KeyboardInput.P and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append("print")
        elif event.input == carb_input.KeyboardInput.ESCAPE and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append("quit")
        return True

    if keyboard is None or input_iface is None:
        raise RuntimeError("Keyboard device unavailable. Please run with a GUI viewport.")

    kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)

    print("[INFO] Sim2real reach reward validation is ready.")
    print("[INFO] Click the viewport first, then use keyboard:")
    print("       W/S or Up/Down: Z +/-")
    print("       A/D or Left/Right: Y +/-")
    print("       E/Q: X +/-")
    print("       J/K: gripper open/close")
    print("       R: reset env, P: print reward once, ESC: quit")
    print(
        f"[INFO] translation_step={float(args_cli.translation_step):.4f} m per step, "
        f"action_units={translation_step_units:.4f}, print_interval_steps={print_interval_steps}"
    )

    step_counter = 0

    try:
        while simulation_app.is_running():
            while command_queue:
                cmd = command_queue.popleft()
                if cmd == "reset":
                    _reset_env(env, args_cli.warmup_steps)
                    step_counter = 0
                elif cmd == "print":
                    _print_reach_metrics(env)
                elif cmd == "quit":
                    return

            dx_dir = (1.0 if pressed["forward"] else 0.0) + (-1.0 if pressed["backward"] else 0.0)
            dy_dir = (-1.0 if pressed["left"] else 0.0) + (1.0 if pressed["right"] else 0.0)
            dz_dir = (1.0 if pressed["up"] else 0.0) + (-1.0 if pressed["down"] else 0.0)
            gripper_dir = (1.0 if pressed["open"] else 0.0) + (-1.0 if pressed["close"] else 0.0)

            actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            actions[:, 0] = dx_dir * translation_step_units
            actions[:, 1] = dy_dir * translation_step_units
            actions[:, 2] = dz_dir * translation_step_units
            actions[:, 3] = gripper_dir

            _step_env(env, actions)

            if step_counter % print_interval_steps == 0:
                _print_reach_metrics(env)

            step_counter += 1
    finally:
        try:
            if kb_sub is not None:
                input_iface.unsubscribe_from_keyboard_events(keyboard, kb_sub)
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
