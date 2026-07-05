from __future__ import annotations

import argparse
import sys
import traceback
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "source"
    for package_root in (
        source_root / "tacex_tasks",
        source_root / "tacex",
        source_root / "tacex_assets",
        source_root / "tacex_uipc",
    ):
        package_root_str = str(package_root)
        if package_root.is_dir() and package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)


parser = argparse.ArgumentParser(
    description="Keyboard teleoperation for UR10 + Robotiq 2F85 with reward diagnostics."
)
parser.add_argument("--translation_step", type=float, default=0.006, help="End-effector target delta per sim step in m.")
parser.add_argument("--print_interval", type=int, default=10, help="Print diagnostics every N sim steps. 0 disables.")
parser.add_argument("--warmup_steps", type=int, default=10, help="Zero-action settle steps after reset.")
parser.add_argument("--zero_action_noise", action="store_true", default=True, help="Disable action noise for debugging.")
parser.add_argument("--no_zero_action_noise", dest="zero_action_noise", action="store_false")
parser.add_argument("--disable_cameras", action="store_true", help="Do not force camera sensors on.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.disable_cameras:
    args_cli.enable_cameras = True
if getattr(args_cli, "headless", False):
    raise RuntimeError("Keyboard teleoperation requires GUI mode. Remove --headless.")

_extend_repo_pythonpath()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import carb.input as carb_input
import omni.appwindow
import torch

import tacex_tasks  # noqa: F401
from tacex_tasks.direct.ur10_robotiq_pickplace.ur10_robotiq_2f85_third_person_pick_place_env import (
    UR10Robotiq2F85ThirdPersonPickPlaceEnv,
    UR10Robotiq2F85ThirdPersonPickPlaceEnvCfg,
)


def _fmt_vec(vec: torch.Tensor) -> str:
    return f"({vec[0].item():+.4f}, {vec[1].item():+.4f}, {vec[2].item():+.4f})"


def _compute_metrics(env: UR10Robotiq2F85ThirdPersonPickPlaceEnv) -> dict[str, torch.Tensor]:
    terms = env._compute_task_terms()
    reach_reward = 1.0 - torch.tanh(6.0 * terms["grip_to_object_dist"])
    lift_reward = torch.clamp(terms["lift_amount"] / float(env.cfg.lift_height), min=0.0, max=1.0)
    success_reward = terms["success"].float()
    total_reward = (
        float(env.cfg.reward_reach_weight) * reach_reward
        + float(env.cfg.reward_lift_weight) * lift_reward
        + float(env.cfg.reward_success_weight) * success_reward
    )

    grip_body_pos_local = env._robot.data.body_pos_w[:, env._grip_body_ids] - env.scene.env_origins[:, None, :]
    finger_min_z = grip_body_pos_local[:, :, 2].min(dim=1).values
    table_top_z = torch.full_like(finger_min_z, env._table_top_z)
    finger_clearance = finger_min_z - table_top_z

    return {
        **terms,
        "reach_reward": reach_reward,
        "lift_reward": lift_reward,
        "success_reward": success_reward,
        "total_reward": total_reward,
        "finger_min_z": finger_min_z,
        "table_top_z": table_top_z,
        "finger_clearance": finger_clearance,
    }


def _print_metrics(env: UR10Robotiq2F85ThirdPersonPickPlaceEnv, prefix: str = "[teleop]") -> None:
    metrics = _compute_metrics(env)
    env_id = 0
    print(
        f"{prefix} "
        f"reward={metrics['total_reward'][env_id].item():+.4f} "
        f"reach={metrics['reach_reward'][env_id].item():.4f} "
        f"lift={metrics['lift_reward'][env_id].item():.4f} "
        f"success={metrics['success_reward'][env_id].item():.0f} "
        f"dist={metrics['grip_to_object_dist'][env_id].item():.4f}m "
        f"lift_amount={metrics['lift_amount'][env_id].item():+.4f}m "
        f"finger_z={metrics['finger_min_z'][env_id].item():+.4f}m "
        f"table_top_z={metrics['table_top_z'][env_id].item():+.4f}m "
        f"clearance={metrics['finger_clearance'][env_id].item():+.4f}m "
        f"grip={_fmt_vec(metrics['grip_pos'][env_id])} "
        f"object={_fmt_vec(metrics['object_pos'][env_id])} "
        f"gripper_target={env._gripper_targets[env_id, 0].item():.2f}deg"
    )


def _step_env(env: UR10Robotiq2F85ThirdPersonPickPlaceEnv, actions: torch.Tensor) -> None:
    env._pre_physics_step(actions)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env.sim.render()


def _reset_env(env: UR10Robotiq2F85ThirdPersonPickPlaceEnv, warmup_steps: int) -> None:
    env.reset()
    actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    for _ in range(max(int(warmup_steps), 0)):
        _step_env(env, actions)
    _print_metrics(env, prefix="[reset]")


def _help_text(translation_step: float, action_units: float) -> str:
    return (
        "[INFO] UR10 + Robotiq 2F85 teleop reward debugger\n"
        "       Click the viewport first.\n"
        "       W/S:      end-effector target +X / -X\n"
        "       A/D:      end-effector target +Y / -Y\n"
        "       R/F:      end-effector target +Z / -Z\n"
        "       J/K:      gripper open / close\n"
        "       Space:    zero action while held\n"
        "       P:        print diagnostics once\n"
        "       C:        toggle periodic diagnostics\n"
        "       T:        reset environment\n"
        "       H:        print this help\n"
        "       ESC:      quit\n"
        f"       translation_step={translation_step:.4f}m, action_units={action_units:.4f}"
    )


def main() -> None:
    env_cfg = UR10Robotiq2F85ThirdPersonPickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.reward_print_interval = 0
    if args_cli.zero_action_noise:
        env_cfg.ee_action_noise_std = 0.0
        env_cfg.gripper_action_noise_std = 0.0

    env = UR10Robotiq2F85ThirdPersonPickPlaceEnv(env_cfg)
    _reset_env(env, args_cli.warmup_steps)

    action_units = float(args_cli.translation_step) / max(float(env.cfg.ee_action_scale), 1.0e-6)
    action_units = max(min(action_units, 1.0), 0.0)
    print_interval = max(int(args_cli.print_interval), 0)
    periodic_print = print_interval > 0

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window is not None else None
    input_iface = carb_input.acquire_input_interface() if keyboard is not None else None
    if keyboard is None or input_iface is None:
        raise RuntimeError("Keyboard unavailable. Run in GUI mode and click the viewport first.")

    pressed = {
        "x_plus": False,
        "x_minus": False,
        "y_plus": False,
        "y_minus": False,
        "z_plus": False,
        "z_minus": False,
        "open": False,
        "close": False,
        "stop": False,
    }
    commands: deque[str] = deque()

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
        is_press = event.type == carb_input.KeyboardEventType.KEY_PRESS

        if event.input in (carb_input.KeyboardInput.W, carb_input.KeyboardInput.UP):
            pressed["x_plus"] = is_down
        elif event.input in (carb_input.KeyboardInput.S, carb_input.KeyboardInput.DOWN):
            pressed["x_minus"] = is_down
        elif event.input in (carb_input.KeyboardInput.D, carb_input.KeyboardInput.RIGHT):
            pressed["y_plus"] = is_down
        elif event.input in (carb_input.KeyboardInput.A, carb_input.KeyboardInput.LEFT):
            pressed["y_minus"] = is_down
        elif event.input == carb_input.KeyboardInput.R:
            pressed["z_plus"] = is_down
        elif event.input == carb_input.KeyboardInput.F:
            pressed["z_minus"] = is_down
        elif event.input == carb_input.KeyboardInput.J:
            pressed["open"] = is_down
        elif event.input == carb_input.KeyboardInput.K:
            pressed["close"] = is_down
        elif event.input == carb_input.KeyboardInput.SPACE:
            pressed["stop"] = is_down
        elif event.input == carb_input.KeyboardInput.P and is_press:
            commands.append("print")
        elif event.input == carb_input.KeyboardInput.C and is_press:
            commands.append("toggle_print")
        elif event.input == carb_input.KeyboardInput.T and is_press:
            commands.append("reset")
        elif event.input == carb_input.KeyboardInput.H and is_press:
            commands.append("help")
        elif event.input == carb_input.KeyboardInput.ESCAPE and is_press:
            commands.append("quit")
        return True

    kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)

    print(_help_text(float(args_cli.translation_step), action_units))
    print(
        f"[INFO] object_spawn_center={env.cfg.object_spawn_center}, table_pos={env.cfg.table_pos}, "
        f"table_size={env.cfg.table_size}, lift_height={env.cfg.lift_height}, "
        f"reward_weights=(reach={env.cfg.reward_reach_weight}, lift={env.cfg.reward_lift_weight}, "
        f"success={env.cfg.reward_success_weight})"
    )

    step = 0
    try:
        while simulation_app.is_running():
            while commands:
                cmd = commands.popleft()
                if cmd == "print":
                    _print_metrics(env, prefix=f"[manual][step={step}]")
                elif cmd == "toggle_print":
                    periodic_print = not periodic_print
                    print(f"[INFO] periodic diagnostics: {'on' if periodic_print else 'off'}")
                elif cmd == "reset":
                    _reset_env(env, args_cli.warmup_steps)
                    step = 0
                elif cmd == "help":
                    print(_help_text(float(args_cli.translation_step), action_units))
                elif cmd == "quit":
                    return

            actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            if not pressed["stop"]:
                actions[:, 0] = ((1.0 if pressed["x_plus"] else 0.0) - (1.0 if pressed["x_minus"] else 0.0)) * action_units
                actions[:, 1] = ((1.0 if pressed["y_plus"] else 0.0) - (1.0 if pressed["y_minus"] else 0.0)) * action_units
                actions[:, 2] = ((1.0 if pressed["z_plus"] else 0.0) - (1.0 if pressed["z_minus"] else 0.0)) * action_units
                actions[:, 3] = (1.0 if pressed["open"] else 0.0) - (1.0 if pressed["close"] else 0.0)

            _step_env(env, actions)

            if periodic_print and print_interval > 0 and step % print_interval == 0:
                _print_metrics(env, prefix=f"[step={step}]")
            step += 1
    finally:
        try:
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
