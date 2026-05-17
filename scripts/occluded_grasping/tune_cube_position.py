from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Keyboard tuner for the occluded_grasping cube position.")
parser.add_argument("--task", type=str, default="TacEx-VT-Drawer-Occlusion-Cube", help="Occluded grasping task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. This tool controls env 0.")
parser.add_argument("--object_step", type=float, default=0.005, help="Object translation step in meters.")
parser.add_argument("--print_every", type=int, default=30, help="Print env 0 cube pose every N sim steps. 0 disables periodic prints.")
parser.add_argument("--no_clamp", action="store_true", help="Do not clamp XY motion to the task reset bounds.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import carb.input as carb_input
import omni.appwindow
from isaaclab_tasks.utils import parse_env_cfg

import tacex_tasks  # noqa: F401


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _object(env):
    base_env = _base_env(env)
    if hasattr(base_env, "_can"):
        return base_env._can
    rigid_objects = getattr(getattr(base_env, "scene", None), "rigid_objects", {})
    if "can" in rigid_objects:
        return rigid_objects["can"]
    raise RuntimeError("Could not find occluded_grasping object 'can' in the environment.")


def _env0_positions(env) -> tuple[torch.Tensor, torch.Tensor]:
    base_env = _base_env(env)
    obj = _object(env)
    world_pos = obj.data.root_pos_w[0].detach().clone()
    local_pos = world_pos - base_env.scene.env_origins[0]
    return world_pos, local_pos


def _xy_bounds(env) -> tuple[float, float, float, float] | None:
    base_env = _base_env(env)
    if hasattr(base_env, "_get_can_reset_xy_bounds"):
        return tuple(float(v) for v in base_env._get_can_reset_xy_bounds())
    return None


def _print_pose(env, prefix: str = "[CUBE]") -> None:
    world_pos, local_pos = _env0_positions(env)
    print(
        f"{prefix} world=({world_pos[0].item():.4f}, {world_pos[1].item():.4f}, {world_pos[2].item():.4f}) "
        f"local=({local_pos[0].item():.4f}, {local_pos[1].item():.4f}, {local_pos[2].item():.4f})"
    )


def _print_bounds(env) -> None:
    bounds = _xy_bounds(env)
    if bounds is None:
        print("[BOUNDS] unavailable")
        return
    x_min, x_max, y_min, y_max = bounds
    print(
        "[BOUNDS] current task reset bounds:\n"
        f"CAN_RANDOM_X_MIN = {x_min:.4f}\n"
        f"CAN_RANDOM_X_MAX = {x_max:.4f}\n"
        f"CAN_RANDOM_Y_MIN = {y_min:.4f}\n"
        f"CAN_RANDOM_Y_MAX = {y_max:.4f}"
    )


def _move_object(env, delta_xyz: tuple[float, float, float], *, clamp_xy: bool) -> None:
    base_env = _base_env(env)
    obj = _object(env)
    env_ids = torch.tensor([0], device=base_env.device, dtype=torch.long)

    root_pose = obj.data.root_state_w[0:1, :7].clone()
    delta = torch.tensor(delta_xyz, device=base_env.device, dtype=root_pose.dtype).view(1, 3)
    root_pose[:, :3] += delta

    local_pos = root_pose[:, :3] - base_env.scene.env_origins[0:1]
    if clamp_xy:
        bounds = _xy_bounds(env)
        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
            local_pos[:, 0] = torch.clamp(local_pos[:, 0], min=x_min, max=x_max)
            local_pos[:, 1] = torch.clamp(local_pos[:, 1], min=y_min, max=y_max)
    root_pose[:, :3] = local_pos + base_env.scene.env_origins[0:1]

    obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=base_env.device), env_ids=env_ids)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = _base_env(env)

    env.reset()
    for _ in range(5):
        base_env.sim.render()
        base_env.scene.update(dt=base_env.physics_dt)

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window is not None else None
    input_iface = carb_input.acquire_input_interface() if keyboard is not None else None
    kb_sub = None
    pressed = {
        "x_plus": False,
        "x_minus": False,
        "y_plus": False,
        "y_minus": False,
        "z_plus": False,
        "z_minus": False,
    }

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
        if event.input == carb_input.KeyboardInput.UP:
            pressed["x_plus"] = is_down
        elif event.input == carb_input.KeyboardInput.DOWN:
            pressed["x_minus"] = is_down
        elif event.input == carb_input.KeyboardInput.LEFT:
            pressed["y_plus"] = is_down
        elif event.input == carb_input.KeyboardInput.RIGHT:
            pressed["y_minus"] = is_down
        elif event.input == carb_input.KeyboardInput.R:
            pressed["z_plus"] = is_down
        elif event.input == carb_input.KeyboardInput.F:
            pressed["z_minus"] = is_down
        elif event.input == carb_input.KeyboardInput.P and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            _print_pose(env, prefix="[CUBE][P]")
        elif event.input == carb_input.KeyboardInput.B and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            _print_bounds(env)
        elif event.input == carb_input.KeyboardInput.H and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            print(_help_text())
        return True

    if keyboard is not None and input_iface is not None:
        kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)
    else:
        print("[WARN] Keyboard unavailable. Run without --headless and click the viewport first.")

    print(_help_text())
    _print_bounds(env)
    _print_pose(env, prefix="[CUBE][INIT]")

    step = 0
    clamp_xy = not args_cli.no_clamp
    try:
        while simulation_app.is_running():
            dx = (1.0 if pressed["x_plus"] else 0.0) - (1.0 if pressed["x_minus"] else 0.0)
            dy = (1.0 if pressed["y_plus"] else 0.0) - (1.0 if pressed["y_minus"] else 0.0)
            dz = (1.0 if pressed["z_plus"] else 0.0) - (1.0 if pressed["z_minus"] else 0.0)
            if dx != 0.0 or dy != 0.0 or dz != 0.0:
                _move_object(
                    env,
                    (dx * args_cli.object_step, dy * args_cli.object_step, dz * args_cli.object_step),
                    clamp_xy=clamp_xy,
                )
                _print_pose(env, prefix=f"[CUBE][MOVE][STEP={step}]")

            base_env.scene.write_data_to_sim()
            base_env.sim.step(render=False)
            base_env.scene.update(dt=base_env.physics_dt)
            base_env.sim.render()

            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                _print_pose(env, prefix=f"[CUBE][STEP={step}]")
            step += 1
    finally:
        if kb_sub is not None and keyboard is not None and input_iface is not None:
            input_iface.unsubscribe_from_keyboard_events(keyboard, kb_sub)
        env.close()


def _help_text() -> str:
    return (
        "[INFO] Occluded cube position tuner\n"
        "       Click the viewport first.\n"
        "       Arrow Up/Down:    cube +X / -X\n"
        "       Arrow Left/Right: cube +Y / -Y\n"
        "       R/F:              cube +Z / -Z\n"
        "       P:                print cube pose\n"
        "       B:                print current reset bounds\n"
        "       H:                print this help"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
