"""Keyboard-control Franka around a PhysX soft object and save inner GelSight images.

This script does not load a trained policy. It creates the occluded-grasping
Soft* task, maps keyboard input to the environment's 5-D action
[dx, dy, dz, dyaw, gripper], and saves tactile RGB frames from the two inner
GelSight sensors on demand.

Example:
    python scripts/occluded_grasping/check_softcube_franka_tactile.py \
        --task TacEx-Alpha-GRU-Self-Occlusion-SoftCube \
        --num_envs 1 \
        --enable_cameras
"""

from __future__ import annotations

import argparse
import csv
import sys
import types
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


parser = argparse.ArgumentParser(description="Keyboard-control Franka around a soft object and save tactile images.")
parser.add_argument("--task", type=str, default="TacEx-Alpha-GRU-Self-Occlusion-SoftCube")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_id", type=int, default=0)
parser.add_argument("--steps", type=int, default=0, help="Maximum control steps. Use 0 to run until ESC/window close.")
parser.add_argument("--save_every", type=int, default=0, help="Save tactile PNGs every N steps. Use 0 for manual save only.")
parser.add_argument("--output_dir", type=str, default="logs/softcube_franka_tactile")
parser.add_argument("--translation_step", type=float, default=0.03, help="End-effector translation command per step in meters.")
parser.add_argument("--yaw_step", type=float, default=0.05, help="End-effector yaw command per step in radians.")
parser.add_argument("--print_every", type=int, default=30, help="Print soft metrics every N steps. Use 0 to disable periodic prints.")
parser.add_argument(
    "--allow_done_resets",
    action="store_true",
    default=False,
    help="Allow normal environment done/reset behavior. Default disables timeout and ground-collision resets for teleop.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if getattr(args_cli, "headless", False):
    raise RuntimeError("Keyboard control requires GUI mode. Remove --headless and click the viewport before pressing keys.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.input as carb_input
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import omni.appwindow
import torch
from isaaclab_tasks.utils import parse_env_cfg

_extend_repo_pythonpath()
import tacex_tasks  # noqa: F401, E402


INNER_SENSOR_ATTRS = {
    "left": "gsmini_left",
    "right": "gsmini_right",
}


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _to_uint8_rgb(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.5:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _read_inner_tactile_rgbs(base_env, env_id: int) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for name, attr in INNER_SENSOR_ATTRS.items():
        sensor = getattr(base_env, attr, None)
        if sensor is None:
            continue
        tactile = sensor.data.output.get("tactile_rgb")
        if tactile is None:
            continue
        if isinstance(tactile, torch.Tensor):
            tactile = tactile[env_id]
        else:
            tactile = np.asarray(tactile)[env_id]
        images[name] = _to_uint8_rgb(tactile)
    return images


def _make_inner_grid(images: dict[str, np.ndarray]) -> np.ndarray:
    if "left" not in images or "right" not in images:
        raise RuntimeError("Expected both inner tactile images: left and right.")
    h = max(images["left"].shape[0], images["right"].shape[0])
    w = max(images["left"].shape[1], images["right"].shape[1])
    padded = []
    for name in ("left", "right"):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        img = images[name]
        canvas[: img.shape[0], : img.shape[1]] = img
        padded.append(canvas)
    return np.concatenate(padded, axis=1)


def _save_tactile_images(base_env, output_dir: Path, step: int, env_id: int) -> None:
    images = _read_inner_tactile_rgbs(base_env, env_id)
    if not images:
        print(f"[WARN] step={step}: no inner tactile_rgb outputs yet")
        return
    step_dir = output_dir / f"step_{step:05d}_env_{env_id}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        imageio.imwrite(step_dir / f"tactile_inner_{name}.png", image)
    if "left" in images and "right" in images:
        imageio.imwrite(step_dir / "tactile_inner_left_right_grid.png", _make_inner_grid(images))
    print(f"[SAVE] tactile images -> {step_dir}")


def _soft_metrics(base_env, initial_nodes: torch.Tensor, env_id: int) -> dict[str, float]:
    nodes = base_env._can.data.nodal_pos_w[env_id]
    disp = nodes - initial_nodes
    z_span = nodes[:, 2].max() - nodes[:, 2].min()
    xy_span = torch.linalg.norm(nodes[:, :2].max(dim=0).values - nodes[:, :2].min(dim=0).values)
    return {
        "root_x": float(base_env._can.data.root_pos_w[env_id, 0].detach().cpu()),
        "root_y": float(base_env._can.data.root_pos_w[env_id, 1].detach().cpu()),
        "root_z": float(base_env._can.data.root_pos_w[env_id, 2].detach().cpu()),
        "max_node_displacement": float(torch.linalg.norm(disp, dim=-1).max().detach().cpu()),
        "mean_node_displacement": float(torch.linalg.norm(disp, dim=-1).mean().detach().cpu()),
        "z_span": float(z_span.detach().cpu()),
        "xy_span": float(xy_span.detach().cpu()),
    }


def _print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix} "
        f"root=({metrics['root_x']:.4f}, {metrics['root_y']:.4f}, {metrics['root_z']:.4f}) "
        f"max_disp={metrics['max_node_displacement']:.5f} "
        f"mean_disp={metrics['mean_node_displacement']:.5f} "
        f"z_span={metrics['z_span']:.5f} "
        f"xy_span={metrics['xy_span']:.5f}"
    )


def _print_help(translation_units: float, yaw_units: float) -> None:
    print("[INFO] Soft-object Franka teleop is ready.")
    print("[INFO] Click the Isaac viewport first, then use keyboard:")
    print("       W/S or Up/Down: Z +/-")
    print("       A/D or Left/Right: Y -/+")
    print("       E/Q: X +/-")
    print("       Z/X: yaw +/-")
    print("       J/K: gripper open/close")
    print("       P: save tactile images, M: print metrics, H: help, ESC: quit")
    print(
        f"[INFO] translation_step={args_cli.translation_step:.4f} m "
        f"(action_units={translation_units:.4f}), yaw_step={args_cli.yaw_step:.4f} rad "
        f"(action_units={yaw_units:.4f})"
    )


def _make_keyboard_action(base_env, pressed: dict[str, bool], translation_units: float, yaw_units: float) -> torch.Tensor:
    action_dim = base_env.cfg.action_space
    action = torch.zeros((base_env.num_envs, action_dim), dtype=torch.float32, device=base_env.device)
    action[:, 0] = ((1.0 if pressed["x_plus"] else 0.0) - (1.0 if pressed["x_minus"] else 0.0)) * translation_units
    action[:, 1] = ((1.0 if pressed["y_plus"] else 0.0) - (1.0 if pressed["y_minus"] else 0.0)) * translation_units
    action[:, 2] = ((1.0 if pressed["z_plus"] else 0.0) - (1.0 if pressed["z_minus"] else 0.0)) * translation_units
    action[:, 3] = ((1.0 if pressed["yaw_plus"] else 0.0) - (1.0 if pressed["yaw_minus"] else 0.0)) * yaw_units
    action[:, 4] = (1.0 if pressed["open"] else 0.0) - (1.0 if pressed["close"] else 0.0)
    return action


def _disable_done_resets(base_env) -> None:
    """Patch this teleop instance so env.step() cannot trigger automatic resets."""

    def _never_done(self):
        zeros = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return zeros, zeros

    base_env._get_dones = types.MethodType(_never_done, base_env)


def main() -> None:
    if args_cli.env_id < 0 or args_cli.env_id >= args_cli.num_envs:
        raise ValueError(f"--env_id must be in [0, {args_cli.num_envs - 1}]")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.action_noise_scale = 0.0
    env_cfg.robot_joint_pos_noise = 0.0
    env_cfg.robot_joint_vel_noise = 0.0
    env_cfg.reset_jitter_max_steps = 1
    if args_cli.allow_done_resets and args_cli.steps > 0:
        env_cfg.max_episode_length = max(args_cli.steps + 50, int(getattr(env_cfg, "max_episode_length", 200)))
    else:
        env_cfg.max_episode_length = max(1000000, int(getattr(env_cfg, "max_episode_length", 200)))
    if not args_cli.allow_done_resets:
        env_cfg.ground_height = -1000.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    object_name = args_cli.task.rsplit("-", 1)[-1].lower()
    csv_path = output_dir / f"{object_name}_teleop_metrics.csv"
    kb_sub = None

    try:
        env.reset()
        base_env = _base_env(env)
        if base_env.__class__.__name__:
            print(f"[INFO] env={base_env.__class__.__name__}")
        print(f"[INFO] object type={type(base_env._can)}")
        if not hasattr(base_env._can.data, "nodal_pos_w"):
            raise RuntimeError("The grasp object does not expose nodal_pos_w; check that the task uses a Soft* object.")
        if not args_cli.allow_done_resets:
            _disable_done_resets(base_env)
            print("[INFO] Automatic done/reset is disabled for this teleop session.")

        initial_nodes = base_env._can.data.nodal_pos_w[args_cli.env_id].detach().clone()
        print(f"[INFO] soft node count={initial_nodes.shape[0]}")
        _save_tactile_images(base_env, output_dir, 0, args_cli.env_id)

        action_scale = max(float(getattr(base_env.cfg, "action_scale", 1.0)), 1.0e-6)
        translation_units = max(min(float(args_cli.translation_step) / action_scale, 1.0), 0.0)
        yaw_units = max(min(float(args_cli.yaw_step) / action_scale, 1.0), 0.0)

        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard() if app_window is not None else None
        input_iface = carb_input.acquire_input_interface() if keyboard is not None else None
        if keyboard is None or input_iface is None:
            raise RuntimeError("Keyboard device unavailable. Run with GUI mode and click the viewport.")

        pressed = {
            "x_plus": False,
            "x_minus": False,
            "y_plus": False,
            "y_minus": False,
            "z_plus": False,
            "z_minus": False,
            "yaw_plus": False,
            "yaw_minus": False,
            "open": False,
            "close": False,
        }
        commands: list[str] = []

        def _set_pressed(event_input, is_down: bool) -> bool:
            if event_input in (carb_input.KeyboardInput.E,):
                pressed["x_plus"] = is_down
            elif event_input in (carb_input.KeyboardInput.Q,):
                pressed["x_minus"] = is_down
            elif event_input in (carb_input.KeyboardInput.D, carb_input.KeyboardInput.RIGHT):
                pressed["y_plus"] = is_down
            elif event_input in (carb_input.KeyboardInput.A, carb_input.KeyboardInput.LEFT):
                pressed["y_minus"] = is_down
            elif event_input in (carb_input.KeyboardInput.W, carb_input.KeyboardInput.UP):
                pressed["z_plus"] = is_down
            elif event_input in (carb_input.KeyboardInput.S, carb_input.KeyboardInput.DOWN):
                pressed["z_minus"] = is_down
            elif event_input == carb_input.KeyboardInput.Z:
                pressed["yaw_plus"] = is_down
            elif event_input == carb_input.KeyboardInput.X:
                pressed["yaw_minus"] = is_down
            elif event_input == carb_input.KeyboardInput.J:
                pressed["open"] = is_down
            elif event_input == carb_input.KeyboardInput.K:
                pressed["close"] = is_down
            else:
                return False
            return True

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
            if _set_pressed(event.input, is_down):
                return True
            if event.type != carb_input.KeyboardEventType.KEY_PRESS:
                return True
            if event.input == carb_input.KeyboardInput.P:
                commands.append("save")
            elif event.input == carb_input.KeyboardInput.M:
                commands.append("metrics")
            elif event.input == carb_input.KeyboardInput.H:
                commands.append("help")
            elif event.input == carb_input.KeyboardInput.ESCAPE:
                commands.append("quit")
            return True

        kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)
        _print_help(translation_units, yaw_units)

        with csv_path.open("w", newline="") as f:
            fieldnames = [
                "step",
                "action_x",
                "action_y",
                "action_z",
                "action_yaw",
                "action_gripper",
                "root_x",
                "root_y",
                "root_z",
                "max_node_displacement",
                "mean_node_displacement",
                "z_span",
                "xy_span",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            step = 0
            while simulation_app.is_running():
                if args_cli.steps > 0 and step >= args_cli.steps:
                    break
                step += 1

                should_quit = False
                while commands:
                    command = commands.pop(0)
                    if command == "save":
                        _save_tactile_images(base_env, output_dir, step, args_cli.env_id)
                    elif command == "metrics":
                        _print_metrics(f"[METRICS {step:05d}]", _soft_metrics(base_env, initial_nodes, args_cli.env_id))
                    elif command == "help":
                        _print_help(translation_units, yaw_units)
                    elif command == "quit":
                        should_quit = True
                if should_quit:
                    break

                action = _make_keyboard_action(base_env, pressed, translation_units, yaw_units)
                env.step(action)

                metrics = _soft_metrics(base_env, initial_nodes, args_cli.env_id)
                action_env = action[args_cli.env_id].detach().cpu().tolist()
                writer.writerow(
                    {
                        "step": step,
                        "action_x": action_env[0],
                        "action_y": action_env[1],
                        "action_z": action_env[2],
                        "action_yaw": action_env[3],
                        "action_gripper": action_env[4],
                        **metrics,
                    }
                )

                if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                    _print_metrics(f"[STEP {step:05d}]", metrics)
                if args_cli.save_every > 0 and step % args_cli.save_every == 0:
                    _save_tactile_images(base_env, output_dir, step, args_cli.env_id)

        print(f"[DONE] metrics -> {csv_path}")
        print(f"[DONE] tactile PNGs -> {output_dir}")
    finally:
        if kb_sub is not None:
            try:
                input_iface.unsubscribe_from_keyboard_events(keyboard, kb_sub)
            except Exception:
                pass
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
