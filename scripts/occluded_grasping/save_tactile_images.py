"""Save four GelSight tactile RGB images from an occluded grasping environment."""

from __future__ import annotations

import argparse
import sys
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


parser = argparse.ArgumentParser(description="Save four tactile_rgb PNGs from an occluded grasping task.")
parser.add_argument(
    "--task",
    type=str,
    default="TacEx-Dual-Cross-Alpha-Downsample-Drawer-Occlusion-Cuboid",
    help="Task ID to instantiate.",
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to save.")
parser.add_argument("--steps", type=int, default=5, help="Number of zero-action steps before saving.")
parser.add_argument("--save_every", type=int, default=1, help="Save every N steps after reset.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/tactile_images",
    help="Output directory for PNG files.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

_extend_repo_pythonpath()
import tacex_tasks  # noqa: F401, E402


SENSOR_ATTRS = {
    "left": "gsmini_left",
    "right": "gsmini_right",
    "left_down": "gsmini_left_down",
    "right_down": "gsmini_right_down",
}


def _to_uint8_rgb(image) -> np.ndarray:
    if image is None:
        raise ValueError("Cannot save None tactile image")
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


def _get_unwrapped_env(env):
    return getattr(env, "unwrapped", env)


def _read_tactile_rgbs(env_obj, env_id: int) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for name, attr in SENSOR_ATTRS.items():
        sensor = getattr(env_obj, attr, None)
        if sensor is None:
            continue
        tactile = sensor.data.output.get("tactile_rgb")
        if tactile is None:
            continue
        if isinstance(tactile, torch.Tensor):
            if tactile.shape[0] <= env_id:
                raise IndexError(f"env_id={env_id} out of range for {attr} tactile_rgb shape={tuple(tactile.shape)}")
            tactile = tactile[env_id]
        else:
            tactile = np.asarray(tactile)[env_id]
        images[name] = _to_uint8_rgb(tactile)
    return images


def _make_grid(images: dict[str, np.ndarray]) -> np.ndarray:
    order = ("left", "right", "left_down", "right_down")
    present = [key for key in order if key in images]
    if not present:
        raise ValueError("No tactile images available for grid")
    h = max(images[key].shape[0] for key in present)
    w = max(images[key].shape[1] for key in present)

    padded = {}
    for key in present:
        img = images[key]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        padded[key] = canvas

    blank = np.zeros((h, w, 3), dtype=np.uint8)
    left = padded.get("left", blank)
    right = padded.get("right", blank)
    left_down = padded.get("left_down", blank)
    right_down = padded.get("right_down", blank)
    return np.concatenate(
        [
            np.concatenate([left, right], axis=1),
            np.concatenate([left_down, right_down], axis=1),
        ],
        axis=0,
    )


def _save_images(images: dict[str, np.ndarray], output_dir: Path, step: int, env_id: int) -> None:
    step_dir = output_dir / f"step_{step:05d}_env_{env_id}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        imageio.imwrite(step_dir / f"tactile_{name}_rgb.png", image)
    imageio.imwrite(step_dir / "tactile_rgb_grid.png", _make_grid(images))
    print(f"saved {len(images)} tactile images -> {step_dir}")


def main() -> None:
    if args_cli.env_id < 0 or args_cli.env_id >= args_cli.num_envs:
        raise ValueError(f"--env_id must be in [0, {args_cli.num_envs - 1}]")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    output_dir = Path(args_cli.output_dir)
    try:
        env.reset()
        env_obj = _get_unwrapped_env(env)
        device = getattr(env_obj, "device", args_cli.device)
        action_dim = env.action_space.shape[0]
        zero_action = torch.zeros((args_cli.num_envs, action_dim), dtype=torch.float32, device=device)

        for step in range(args_cli.steps + 1):
            if step > 0:
                env.step(zero_action)
            if step % max(1, args_cli.save_every) != 0:
                continue
            images = _read_tactile_rgbs(env_obj, args_cli.env_id)
            if not images:
                raise RuntimeError("No tactile_rgb outputs found on GelSight sensors")
            _save_images(images, output_dir, step, args_cli.env_id)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
