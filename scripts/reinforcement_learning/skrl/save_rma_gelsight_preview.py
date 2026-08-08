"""Save left/right GelSight tactile RGB previews from a GelSight RMA task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for package_root in (
        repo_root / "source" / "tacex_tasks",
        repo_root / "source" / "tacex",
        repo_root / "source" / "tacex_assets",
        repo_root / "source" / "tacex_uipc",
    ):
        value = str(package_root)
        if package_root.is_dir() and value not in sys.path:
            sys.path.insert(0, value)


_extend_repo_pythonpath()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Teacher-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument(
    "--output_dir",
    default="logs/skrl/rma_gelsight_preview",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from PIL import Image

import tacex_tasks  # noqa: F401


def _to_uint8_hwc(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected one HWC/CHW image, got {tuple(image.shape)}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = image.permute(1, 2, 0).contiguous()
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 3)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.dtype.is_floating_point:
        if image.numel() > 0 and image.max().item() <= 1.0 + 1.0e-6:
            image = image * 255.0
        image = image.round().clamp(0, 255).to(torch.uint8)
    else:
        image = image.clamp(0, 255).to(torch.uint8)
    return image


def _save_sensor_frame(sensor, output_dir: Path, name: str) -> Path | None:
    tactile = sensor.data.output.get("tactile_rgb")
    if tactile is None or tactile.numel() == 0:
        return None
    image = _to_uint8_hwc(tactile[0])
    path = output_dir / f"{name}_env000_{image.shape[1]}x{image.shape[0]}.png"
    Image.fromarray(image.numpy()).save(path)
    return path


def main() -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.rma_gelsight_tactile_sensor_enabled = True
    env = gym.make(args.task, cfg=env_cfg)
    try:
        env.reset()
        base_env = env.unwrapped
        zero_actions = torch.zeros(
            (base_env.num_envs, int(base_env.cfg.action_space)),
            device=base_env.device,
        )
        for _ in range(max(0, int(args.steps))):
            env.step(zero_actions)
            if base_env.sim.has_rtx_sensors():
                base_env.sim.render()
            base_env.scene.update(dt=base_env.physics_dt)

        saved = []
        for name in ("gsmini_left", "gsmini_right"):
            sensor = base_env.scene.sensors.get(name)
            if sensor is None:
                raise RuntimeError(f"Missing GelSight sensor: {name}")
            path = _save_sensor_frame(sensor, output_dir, name)
            if path is not None:
                saved.append(path)
        if not saved:
            raise RuntimeError("GelSight sensors were present but no tactile_rgb frame was available")
        for path in saved:
            print(path)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
