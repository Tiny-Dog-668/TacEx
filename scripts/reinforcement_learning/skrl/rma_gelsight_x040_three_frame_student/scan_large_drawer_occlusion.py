"""Scan initial bbox occlusion for the Large Drawer Cylinder comparison task."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    root = Path(__file__).resolve().parents[4]
    for relative in ("source/tacex_tasks", "source/tacex", "source/tacex_assets"):
        value = str(root / relative)
        if value not in sys.path:
            sys.path.insert(0, value)


_extend_repo_pythonpath()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--geometry_seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
parser.add_argument("--x_min", type=float, default=0.4775)
parser.add_argument("--x_max", type=float, default=0.5775)
parser.add_argument("--y_min", type=float, default=-0.05)
parser.add_argument("--y_max", type=float, default=0.05)
parser.add_argument("--step", type=float, default=0.01)
parser.add_argument("--warmup_steps", type=int, default=3)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma.large_drawer_bbox import (
    LargeDrawerBBoxReader,
    enable_target_object_semantics,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_downsample_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
)


def _grid() -> list[tuple[int, float, float]]:
    if args.step <= 0.0:
        raise ValueError("--step must be positive")
    x_count = int(math.floor((args.x_max - args.x_min) / args.step + 1.0e-9)) + 1
    y_count = int(math.floor((args.y_max - args.y_min) / args.step + 1.0e-9)) + 1
    points = [
        (0, round(args.x_min + xi * args.step, 10), round(args.y_min + yi * args.step, 10))
        for xi in range(x_count)
        for yi in range(y_count)
    ]
    return [(index, x, y) for index, (_, x, y) in enumerate(points)]


def _warmup(base_env) -> None:
    for _ in range(max(0, args.warmup_steps)):
        base_env.scene.write_data_to_sim()
        base_env.sim.step(render=True)
        base_env.scene.update(dt=base_env.physics_dt)


def _scan_seed(seed: int, writer: csv.DictWriter, points) -> None:
    cfg = parse_env_cfg(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
        device=args.device,
        num_envs=args.num_envs,
    )
    cfg.seed = int(seed)
    cfg.camera_pose_randomization_enabled = False
    cfg.wrist_visual_randomization_enabled = False
    cfg.light_randomization_enabled = False
    cfg.pulled_drawer_appearance_randomization_enabled = False
    enable_target_object_semantics(cfg)
    env = gym.make(cfg.rma_task_id, cfg=cfg)
    base_env = env.unwrapped
    reader = LargeDrawerBBoxReader(base_env)
    try:
        env.reset()
        _warmup(base_env)
        reader.initialize()
        for start in range(0, len(points), args.num_envs):
            batch = points[start : start + args.num_envs]
            reset_seed = (int(seed) * 1_000_003 + int(start)) % 2_147_483_647 or 1
            env_ids = torch.arange(len(batch), device=base_env.device, dtype=torch.long)
            state = base_env._cube.data.default_root_state[env_ids].clone()
            state[:, 0] = torch.tensor([row[1] for row in batch], device=base_env.device)
            state[:, 1] = torch.tensor([row[2] for row in batch], device=base_env.device)
            state[:, :3] += base_env.scene.env_origins[env_ids]
            base_env._cube.write_root_state_to_sim(state, env_ids=env_ids)
            base_env._cube.write_root_velocity_to_sim(
                torch.zeros((len(batch), 6), device=base_env.device), env_ids=env_ids
            )
            _warmup(base_env)
            occlusion, found = reader.read()
            bucket_ids = base_env._active_cube_bucket_ids
            for local_index, (grid_index, x, y) in enumerate(batch):
                env_id = int(env_ids[local_index].item())
                writer.writerow(
                    {
                        "geometry_seed": seed,
                        "grid_index": grid_index,
                        "x": x,
                        "y": y,
                        "occlusion_ratio": float(occlusion[env_id].item()),
                        "bbox_found": int(found[env_id].item()),
                        "env_id": env_id,
                        "scale_bucket_id": int(bucket_ids[env_id].item()),
                        "reset_seed": reset_seed,
                    }
                )
    finally:
        reader.close()
        env.close()


def main() -> None:
    if args.num_envs <= 0 or args.num_envs % 8 != 0:
        raise ValueError("--num_envs must be a positive multiple of 8")
    points = _grid()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite scan CSV: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "geometry_seed", "grid_index", "x", "y", "occlusion_ratio",
            "bbox_found", "env_id", "scale_bucket_id", "reset_seed",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for seed in args.geometry_seeds:
            _scan_seed(seed, writer, points)
            stream.flush()
            print(f"[INFO] Scanned geometry_seed={seed}", flush=True)
    os.replace(temporary, output)
    print(f"[INFO] Saved Large Drawer occlusion scan: {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
