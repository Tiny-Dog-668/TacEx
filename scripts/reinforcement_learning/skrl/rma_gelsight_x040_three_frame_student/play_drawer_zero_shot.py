"""Evaluate a paired Drawer Student or an explicitly requested X040 zero-shot policy."""

from __future__ import annotations

import argparse
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
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=1_000)
parser.add_argument("--metrics_interval", type=int, default=100)
parser.add_argument(
    "--x040_zero_shot",
    action="store_true",
    help="Explicitly load an X040 Student instead of a paired Pulled-Drawer checkpoint.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma import (
    rma_gelsight_pulled_drawer_artifacts as drawer_artifacts,
)
from tacex_tasks.sim2real_gelsight_rma import (
    rma_gelsight_x040_three_frame_artifacts as x040_artifacts,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_pulled_drawer_env import (
    GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_pulled_drawer_progress_binary_tactile_env import (
    GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_x040_progress_binary_tactile_env import (
    GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
)
from tacex_tasks.sim2real_gelsight_rma.large_drawer_fusion_runtime import (
    initial_recurrent_state,
    run_student_model,
    update_recurrent_state,
)


def main() -> None:
    if args.num_envs <= 0 or args.num_envs % 8 != 0:
        raise ValueError(
            "Pulled-Drawer requires num_envs to be a positive multiple of 8"
        )
    artifacts = x040_artifacts if args.x040_zero_shot else drawer_artifacts
    payload = artifacts.load_student_checkpoint(args.student_checkpoint, device="cpu")
    if args.x040_zero_shot:
        target_task = (
            GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK
            if payload.get("task")
            == GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK
            else GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK
        )
    else:
        target_task = str(payload["task"])
    env_cfg = parse_env_cfg(
        target_task,
        device=args.device,
        num_envs=args.num_envs,
    )
    env = gym.make(target_task, cfg=env_cfg)
    try:
        device = torch.device(env.unwrapped.device)
        model = artifacts.make_student_model_for_checkpoint(
            payload, pretrained_backbone=False
        ).to(device).eval()
        artifacts.load_student_model_state(model, payload["model"])
        observations, _ = env.reset()
        recurrent_state, reset_mask = initial_recurrent_state(
            target_task, args.num_envs, device
        )
        reward_sum = 0.0
        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                obs = observations["policy"]
                result = run_student_model(
                    model,
                    target_task,
                    obs,
                    recurrent_state=recurrent_state,
                    reset_mask=reset_mask,
                )
                actions = result["action"]
                observations, rewards, terminated, truncated, _ = env.step(actions)
                recurrent_state, reset_mask = update_recurrent_state(
                    result, terminated, truncated
                )
                reward_sum += rewards.mean().item()
                if step % args.metrics_interval == 0 or step == args.steps:
                    metrics = env.unwrapped._episode_success_statistics()
                    print(
                        f"[Drawer play] {step}/{args.steps} "
                        f"mean_reward={reward_sum / step:.3f} "
                        f"contact_prob={None if result['contact_probability'] is None else result['contact_probability'].mean(dim=0).tolist()} "
                        f"cube_position_root_m={result['cube_position_root_m'].mean(dim=0).tolist()} "
                        f"alpha={None if result['alpha'] is None else result['alpha'].mean().item()} "
                        f"success={metrics['cumulative_rate'].item():.3f}",
                        flush=True,
                    )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
