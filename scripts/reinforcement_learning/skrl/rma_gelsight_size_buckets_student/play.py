"""Roll out a fixed-size GelSight reference-delta Student checkpoint."""

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
parser.add_argument(
    "--task",
    default="TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Size-Buckets-Student-DR-v0",
)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--metrics_interval", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_size_buckets_artifacts import (
    load_student_checkpoint,
    load_student_model_state,
    state_dict_sha256,
    validate_live_env_contract,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_size_buckets_models import (
    RMAGelSightReferenceStudent,
)
from tacex_tasks.sim2real_grasp.rma_models import RMAActorCore


def main() -> None:
    if args.num_envs <= 0 or args.num_envs % 8:
        raise ValueError("num_envs must be a positive multiple of 8")
    payload = load_student_checkpoint(
        args.student_checkpoint, device="cpu", expected_task=args.task
    )
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    validate_live_env_contract(env_cfg, payload["teacher_manifest"])
    env = gym.make(args.task, cfg=env_cfg)
    device = torch.device(env.unwrapped.device)
    model = RMAGelSightReferenceStudent(
        RMAActorCore(), pretrained_backbone=False
    ).to(device).eval()
    load_student_model_state(model, payload["model"])
    if state_dict_sha256(model.actor_core.state_dict()) != payload[
        "teacher_actor_state_dict_sha256"
    ]:
        raise RuntimeError("Student Teacher Actor hash mismatch")
    if state_dict_sha256(model.tactile_contact_head.state_dict()) != payload[
        "tactile_contact_head_state_dict_sha256"
    ]:
        raise RuntimeError("Student tactile head hash mismatch")

    observations, _ = env.reset()
    reward_sum = 0.0
    try:
        for step in range(1, args.steps + 1):
            obs = observations["policy"]
            with torch.inference_mode():
                actions = model(
                    obs["wrist_rgb"],
                    obs["proprio_obs"].float(),
                    obs["action_history"].float(),
                    obs["gsmini_left_rgb"],
                    obs["gsmini_right_rgb"],
                    obs["gsmini_left_reference_rgb"],
                    obs["gsmini_right_reference_rgb"],
                )
            observations, rewards, _, _, _ = env.step(actions)
            reward_sum += rewards.mean().item()
            if step % args.metrics_interval == 0 or step == args.steps:
                metrics = env.unwrapped._episode_success_statistics()
                print(
                    f"[GelSight Student play] {step}/{args.steps} "
                    f"mean_reward={reward_sum/step:.3f} "
                    f"success={metrics['cumulative_rate'].item():.3f} "
                    f"({int(metrics['success_count'].item())}/"
                    f"{int(metrics['completed_count'].item())})",
                    flush=True,
                )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
