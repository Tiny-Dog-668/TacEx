"""Replay and evaluate a distilled Real-Alignment RMA visual Student.

This script intentionally does not construct a skrl Agent. RMA Student
checkpoints are online-distillation artifacts containing an ``RMAVisualStudent``
state_dict, rather than a PPO policy/value checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
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
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument(
    "--task",
    default=None,
    help="Optional RMA Student task. It must match the task recorded in the checkpoint.",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=3_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--metrics_interval", type=int, default=200)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.rma_artifacts import (
    RMA_STUDENT_TASKS,
    load_student_checkpoint,
    load_student_model_state,
    sha256_file,
    state_dict_sha256,
    validate_live_env_contract,
)
from tacex_tasks.sim2real_grasp.rma_models import RMAActorCore, RMAVisualStudent


def _atomic_json_dump(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _statistics(base_env) -> dict[str, int | float]:
    metrics = base_env._episode_success_statistics()
    return {
        "cumulative_success_rate": float(metrics["cumulative_rate"].item()),
        "recent_success_rate": float(metrics["window_rate"].item()),
        "completed_episodes": int(metrics["completed_count"].item()),
        "successful_episodes": int(metrics["success_count"].item()),
    }


def main() -> None:
    if args.num_envs <= 0 or args.steps <= 0 or args.metrics_interval <= 0:
        raise ValueError("num_envs, steps, and metrics_interval must be positive")

    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_student_checkpoint(checkpoint, device="cpu")
    task = str(payload["task"])
    if task not in RMA_STUDENT_TASKS:
        raise RuntimeError(f"Unsupported RMA Student task: {task!r}")
    if args.task is not None and args.task != task:
        raise RuntimeError(
            f"--task {args.task!r} differs from checkpoint task {task!r}"
        )

    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # Evaluation covers the full Teacher-compatible cube XY distribution.
    env_cfg.cube_position_curriculum_force_full_range = True
    validate_live_env_contract(env_cfg, payload["teacher_manifest"])

    run_dir = checkpoint.parent.parent
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "metrics" / "rma_student_play"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv = output_dir / f"play_metrics_{run_tag}.csv"
    summary_json = output_dir / f"play_summary_{run_tag}.json"

    print(
        "[RMA Student play] start | "
        f"task={task} | envs={args.num_envs} | steps={args.steps} | seed={args.seed} | "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    base_env = env.unwrapped
    device = torch.device(base_env.device)
    student = RMAVisualStudent(RMAActorCore(), pretrained_backbone=False).to(device).eval()
    load_student_model_state(student, payload["model"])
    if state_dict_sha256(student.vision_encoder.state_dict()) != payload.get(
        "vision_encoder_state_dict_sha256"
    ):
        raise RuntimeError("Student checkpoint vision encoder hash mismatch")
    if state_dict_sha256(student.actor_core.state_dict()) != payload.get(
        "teacher_actor_state_dict_sha256"
    ):
        raise RuntimeError("Student checkpoint Teacher Actor hash mismatch")

    if args.video:
        video_length = min(args.video_length, args.steps)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(run_dir / "videos" / "rma_student_play"),
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )

    observations, _ = env.reset()
    reward_sum = 0.0
    final_stats = None
    try:
        with metrics_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "step",
                    "mean_step_reward",
                    "cumulative_success_rate",
                    "recent_success_rate",
                    "successful_episodes",
                    "completed_episodes",
                ]
            )
            for step in range(1, args.steps + 1):
                with torch.inference_mode():
                    obs = observations["policy"]
                    actions = student(
                        obs["wrist_rgb"],
                        obs["proprio_obs"].to(torch.float32),
                        obs["action_history"].to(torch.float32),
                    )
                    observations, rewards, _, _, _ = env.step(actions)
                reward_sum += float(rewards.mean().item())

                if step % args.metrics_interval == 0 or step == args.steps:
                    stats = _statistics(base_env)
                    final_stats = stats
                    mean_step_reward = reward_sum / step
                    writer.writerow(
                        [
                            step,
                            mean_step_reward,
                            stats["cumulative_success_rate"],
                            stats["recent_success_rate"],
                            stats["successful_episodes"],
                            stats["completed_episodes"],
                        ]
                    )
                    file.flush()
                    print(
                        f"[RMA Student play] step={step:,}/{args.steps:,} | "
                        f"mean_reward={mean_step_reward:.3f} | "
                        f"success(cumulative/recent)="
                        f"{stats['cumulative_success_rate']:.3f}/{stats['recent_success_rate']:.3f} "
                        f"({stats['successful_episodes']}/{stats['completed_episodes']} completed)",
                        flush=True,
                    )
    finally:
        if final_stats is None:
            final_stats = _statistics(base_env)
        env.close()

    summary = {
        "kind": "tacex_rma_student_play",
        "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": sha256_file(checkpoint),
        "task": task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "mean_step_reward": reward_sum / args.steps,
        **final_stats,
        "metrics_csv": str(metrics_csv),
    }
    _atomic_json_dump(summary, summary_json)
    print(f"[RMA Student play] summary: {summary_json}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
