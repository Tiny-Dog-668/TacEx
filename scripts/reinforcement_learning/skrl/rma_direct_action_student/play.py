"""Replay a three-input direct-action RMA Student and write aggregate metrics."""

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
    root = Path(__file__).resolve().parents[4]
    for name in ("tacex_tasks", "tacex", "tacex_assets", "tacex_uipc"):
        path = root / "source" / name
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_extend_repo_pythonpath()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=3_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--metrics_interval", type=int, default=200)
parser.add_argument("--output_dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.rma_direct_action_student.artifacts import (
    load_student_checkpoint, load_student_model_state, sha256_file, validate_live_env_contract,
)
from tacex_tasks.sim2real_grasp.rma_direct_action_student.models import RMADirectActionVisualStudent
from tacex_tasks.sim2real_grasp.rma_xy_artifacts import load_teacher_policy_state
from tacex_tasks.sim2real_grasp.rma_xy_models import RMAXYActorCore, extract_xy_actor_core_state_dict


def _atomic_json_dump(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if min(args.num_envs, args.steps, args.metrics_interval) <= 0:
        raise ValueError("num_envs, steps and metrics_interval must be positive")
    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_student_checkpoint(checkpoint, device="cpu")
    task = str(payload["task"])
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.cube_position_curriculum_force_full_range = True
    validate_live_env_contract(env_cfg, payload["teacher_manifest"])
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        checkpoint.parent.parent / "metrics" / "direct_action_play"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"play_metrics_{tag}.csv"
    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    student = RMADirectActionVisualStudent().to(base_env.device).eval()
    load_student_model_state(student, payload["model"])
    teacher_state = load_teacher_policy_state(payload["teacher_checkpoint"], base_env.device)
    teacher = RMAXYActorCore().to(base_env.device).eval()
    teacher.load_state_dict(extract_xy_actor_core_state_dict(teacher_state), strict=True)
    observations, _ = env.reset()
    reward_sum = 0.0
    action_mse_sum = 0.0
    action_elements = 0
    previous_actions = None
    action_delta_sum = 0.0
    action_delta_elements = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", "mean_step_reward", "teacher_action_rmse", "student_action_delta_rmse", "cumulative_success_rate", "recent_success_rate", "successful_episodes", "completed_episodes"))
            with torch.inference_mode():
                for step in range(1, args.steps + 1):
                    obs = observations["policy"]
                    actions = student(obs["wrist_rgb"], obs["proprio_obs"].float(), obs["action_history"].float())
                    # Analysis only; teacher labels are not part of the Student call.
                    teacher_actions = teacher(
                        obs["proprio_obs"].float(), obs["action_history"].float(),
                        obs["rma_cube_xy"].float(), obs["rma_contact_force"].float(),
                    )
                    action_mse_sum += float(torch.sum((actions - teacher_actions).square()).item())
                    action_elements += int(actions.numel())
                    observations, rewards, _, _, _ = env.step(actions)
                    reward_sum += float(rewards.mean().item())
                    if previous_actions is not None:
                        action_delta_sum += float(torch.sum((actions - previous_actions).square()).item())
                        action_delta_elements += int(actions.numel())
                    previous_actions = actions
                    if step % args.metrics_interval == 0 or step == args.steps:
                        stats = base_env._episode_success_statistics()
                        delta_rmse = (action_delta_sum / action_delta_elements) ** 0.5 if action_delta_elements else 0.0
                        teacher_rmse = (action_mse_sum / action_elements) ** 0.5
                        writer.writerow((step, reward_sum / step, teacher_rmse, delta_rmse, float(stats["cumulative_rate"].item()), float(stats["window_rate"].item()), int(stats["success_count"].item()), int(stats["completed_count"].item())))
                        stream.flush()
    finally:
        env.close()
    _atomic_json_dump({
        "kind": "tacex_rma_direct_action_student_play", "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": sha256_file(checkpoint), "task": task, "seed": args.seed,
        "num_envs": args.num_envs, "steps": args.steps, "metrics_csv": str(csv_path),
        "student_runtime_inputs": ["wrist_rgb", "proprio_obs", "action_history"],
        "teacher_action_rmse": (action_mse_sum / action_elements) ** 0.5 if action_elements else None,
    }, output_dir / f"play_summary_{tag}.json")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
