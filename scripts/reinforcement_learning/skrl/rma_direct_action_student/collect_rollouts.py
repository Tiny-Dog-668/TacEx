"""Collect deployment-input-only rollouts for a direct-action RMA Student."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--frame_stride", type=int, default=1)
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


def _to_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if args.episodes <= 0 or args.frame_stride <= 0:
        raise ValueError("episodes and frame_stride must be positive")
    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_student_checkpoint(checkpoint, device="cpu")
    env_cfg = parse_env_cfg(payload["task"], device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.cube_position_curriculum_force_full_range = True
    validate_live_env_contract(env_cfg, payload["teacher_manifest"])
    task_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else checkpoint.parent.parent / "rollouts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    task_dir.mkdir(parents=True, exist_ok=False)
    env = gym.make(payload["task"], cfg=env_cfg)
    base_env = env.unwrapped
    student = RMADirectActionVisualStudent().to(base_env.device).eval()
    load_student_model_state(student, payload["model"])
    teacher_state = load_teacher_policy_state(payload["teacher_checkpoint"], base_env.device)
    teacher = RMAXYActorCore().to(base_env.device).eval()
    teacher.load_state_dict(extract_xy_actor_core_state_dict(teacher_state), strict=True)
    summaries: list[dict] = []
    observations, _ = env.reset()
    try:
        for episode in range(args.episodes):
            records: dict[str, list[np.ndarray]] = {name: [] for name in (
                "proprio_obs", "action_history", "student_action", "teacher_action_analysis", "reward",
                "terminated", "truncated", "success", "table_collision",
            )}
            frames: list[np.ndarray] = []
            frame_steps: list[np.ndarray] = []
            while True:
                obs = observations["policy"]
                step = len(records["reward"])
                proprio, history = obs["proprio_obs"].float(), obs["action_history"].float()
                if step % args.frame_stride == 0:
                    frames.append(_to_numpy(obs["wrist_rgb"][0]))
                    frame_steps.append(np.asarray(step, dtype=np.int32))
                with torch.inference_mode():
                    action = student(obs["wrist_rgb"], proprio, history)
                    # Stored only to quantify distillation; its privileged source
                    # tensors are not written to the rollout dataset.
                    teacher_action = teacher(proprio, history, obs["rma_cube_xy"].float(), obs["rma_contact_force"].float())
                    observations, rewards, terminated, truncated, _ = env.step(action)
                records["proprio_obs"].append(_to_numpy(proprio[0]))
                records["action_history"].append(_to_numpy(history[0]))
                records["student_action"].append(_to_numpy(action[0]))
                records["teacher_action_analysis"].append(_to_numpy(teacher_action[0]))
                records["reward"].append(_to_numpy(rewards[0]))
                records["terminated"].append(_to_numpy(terminated[0]))
                records["truncated"].append(_to_numpy(truncated[0]))
                records["success"].append(_to_numpy(getattr(base_env, "_last_rma_success_nonterminal")[0]))
                records["table_collision"].append(_to_numpy(getattr(base_env, "_last_table_collision")[0]))
                if bool(torch.logical_or(terminated, truncated)[0].item()):
                    break
            arrays = {key: np.stack(value) for key, value in records.items()}
            arrays["wrist_rgb"] = np.stack(frames)
            arrays["frame_step_indices"] = np.stack(frame_steps)
            output = task_dir / f"episode_{episode:04d}.npz"
            _save_npz(output, arrays)
            error = arrays["student_action"] - arrays["teacher_action_analysis"]
            summaries.append({
                "episode": episode, "steps": int(arrays["reward"].shape[0]),
                "return": float(arrays["reward"].sum()), "success_ever": int(arrays["success"].any()),
                "table_collision_ever": int(arrays["table_collision"].any()),
                "teacher_action_mse": float(np.mean(np.square(error))), "frame_count": int(arrays["wrist_rgb"].shape[0]),
                "file": output.name,
            })
    finally:
        env.close()
    with (task_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    _save_json(task_dir / "manifest.json", {
        "kind": "tacex_rma_direct_action_student_rollout_dataset", "version": 1,
        "task": payload["task"], "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": sha256_file(checkpoint), "episodes": args.episodes,
        "frame_stride": args.frame_stride, "student_runtime_inputs": payload["student_input_contract"],
        "stored_transition_inputs": ["wrist_rgb", "proprio_obs", "action_history"],
        "excluded_privileged_inputs": ["rma_cube_xy", "rma_contact_force"],
        "teacher_action_analysis": "derived during simulation for action-error analysis; no privileged tensor stored",
        "episode_files": [row["file"] for row in summaries],
    })
    print(f"[INFO] Saved direct-action rollout dataset: {task_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
