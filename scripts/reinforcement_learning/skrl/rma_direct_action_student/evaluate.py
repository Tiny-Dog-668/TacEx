"""Evaluate a direct-action Student on the deterministic RMA 10x10 XY grid."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
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
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default=None)
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


def _install_grid_sampler(base_env) -> list[tuple[float, float]]:
    positions = [(float(x), float(y)) for x in torch.linspace(0.45, 0.55, 10).tolist() for y in torch.linspace(-0.05, 0.05, 10).tolist()]
    nominal = (float(base_env.cfg.cube.init_state.pos[0]), float(base_env.cfg.cube.init_state.pos[1]))
    cursor = {"value": 0}
    def sample(self, count: int) -> torch.Tensor:
        result = torch.empty((count, 2), device=self.device)
        for row in range(count):
            x, y = positions[cursor["value"] % len(positions)]
            cursor["value"] += 1
            result[row] = torch.tensor((x - nominal[0], y - nominal[1]), device=self.device)
        return result
    base_env._sample_cube_xy_offsets = types.MethodType(sample, base_env)
    return positions


def main() -> None:
    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_student_checkpoint(checkpoint, device="cpu")
    env_cfg = parse_env_cfg(payload["task"], device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.cube_position_curriculum_force_full_range = True
    validate_live_env_contract(env_cfg, payload["teacher_manifest"])
    env = gym.make(payload["task"], cfg=env_cfg)
    base_env = env.unwrapped
    positions = _install_grid_sampler(base_env)
    student = RMADirectActionVisualStudent().to(base_env.device).eval()
    load_student_model_state(student, payload["model"])
    teacher_state = load_teacher_policy_state(payload["teacher_checkpoint"], base_env.device)
    teacher = RMAXYActorCore().to(base_env.device).eval()
    teacher.load_state_dict(extract_xy_actor_core_state_dict(teacher_state), strict=True)
    observations, _ = env.reset()
    rows: list[dict[str, float | int]] = []
    episode_return = 0.0
    episode_steps = 0
    action_squared_sum = 0.0
    action_count = 0
    student_delta_squared_sum = 0.0
    student_delta_count = 0
    previous_action = None
    try:
        with torch.inference_mode():
            while len(rows) < len(positions):
                obs = observations["policy"]
                action = student(obs["wrist_rgb"], obs["proprio_obs"].float(), obs["action_history"].float())
                oracle = teacher(obs["proprio_obs"].float(), obs["action_history"].float(), obs["rma_cube_xy"].float(), obs["rma_contact_force"].float())
                action_squared_sum += float(torch.sum((action - oracle).square()).item())
                action_count += int(action.numel())
                if previous_action is not None:
                    student_delta_squared_sum += float(torch.sum((action - previous_action).square()).item())
                    student_delta_count += int(action.numel())
                previous_action = action
                observations, rewards, terminated, truncated, _ = env.step(action)
                episode_return += float(rewards[0].item())
                episode_steps += 1
                if not bool(torch.logical_or(terminated, truncated)[0].item()):
                    continue
                x, y = positions[len(rows)]
                success_state = getattr(
                    base_env,
                    "_last_rma_success_terminal",
                    getattr(base_env, "_last_rma_success_nonterminal", None),
                )
                if success_state is None:
                    raise RuntimeError("RMA environment exposes no success diagnostic")
                success = bool(success_state[0].item())
                rows.append({"episode": len(rows), "cube_x": x, "cube_y": y, "success": int(success), "steps": episode_steps, "return": episode_return})
                episode_return = 0.0
                episode_steps = 0
                previous_action = None
    finally:
        env.close()
    output = Path(args.output).expanduser().resolve() if args.output else checkpoint.parent.parent / "metrics" / "direct_action_grid.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "kind": "tacex_rma_direct_action_student_grid_evaluation", "protocol": "fixed_10x10_xy_grid_v1",
        "student_checkpoint": str(checkpoint), "student_checkpoint_sha256": sha256_file(checkpoint),
        "episodes": len(rows), "success_rate": sum(row["success"] for row in rows) / len(rows),
        "mean_return": sum(float(row["return"]) for row in rows) / len(rows),
        "teacher_action_rmse": (action_squared_sum / action_count) ** 0.5,
        "student_action_delta_rmse": (student_delta_squared_sum / student_delta_count) ** 0.5 if student_delta_count else 0.0,
        "episodes_csv": str(csv_path),
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
