"""Evaluate Real-Alignment RMA teacher/student on one deterministic 10x10 XY grid."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for name in ("tacex_tasks", "tacex", "tacex_assets", "tacex_uipc"):
        path = repo_root / "source" / name
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_extend_repo_pythonpath()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher_checkpoint", default=None)
parser.add_argument("--student_checkpoint", default=None)
parser.add_argument("--output", default=None, help="Summary JSON path")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--episodes", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.student_checkpoint:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.rma_artifacts import (
    RMA_TEACHER_TASK,
    load_student_checkpoint,
    load_student_model_state,
    load_teacher_manifest,
    load_teacher_policy_state,
    sha256_file,
    state_dict_sha256,
    validate_live_env_contract,
)
from tacex_tasks.sim2real_grasp.rma_models import (
    RMAActorCore,
    RMAVisualStudent,
    extract_actor_core_state_dict,
)


def _grid(episodes: int) -> list[tuple[float, float]]:
    if episodes != 100:
        raise ValueError("The acceptance grid is fixed at exactly 100 episodes")
    xs = torch.linspace(0.45, 0.55, 10).tolist()
    ys = torch.linspace(-0.05, 0.05, 10).tolist()
    return [(float(x), float(y)) for x in xs for y in ys]


def _install_grid_sampler(base_env, positions: list[tuple[float, float]]) -> None:
    nominal_x = float(base_env.cfg.cube.init_state.pos[0])
    nominal_y = float(base_env.cfg.cube.init_state.pos[1])
    cursor = {"value": 0}

    def sample(self, count: int) -> torch.Tensor:
        offsets = torch.empty((count, 2), device=self.device, dtype=torch.float32)
        for row in range(count):
            x, y = positions[cursor["value"] % len(positions)]
            cursor["value"] += 1
            offsets[row, 0] = x - nominal_x
            offsets[row, 1] = y - nominal_y
        return offsets

    base_env._sample_cube_xy_offsets = types.MethodType(sample, base_env)


def _load_actor(teacher_checkpoint: Path, device: torch.device) -> RMAActorCore:
    policy_state = load_teacher_policy_state(teacher_checkpoint, device)
    actor = RMAActorCore().to(device)
    actor.load_state_dict(extract_actor_core_state_dict(policy_state), strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _evaluate(
    role: str,
    teacher_checkpoint: Path,
    student_checkpoint: Path | None,
    positions: list[tuple[float, float]],
) -> tuple[dict, list[dict]]:
    student_payload = None
    if role == "student":
        student_payload = load_student_checkpoint(
            student_checkpoint,
            device="cpu",
            expected_teacher_checkpoint=teacher_checkpoint,
        )
        task = str(student_payload["task"])
    else:
        task = RMA_TEACHER_TASK
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.cube_position_curriculum_force_full_range = True
    teacher_manifest = load_teacher_manifest(teacher_checkpoint)
    validate_live_env_contract(env_cfg, teacher_manifest)
    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    device = torch.device(base_env.device)
    _install_grid_sampler(base_env, positions)
    actor = _load_actor(teacher_checkpoint, device)

    student = None
    if role == "student":
        student = RMAVisualStudent(RMAActorCore(), pretrained_backbone=False).to(device)
        load_student_model_state(student, student_payload["model"])
        student.eval()
        if state_dict_sha256(student.vision_encoder.state_dict()) != student_payload.get(
            "vision_encoder_state_dict_sha256"
        ):
            raise RuntimeError("Student vision encoder hash mismatch")
        if state_dict_sha256(student.actor_core.state_dict()) != student_payload.get(
            "teacher_actor_state_dict_sha256"
        ):
            raise RuntimeError("Student teacher Actor hash mismatch")

    observations, _ = env.reset()
    records: list[dict] = []
    episode_return = 0.0
    episode_steps = 0
    episode_collision = False
    episode_bilateral_contact_steps = 0
    episode_predicted_bilateral_contact_steps = 0
    squared_position_error: list[torch.Tensor] = []
    action_squared_error_sum = 0.0
    action_error_count = 0
    contact_correct = 0
    contact_count = 0
    contact_true_positive = 0
    contact_false_positive = 0
    contact_false_negative = 0
    max_steps = len(positions) * (int(base_env.max_episode_length) + 5)

    try:
        with torch.inference_mode():
            for _ in range(max_steps):
                obs = observations["policy"]
                proprio = obs["proprio_obs"].float()
                history = obs["action_history"].float()
                cube_position = obs["rma_cube_pos"].float()
                contact_state = obs["rma_contact_state"].float()
                teacher_action = actor(proprio, history, cube_position, contact_state)
                episode_bilateral_contact_steps += int(
                    torch.all(contact_state >= 0.5, dim=-1)[0].item()
                )
                if student is None:
                    actions = teacher_action
                else:
                    predicted_normalized, contact_logits = student.predict_adaptation(
                        obs["wrist_rgb"]
                    )
                    predicted_position = student.actor_core.normalizer.denormalize_position(
                        predicted_normalized
                    )
                    predicted_contact = torch.sigmoid(contact_logits)
                    actions = student.action_from_normalized_position(
                        proprio,
                        history,
                        predicted_normalized,
                        predicted_contact,
                    )
                    squared_position_error.append((predicted_position - cube_position).square().cpu())
                    action_squared_error_sum += float(
                        torch.sum((actions - teacher_action).square()).item()
                    )
                    action_error_count += int(actions.numel())
                    predicted_binary = predicted_contact >= 0.5
                    target_binary = contact_state >= 0.5
                    contact_correct += int((predicted_binary == target_binary).sum().item())
                    contact_count += int(target_binary.numel())
                    contact_true_positive += int(
                        torch.logical_and(predicted_binary, target_binary).sum().item()
                    )
                    contact_false_positive += int(
                        torch.logical_and(predicted_binary, ~target_binary).sum().item()
                    )
                    contact_false_negative += int(
                        torch.logical_and(~predicted_binary, target_binary).sum().item()
                    )
                    episode_predicted_bilateral_contact_steps += int(
                        torch.all(predicted_binary, dim=-1)[0].item()
                    )

                observations, rewards, terminated, truncated, _ = env.step(actions)
                episode_return += float(rewards[0].item())
                episode_steps += 1
                collision = getattr(base_env, "_last_table_collision", None)
                if collision is not None:
                    episode_collision |= bool(collision[0].item())
                done = bool(torch.logical_or(terminated, truncated)[0].item())
                if not done:
                    continue

                success_state = getattr(base_env, "_last_rma_success_terminal", None)
                success = bool(success_state[0].item()) if success_state is not None else False
                index = len(records)
                x, y = positions[index]
                records.append(
                    {
                        "episode": index,
                        "cube_x": x,
                        "cube_y": y,
                        "success": int(success),
                        "steps": episode_steps,
                        "return": episode_return,
                        "table_collision": int(episode_collision),
                        "bilateral_contact_fraction": (
                            episode_bilateral_contact_steps / episode_steps
                        ),
                        **(
                            {
                                "predicted_bilateral_contact_fraction": (
                                    episode_predicted_bilateral_contact_steps / episode_steps
                                )
                            }
                            if student is not None
                            else {}
                        ),
                    }
                )
                episode_return = 0.0
                episode_steps = 0
                episode_collision = False
                episode_bilateral_contact_steps = 0
                episode_predicted_bilateral_contact_steps = 0
                if len(records) >= len(positions):
                    break
    finally:
        env.close()

    if len(records) != len(positions):
        raise RuntimeError(f"Evaluation completed only {len(records)}/{len(positions)} episodes")
    success_rate = sum(row["success"] for row in records) / len(records)
    summary = {
        "role": role,
        "episodes": len(records),
        "success_rate": success_rate,
        "mean_return": sum(row["return"] for row in records) / len(records),
        "table_collision_rate": sum(row["table_collision"] for row in records) / len(records),
        "mean_bilateral_contact_fraction": sum(
            row["bilateral_contact_fraction"] for row in records
        ) / len(records),
        "checkpoint": str(teacher_checkpoint if role == "teacher" else student_checkpoint),
        "checkpoint_sha256": sha256_file(
            teacher_checkpoint if role == "teacher" else student_checkpoint
        ),
    }
    if squared_position_error:
        squared = torch.cat(squared_position_error, dim=0)
        axis_rmse = torch.sqrt(squared.mean(dim=0))
        summary.update(
            {
                "position_rmse_xyz_m": axis_rmse.tolist(),
                "position_rmse_3d_m": float(torch.sqrt(squared.sum(dim=-1).mean()).item()),
                "action_rmse": (action_squared_error_sum / action_error_count) ** 0.5,
                "contact_accuracy": contact_correct / contact_count,
                "contact_precision": contact_true_positive
                / max(contact_true_positive + contact_false_positive, 1),
                "contact_recall": contact_true_positive
                / max(contact_true_positive + contact_false_negative, 1),
            }
        )
        precision = summary["contact_precision"]
        recall = summary["contact_recall"]
        summary["contact_f1"] = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        summary["mean_predicted_bilateral_contact_fraction"] = sum(
            row["predicted_bilateral_contact_fraction"] for row in records
        ) / len(records)
    return summary, records


def main() -> None:
    if not args.teacher_checkpoint:
        raise ValueError("--teacher_checkpoint is required")
    teacher_checkpoint = Path(args.teacher_checkpoint).expanduser().resolve()
    student_checkpoint = (
        Path(args.student_checkpoint).expanduser().resolve() if args.student_checkpoint else None
    )
    positions = _grid(args.episodes)
    report: dict[str, object] = {
        "protocol": "fixed_10x10_xy_grid_v1",
        "seed": args.seed,
        "acceptance": {
            "teacher_success_rate_min": 0.80,
            "student_teacher_success_ratio_min": 0.90,
            "student_position_rmse_3d_m_max": 0.015,
        },
    }
    teacher_summary, teacher_rows = _evaluate(
        "teacher", teacher_checkpoint, None, positions
    )
    report["teacher"] = teacher_summary

    output = Path(args.output).expanduser().resolve() if args.output else (
        teacher_checkpoint.parent / "rma_evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".teacher.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=teacher_rows[0].keys())
        writer.writeheader()
        writer.writerows(teacher_rows)

    if student_checkpoint is not None:
        student_summary, student_rows = _evaluate(
            "student", teacher_checkpoint, student_checkpoint, positions
        )
        report["student"] = student_summary
        report["passed"] = bool(
            teacher_summary["success_rate"] >= 0.80
            and student_summary["success_rate"] >= 0.90 * teacher_summary["success_rate"]
            and student_summary["position_rmse_3d_m"] <= 0.015
        )
        student_csv = output.with_suffix(".student.csv")
        with student_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=student_rows[0].keys())
            writer.writeheader()
            writer.writerows(student_rows)
    else:
        report["passed"] = bool(teacher_summary["success_rate"] >= 0.80)

    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[INFO] Saved RMA evaluation: {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
