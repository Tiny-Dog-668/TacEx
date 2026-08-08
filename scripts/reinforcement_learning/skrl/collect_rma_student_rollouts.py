"""Collect per-episode Real-Alignment RMA Student rollout datasets.

Each ``--run TASK_ID=CHECKPOINT`` replays one fail-closed Student artifact in
its matching task preset.  The resulting NPZ files keep pre-action Student
inputs/predictions aligned with the reward and termination produced by that
action, while optional image streams use ``frame_step_indices`` for alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


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


@dataclass(frozen=True)
class RunSpec:
    """One explicit task/checkpoint pairing requested by ``--run``."""

    task: str
    checkpoint: Path


def parse_run_spec(value: str) -> RunSpec:
    """Parse ``TASK_ID=CHECKPOINT_PATH`` without guessing either component."""
    task, separator, checkpoint = value.partition("=")
    if not separator or not task or not checkpoint:
        raise argparse.ArgumentTypeError(
            "--run must be TASK_ID=CHECKPOINT_PATH, for example "
            "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-v0=/path/student.pt"
        )
    return RunSpec(task=task, checkpoint=Path(checkpoint).expanduser().resolve())


def validate_run_payload(
    run: RunSpec,
    payload: Mapping[str, Any],
    supported_tasks: frozenset[str],
) -> None:
    """Reject unsupported task presets and cross-task Student artifacts."""
    if run.task not in supported_tasks:
        supported = ", ".join(sorted(supported_tasks))
        raise RuntimeError(f"Unsupported RMA Student task {run.task!r}; supported: {supported}")
    checkpoint_task = str(payload.get("task", ""))
    if checkpoint_task != run.task:
        raise RuntimeError(
            "Run task differs from the Student checkpoint task: "
            f"{run.task!r} != {checkpoint_task!r}"
        )


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_episode_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically save one episode without allowing NumPy to change the temp name."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _atomic_csv_dump(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _to_numpy(value: Any) -> np.ndarray:
    """Copy a Tensor-like rollout value to a CPU NumPy array."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _episode_dr_parameters(base_env) -> dict[str, np.ndarray]:
    """Snapshot episode-fixed DR draws. Clean tasks deliberately return no keys."""
    tensor_fields = {
        "dr_brightness": "_dr_brightness",
        "dr_contrast": "_dr_contrast",
        "dr_saturation": "_dr_saturation",
        "dr_gamma": "_dr_gamma",
        "dr_hue_rad": "_dr_hue_rad",
        "dr_white_balance": "_dr_white_balance",
        "dr_noise_std": "_dr_noise_std",
        "dr_blur_mask": "_dr_blur_mask",
        "dr_focal_scale": "_dr_focal_scale",
        "dr_principal_shift_px": "_dr_principal_shift_px",
        "dr_camera_delta_pos_m": "_dr_camera_delta_pos",
        "dr_camera_delta_rpy_rad": "_dr_camera_delta_rpy_rad",
        "dr_camera_position_root_m": "_dr_camera_pos_w",
        "dr_camera_quaternion_wxyz": "_dr_camera_quat_w",
        "dr_plate_color": "_dr_plate_colors",
        "dr_backdrop_color": "_dr_backdrop_colors",
    }
    if not bool(getattr(base_env.cfg, "wrist_visual_randomization_enabled", False)):
        return {}

    result: dict[str, np.ndarray] = {}
    for key, attribute in tensor_fields.items():
        value = getattr(base_env, attribute, None)
        if value is None:
            continue
        if key == "dr_camera_position_root_m":
            value = value - base_env.scene.env_origins
        result[key] = _to_numpy(value[0])
    result["dr_curriculum_scale"] = np.asarray(
        float(getattr(base_env, "_dr_current_curriculum_scale", 1.0)), dtype=np.float32
    )
    result["dr_global_light_intensity"] = np.asarray(
        float(getattr(base_env, "_dr_global_light_intensity", 0.0)), dtype=np.float32
    )
    result["dr_global_light_color_temperature"] = np.asarray(
        float(getattr(base_env, "_dr_global_light_color_temperature", 0.0)), dtype=np.float32
    )
    result["dr_global_light_color"] = np.asarray(
        getattr(base_env, "_dr_global_light_color", (0.0, 0.0, 0.0)), dtype=np.float32
    )
    return result


def _stack(records: Mapping[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack(values, axis=0) for name, values in records.items() if values}


def _snapshot_transition_diagnostics(base_env, reference):
    """Return post-action diagnostics with stable zero defaults for every RMA task."""
    import torch

    zeros = torch.zeros_like(reference)
    false = torch.zeros_like(reference, dtype=torch.bool)
    contact_force = getattr(base_env, "_last_rma_contact_forces", None)
    if contact_force is None:
        contact_force = torch.zeros((reference.shape[0], 2), device=reference.device)
    return {
        "success": getattr(base_env, "_last_rma_success_nonterminal", false),
        "contact_force_n": contact_force,
        "contact_state_post": getattr(base_env, "_last_rma_contact_state", torch.zeros((reference.shape[0], 2), device=reference.device)),
        "table_collision": getattr(base_env, "_last_table_collision", false),
        "table_collision_force_n": getattr(base_env, "_last_table_collision_force", zeros),
        "table_collision_penalty": getattr(base_env, "_last_table_collision_penalty", zeros),
    }


def _load_student_for_rollout(payload: Mapping[str, Any], device, *, use_tactile_contact: bool):
    import torch

    from tacex_tasks.sim2real_grasp.rma_artifacts import load_student_model_state, state_dict_sha256
    from tacex_tasks.sim2real_grasp.rma_models import RMAActorCore, RMAVisualStudent

    student = RMAVisualStudent(
        RMAActorCore(),
        pretrained_backbone=False,
        use_tactile_contact=use_tactile_contact,
    ).to(device).eval()
    load_student_model_state(student, payload["model"])
    if state_dict_sha256(student.vision_encoder.state_dict()) != payload.get(
        "vision_encoder_state_dict_sha256"
    ):
        raise RuntimeError("Student vision encoder hash mismatch")
    if state_dict_sha256(student.actor_core.state_dict()) != payload.get(
        "teacher_actor_state_dict_sha256"
    ):
        raise RuntimeError("Student checkpoint Teacher Actor hash mismatch")
    if use_tactile_contact and state_dict_sha256(student.tactile_contact_head.state_dict()) != payload.get(
        "tactile_contact_head_state_dict_sha256"
    ):
        raise RuntimeError("Student tactile contact head hash mismatch")
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student


def _collect_task(run: RunSpec, args, *, run_index: int) -> Path:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import tacex_tasks  # noqa: F401
    from tacex_tasks.sim2real_grasp.rma_artifacts import (
        RMA_STUDENT_TASKS,
        RMA_STUDENT_CHECKPOINT_VERSION,
        load_student_checkpoint,
        sha256_file,
        validate_live_env_contract,
    )

    print(
        f"[INFO] Validating RMA Student checkpoint for task {run.task}: {run.checkpoint}",
        flush=True,
    )
    payload = load_student_checkpoint(run.checkpoint, device="cpu")
    validate_run_payload(run, payload, RMA_STUDENT_TASKS)
    if int(payload.get("version", -1)) != RMA_STUDENT_CHECKPOINT_VERSION:
        raise RuntimeError("Student checkpoint version changed while loading rollout collector")

    task_seed = int(args.seed) + run_index
    torch.manual_seed(task_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(task_seed)

    env_cfg = parse_env_cfg(run.task, device=args.device, num_envs=1)
    env_cfg.seed = task_seed
    env_cfg.cube_position_curriculum_force_full_range = True
    validate_live_env_contract(env_cfg, payload["teacher_manifest"])
    student_input_contract = payload.get("student_input_contract", {})
    use_tactile_contact = (
        isinstance(student_input_contract, Mapping)
        and student_input_contract.get("contact_observation_source") == "gelsight_tactile_rgb"
    )
    if use_tactile_contact:
        env_cfg.rma_gelsight_tactile_sensor_enabled = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir = Path(args.output_dir).expanduser().resolve() / timestamp / run.task
    task_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"[INFO] Starting {args.episodes} episode(s) for {run.task} "
        f"(seed={task_seed}, frame_stride={args.frame_stride}); output: {task_dir}",
        flush=True,
    )

    env = gym.make(run.task, cfg=env_cfg)
    base_env = env.unwrapped
    device = torch.device(base_env.device)
    student = _load_student_for_rollout(
        payload, device, use_tactile_contact=use_tactile_contact
    )
    episode_rows: list[dict[str, Any]] = []
    try:
        observations, _ = env.reset()
        for episode_index in range(int(args.episodes)):
            print(
                f"[INFO] {run.task}: collecting episode "
                f"{episode_index + 1}/{args.episodes}",
                flush=True,
            )
            records: dict[str, list[np.ndarray]] = {
                "proprio_obs": [],
                "action_history": [],
                "rma_cube_pos": [],
                "rma_contact_state": [],
                "student_predicted_cube_pos": [],
                "student_contact_logits": [],
                "student_contact_probability": [],
                "student_action": [],
                "oracle_action": [],
                "reward": [],
                "terminated": [],
                "truncated": [],
                "success": [],
                "contact_force_n": [],
                "contact_state_post": [],
                "table_collision": [],
                "table_collision_force_n": [],
                "table_collision_penalty": [],
            }
            frames: dict[str, list[np.ndarray]] = {"frame_step_indices": []}
            initial_cube_position = _to_numpy(observations["policy"]["rma_cube_pos"][0])
            episode_dr = _episode_dr_parameters(base_env)

            while True:
                observation = observations["policy"]
                episode_step = len(records["reward"])
                proprio = observation["proprio_obs"].to(torch.float32)
                history = observation["action_history"].to(torch.float32)
                cube_position = observation["rma_cube_pos"].to(torch.float32)
                contact_state = observation["rma_contact_state"].to(torch.float32)
                tactile_left = observation.get("gsmini_left_rgb")
                tactile_right = observation.get("gsmini_right_rgb")

                if episode_step % int(args.frame_stride) == 0:
                    frames["frame_step_indices"].append(np.asarray(episode_step, dtype=np.int32))
                    frames.setdefault("wrist_rgb", []).append(_to_numpy(observation["wrist_rgb"][0]))
                    if tactile_left is not None and tactile_right is not None:
                        frames.setdefault("gsmini_left_rgb", []).append(_to_numpy(tactile_left[0]))
                        frames.setdefault("gsmini_right_rgb", []).append(_to_numpy(tactile_right[0]))

                with torch.inference_mode():
                    normalized_position, contact_logits = student.predict_adaptation(
                        observation["wrist_rgb"], tactile_left, tactile_right
                    )
                    predicted_position = student.actor_core.normalizer.denormalize_position(
                        normalized_position
                    )
                    predicted_contact = torch.sigmoid(contact_logits)
                    student_action = student.action_from_normalized_position(
                        proprio, history, normalized_position, predicted_contact
                    )
                    oracle_action = student.actor_core(
                        proprio, history, cube_position, contact_state
                    )
                    observations, rewards, terminated, truncated, _ = env.step(student_action)

                diagnostics = _snapshot_transition_diagnostics(base_env, rewards)
                records["proprio_obs"].append(_to_numpy(proprio[0]))
                records["action_history"].append(_to_numpy(history[0]))
                records["rma_cube_pos"].append(_to_numpy(cube_position[0]))
                records["rma_contact_state"].append(_to_numpy(contact_state[0]))
                records["student_predicted_cube_pos"].append(_to_numpy(predicted_position[0]))
                records["student_contact_logits"].append(_to_numpy(contact_logits[0]))
                records["student_contact_probability"].append(_to_numpy(predicted_contact[0]))
                records["student_action"].append(_to_numpy(student_action[0]))
                records["oracle_action"].append(_to_numpy(oracle_action[0]))
                records["reward"].append(_to_numpy(rewards[0]))
                records["terminated"].append(_to_numpy(terminated[0]))
                records["truncated"].append(_to_numpy(truncated[0]))
                for name, value in diagnostics.items():
                    records[name].append(_to_numpy(value[0]))

                done = bool(torch.logical_or(terminated, truncated)[0].item())
                if done:
                    break

            arrays = _stack(records)
            arrays.update(_stack(frames))
            arrays.update(episode_dr)
            output = task_dir / f"episode_{episode_index:04d}.npz"
            save_episode_npz(output, arrays)
            action_error = arrays["student_action"] - arrays["oracle_action"]
            episode_summary = {
                "episode": episode_index,
                "steps": int(arrays["reward"].shape[0]),
                "initial_cube_x_m": float(initial_cube_position[0]),
                "initial_cube_y_m": float(initial_cube_position[1]),
                "return": float(np.sum(arrays["reward"])),
                "success_ever": int(np.any(arrays["success"])),
                "table_collision_ever": int(np.any(arrays["table_collision"])),
                "bilateral_contact_fraction": float(
                    np.mean(np.all(arrays["rma_contact_state"] >= 0.5, axis=-1))
                ),
                "student_oracle_action_mse": float(np.mean(np.square(action_error))),
                "frame_count": int(arrays["frame_step_indices"].shape[0]),
                "file": output.name,
            }
            episode_rows.append(episode_summary)
            print(
                f"[INFO] {run.task}: saved {output.name} "
                f"(steps={episode_summary['steps']}, "
                f"return={episode_summary['return']:.4f}, "
                f"success={episode_summary['success_ever']}, "
                f"frames={episode_summary['frame_count']})",
                flush=True,
            )
    finally:
        env.close()

    _atomic_csv_dump(episode_rows, task_dir / "episodes.csv")
    _atomic_json_dump(
        {
            "kind": "tacex_rma_student_rollout_dataset",
            "version": 1,
            "task": run.task,
            "task_seed": task_seed,
            "episodes": int(args.episodes),
            "frame_stride": int(args.frame_stride),
            "policy_frequency_hz": 1.0 / (float(env_cfg.sim.dt) * int(env_cfg.decimation)),
            "student_checkpoint": str(run.checkpoint),
            "student_checkpoint_sha256": sha256_file(run.checkpoint),
            "student_input_contract": student_input_contract,
            "teacher_manifest": payload["teacher_manifest"],
            "transition_alignment": "pre_action_observation_and_predictions; post_action_reward_and_diagnostics",
            "image_alignment": "image arrays index policy steps through frame_step_indices",
            "episode_files": [row["file"] for row in episode_rows],
        },
        task_dir / "manifest.json",
    )
    print(
        f"[INFO] {run.task}: completed {len(episode_rows)} episode(s); "
        "wrote episodes.csv and manifest.json",
        flush=True,
    )
    return task_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        metavar="TASK_ID=CHECKPOINT_PATH",
        help="Repeat for each Student task/checkpoint pair to collect.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output-dir", default="logs/skrl/rma_student_rollouts")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.frame_stride <= 0:
        raise ValueError("--episodes and --frame-stride must be positive")
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        task_dirs = [
            _collect_task(run, args, run_index=index)
            for index, run in enumerate(args.run)
        ]
        for task_dir in task_dirs:
            print(f"[INFO] Saved RMA Student rollout dataset: {task_dir}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
