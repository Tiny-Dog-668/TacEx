"""Fail-closed manifests and checkpoints for Real-Alignment RMA experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .rma_models import RMA_MODEL_VERSION, RMAActorCore, RMAObservationNormalizer


RMA_TEACHER_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0"
RMA_GELSIGHT_TEACHER_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Teacher-v0"
)
RMA_STUDENT_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-v0"
RMA_STUDENT_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0"
RMA_STUDENT_HEATMAP_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-v0"
RMA_STUDENT_HEATMAP_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0"
RMA_GELSIGHT_STUDENT_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-v0"
)
RMA_GELSIGHT_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-DR-v0"
)
RMA_GELSIGHT_STUDENT_HEATMAP_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-v0"
)
RMA_GELSIGHT_STUDENT_HEATMAP_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-DR-v0"
)
RMA_TEACHER_TASKS = frozenset((RMA_TEACHER_TASK, RMA_GELSIGHT_TEACHER_TASK))
RMA_STUDENT_TASKS = frozenset(
    (
        RMA_STUDENT_TASK,
        RMA_STUDENT_DR_TASK,
        RMA_STUDENT_HEATMAP_TASK,
        RMA_STUDENT_HEATMAP_DR_TASK,
        RMA_GELSIGHT_STUDENT_TASK,
        RMA_GELSIGHT_STUDENT_DR_TASK,
        RMA_GELSIGHT_STUDENT_HEATMAP_TASK,
        RMA_GELSIGHT_STUDENT_HEATMAP_DR_TASK,
    )
)
RMA_MANIFEST_FILENAME = "rma_manifest.json"
RMA_TEACHER_MANIFEST_VERSION = 10
RMA_STUDENT_CHECKPOINT_VERSION = 6


def _is_gelsight_task(task: str) -> bool:
    return task in {
        RMA_GELSIGHT_TEACHER_TASK,
        RMA_GELSIGHT_STUDENT_TASK,
        RMA_GELSIGHT_STUDENT_DR_TASK,
        RMA_GELSIGHT_STUDENT_HEATMAP_TASK,
        RMA_GELSIGHT_STUDENT_HEATMAP_DR_TASK,
    }


def _actor_core_for_task(task: str) -> RMAActorCore:
    """Select the FK contract without changing non-GelSight RMA artifacts."""
    if _is_gelsight_task(task):
        from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_models import (
            RMAGelSightActorCore,
        )

        return RMAGelSightActorCore()
    return RMAActorCore()


def _actor_core_for_cfg(cfg: Any) -> RMAActorCore:
    task = str(getattr(cfg, "rma_task_id", RMA_TEACHER_TASK))
    return _actor_core_for_task(task)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_run_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if path.parent.name != "checkpoints":
        raise RuntimeError(f"Expected checkpoint under a checkpoints/ directory: {path}")
    return path.parent.parent


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _env_contract(cfg: Any) -> dict[str, Any]:
    hand_actuator = cfg.robot.actuators["panda_hand"]

    def optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    contract = {
        "robot_profile": str(getattr(cfg, "rma_robot_profile", "franka_panda_hand")),
        "gelsight_enabled": bool(getattr(cfg, "rma_gelsight_enabled", False)),
        "gelsight_sensor_names": [
            str(value) for value in getattr(cfg, "rma_gelsight_sensor_names", ())
        ],
        "gelsight_sensor_prims": [
            str(value) for value in getattr(cfg, "rma_gelsight_sensor_prims", ())
        ],
        "gelsight_actor_observation": str(
            getattr(cfg, "rma_gelsight_actor_observation", "none")
        ),
        "contact_filter_prim_paths": [
            str(value)
            for value in getattr(cfg.rma_cube_contact_sensor, "filter_prim_paths_expr", ())
        ],
        "action_dim": int(cfg.action_space),
        "action_scales": [
            float(cfg.action_scale),
            float(cfg.action_scale),
            float(cfg.action_scale),
            float(cfg.gripper_width_delta_scale),
        ],
        "gripper_control_mode": str(cfg.gripper_control_mode),
        "gripper_actuator": {
            "joint_names_expr": list(hand_actuator.joint_names_expr),
            "effort_limit_sim": optional_float(hand_actuator.effort_limit_sim),
            "velocity_limit_sim": optional_float(hand_actuator.velocity_limit_sim),
            "stiffness": optional_float(hand_actuator.stiffness),
            "damping": optional_float(hand_actuator.damping),
        },
        # The articulated root and the standalone camera have different
        # parents in USD. Record both geometry contracts so a Teacher trained
        # with another base elevation cannot silently supervise a Student.
        "robot_base_world_position_m": [float(value) for value in cfg.robot.init_state.pos],
        "camera_base_position_m": [float(value) for value in cfg.camera_base_position_m],
        "camera_world_position_m": [float(value) for value in cfg.wrist_camera.offset.pos],
        "camera_rotation_wxyz": [float(value) for value in cfg.wrist_camera.offset.rot],
        "camera_convention": str(cfg.wrist_camera.offset.convention),
        "cube_nominal_position": [float(value) for value in cfg.cube.init_state.pos],
        "cube_xy_half_ranges": [float(cfg.cube_x_pos_range), float(cfg.cube_y_pos_range)],
        "position_frame": str(cfg.rma_position_frame),
        "actor_feature_dim": int(cfg.rma_actor_feature_dim),
        "end_effector_position_source": str(cfg.rma_end_effector_position_source),
        "object_pose_components": str(cfg.rma_object_pose_components),
        "contact_components": str(cfg.rma_contact_components),
        "contact_force_threshold_n": float(cfg.rma_contact_force_threshold_n),
        "contact_reward_weight": float(cfg.rma_contact_reward_weight),
        "single_contact_reward_fraction": float(cfg.rma_single_contact_reward_fraction),
        "action_rate_penalty_weight": float(cfg.rma_action_rate_penalty_weight),
        "action_rate_penalty_scales": str(cfg.rma_action_rate_penalty_scales),
        "policy_frequency_hz": 1.0 / (float(cfg.sim.dt) * int(cfg.decimation)),
        "episode_length_s": float(cfg.episode_length_s),
        "success_lift_delta_m": float(cfg.success_lift_delta),
        "success_hold_steps": int(cfg.success_hold_steps),
        "success_terminates_episode": bool(cfg.rma_success_terminates_episode),
    }
    if bool(getattr(cfg, "rma_gelsight_enabled", False)):
        from tacex_tasks.sim2real_gelsight_rma.gelsight_geometry import geometry_contract

        contract["gelsight_geometry"] = geometry_contract()
    return contract


def write_teacher_manifest(base_env: Any, params_dir: str | Path, agent_cfg: Mapping[str, Any]) -> Path:
    params_path = Path(params_dir)
    normalizer = RMAObservationNormalizer()
    config_hashes = {}
    for name in ("agent.yaml", "env.yaml"):
        path = params_path / name
        if not path.is_file():
            raise FileNotFoundError(f"RMA run config is missing: {path}")
        config_hashes[name] = sha256_file(path)
    manifest = {
        "kind": "tacex_rma_teacher",
        "version": RMA_TEACHER_MANIFEST_VERSION,
        "task": str(getattr(base_env.cfg, "rma_task_id", RMA_TEACHER_TASK)),
        "model_version": RMA_MODEL_VERSION,
        "actor_inputs": {
            "proprio_obs": 15,
            "action_history": 4,
            "rma_cube_pos": 3,
            "rma_contact_state": 2,
        },
        "actor_contract": _actor_core_for_cfg(base_env.cfg).contract(),
        "actor_output": {"mean_actions": 4, "transform": "tanh"},
        "normalization": normalizer.contract(),
        "environment_contract": _env_contract(base_env.cfg),
        "run_config_sha256": config_hashes,
        "trainer_timesteps": int(agent_cfg["trainer"]["timesteps"]),
    }
    output = params_path / RMA_MANIFEST_FILENAME
    _atomic_json_dump(manifest, output)
    return output


def load_teacher_manifest(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    run_dir = checkpoint_run_dir(checkpoint_path)
    manifest_path = run_dir / "params" / RMA_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Teacher RMA manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "tacex_rma_teacher" or manifest.get("task") not in RMA_TEACHER_TASKS:
        raise RuntimeError("Checkpoint is not a compatible Real-Alignment RMA teacher")
    if manifest.get("version") != RMA_TEACHER_MANIFEST_VERSION:
        raise RuntimeError(f"Unsupported RMA teacher manifest version: {manifest.get('version')}")
    if manifest.get("model_version") != RMA_MODEL_VERSION:
        raise RuntimeError(f"Unsupported RMA teacher model version: {manifest.get('model_version')}")
    for name, expected_hash in manifest.get("run_config_sha256", {}).items():
        path = run_dir / "params" / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Teacher run config hash mismatch: {path}")
    expected_contract = RMAObservationNormalizer().contract()
    if manifest.get("normalization") != expected_contract:
        raise RuntimeError("Teacher normalization contract differs from the current RMA model")
    if manifest.get("actor_contract") != _actor_core_for_task(str(manifest["task"])).contract():
        raise RuntimeError("Teacher Actor feature/FK contract differs from the current RMA model")
    return manifest


def _normalize_environment_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(contract)
    normalized.setdefault("robot_profile", "franka_panda_hand")
    normalized.setdefault("gelsight_enabled", False)
    normalized.setdefault("gelsight_sensor_names", [])
    normalized.setdefault("gelsight_sensor_prims", [])
    normalized.setdefault("gelsight_actor_observation", "none")
    normalized.setdefault(
        "contact_filter_prim_paths",
        [
            "/World/envs/env_.*/Robot/panda_leftfinger",
            "/World/envs/env_.*/Robot/panda_rightfinger",
        ],
    )
    return normalized


def validate_live_env_contract(cfg: Any, manifest: Mapping[str, Any]) -> None:
    saved_contract = manifest.get("environment_contract")
    if not isinstance(saved_contract, Mapping):
        raise RuntimeError("Teacher manifest has no environment contract")
    if _normalize_environment_contract(saved_contract) != _env_contract(cfg):
        raise RuntimeError("Live RMA environment differs from the teacher environment contract")


def load_teacher_policy_state(checkpoint: str | Path, device: str | torch.device) -> dict[str, torch.Tensor]:
    load_teacher_manifest(checkpoint)
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    policy_state = payload.get("policy") if isinstance(payload, Mapping) else None
    if not isinstance(policy_state, Mapping):
        raise RuntimeError("Teacher checkpoint has no policy state_dict")
    return dict(policy_state)


def load_student_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_teacher_checkpoint: str | Path | None = None,
    expected_task: str | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "tacex_rma_student":
        raise RuntimeError("Checkpoint is not a TacEx RMA student artifact")
    if payload.get("version") != RMA_STUDENT_CHECKPOINT_VERSION:
        raise RuntimeError(f"Unsupported RMA student checkpoint version: {payload.get('version')}")
    if payload.get("task") not in RMA_STUDENT_TASKS:
        raise RuntimeError(f"RMA student task mismatch: {payload.get('task')!r}")
    if expected_task is not None and payload.get("task") != expected_task:
        raise RuntimeError(
            "RMA student checkpoint task differs from the selected Student environment"
        )
    if payload.get("model_version") != RMA_MODEL_VERSION:
        raise RuntimeError(f"RMA student model version mismatch: {payload.get('model_version')}")
    if payload.get("normalization") != RMAObservationNormalizer().contract():
        raise RuntimeError("Student normalization contract differs from the current RMA model")
    task = str(payload["task"])
    if _is_gelsight_task(task):
        teacher_manifest = payload.get("teacher_manifest")
        if not isinstance(teacher_manifest, Mapping) or teacher_manifest.get(
            "actor_contract"
        ) != _actor_core_for_task(task).contract():
            raise RuntimeError("GelSight Student checkpoint uses the obsolete Panda FK contract")
    if not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("Student checkpoint has no model state_dict")
    if expected_teacher_checkpoint is not None:
        expected_hash = sha256_file(expected_teacher_checkpoint)
        if payload.get("teacher_checkpoint_sha256") != expected_hash:
            raise RuntimeError("Student checkpoint was distilled from a different teacher")
    return payload


def load_student_model_state(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Load Student weights while allowing newly added optional head parameters."""
    result = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        key for key in model.state_dict()
        if key.startswith("heatmap_head.") or key.startswith("tactile_contact_head.")
    }
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    if missing - allowed_missing or unexpected:
        raise RuntimeError(
            "RMA Student model state_dict mismatch: "
            f"missing={sorted(missing - allowed_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
