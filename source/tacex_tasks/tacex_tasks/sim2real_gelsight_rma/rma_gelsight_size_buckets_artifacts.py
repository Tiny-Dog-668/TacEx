"""Fail-closed artifacts for fixed-size GelSight Teacher/Student experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tacex_tasks.sim2real_grasp.rma_models import (
    RMA_MODEL_VERSION,
    RMAObservationNormalizer,
)

from .gelsight_geometry import geometry_contract
from .rma_gelsight_models import RMAGelSightActorCore
from .sim2real_cube_real_alignment_gelsight_size_buckets_env import (
    GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK,
    GELSIGHT_SIZE_BUCKETS_TEACHER_TASK,
)
from .rma_gelsight_size_buckets_models import gelsight_student_model_contract


MANIFEST_FILENAME = "rma_gelsight_size_buckets_manifest.json"
TEACHER_MANIFEST_VERSION = 4
STUDENT_CHECKPOINT_VERSION = 5
STUDENT_MODEL_VERSION = 2


def student_input_contract() -> dict[str, Any]:
    return {
        "input_order": [
            "wrist_rgb",
            "proprio_obs",
            "action_history",
            "gsmini_left_rgb",
            "gsmini_right_rgb",
            "gsmini_left_reference_rgb",
            "gsmini_right_reference_rgb",
        ],
        "wrist_rgb": [224, 224, 3],
        "tactile_rgb": [96, 128, 3],
        "tactile_delta": "signed_float32_current_minus_reference_div_255",
        "reference_capture": "first_post_reset_frame_per_environment",
        "visual_feature": "resnet18_layer4_global_average_pool_512",
        "tactile_feature": "shared_cnn_256_per_side",
        "actor_fusion_dim": 1043,
        "training_only_labels": [
            "rma_cube_pos",
            "rma_contact_state",
            "projected_cube_center_heatmap",
        ],
    }


def infer_teacher_checkpoint_policy_step(checkpoint: str | Path) -> int:
    """Infer cumulative policy steps from the run offset and ``agent_<step>.pt``."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    stem = checkpoint_path.stem
    prefix = "agent_"
    suffix = stem[len(prefix) :] if stem.startswith(prefix) else ""
    if not suffix.isdigit():
        raise RuntimeError(
            "Cannot infer the GelSight collision curriculum step from checkpoint "
            f"'{checkpoint}'. Use an agent_<step>.pt checkpoint or pass "
            "--gelsight_collision_curriculum_step_offset explicitly."
        )
    local_step = int(suffix)
    if checkpoint_path.parent.name != "checkpoints":
        return local_step
    manifest_path = _run_dir(checkpoint_path) / "params" / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return local_step
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    offset = manifest.get("curriculum_policy_step_offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise RuntimeError(
            f"Invalid curriculum_policy_step_offset in Teacher manifest: {manifest_path}"
        )
    return offset + local_step


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
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _run_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if path.parent.name != "checkpoints":
        raise RuntimeError(f"Expected checkpoint under checkpoints/: {path}")
    return path.parent.parent


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def environment_contract(cfg: Any) -> dict[str, Any]:
    return {
        "profile": "rma_gelsight_fixed_size_buckets_v3",
        "robot_profile": str(cfg.rma_robot_profile),
        "robot_asset_filename": Path(str(cfg.robot.spawn.usd_path)).name,
        "robot_base_world_position_m": [float(value) for value in cfg.robot.init_state.pos],
        "camera_world_position_m": [float(value) for value in cfg.wrist_camera.offset.pos],
        "action_dim": int(cfg.action_space),
        "action_scales": [float(cfg.action_scale)] * 3
        + [float(cfg.gripper_width_delta_scale)],
        "position_frame": str(cfg.rma_position_frame),
        "gelsight_geometry": geometry_contract(),
        "actor_feature_dim": int(cfg.rma_actor_feature_dim),
        "cube_sizes_m": [float(value) for value in cfg.cube_size_buckets_m],
        "cube_size_assignment": str(cfg.cube_size_assignment),
        "cube_count_per_env": int(cfg.cube_size_object_count_per_environment),
        "requires_num_envs_multiple_of_8": bool(
            cfg.cube_size_requires_num_envs_multiple_of_bucket_count
        ),
        "cube_center_z_m": [
            float(cfg.plate_top_height_m) + 0.5 * float(value)
            for value in cfg.cube_size_buckets_m
        ],
        "env_spacing_m": float(cfg.scene.env_spacing),
        "replicate_physics": bool(cfg.scene.replicate_physics),
        "shared_ground_visible": bool(cfg.ground.spawn.visible),
        "gelpad_contact_filters": list(cfg.rma_cube_contact_sensor.filter_prim_paths_expr),
        "contact_force_threshold_n": float(cfg.rma_contact_force_threshold_n),
        "single_contact_reward": 0.5
        * float(cfg.rma_contact_reward_weight)
        * float(cfg.rma_single_contact_reward_fraction),
        "bilateral_contact_reward": float(cfg.rma_contact_reward_weight),
        "cube_illegal_filters": list(
            cfg.cube_illegal_contact_sensor.filter_prim_paths_expr
        ),
        "table_illegal_filters": list(cfg.table_contact_sensor.filter_prim_paths_expr),
        "illegal_collision_reward_name": "illegal_collision",
        "illegal_collision_penalty_threshold_curriculum": {
            "schedule": str(cfg.illegal_collision_curriculum_schedule),
            "start_n": float(cfg.illegal_collision_penalty_threshold_start_n),
            "end_n": float(cfg.illegal_collision_penalty_threshold_end_n),
            "start_step": int(cfg.illegal_collision_curriculum_start_step),
            "end_step": int(cfg.illegal_collision_curriculum_end_step),
        },
        "illegal_collision_penalty": float(cfg.illegal_collision_penalty),
        "illegal_collision_termination_threshold_n": float(
            cfg.illegal_collision_termination_threshold_n
        ),
        "cube_table_contact": "neutral",
        "success_terminates_episode": bool(cfg.rma_success_terminates_episode),
        "policy_frequency_hz": 1.0 / (float(cfg.sim.dt) * int(cfg.decimation)),
        "episode_length_s": float(cfg.episode_length_s),
        "max_episode_length_steps": int(
            round(float(cfg.episode_length_s) / (float(cfg.sim.dt) * int(cfg.decimation)))
        ),
        "timeout_semantics": "truncated_only",
    }


def write_teacher_manifest(
    base_env: Any,
    params_dir: str | Path,
    agent_cfg: Mapping[str, Any],
) -> Path:
    params = Path(params_dir)
    hashes = {}
    for name in ("agent.yaml", "env.yaml"):
        path = params / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Teacher run config: {path}")
        hashes[name] = sha256_file(path)
    manifest = {
        "kind": "tacex_rma_gelsight_size_buckets_teacher",
        "version": TEACHER_MANIFEST_VERSION,
        "task": GELSIGHT_SIZE_BUCKETS_TEACHER_TASK,
        "model_version": RMA_MODEL_VERSION,
        "actor_inputs": {
            "proprio_obs": 15,
            "action_history": 4,
            "rma_cube_pos": 3,
            "rma_contact_state": 2,
        },
        "actor_contract": RMAGelSightActorCore().contract(),
        "normalization": RMAObservationNormalizer().contract(),
        "environment_contract": environment_contract(base_env.cfg),
        "curriculum_policy_step_offset": int(
            base_env.cfg.illegal_collision_curriculum_step_offset
        ),
        "run_config_sha256": hashes,
        "trainer_timesteps": int(agent_cfg["trainer"]["timesteps"]),
    }
    output = params / MANIFEST_FILENAME
    _atomic_json_dump(manifest, output)
    return output


def load_teacher_manifest(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint}")
    run_dir = _run_dir(checkpoint)
    path = run_dir / "params" / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"GelSight Size-Buckets manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "tacex_rma_gelsight_size_buckets_teacher":
        raise RuntimeError("Checkpoint is not a GelSight Size-Buckets Teacher")
    if manifest.get("version") != TEACHER_MANIFEST_VERSION:
        raise RuntimeError("Unsupported GelSight Size-Buckets Teacher manifest version")
    if manifest.get("task") != GELSIGHT_SIZE_BUCKETS_TEACHER_TASK:
        raise RuntimeError("GelSight Size-Buckets Teacher task mismatch")
    if manifest.get("model_version") != RMA_MODEL_VERSION:
        raise RuntimeError("Teacher RMA model version mismatch")
    if manifest.get("actor_contract") != RMAGelSightActorCore().contract():
        raise RuntimeError("Teacher Actor contract mismatch")
    if manifest.get("normalization") != RMAObservationNormalizer().contract():
        raise RuntimeError("Teacher normalization contract mismatch")
    curriculum_offset = manifest.get("curriculum_policy_step_offset", 0)
    if (
        isinstance(curriculum_offset, bool)
        or not isinstance(curriculum_offset, int)
        or curriculum_offset < 0
    ):
        raise RuntimeError("Teacher curriculum policy-step offset is invalid")
    for name, expected in manifest.get("run_config_sha256", {}).items():
        config_path = run_dir / "params" / name
        if not config_path.is_file() or sha256_file(config_path) != expected:
            raise RuntimeError(f"Teacher run config hash mismatch: {config_path}")
    return manifest


def validate_live_env_contract(cfg: Any, manifest: Mapping[str, Any]) -> None:
    if manifest.get("environment_contract") != environment_contract(cfg):
        raise RuntimeError("Live GelSight Size-Buckets environment contract mismatch")


def load_teacher_policy_state(
    checkpoint: str | Path, device: str | torch.device
) -> dict[str, torch.Tensor]:
    load_teacher_manifest(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = payload.get("policy") if isinstance(payload, Mapping) else None
    if not isinstance(policy, Mapping):
        raise RuntimeError("Teacher checkpoint has no policy state_dict")
    return dict(policy)


def load_student_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_task: str | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != (
        "tacex_rma_gelsight_size_buckets_student"
    ):
        raise RuntimeError("Not a GelSight Size-Buckets Student checkpoint")
    if payload.get("version") != STUDENT_CHECKPOINT_VERSION:
        raise RuntimeError("Student checkpoint version mismatch")
    if payload.get("model_version") != STUDENT_MODEL_VERSION:
        raise RuntimeError("Student model version mismatch")
    if payload.get("task") != GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK:
        raise RuntimeError("Student task mismatch")
    if expected_task is not None and payload.get("task") != expected_task:
        raise RuntimeError("Selected task differs from Student checkpoint")
    if payload.get("normalization") != RMAObservationNormalizer().contract():
        raise RuntimeError("Student normalization contract mismatch")
    if payload.get("student_input_contract") != student_input_contract():
        raise RuntimeError("Student runtime input contract mismatch")
    if payload.get("student_model_contract") != gelsight_student_model_contract():
        raise RuntimeError("Student fusion model contract mismatch")
    if not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("Student checkpoint has no model state_dict")
    state = payload["model"]
    state_prefix_hashes = {
        "vision_encoder.": "vision_encoder_state_dict_sha256",
        "tactile_encoder.": "tactile_encoder_state_dict_sha256",
        "action_head.": "action_head_state_dict_sha256",
        "position_head.": "position_head_state_dict_sha256",
        "heatmap_head.": "heatmap_head_state_dict_sha256",
    }
    for prefix, hash_key in state_prefix_hashes.items():
        component = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not component or payload.get(hash_key) != state_dict_sha256(component):
            raise RuntimeError(f"Student component hash mismatch: {prefix[:-1]}")
    if not isinstance(payload.get("teacher_actor_state_dict_sha256"), str):
        raise RuntimeError("Student Teacher Actor provenance is missing")
    return payload


def load_student_model_state(
    model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> None:
    model.load_state_dict(state_dict, strict=True)


__all__ = (
    "GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK",
    "GELSIGHT_SIZE_BUCKETS_TEACHER_TASK",
    "MANIFEST_FILENAME",
    "STUDENT_CHECKPOINT_VERSION",
    "STUDENT_MODEL_VERSION",
    "environment_contract",
    "infer_teacher_checkpoint_policy_step",
    "load_student_checkpoint",
    "load_student_model_state",
    "load_teacher_manifest",
    "load_teacher_policy_state",
    "sha256_file",
    "state_dict_sha256",
    "student_input_contract",
    "validate_live_env_contract",
    "write_teacher_manifest",
)
