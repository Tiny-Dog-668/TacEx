"""Fail-closed artifacts for the independent X040-Wide XYZ/no-contact profile."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .rma_direct_action_student.artifacts import (
    load_encoder_initialization_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from .rma_x040_wide_models import (
    RMA_X040_WIDE_DIRECT_STUDENT_MODEL_VERSION,
    RMA_X040_WIDE_MODEL_VERSION,
    RMAX040WideActorCore,
    RMAX040WideDirectActionVisualStudent,
    RMAX040WideObservationNormalizer,
)


RMA_X040_WIDE_TEACHER_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Teacher-v0"
RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Direct-Action-Student-DR-v0"
RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Size-Buckets-Teacher-v0"
)
RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Size-Buckets-Direct-Action-Student-DR-v0"
)
RMA_X040_WIDE_TEACHER_TASKS = (
    RMA_X040_WIDE_TEACHER_TASK,
    RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK,
)
RMA_X040_WIDE_STUDENT_TASKS = (
    RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK,
    RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK,
)
RMA_X040_WIDE_TEACHER_BY_STUDENT_TASK = {
    RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK: RMA_X040_WIDE_TEACHER_TASK,
    RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK: RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK,
}
RMA_X040_WIDE_MANIFEST_FILENAME = "rma_x040_wide_manifest.json"
RMA_X040_WIDE_MANIFEST_VERSION = 1
RMA_X040_WIDE_STUDENT_KIND = "tacex_rma_x040_wide_direct_action_student"
# v3 changes the Student-only base LED from fixed green to independent
# per-environment, per-episode HSV randomization. Older fixed-color policies
# must not be resumed or deployed in the new visual domain.
RMA_X040_WIDE_STUDENT_VERSION = 3
_SIZE_BUCKET_RENDERING_CONTRACT = {
    "selection": "selected_bucket_per_environment_per_reset",
    "inactive_bucket_exclusion": "parked_behind_all_cameras_using_env_x_extent",
    "usd_visibility_mutation": "none",
}
_STATIC_SIZE_BUCKET_RENDERING_CONTRACT = {
    "selection": "fixed_bucket_by_environment",
    "assignment": "env_id_mod_bucket_count",
    "object_count_per_environment": 1,
    "requires_num_envs_multiple_of_bucket_count": True,
    "usd_visibility_mutation": "none",
}


def _checkpoint_run_dir(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.parent.name != "checkpoints":
        raise RuntimeError(f"Expected checkpoint under checkpoints/: {checkpoint}")
    return checkpoint.parent.parent


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _shared_environment_contract(cfg: Any) -> dict[str, Any]:
    hand = cfg.robot.actuators["panda_hand"]
    contract = {
        "profile": "rma_x040_wide_xyz_no_contact_v1",
        "robot_profile": "franka_panda_hand",
        "action_dim": int(cfg.action_space),
        "action_scales": [float(cfg.action_scale)] * 3 + [float(cfg.gripper_width_delta_scale)],
        "gripper_control_mode": str(cfg.gripper_control_mode),
        "gripper_actuator": {"effort_limit_sim": float(hand.effort_limit_sim), "stiffness": float(hand.stiffness), "damping": float(hand.damping)},
        "cube_nominal_position_root_m": [float(value) for value in cfg.cube.init_state.pos],
        "cube_reset_half_range_xy_m": [float(cfg.cube_x_pos_range), float(cfg.cube_y_pos_range)],
        "cube_position_curriculum_enabled": bool(cfg.cube_position_curriculum_enabled),
        "cube_position_curriculum_force_full_range": bool(cfg.cube_position_curriculum_force_full_range),
        "tcp_table_clearance_min_z_m": float(cfg.tcp_table_clearance_min_z_m),
        "tcp_table_clearance_penalty": float(cfg.tcp_table_clearance_penalty),
        "table_top_height_m": float(cfg.plate_top_height_m),
        "success_terminates_episode": bool(cfg.rma_success_terminates_episode),
        "position_frame": str(cfg.rma_position_frame),
        "teacher_actor_feature_dim": int(cfg.rma_actor_feature_dim),
        "teacher_contact_input": "none",
    }
    if hasattr(cfg, "cube_size_buckets_m"):
        sizes = [float(value) for value in cfg.cube_size_buckets_m]
        fixed_assignment = hasattr(cfg, "cube_size_assignment")
        contract.update({
            "profile": (
                "rma_x040_wide_xyz_no_contact_static_size_buckets_v3"
                if fixed_assignment
                else "rma_x040_wide_xyz_no_contact_size_buckets_v2"
            ),
            "cube_size_buckets_m": sizes,
            "cube_size_sampling": str(cfg.cube_size_sampling),
            "cube_center_z_buckets_root_m": [
                float(cfg.plate_top_height_m) + 0.5 * size for size in sizes
            ],
            "arm_joint_reset_noise": {
                "joint_count": 7,
                "distribution": str(cfg.arm_joint_reset_noise_distribution),
                "mean_rad": 0.0,
                "std_rad": float(cfg.arm_joint_reset_noise_std_rad),
                "clip_abs_rad": float(cfg.arm_joint_reset_noise_clip_rad),
                "finger_joint_noise": "none",
                "joint_velocity_reset_rad_s": 0.0,
            },
        })
        if fixed_assignment:
            contract["cube_size_assignment"] = str(cfg.cube_size_assignment)
            contract["cube_size_object_count_per_environment"] = int(
                cfg.cube_size_object_count_per_environment
            )
            contract["cube_size_requires_num_envs_multiple_of_bucket_count"] = bool(
                cfg.cube_size_requires_num_envs_multiple_of_bucket_count
            )
    return contract


def student_environment_contract(cfg: Any) -> dict[str, Any]:
    contract = _shared_environment_contract(cfg)
    contract.update({
        "camera_position_delta_max_m": [float(value) for value in cfg.camera_position_delta_max_m],
        "camera_rotation_delta_max_deg": [float(value) for value in cfg.camera_rotation_delta_max_deg],
        "dr_curriculum_enabled": bool(cfg.dr_curriculum_enabled),
        "wrist_rgb_shape": [224, 224, 3],
        "student_base_led": {
            "subset_path": str(cfg.student_base_led_subset_path),
            "nominal_emissive_color": [float(value) for value in cfg.student_base_led_emissive_color],
            "sampling": "per_environment_per_episode_hsv",
            "hue_deg_range": [float(value) for value in cfg.student_base_led_hue_deg_range],
            "saturation_range": [float(value) for value in cfg.student_base_led_saturation_range],
            "value_range": [float(value) for value in cfg.student_base_led_value_range],
        },
    })
    if hasattr(cfg, "cube_size_buckets_m"):
        if hasattr(cfg, "cube_size_assignment"):
            contract["cube_bucket_rendering"] = {
                **_STATIC_SIZE_BUCKET_RENDERING_CONTRACT,
                "assignment": str(cfg.cube_size_assignment),
                "object_count_per_environment": int(
                    cfg.cube_size_object_count_per_environment
                ),
                "requires_num_envs_multiple_of_bucket_count": bool(
                    cfg.cube_size_requires_num_envs_multiple_of_bucket_count
                ),
            }
        else:
            contract["cube_bucket_rendering"] = {
                **_SIZE_BUCKET_RENDERING_CONTRACT,
                "parking_strategy": str(cfg.cube_bucket_parking_strategy),
                "camera_margin_m": float(cfg.cube_bucket_parking_camera_margin_m),
            }
    return contract


def write_teacher_manifest(
    base_env: Any,
    params_dir: str | Path,
    agent_cfg: Mapping[str, Any],
    *,
    task: str = RMA_X040_WIDE_TEACHER_TASK,
) -> Path:
    if task not in RMA_X040_WIDE_TEACHER_TASKS:
        raise ValueError(f"Unsupported X040-Wide Teacher task: {task}")
    params = Path(params_dir)
    hashes = {name: sha256_file(params / name) for name in ("agent.yaml", "env.yaml")}
    manifest = {
        "kind": "tacex_rma_x040_wide_teacher", "version": RMA_X040_WIDE_MANIFEST_VERSION,
        "task": task, "model_version": RMA_X040_WIDE_MODEL_VERSION,
        "actor_inputs": {"proprio_obs": 15, "action_history": 4, "rma_cube_pos": 3},
        "actor_contract": RMAX040WideActorCore().contract(),
        "normalization": RMAX040WideObservationNormalizer().contract(),
        "environment_contract": _shared_environment_contract(base_env.cfg),
        "run_config_sha256": hashes, "trainer_timesteps": int(agent_cfg["trainer"]["timesteps"]),
    }
    output = params / RMA_X040_WIDE_MANIFEST_FILENAME
    _atomic_json_dump(manifest, output)
    return output


def load_teacher_manifest(
    checkpoint: str | Path,
    *,
    expected_task: str | None = None,
) -> dict[str, Any]:
    run_dir = _checkpoint_run_dir(checkpoint)
    path = run_dir / "params" / RMA_X040_WIDE_MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"X040-Wide Teacher manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "tacex_rma_x040_wide_teacher" or manifest.get("version") != RMA_X040_WIDE_MANIFEST_VERSION:
        raise RuntimeError("Unsupported X040-Wide Teacher manifest")
    if manifest.get("task") not in RMA_X040_WIDE_TEACHER_TASKS or manifest.get("model_version") != RMA_X040_WIDE_MODEL_VERSION:
        raise RuntimeError("X040-Wide Teacher manifest task/model mismatch")
    if expected_task is not None and manifest.get("task") != expected_task:
        raise RuntimeError("X040-Wide Teacher manifest does not match the requested task")
    if manifest.get("actor_contract") != RMAX040WideActorCore().contract() or manifest.get("normalization") != RMAX040WideObservationNormalizer().contract():
        raise RuntimeError("X040-Wide Teacher model contract mismatch")
    for name, expected_hash in manifest.get("run_config_sha256", {}).items():
        candidate = run_dir / "params" / name
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise RuntimeError(f"X040-Wide Teacher run config hash mismatch: {candidate}")
    return manifest


def validate_live_teacher_contract(
    cfg: Any,
    manifest: Mapping[str, Any],
    *,
    allow_static_size_bucket_assignment: bool = False,
) -> None:
    live_contract = _shared_environment_contract(cfg)
    manifest_contract = manifest.get("environment_contract")
    if manifest_contract == live_contract:
        return
    if allow_static_size_bucket_assignment and live_contract.get("profile") == (
        "rma_x040_wide_xyz_no_contact_static_size_buckets_v3"
    ):
        # The Teacher policy contract is unchanged: it observes cube XYZ, not
        # bucket ID. Accept its exact legacy Size-Buckets dynamics contract when
        # only the Student rollout allocation changes from random-per-reset to
        # deterministic balanced env assignment.
        legacy_equivalent = dict(live_contract)
        legacy_equivalent["profile"] = (
            "rma_x040_wide_xyz_no_contact_size_buckets_v2"
        )
        legacy_equivalent["cube_size_sampling"] = (
            "uniform_discrete_per_environment_per_reset"
        )
        legacy_equivalent.pop("cube_size_assignment", None)
        legacy_equivalent.pop("cube_size_object_count_per_environment", None)
        legacy_equivalent.pop(
            "cube_size_requires_num_envs_multiple_of_bucket_count",
            None,
        )
        if manifest_contract == legacy_equivalent:
            return
    raise RuntimeError("Live X040-Wide environment differs from Teacher contract")


def load_teacher_policy_state(checkpoint: str | Path, device: str | torch.device) -> dict[str, torch.Tensor]:
    load_teacher_manifest(checkpoint)
    payload = torch.load(Path(checkpoint).expanduser().resolve(), map_location=device, weights_only=False)
    policy = payload.get("policy") if isinstance(payload, Mapping) else None
    if not isinstance(policy, Mapping):
        raise RuntimeError("X040-Wide Teacher checkpoint has no policy state")
    return dict(policy)


def direct_action_input_contract() -> dict[str, object]:
    return {"wrist_rgb": [224, 224, 3], "proprio_obs": [15], "action_history": [4]}


def make_student_payload(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer, global_step: int,
                         task: str, teacher_checkpoint: str | Path, teacher_manifest: Mapping[str, Any],
                         encoder_init_checkpoint: str | Path, encoder_init_payload: Mapping[str, Any],
                         student_env_contract: Mapping[str, Any], loss: Mapping[str, Any], optimizer_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": RMA_X040_WIDE_STUDENT_KIND, "version": RMA_X040_WIDE_STUDENT_VERSION,
        "model_version": RMA_X040_WIDE_DIRECT_STUDENT_MODEL_VERSION, "task": task,
        "global_step": int(global_step), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "student_input_contract": direct_action_input_contract(), "student_model_contract": model.contract(),
        "normalization": model.normalizer.contract(), "student_environment_contract": dict(student_env_contract),
        "teacher_checkpoint": str(Path(teacher_checkpoint).expanduser().resolve()), "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_manifest": dict(teacher_manifest), "encoder_init_checkpoint": str(Path(encoder_init_checkpoint).expanduser().resolve()),
        "encoder_init_checkpoint_sha256": sha256_file(encoder_init_checkpoint), "encoder_init_task": encoder_init_payload.get("task"),
        "encoder_init_state_dict_sha256": encoder_init_payload.get("vision_encoder_state_dict_sha256"),
        "vision_encoder_state_dict_sha256": state_dict_sha256(model.vision_encoder.state_dict()),
        "loss": dict(loss), "optimizer_config": dict(optimizer_config),
        "training_privileged_inputs": {"rma_cube_pos": [3], "purpose": "Teacher action and visual-position MSE labels only; never runtime input"},
    }


def load_student_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_teacher_checkpoint: str | Path | None = None,
    expected_task: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint).expanduser().resolve(), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != RMA_X040_WIDE_STUDENT_KIND:
        raise RuntimeError("Checkpoint is not an X040-Wide direct-action Student artifact")
    if payload.get("version") != RMA_X040_WIDE_STUDENT_VERSION or payload.get("model_version") != RMA_X040_WIDE_DIRECT_STUDENT_MODEL_VERSION:
        raise RuntimeError("Unsupported X040-Wide direct-action Student version")
    if payload.get("task") not in RMA_X040_WIDE_STUDENT_TASKS or payload.get("student_input_contract") != direct_action_input_contract():
        raise RuntimeError("X040-Wide Student task/input contract mismatch")
    if expected_task is not None and payload.get("task") != expected_task:
        raise RuntimeError("X040-Wide Student checkpoint does not match the requested task")
    if payload.get("task") == RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK:
        rendering_contract = payload.get("student_environment_contract", {}).get(
            "cube_bucket_rendering"
        )
        expected_rendering_contract = {
            **_SIZE_BUCKET_RENDERING_CONTRACT,
            "parking_strategy": "beyond_positive_x_env_extent_behind_all_cameras",
            "camera_margin_m": 0.5,
        }
        if rendering_contract != expected_rendering_contract:
            raise RuntimeError("Size-Buckets Student rendering contract mismatch")
    model = RMAX040WideDirectActionVisualStudent()
    if payload.get("student_model_contract") != model.contract() or payload.get("normalization") != model.normalizer.contract() or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("X040-Wide Student model contract mismatch")
    if payload.get("encoder_init_checkpoint_sha256") is None or payload.get("encoder_init_state_dict_sha256") is None:
        raise RuntimeError("X040-Wide Student encoder initialization provenance missing")
    vision_state = {key[len("vision_encoder."):]: value for key, value in payload["model"].items() if key.startswith("vision_encoder.")}
    if not vision_state or payload.get("vision_encoder_state_dict_sha256") != state_dict_sha256(vision_state):
        raise RuntimeError("X040-Wide Student vision encoder hash mismatch")
    if expected_teacher_checkpoint is not None:
        if payload.get("teacher_checkpoint_sha256") != sha256_file(expected_teacher_checkpoint) or payload.get("teacher_manifest") != load_teacher_manifest(expected_teacher_checkpoint):
            raise RuntimeError("X040-Wide Student was distilled from a different Teacher")
    return payload


def load_student_model_state(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    result = model.load_state_dict(dict(state_dict), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("X040-Wide Student state mismatch")


__all__ = (
    "RMA_X040_WIDE_TEACHER_TASK", "RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK",
    "RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK", "RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK",
    "RMA_X040_WIDE_TEACHER_TASKS", "RMA_X040_WIDE_STUDENT_TASKS", "RMA_X040_WIDE_TEACHER_BY_STUDENT_TASK",
    "RMA_X040_WIDE_MANIFEST_FILENAME", "load_encoder_initialization_checkpoint", "load_teacher_manifest",
    "load_teacher_policy_state", "validate_live_teacher_contract", "student_environment_contract",
    "make_student_payload", "load_student_checkpoint", "load_student_model_state", "sha256_file", "state_dict_sha256",
)
