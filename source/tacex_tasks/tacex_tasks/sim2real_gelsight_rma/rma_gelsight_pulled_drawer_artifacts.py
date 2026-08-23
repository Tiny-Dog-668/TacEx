"""Fail-closed Teacher/Student artifacts for the Pulled-Drawer profile."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .rma_gelsight_pulled_drawer_models import (
    PULLED_DRAWER_STUDENT_MODEL_VERSION,
    PULLED_DRAWER_TEACHER_MODEL_VERSION,
    RMAGelSightPulledDrawerActorCore,
    RMAGelSightPulledDrawerObservationNormalizer,
    RMAGelSightPulledDrawerThreeFrameStudent,
    pulled_drawer_student_model_contract,
)
from .rma_gelsight_x040_three_frame_artifacts import (
    load_encoder_initialization_checkpoint,
    sha256_file,
    state_dict_sha256,
    student_input_contract,
)
from tacex_tasks.sim2real_grasp.rma_xy_artifacts import (
    RMA_XY_STUDENT_HEATMAP_DR_TASK,
)
from .sim2real_cube_real_alignment_gelsight_pulled_drawer_env import (
    GELSIGHT_PULLED_DRAWER_TEACHER_TASK,
    GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
    pulled_drawer_geometry_contract,
)


MANIFEST_FILENAME = "rma_gelsight_pulled_drawer_manifest.json"
TEACHER_KIND = "tacex_rma_gelsight_pulled_drawer_teacher"
TEACHER_MANIFEST_VERSION = 9
STUDENT_KIND = "tacex_rma_gelsight_pulled_drawer_three_frame_student"
STUDENT_CHECKPOINT_VERSION = 9


def _run_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if path.parent.name != "checkpoints":
        raise RuntimeError(f"Checkpoint must be below a checkpoints directory: {path}")
    return path.parent.parent


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cfg_seed(cfg: Any) -> int:
    return int(getattr(cfg, "seed", 42) or 42)


def teacher_environment_contract(cfg: Any) -> dict[str, Any]:
    hand = cfg.robot.actuators["panda_hand"]
    return {
        "profile": "rma_gelsight_pulled_drawer_contiguous_rigid_tray_v9",
        "task_family": "paired_teacher_student",
        "robot_profile": str(cfg.rma_robot_profile),
        "robot_asset": {
            "profile": str(cfg.pulled_drawer_robot_asset_profile),
            "filename": Path(str(cfg.robot.spawn.usd_path)).name,
            "panda_hand_to_finger_gelsight_extension_local_z_m": float(
                cfg.pulled_drawer_finger_extension_local_z_m
            ),
        },
        "robot_base_world_position_m": [float(value) for value in cfg.robot.init_state.pos],
        "camera_world_position_m": [float(value) for value in cfg.wrist_camera.offset.pos],
        "action_dim": int(cfg.action_space),
        "action_scales": [float(cfg.action_scale)] * 3
        + [float(cfg.gripper_width_delta_scale)],
        "gripper_actuator": {
            "effort_limit_sim": float(hand.effort_limit_sim),
            "stiffness": float(hand.stiffness),
            "damping": float(hand.damping),
        },
        "teacher_actor_inputs": {
            "proprio_obs": 15,
            "action_history": 4,
            "rma_cube_pos": 3,
            "rma_contact_state": 2,
        },
        "position_frame": str(cfg.rma_position_frame),
        "cube_reset_half_range_xy_m": [
            float(cfg.cube_x_pos_range),
            float(cfg.cube_y_pos_range),
        ],
        "cube_position_curriculum_enabled": bool(cfg.cube_position_curriculum_enabled),
        "cube_size_buckets_m": [float(value) for value in cfg.cube_size_buckets_m],
        "cube_size_assignment": "seeded_balanced_permutation_per_group_of_8",
        "cube_size_object_count_per_environment": int(cfg.cube_size_object_count_per_environment),
        "physics_layout": {
            "gpu_dynamics": bool(cfg.enable_gpu_dynamics),
            "replicate_physics": bool(cfg.scene.replicate_physics),
            "env_spacing_m": float(cfg.scene.env_spacing),
            "robot_articulation_solver": {
                "position_iterations": int(
                    cfg.robot.spawn.articulation_props.solver_position_iteration_count
                ),
                "velocity_iterations": int(
                    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count
                ),
            },
        },
        "geometry": pulled_drawer_geometry_contract(),
        "geometry_seed": _cfg_seed(cfg),
        "gelpad_contact_filters": list(cfg.rma_cube_contact_sensor.filter_prim_paths_expr),
        "cube_illegal_filters": list(cfg.cube_illegal_contact_sensor.filter_prim_paths_expr),
        "surface_robot_collision_filters": {
            name: list(sensor_cfg.filter_prim_paths_expr)
            for name, sensor_cfg in cfg.pulled_drawer_surface_contact_sensors.items()
        },
        "illegal_collision_scope": str(cfg.pulled_drawer_collision_scope),
        "illegal_collision_penalty": float(cfg.illegal_collision_penalty),
        "illegal_collision_penalty_threshold_n": {
            "schedule": "linear_clamped_global_policy_step",
            "start_n": float(cfg.illegal_collision_penalty_threshold_start_n),
            "end_n": float(cfg.illegal_collision_penalty_threshold_end_n),
            "start_step": int(cfg.illegal_collision_curriculum_start_step),
            "end_step": int(cfg.illegal_collision_curriculum_end_step),
        },
        "illegal_collision_termination_threshold_n": {
            "schedule": "linear_clamped_global_policy_step",
            "start_n": float(cfg.illegal_collision_termination_threshold_start_n),
            "end_n": float(cfg.illegal_collision_termination_threshold_end_n),
            "start_step": int(cfg.illegal_collision_curriculum_start_step),
            "end_step": int(cfg.illegal_collision_curriculum_end_step),
        },
        "success_terminates_episode": bool(cfg.rma_success_terminates_episode),
        "timeout_semantics": "truncated_only",
    }


def student_environment_contract(cfg: Any) -> dict[str, Any]:
    contract = teacher_environment_contract(cfg)
    contract.update(
        {
            "task": GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
            "num_envs": int(cfg.scene.num_envs),
            "camera_position_delta_max_m": [
                float(value) for value in cfg.camera_position_delta_max_m
            ],
            "camera_rotation_delta_max_deg": [
                float(value) for value in cfg.camera_rotation_delta_max_deg
            ],
            "dr_curriculum_enabled": bool(cfg.dr_curriculum_enabled),
            "wrist_rgb_history": {
                "shape": [3, 224, 224, 3],
                "dtype": "uint8",
                "order": str(cfg.wrist_rgb_history_order),
                "stride_policy_steps": int(cfg.wrist_rgb_history_stride_policy_steps),
                "reset_fill": str(cfg.wrist_rgb_history_reset_fill),
            },
            "gelsight_reference": "first_post_reset_frame_per_environment",
            "appearance_randomization_enabled": bool(
                cfg.pulled_drawer_appearance_randomization_enabled
            ),
            "opacity_randomization_enabled": bool(
                cfg.pulled_drawer_opacity_randomization_enabled
            ),
            "tray_opacity": float(cfg.pulled_drawer_tray_opacity),
        }
    )
    return contract


def geometry_instance_sha256(base_env: Any) -> str:
    layout = base_env._pulled_drawer_layout
    serializable = {
        "seed": _cfg_seed(base_env.cfg),
        "num_envs": int(base_env.num_envs),
        "cube_bucket_ids": base_env._active_cube_bucket_ids.detach().cpu().tolist(),
        **{
            key: value.detach().cpu().tolist()
            for key, value in sorted(layout.items())
        },
    }
    encoded = json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "kind": TEACHER_KIND,
        "version": TEACHER_MANIFEST_VERSION,
        "task": GELSIGHT_PULLED_DRAWER_TEACHER_TASK,
        "model_version": PULLED_DRAWER_TEACHER_MODEL_VERSION,
        "actor_contract": RMAGelSightPulledDrawerActorCore().contract(),
        "normalization": RMAGelSightPulledDrawerObservationNormalizer().contract(),
        "environment_contract": teacher_environment_contract(base_env.cfg),
        "geometry_instance_sha256": geometry_instance_sha256(base_env),
        "num_envs": int(base_env.num_envs),
        "curriculum_policy_step_offset": int(base_env.cfg.illegal_collision_curriculum_step_offset),
        "run_config_sha256": hashes,
        "trainer_timesteps": int(agent_cfg["trainer"]["timesteps"]),
    }
    output = params / MANIFEST_FILENAME
    _atomic_json_dump(manifest, output)
    return output


def load_teacher_manifest(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    path = _run_dir(checkpoint_path) / "params" / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Pulled-Drawer Teacher manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != TEACHER_KIND or manifest.get("version") != TEACHER_MANIFEST_VERSION:
        raise RuntimeError("Unsupported Pulled-Drawer Teacher manifest")
    if manifest.get("task") != GELSIGHT_PULLED_DRAWER_TEACHER_TASK:
        raise RuntimeError("Pulled-Drawer Teacher task mismatch")
    if manifest.get("model_version") != PULLED_DRAWER_TEACHER_MODEL_VERSION:
        raise RuntimeError("Pulled-Drawer Teacher model mismatch")
    if manifest.get("actor_contract") != RMAGelSightPulledDrawerActorCore().contract():
        raise RuntimeError("Pulled-Drawer Teacher Actor contract mismatch")
    if manifest.get("normalization") != RMAGelSightPulledDrawerObservationNormalizer().contract():
        raise RuntimeError("Pulled-Drawer Teacher normalization mismatch")
    for name, expected_hash in manifest.get("run_config_sha256", {}).items():
        config_path = _run_dir(checkpoint_path) / "params" / name
        if not config_path.is_file() or sha256_file(config_path) != expected_hash:
            raise RuntimeError(f"Teacher run config hash mismatch: {config_path}")
    return manifest


def validate_live_teacher_contract(
    cfg_or_env: Any,
    manifest: Mapping[str, Any],
    *,
    require_instance: bool = False,
) -> None:
    cfg = cfg_or_env.cfg if hasattr(cfg_or_env, "cfg") else cfg_or_env
    if manifest.get("environment_contract") != teacher_environment_contract(cfg):
        raise RuntimeError("Live Pulled-Drawer environment differs from Teacher contract")
    if require_instance:
        if int(manifest.get("num_envs", -1)) != int(cfg_or_env.num_envs):
            raise RuntimeError("Pulled-Drawer Teacher resume num_envs mismatch")
        if manifest.get("geometry_instance_sha256") != geometry_instance_sha256(cfg_or_env):
            raise RuntimeError("Pulled-Drawer Teacher geometry instance mismatch")


def load_teacher_policy_state(
    checkpoint: str | Path, device: str | torch.device
) -> dict[str, torch.Tensor]:
    load_teacher_manifest(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = payload.get("policy") if isinstance(payload, Mapping) else None
    if not isinstance(policy, Mapping):
        raise RuntimeError("Pulled-Drawer Teacher checkpoint has no policy state_dict")
    return dict(policy)


def infer_teacher_checkpoint_policy_step(checkpoint: str | Path) -> int:
    path = Path(checkpoint).expanduser().resolve()
    suffix = path.stem.removeprefix("agent_")
    if not suffix.isdigit():
        raise RuntimeError("Teacher checkpoint must be named agent_<step>.pt")
    return int(load_teacher_manifest(path)["curriculum_policy_step_offset"]) + int(suffix)


def make_student_payload(
    *,
    model: RMAGelSightPulledDrawerThreeFrameStudent,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    teacher_checkpoint: str | Path,
    teacher_manifest: Mapping[str, Any],
    teacher_actor_state_dict: Mapping[str, torch.Tensor],
    encoder_init_checkpoint: str | Path,
    encoder_init_payload: Mapping[str, Any],
    student_env_contract: Mapping[str, Any],
    geometry_instance_hash: str,
    loss: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
) -> dict[str, Any]:
    state = model.state_dict()
    component_hashes = {}
    for prefix, name in (
        ("vision_encoder.", "vision_encoder"),
        ("temporal_fusion.", "temporal_fusion"),
        ("tactile_encoder.", "tactile_encoder"),
        ("position_head.", "position_head"),
        ("action_head.", "action_head"),
    ):
        component = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        component_hashes[f"{name}_state_dict_sha256"] = state_dict_sha256(component)
    return {
        "kind": STUDENT_KIND,
        "version": STUDENT_CHECKPOINT_VERSION,
        "model_version": PULLED_DRAWER_STUDENT_MODEL_VERSION,
        "task": GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
        "global_step": int(global_step),
        "model": state,
        "optimizer": optimizer.state_dict(),
        "student_input_contract": student_input_contract(),
        "student_model_contract": pulled_drawer_student_model_contract(),
        "normalization": model.normalizer.contract(),
        "student_environment_contract": dict(student_env_contract),
        "geometry_instance_sha256": str(geometry_instance_hash),
        "teacher_checkpoint": str(Path(teacher_checkpoint).expanduser().resolve()),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_manifest": dict(teacher_manifest),
        "teacher_actor_state_dict_sha256": state_dict_sha256(teacher_actor_state_dict),
        "encoder_init_checkpoint": str(Path(encoder_init_checkpoint).expanduser().resolve()),
        "encoder_init_checkpoint_sha256": sha256_file(encoder_init_checkpoint),
        "encoder_init_task": encoder_init_payload.get("task"),
        "encoder_init_state_dict_sha256": encoder_init_payload.get("vision_encoder_state_dict_sha256"),
        "loss": dict(loss),
        "optimizer_config": dict(optimizer_config),
        **component_hashes,
    }


def load_student_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_teacher_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint).expanduser().resolve(), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != STUDENT_KIND:
        raise RuntimeError("Not a Pulled-Drawer Student checkpoint")
    if payload.get("version") != STUDENT_CHECKPOINT_VERSION:
        raise RuntimeError("Pulled-Drawer Student checkpoint version mismatch")
    if payload.get("model_version") != PULLED_DRAWER_STUDENT_MODEL_VERSION:
        raise RuntimeError("Pulled-Drawer Student model mismatch")
    if payload.get("task") != GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK:
        raise RuntimeError("Pulled-Drawer Student task mismatch")
    if payload.get("student_input_contract") != student_input_contract():
        raise RuntimeError("Pulled-Drawer Student input contract mismatch")
    if payload.get("student_model_contract") != pulled_drawer_student_model_contract():
        raise RuntimeError("Pulled-Drawer Student model contract mismatch")
    if payload.get("normalization") != RMAGelSightPulledDrawerObservationNormalizer().contract():
        raise RuntimeError("Pulled-Drawer Student normalization mismatch")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("Pulled-Drawer Student has no model state_dict")
    for prefix, name in (
        ("vision_encoder.", "vision_encoder"),
        ("temporal_fusion.", "temporal_fusion"),
        ("tactile_encoder.", "tactile_encoder"),
        ("position_head.", "position_head"),
        ("action_head.", "action_head"),
    ):
        component = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not component or payload.get(f"{name}_state_dict_sha256") != state_dict_sha256(component):
            raise RuntimeError(f"Pulled-Drawer Student component hash mismatch: {name}")
    if not isinstance(payload.get("teacher_actor_state_dict_sha256"), str):
        raise RuntimeError("Pulled-Drawer Student Teacher Actor provenance is missing")
    if (
        not isinstance(payload.get("encoder_init_checkpoint_sha256"), str)
        or not isinstance(payload.get("encoder_init_state_dict_sha256"), str)
        or payload.get("encoder_init_task") != RMA_XY_STUDENT_HEATMAP_DR_TASK
    ):
        raise RuntimeError("Pulled-Drawer Student encoder initialization provenance is missing")
    if expected_teacher_checkpoint is not None:
        if payload.get("teacher_checkpoint_sha256") != sha256_file(expected_teacher_checkpoint):
            raise RuntimeError("Student was distilled from a different Teacher checkpoint")
        if payload.get("teacher_manifest") != load_teacher_manifest(expected_teacher_checkpoint):
            raise RuntimeError("Student Teacher manifest mismatch")
    return payload


def make_student_model_for_checkpoint(
    payload: Mapping[str, Any], *, pretrained_backbone: bool = False
) -> RMAGelSightPulledDrawerThreeFrameStudent:
    if payload.get("kind") != STUDENT_KIND or payload.get("version") != STUDENT_CHECKPOINT_VERSION:
        raise RuntimeError("Unsupported Pulled-Drawer Student checkpoint")
    return RMAGelSightPulledDrawerThreeFrameStudent(pretrained_backbone=pretrained_backbone)


def load_student_model_state(
    model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> None:
    model.load_state_dict(dict(state_dict), strict=True)


__all__ = (
    "GELSIGHT_PULLED_DRAWER_TEACHER_TASK",
    "GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK",
    "MANIFEST_FILENAME",
    "STUDENT_CHECKPOINT_VERSION",
    "geometry_instance_sha256",
    "infer_teacher_checkpoint_policy_step",
    "load_encoder_initialization_checkpoint",
    "load_student_checkpoint",
    "load_student_model_state",
    "load_teacher_manifest",
    "load_teacher_policy_state",
    "make_student_model_for_checkpoint",
    "make_student_payload",
    "sha256_file",
    "state_dict_sha256",
    "student_environment_contract",
    "teacher_environment_contract",
    "validate_live_teacher_contract",
    "write_teacher_manifest",
)
