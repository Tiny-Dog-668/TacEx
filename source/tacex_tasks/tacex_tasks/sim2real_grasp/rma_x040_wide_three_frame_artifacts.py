"""Fail-closed artifacts for the X040-Wide Size-Buckets three-frame Student."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .rma_x040_wide_artifacts import (
    RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK,
    load_encoder_initialization_checkpoint,
    load_teacher_manifest,
    load_teacher_policy_state,
    sha256_file,
    state_dict_sha256,
    student_environment_contract,
    validate_live_teacher_contract,
)
from .rma_x040_wide_three_frame_models import (
    RMA_X040_WIDE_THREE_FRAME_DIRECT_STUDENT_MODEL_VERSION,
    RMAX040WideThreeFrameDirectActionVisualStudent,
)


RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Size-Buckets-"
    "Three-Frame-Direct-Action-Student-DR-v0"
)
RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Size-Buckets-"
    "Three-Frame-Direct-Action-Student-Appearance-DR-v0"
)
RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASKS = (
    RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK,
    RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK,
)
RMA_X040_WIDE_THREE_FRAME_STUDENT_KIND = (
    "tacex_rma_x040_wide_three_frame_direct_action_student"
)
RMA_X040_WIDE_THREE_FRAME_STUDENT_VERSION = 1


def three_frame_input_contract() -> dict[str, object]:
    return {
        "wrist_rgb_history": [3, 224, 224, 3],
        "proprio_obs": [15],
        "action_history": [4],
    }


def appearance_randomization_contract(cfg: Any) -> dict[str, Any] | None:
    """Return the strict visual-domain contract for appearance-enabled tasks."""
    profile = getattr(cfg, "appearance_profile", None)
    if profile is None:
        return None

    def normalize_material(entry: Mapping[str, Any]) -> dict[str, Any]:
        normalized = {
            "id": str(entry["id"]),
            "kind": str(entry["kind"]),
            "weight": float(entry["weight"]),
        }
        if entry["kind"] == "preview_surface":
            normalized.update(
                {
                    "diffuse_color": [float(value) for value in entry["diffuse_color"]],
                    "metallic": float(entry["metallic"]),
                    "roughness": float(entry["roughness"]),
                }
            )
        else:
            normalized.update(
                {
                    "mdl_path": str(entry["mdl_path"]),
                    "project_uvw": True,
                    "texture_scale": [float(value) for value in entry["texture_scale"]],
                }
            )
        return normalized

    contract = {
        "enabled": bool(cfg.appearance_randomization_enabled),
        "profile": str(profile),
        "sampling_frequency": str(cfg.appearance_sampling_frequency),
        "sampling_scope": str(cfg.appearance_sampling_scope),
        "material_instance_scope": str(cfg.appearance_material_instance_scope),
        "full_strength_from_step": int(cfg.appearance_full_strength_from_step),
        "material_source": str(cfg.appearance_material_source),
        "categories_sampled_independently": True,
        "plate": [normalize_material(entry) for entry in cfg.appearance_plate_materials],
        "backdrop": [
            normalize_material(entry) for entry in cfg.appearance_backdrop_materials
        ],
        "cube": [normalize_material(entry) for entry in cfg.appearance_cube_materials],
    }
    if hasattr(cfg, "appearance_timeout_termination_enabled"):
        contract["timeout_termination_enabled"] = bool(
            cfg.appearance_timeout_termination_enabled
        )
    if hasattr(cfg, "appearance_scene_layout_profile"):
        contract["scene_layout"] = {
            "profile": str(cfg.appearance_scene_layout_profile),
            "env_spacing_m": float(cfg.scene.env_spacing),
            "plate_size_m": [float(value) for value in cfg.plate.spawn.size],
            "backdrop_size_m": [float(value) for value in cfg.backdrop.spawn.size],
            "min_inter_env_clearance_m": float(
                cfg.appearance_min_inter_env_clearance_m
            ),
            "global_ground_visible": bool(
                cfg.appearance_global_ground_visible
            ),
        }
    return contract


def three_frame_student_environment_contract(cfg: Any) -> dict[str, Any]:
    contract = student_environment_contract(cfg)
    contract.pop("wrist_rgb_shape", None)
    contract["wrist_rgb_history"] = {
        "shape": [3, 224, 224, 3],
        "dtype": "uint8",
        "order": str(cfg.wrist_rgb_history_order),
        "stride_policy_steps": int(cfg.wrist_rgb_history_stride_policy_steps),
        "policy_frequency_hz": 30,
        "reset_fill": str(cfg.wrist_rgb_history_reset_fill),
    }
    appearance_contract = appearance_randomization_contract(cfg)
    if appearance_contract is not None:
        contract["appearance_randomization"] = appearance_contract
    return contract


def _is_appearance_v1_evaluation_compatible(
    task: str,
    checkpoint_contract: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> bool:
    """Accept a verified legacy Appearance contract for evaluation only."""
    if task != RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK:
        return False
    from .sim2real_cube_real_alignment_rma_x040_wide_three_frame_appearance_env import (
        Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameAppearanceStudentDREnvCfg,
    )

    legacy_contract = copy.deepcopy(dict(checkpoint_contract))
    appearance = legacy_contract.get("appearance_randomization")
    if not isinstance(appearance, dict):
        return False
    if appearance.get("profile") == "x040_three_frame_realistic_material_v1":
        if "sampling_scope" in appearance or "material_instance_scope" in appearance:
            return False
        appearance["profile"] = "x040_three_frame_realistic_material_v2"
        appearance["sampling_scope"] = "per_environment"
        appearance["material_instance_scope"] = "per_environment"
    legacy_expected = three_frame_student_environment_contract(
        Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameAppearanceStudentDREnvCfg()
    )
    if legacy_contract != legacy_expected:
        return False
    from .sim2real_cube_real_alignment_rma_x040_wide_three_frame_independent_appearance_env import (
        Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnvCfg,
    )

    # The migration is evaluation-only and accepts exactly the current static
    # one-Cube runtime contract after the checkpoint has independently matched
    # the complete legacy v1/v2 contract above.
    current_expected = three_frame_student_environment_contract(
        Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnvCfg()
    )
    return dict(expected_contract) == current_expected


def make_three_frame_student_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    task: str,
    teacher_checkpoint: str | Path,
    teacher_manifest: Mapping[str, Any],
    encoder_init_checkpoint: str | Path,
    encoder_init_payload: Mapping[str, Any],
    student_env_contract: Mapping[str, Any],
    loss: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
) -> dict[str, Any]:
    if task not in RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASKS:
        raise ValueError(f"Unsupported X040-Wide three-frame Student task: {task}")
    model_state = model.state_dict()
    return {
        "kind": RMA_X040_WIDE_THREE_FRAME_STUDENT_KIND,
        "version": RMA_X040_WIDE_THREE_FRAME_STUDENT_VERSION,
        "model_version": RMA_X040_WIDE_THREE_FRAME_DIRECT_STUDENT_MODEL_VERSION,
        "task": task,
        "global_step": int(global_step),
        "model": model_state,
        "model_state_dict_sha256": state_dict_sha256(model_state),
        "optimizer": optimizer.state_dict(),
        "student_input_contract": three_frame_input_contract(),
        "student_model_contract": model.contract(),
        "normalization": model.normalizer.contract(),
        "student_environment_contract": dict(student_env_contract),
        "teacher_checkpoint": str(Path(teacher_checkpoint).expanduser().resolve()),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_manifest": dict(teacher_manifest),
        "encoder_init_checkpoint": str(
            Path(encoder_init_checkpoint).expanduser().resolve()
        ),
        "encoder_init_checkpoint_sha256": sha256_file(encoder_init_checkpoint),
        "encoder_init_task": encoder_init_payload.get("task"),
        "encoder_init_state_dict_sha256": encoder_init_payload.get(
            "vision_encoder_state_dict_sha256"
        ),
        "vision_encoder_state_dict_sha256": state_dict_sha256(
            model.vision_encoder.state_dict()
        ),
        "loss": dict(loss),
        "optimizer_config": dict(optimizer_config),
        "training_privileged_inputs": {
            "rma_cube_pos": [3],
            "purpose": (
                "Teacher action and fused-visual position MSE labels only; "
                "never a runtime input"
            ),
        },
    }


def load_three_frame_student_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_teacher_checkpoint: str | Path | None = None,
    expected_task: str | None = None,
    allow_appearance_v1_evaluation: bool = False,
) -> dict[str, Any]:
    payload = torch.load(
        Path(checkpoint).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != RMA_X040_WIDE_THREE_FRAME_STUDENT_KIND
    ):
        raise RuntimeError("Checkpoint is not an X040-Wide three-frame Student artifact")
    if (
        payload.get("version") != RMA_X040_WIDE_THREE_FRAME_STUDENT_VERSION
        or payload.get("model_version")
        != RMA_X040_WIDE_THREE_FRAME_DIRECT_STUDENT_MODEL_VERSION
    ):
        raise RuntimeError("Unsupported X040-Wide three-frame Student version")
    if (
        payload.get("task")
        not in RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASKS
        or payload.get("student_input_contract") != three_frame_input_contract()
    ):
        raise RuntimeError("X040-Wide three-frame Student task/input mismatch")
    if expected_task is not None and payload.get("task") != expected_task:
        raise RuntimeError("X040-Wide three-frame Student checkpoint task mismatch")

    environment_contract = payload.get("student_environment_contract", {})
    if environment_contract.get("wrist_rgb_history") != {
        "shape": [3, 224, 224, 3],
        "dtype": "uint8",
        "order": "oldest_to_newest",
        "stride_policy_steps": 1,
        "policy_frequency_hz": 30,
        "reset_fill": "repeat_first_post_reset_frame",
    }:
        raise RuntimeError("X040-Wide three-frame history contract mismatch")
    rendering = environment_contract.get("cube_bucket_rendering")
    supported_rendering_contracts = (
        {
            "selection": "selected_bucket_per_environment_per_reset",
            "inactive_bucket_exclusion": "parked_behind_all_cameras_using_env_x_extent",
            "usd_visibility_mutation": "none",
            "parking_strategy": "beyond_positive_x_env_extent_behind_all_cameras",
            "camera_margin_m": 0.5,
        },
        {
            "selection": "fixed_bucket_by_environment",
            "assignment": "env_id_mod_bucket_count",
            "object_count_per_environment": 1,
            "requires_num_envs_multiple_of_bucket_count": True,
            "usd_visibility_mutation": "none",
        },
    )
    if rendering not in supported_rendering_contracts:
        raise RuntimeError("X040-Wide three-frame Size-Buckets rendering mismatch")

    if payload["task"] == RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK:
        from .sim2real_cube_real_alignment_rma_x040_wide_three_frame_env import (
            Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnvCfg,
        )

        expected_environment_contract = three_frame_student_environment_contract(
            Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnvCfg()
        )
    else:
        from .sim2real_cube_real_alignment_rma_x040_wide_three_frame_independent_appearance_env import (
            Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnvCfg,
        )

        expected_environment_contract = three_frame_student_environment_contract(
            Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnvCfg()
        )
    appearance_v1_evaluation_compatible = (
        allow_appearance_v1_evaluation
        and _is_appearance_v1_evaluation_compatible(
            str(payload["task"]),
            environment_contract,
            expected_environment_contract,
        )
    )
    if environment_contract != expected_environment_contract and not (
        appearance_v1_evaluation_compatible
    ):
        raise RuntimeError("X040-Wide three-frame task environment contract mismatch")

    model = RMAX040WideThreeFrameDirectActionVisualStudent()
    model_state = payload.get("model")
    if (
        payload.get("student_model_contract") != model.contract()
        or payload.get("normalization") != model.normalizer.contract()
        or not isinstance(model_state, Mapping)
    ):
        raise RuntimeError("X040-Wide three-frame Student model contract mismatch")
    if payload.get("model_state_dict_sha256") != state_dict_sha256(model_state):
        raise RuntimeError("X040-Wide three-frame Student model hash mismatch")
    vision_state = {
        key[len("vision_encoder.") :]: value
        for key, value in model_state.items()
        if key.startswith("vision_encoder.")
    }
    if (
        not vision_state
        or payload.get("vision_encoder_state_dict_sha256")
        != state_dict_sha256(vision_state)
    ):
        raise RuntimeError("X040-Wide three-frame Student vision encoder hash mismatch")
    if payload.get("teacher_manifest", {}).get("task") != RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK:
        raise RuntimeError("X040-Wide three-frame Student Teacher task mismatch")
    if expected_teacher_checkpoint is not None:
        if (
            payload.get("teacher_checkpoint_sha256")
            != sha256_file(expected_teacher_checkpoint)
            or payload.get("teacher_manifest")
            != load_teacher_manifest(
                expected_teacher_checkpoint,
                expected_task=RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK,
            )
        ):
            raise RuntimeError("Three-frame Student was distilled from a different Teacher")
    return payload


def load_three_frame_student_model_state(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    result = model.load_state_dict(dict(state_dict), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("X040-Wide three-frame Student state mismatch")


__all__ = (
    "RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK",
    "RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK",
    "RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASKS",
    "RMA_X040_WIDE_THREE_FRAME_STUDENT_KIND",
    "RMA_X040_WIDE_THREE_FRAME_STUDENT_VERSION",
    "load_encoder_initialization_checkpoint",
    "load_teacher_manifest",
    "load_teacher_policy_state",
    "validate_live_teacher_contract",
    "three_frame_input_contract",
    "appearance_randomization_contract",
    "three_frame_student_environment_contract",
    "make_three_frame_student_payload",
    "load_three_frame_student_checkpoint",
    "load_three_frame_student_model_state",
    "sha256_file",
    "state_dict_sha256",
)
