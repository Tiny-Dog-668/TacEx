"""Fail-closed Teacher/Student artifacts for the Pulled-Drawer profile."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_USD,
)

from .rma_gelsight_pulled_drawer_models import (
    PULLED_DRAWER_STUDENT_MODEL_VERSION,
    PULLED_DRAWER_TEACHER_MODEL_VERSION,
    RMAGelSightPulledDrawerActorCore,
    RMAGelSightPulledDrawerObservationNormalizer,
    RMAGelSightPulledDrawerThreeFrameStudent,
    pulled_drawer_student_model_contract,
)
from .rma_gelsight_pulled_drawer_binary_tactile_models import (
    PULLED_DRAWER_BINARY_TACTILE_STUDENT_MODEL_VERSION,
    RMAGelSightPulledDrawerBinaryTactileThreeFrameStudent,
    pulled_drawer_binary_tactile_student_model_contract,
)
from .rma_gelsight_pulled_drawer_four_tactile_models import (
    FOUR_TACTILE_STUDENT_MODEL_VERSION,
    FOUR_TACTILE_TEACHER_MODEL_VERSION,
    RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent,
    RMAGelSightPulledDrawerFourTactileActorCore,
    four_tactile_student_model_contract,
)
from .rma_gelsight_large_drawer_fusion_models import (
    LARGE_DRAWER_FUSION_MODEL_CLASSES,
    LARGE_DRAWER_FUSION_MODEL_VERSION,
    LARGE_DRAWER_FUSION_VARIANTS,
    TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE,
    TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE,
    TACTILE_CROSS_ALPHA_DOWNSAMPLE,
    TACTILE_CROSS_DOWNSAMPLE,
    VISION_ONLY_DOWNSAMPLE,
    VT_DOWNSAMPLE,
    is_aux_variant,
    is_recurrent_variant,
    is_tactile_variant,
    large_drawer_fusion_model_contract,
    make_large_drawer_fusion_student,
)
from .sim2real_cube_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
    GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
    GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
)
from .sim2real_cylinder_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
    CYLINDER_DENSITY_KG_M3,
    CYLINDER_NOMINAL_HEIGHT_M,
    CYLINDER_NOMINAL_RADIUS_M,
    CYLINDER_SCALE_BUCKETS,
    GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
    GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
)
from .sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
    large_pulled_drawer_geometry_contract,
)
from .sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_downsample_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK,
)
from .sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_fusion_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_DOWNSAMPLE_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_DOWNSAMPLE_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_VISION_ONLY_DOWNSAMPLE_STUDENT_TASK,
)
from .sim2real_cube_real_alignment_gelsight_rma_env import (
    gelsight_compliant_grasp_contract,
)
from .rma_gelsight_x040_three_frame_artifacts import (
    load_encoder_initialization_checkpoint,
    sha256_file,
    state_dict_sha256,
    student_input_contract as x040_student_input_contract,
)
from .large_drawer_occlusion import validate_occlusion_predictor_metadata
from tacex_tasks.sim2real_grasp.rma_xy_artifacts import (
    RMA_XY_STUDENT_HEATMAP_DR_TASK,
)
from .sim2real_cube_real_alignment_gelsight_pulled_drawer_env import (
    GELSIGHT_PULLED_DRAWER_TEACHER_TASK,
    GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
    pulled_drawer_geometry_contract,
)
from .sim2real_cube_real_alignment_gelsight_pulled_drawer_progress_binary_tactile_env import (
    GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
    GELSIGHT_PULLED_DRAWER_PROGRESS_TEACHER_TASK,
)


MANIFEST_FILENAME = "rma_gelsight_pulled_drawer_manifest.json"
TEACHER_KIND = "tacex_rma_gelsight_pulled_drawer_teacher"
TEACHER_MANIFEST_VERSION = 11
STUDENT_KIND = "tacex_rma_gelsight_pulled_drawer_three_frame_student"
STUDENT_CHECKPOINT_VERSION = 11
PROGRESS_TEACHER_KIND = "tacex_rma_gelsight_pulled_drawer_progress_teacher"
PROGRESS_TEACHER_MANIFEST_VERSION = 2
PROGRESS_BINARY_STUDENT_KIND = (
    "tacex_rma_gelsight_pulled_drawer_progress_binary_tactile_three_frame_student"
)
PROGRESS_BINARY_STUDENT_CHECKPOINT_VERSION = 2
FOUR_TACTILE_TEACHER_KIND = "tacex_rma_gelsight_pulled_drawer_progress_four_tactile_teacher"
FOUR_TACTILE_TEACHER_MANIFEST_VERSION = 6
FOUR_TACTILE_STUDENT_KIND = "tacex_rma_gelsight_pulled_drawer_progress_four_binary_tactile_three_frame_student"
FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION = 6
CYLINDER_FOUR_TACTILE_TEACHER_KIND = (
    "tacex_rma_gelsight_pulled_drawer_progress_four_tactile_cylinder_teacher"
)
CYLINDER_FOUR_TACTILE_TEACHER_MANIFEST_VERSION = 2
CYLINDER_FOUR_TACTILE_STUDENT_KIND = (
    "tacex_rma_gelsight_pulled_drawer_progress_four_binary_tactile_cylinder_three_frame_student"
)
CYLINDER_FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION = 2
LARGE_DRAWER_CYLINDER_FOUR_TACTILE_TEACHER_KIND = (
    "tacex_rma_gelsight_large_pulled_drawer_progress_four_tactile_cylinder_teacher"
)
LARGE_DRAWER_CYLINDER_FOUR_TACTILE_TEACHER_MANIFEST_VERSION = 2
LARGE_DRAWER_CYLINDER_FOUR_TACTILE_STUDENT_KIND = (
    "tacex_rma_gelsight_large_pulled_drawer_progress_four_binary_tactile_cylinder_three_frame_student"
)
LARGE_DRAWER_CYLINDER_FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION = 2
LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_TEACHER_KIND = (
    "tacex_rma_gelsight_large_pulled_drawer_progress_four_tactile_cylinder_downsample_teacher"
)
LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_TEACHER_MANIFEST_VERSION = 1
LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_KIND = (
    "tacex_rma_gelsight_large_pulled_drawer_progress_four_binary_tactile_cylinder_downsample_three_frame_student"
)
LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_CHECKPOINT_VERSION = 2

_LARGE_DRAWER_FUSION_STUDENT_IDENTITY = {
    VISION_ONLY_DOWNSAMPLE: (
        "tacex_rma_gelsight_large_drawer_vision_only_downsample_student",
        1,
    ),
    VT_DOWNSAMPLE: (
        LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_KIND,
        LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_CHECKPOINT_VERSION,
    ),
    TACTILE_CROSS_DOWNSAMPLE: (
        "tacex_rma_gelsight_large_drawer_tactile_cross_downsample_student",
        1,
    ),
    TACTILE_CROSS_ALPHA_DOWNSAMPLE: (
        "tacex_rma_gelsight_large_drawer_tactile_cross_alpha_downsample_student",
        1,
    ),
    TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE: (
        "tacex_rma_gelsight_large_drawer_tactile_cross_alpha_aux_downsample_student",
        1,
    ),
    TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE: (
        "tacex_rma_gelsight_large_drawer_tactile_cross_alpha_aux_gru_downsample_student",
        1,
    ),
}

LARGE_DRAWER_FUSION_VARIANT_BY_TASK = {
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_VISION_ONLY_DOWNSAMPLE_STUDENT_TASK: VISION_ONLY_DOWNSAMPLE,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK: VT_DOWNSAMPLE,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_DOWNSAMPLE_STUDENT_TASK: TACTILE_CROSS_DOWNSAMPLE,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_DOWNSAMPLE_STUDENT_TASK: TACTILE_CROSS_ALPHA_DOWNSAMPLE,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE_STUDENT_TASK: TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE_STUDENT_TASK: TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE,
}
LARGE_DRAWER_FUSION_STUDENT_TASKS = frozenset(LARGE_DRAWER_FUSION_VARIANT_BY_TASK)
LARGE_DRAWER_FUSION_REGISTRY = {
    task: {
        "variant": variant,
        "model_class": LARGE_DRAWER_FUSION_MODEL_CLASSES[variant],
        "runtime_input_order": tuple(
            large_drawer_fusion_model_contract(variant)["runtime_input_order"]
        ),
        "runtime_output": dict(
            large_drawer_fusion_model_contract(variant)["runtime_output"]
        ),
        "training_losses": tuple(
            ["action_mse", "action_smoothness_mse", "position_smooth_l1"]
            + (["contact_bce"] if is_tactile_variant(variant) else [])
            + (["occlusion_mse"] if is_aux_variant(variant) else [])
        ),
        "recurrent": is_recurrent_variant(variant),
        "artifact_kind": _LARGE_DRAWER_FUSION_STUDENT_IDENTITY[variant][0],
        "artifact_version": _LARGE_DRAWER_FUSION_STUDENT_IDENTITY[variant][1],
        "log_directory": (
            "logs/skrl/sim2real_cylinder_real_alignment_rma_gelsight_"
            f"large_pulled_drawer_fusion_{variant}_student"
        ),
    }
    for task, variant in LARGE_DRAWER_FUSION_VARIANT_BY_TASK.items()
}

_FOUR_TACTILE_TASKS = frozenset(
    {
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
        *LARGE_DRAWER_FUSION_STUDENT_TASKS,
    }
)
_CYLINDER_TASKS = frozenset(
    {
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
        *LARGE_DRAWER_FUSION_STUDENT_TASKS,
    }
)
_LARGE_DRAWER_TASKS = frozenset(
    {
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
        *LARGE_DRAWER_FUSION_STUDENT_TASKS,
    }
)
_DOWNSAMPLE_TASKS = frozenset(
    {
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
        *LARGE_DRAWER_FUSION_STUDENT_TASKS,
    }
)
_PROGRESS_TASKS = frozenset(
    {
        GELSIGHT_PULLED_DRAWER_PROGRESS_TEACHER_TASK,
        GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
        *LARGE_DRAWER_FUSION_STUDENT_TASKS,
    }
)

GELSIGHT_PULLED_DRAWER_TEACHER_TASKS = frozenset(
    {
        GELSIGHT_PULLED_DRAWER_TEACHER_TASK,
        GELSIGHT_PULLED_DRAWER_PROGRESS_TEACHER_TASK,
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK,
    }
)
GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK = {
    GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK: (
        GELSIGHT_PULLED_DRAWER_TEACHER_TASK
    ),
    GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK: (
        GELSIGHT_PULLED_DRAWER_PROGRESS_TEACHER_TASK
    ),
    GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK: (
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    ),
    GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK: (
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    ),
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK: (
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    ),
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK: (
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    ),
    **{
        task: GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
        for task in LARGE_DRAWER_FUSION_STUDENT_TASKS
    },
}
GELSIGHT_PULLED_DRAWER_STUDENT_TASKS = frozenset(
    GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK
)


def _is_binary_student_task(task: str) -> bool:
    return (
        task in {
            GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
            GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
            GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
        }
        or (
            task in LARGE_DRAWER_FUSION_VARIANT_BY_TASK
            and LARGE_DRAWER_FUSION_VARIANT_BY_TASK[task] != VISION_ONLY_DOWNSAMPLE
        )
    )


def _teacher_artifact_identity(task: str) -> tuple[str, int]:
    if task == GELSIGHT_PULLED_DRAWER_TEACHER_TASK:
        return TEACHER_KIND, TEACHER_MANIFEST_VERSION
    if task == GELSIGHT_PULLED_DRAWER_PROGRESS_TEACHER_TASK:
        return PROGRESS_TEACHER_KIND, PROGRESS_TEACHER_MANIFEST_VERSION
    if task == GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK:
        return FOUR_TACTILE_TEACHER_KIND, FOUR_TACTILE_TEACHER_MANIFEST_VERSION
    if task == GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK:
        return (
            CYLINDER_FOUR_TACTILE_TEACHER_KIND,
            CYLINDER_FOUR_TACTILE_TEACHER_MANIFEST_VERSION,
        )
    if task == GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK:
        return (
            LARGE_DRAWER_CYLINDER_FOUR_TACTILE_TEACHER_KIND,
            LARGE_DRAWER_CYLINDER_FOUR_TACTILE_TEACHER_MANIFEST_VERSION,
        )
    if task == GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK:
        return (
            LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_TEACHER_KIND,
            LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_TEACHER_MANIFEST_VERSION,
        )
    raise RuntimeError(f"Unsupported Pulled-Drawer Teacher task: {task}")


def _student_artifact_identity(task: str) -> tuple[str, int, int]:
    if task in LARGE_DRAWER_FUSION_VARIANT_BY_TASK:
        variant = LARGE_DRAWER_FUSION_VARIANT_BY_TASK[task]
        kind, version = _LARGE_DRAWER_FUSION_STUDENT_IDENTITY[variant]
        return kind, version, LARGE_DRAWER_FUSION_MODEL_VERSION
    if task == GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK:
        return STUDENT_KIND, STUDENT_CHECKPOINT_VERSION, PULLED_DRAWER_STUDENT_MODEL_VERSION
    if _is_binary_student_task(task):
        if task == GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK:
            return (
                LARGE_DRAWER_CYLINDER_FOUR_TACTILE_STUDENT_KIND,
                LARGE_DRAWER_CYLINDER_FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION,
                FOUR_TACTILE_STUDENT_MODEL_VERSION,
            )
        if task == GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK:
            return (
                CYLINDER_FOUR_TACTILE_STUDENT_KIND,
                CYLINDER_FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION,
                FOUR_TACTILE_STUDENT_MODEL_VERSION,
            )
        if task == GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK:
            return FOUR_TACTILE_STUDENT_KIND, FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION, FOUR_TACTILE_STUDENT_MODEL_VERSION
        return (
            PROGRESS_BINARY_STUDENT_KIND,
            PROGRESS_BINARY_STUDENT_CHECKPOINT_VERSION,
            PULLED_DRAWER_BINARY_TACTILE_STUDENT_MODEL_VERSION,
        )
    raise RuntimeError(f"Unsupported Pulled-Drawer Student task: {task}")


def student_input_contract(
    student_task: str = GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
) -> dict[str, Any]:
    contract = x040_student_input_contract()
    if student_task in LARGE_DRAWER_FUSION_VARIANT_BY_TASK:
        variant = LARGE_DRAWER_FUSION_VARIANT_BY_TASK[student_task]
        model_contract = large_drawer_fusion_model_contract(variant)
        contract["input_order"] = list(model_contract["runtime_input_order"])
        contract["runtime_output"] = dict(model_contract["runtime_output"])
        contract["training_only_labels"] = ["rma_cube_pos"]
        if variant == VISION_ONLY_DOWNSAMPLE:
            for key in (
                "tactile_rgb",
                "tactile_delta",
                "tactile_network_input",
                "tactile_binary_threshold_u8",
                "tactile_binary_values",
            ):
                contract.pop(key, None)
        else:
            contract["tactile_delta"] = (
                "float32_single_channel_max_abs_rgb_current_minus_reference_"
                "strict_gt_5_u8"
            )
            contract["tactile_network_input"] = [1, 96, 128]
            contract["tactile_binary_threshold_u8"] = 5.0
            contract["tactile_binary_values"] = [0.0, 1.0]
            contract["contact_order"] = list(model_contract["contact_order"])
            contract["training_only_labels"].append("rma_contact_state")
        if variant == TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE:
            contract["tactile_feature_history"] = [4, 9, 256]
            contract["reset_mask"] = [1]
        if is_aux_variant(variant):
            contract["training_only_labels"].append("xy_occlusion_pseudo_target")
        return contract
    if _is_binary_student_task(student_task):
        contract["tactile_delta"] = (
            "float32_single_channel_max_abs_rgb_current_minus_reference_"
            "strict_gt_5_u8"
        )
        contract["tactile_network_input"] = [1, 96, 128]
        contract["tactile_binary_threshold_u8"] = 5.0
        contract["tactile_binary_values"] = [0.0, 1.0]
    if student_task in _FOUR_TACTILE_TASKS:
        contract["input_order"] = four_tactile_student_model_contract()["runtime_input_order"]
        contract["runtime_output"] = four_tactile_student_model_contract()["runtime_output"]
        contract["contact_order"] = four_tactile_student_model_contract()["contact_order"]
    return contract


def _student_model_contract(task: str) -> dict[str, object]:
    if task in LARGE_DRAWER_FUSION_VARIANT_BY_TASK:
        return large_drawer_fusion_model_contract(
            LARGE_DRAWER_FUSION_VARIANT_BY_TASK[task]
        )
    if task in _FOUR_TACTILE_TASKS:
        return four_tactile_student_model_contract()
    if _is_binary_student_task(task):
        return pulled_drawer_binary_tactile_student_model_contract()
    if task == GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK:
        return pulled_drawer_student_model_contract()
    raise RuntimeError(f"Unsupported Pulled-Drawer Student task: {task}")


def _environment_profile(task: str) -> str:
    if task == GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK:
        return "rma_gelsight_large_pulled_drawer_progress_four_tactile_cylinder_downsample_v1"
    if task in _DOWNSAMPLE_TASKS:
        return "rma_gelsight_large_pulled_drawer_progress_four_tactile_cylinder_downsample_v2"
    if task in _LARGE_DRAWER_TASKS:
        return "rma_gelsight_large_pulled_drawer_progress_four_tactile_cylinder_v2"
    if task in _CYLINDER_TASKS:
        return "rma_gelsight_pulled_drawer_progress_four_tactile_cylinder_v2"
    if task in _FOUR_TACTILE_TASKS:
        return "rma_gelsight_pulled_drawer_progress_four_tactile_v6"
    if task in _PROGRESS_TASKS:
        return "rma_gelsight_pulled_drawer_progress_v2"
    return "rma_gelsight_pulled_drawer_contiguous_rigid_tray_v11"


def _geometry_contract_for_task(task: str) -> dict[str, object]:
    if task in _LARGE_DRAWER_TASKS:
        return large_pulled_drawer_geometry_contract()
    return pulled_drawer_geometry_contract()


def _validate_environment_contract_for_task(
    task: str, contract: Any, *, student: bool
) -> None:
    if not isinstance(contract, Mapping):
        raise RuntimeError("Pulled-Drawer environment contract is missing")
    four_tactile = task in _FOUR_TACTILE_TASKS
    cylinder = task in _CYLINDER_TASKS
    progress = task in _PROGRESS_TASKS
    expected_profile = _environment_profile(task)
    if (
        contract.get("profile") != expected_profile
        or contract.get("geometry") != _geometry_contract_for_task(task)
        or contract.get("action_dim") != 4
        or contract.get("position_frame") != "robot_root"
        or (
            not cylinder
            and contract.get("cube_size_assignment")
            != "seeded_balanced_permutation_per_group_of_8"
        )
        or contract.get("teacher_actor_inputs")
        != {
            "proprio_obs": 15,
            "action_history": 4,
            "rma_cube_pos": 3,
            "rma_contact_state": 4 if four_tactile else 2,
        }
    ):
        raise RuntimeError("Pulled-Drawer geometry/action/observation contract mismatch")
    if task in _LARGE_DRAWER_TASKS and contract.get(
        "cube_reset_half_range_xy_m"
    ) != [0.05, 0.05]:
        raise RuntimeError("Large Pulled-Drawer object reset range mismatch")
    if cylinder:
        expected_target = {
            "shape": "cylinder",
            "internal_compatibility_slot": "cube",
            "nominal_height_m": CYLINDER_NOMINAL_HEIGHT_M,
            "nominal_radius_m": CYLINDER_NOMINAL_RADIUS_M,
            "density_kg_m3": CYLINDER_DENSITY_KG_M3,
            "scale_buckets": list(CYLINDER_SCALE_BUCKETS),
            "scale_mode": "isotropic",
            "scale_assignment": "seeded_balanced_permutation_per_group_of_8",
            "scale_lifetime": "fixed_per_environment",
            "success_requires_upright": False,
        }
        if contract.get("target_object") != expected_target:
            raise RuntimeError("Pulled-Drawer Cylinder target contract mismatch")
    if progress:
        reward = contract.get("progress_reward")
        excess_force = (
            reward.get("excess_contact_force_penalty")
            if isinstance(reward, Mapping)
            else None
        )
        collision_termination = contract.get(
            "illegal_collision_termination_threshold_n"
        )
        expected_collision_termination = not four_tactile
        if (
            contract.get("success_terminates_episode") is not True
            or contract.get("success_reward_done_alignment")
            != "same_transition_after_committed_hold_counter"
            or contract.get("illegal_collision_terminates_episode")
            is not expected_collision_termination
            or (four_tactile and collision_termination is not None)
            or (
                not four_tactile
                and (
                    not isinstance(collision_termination, Mapping)
                    or collision_termination.get("start_n") != 200.0
                    or collision_termination.get("end_n") != 20.0
                )
            )
            or not isinstance(reward, Mapping)
            or reward.get("reach") != "absolute_normalized_proximity_per_step"
            or reward.get("reach_weight") != 2.5
            or reward.get("lift") != "absolute_normalized_progress_per_step"
            or reward.get("lift_weight") != 2.5
            or reward.get("contact") != (
                "absolute_weighted_contact_state_per_step"
                if cylinder
                else "signed_contact_acquisition_delta"
            )
            or reward.get("contact_holding_state_repeats_reward") is not cylinder
            or (
                cylinder
                and reward.get("contact_absolute_scale") != 0.2
            )
            or reward.get("success") != "once_on_confirmed_terminal_success"
            or reward.get("success_reward_weight") != 1000.0
            or (cylinder and reward.get("success_requires_upright") is not False)
            or reward.get("action_magnitude_penalty_weight") != 0.05
            or not isinstance(excess_force, Mapping)
            or excess_force.get("mode") != (
                "quadratic_normalized_max_four_tactile_excess" if four_tactile
                else "quadratic_normalized_max_side_excess"
            )
            or excess_force.get("threshold_n") != 15.0
            or excess_force.get("quadratic_weight") != 5.0
        ):
            raise RuntimeError("Pulled-Drawer Progress reward/done contract mismatch")
        if four_tactile and (
            reward.get("contact_order")
            != ["left_inner", "right_inner", "left_down", "right_down"]
            or reward.get("contact_per_sensor_weights") != [1.5, 1.5, 1.0, 1.0]
            or len(contract.get("gelpad_contact_filters", [])) != 4
        ):
            raise RuntimeError("Pulled-Drawer four-tactile contact contract mismatch")
        clearance = contract.get("four_tactile_clearance")
        if four_tactile and (
            not isinstance(clearance, Mapping)
            or clearance.get("source")
            != "minimum_world_z_of_down_gelpad_face_corners"
            or clearance.get("lowest_point_max_hand_z_m") != 0.18331
            or clearance.get("world_reduction") != "minimum_z"
            or len(clearance.get("points_hand_m", [])) != 8
        ):
            raise RuntimeError("Pulled-Drawer four-tactile clearance contract mismatch")
        if four_tactile and "four_tactile_down_collision" in contract:
            raise RuntimeError("Pulled-Drawer contains removed downward collision contract")
    elif "progress_reward" in contract:
        raise RuntimeError("Legacy Pulled-Drawer contract contains Progress rewards")
    if student:
        if (
            contract.get("task") != task
            or contract.get("wrist_rgb_history", {}).get("shape")
            != [3, 224, 224, 3]
            or contract.get("gelsight_reference")
            != "first_post_reset_frame_per_environment"
        ):
            raise RuntimeError("Pulled-Drawer Student runtime contract mismatch")
        if progress and contract.get("visual_domain_randomization") != {
            "full_strength_from_first_step": True,
            "camera_pose_enabled": True,
            "wrist_rgb_enabled": True,
            "dome_light_enabled": True,
            "drawer_appearance_enabled": True,
        }:
            raise RuntimeError("Pulled-Drawer Student visual DR contract mismatch")
        expected_degradation = {
            "mode": "bilinear_downsample_then_bilinear_upsample",
            "input_output_size_hw": [224, 224],
            "downsample_size_hw": [32, 32],
            "align_corners": False,
            "applied_after": "wrist_color_and_intrinsic_dr_before_sensor_noise",
            "random_gaussian_blur_enabled": False,
        }
        if task in _DOWNSAMPLE_TASKS:
            if contract.get("wrist_rgb_degradation") != expected_degradation:
                raise RuntimeError("Pulled-Drawer Downsample Student visual contract mismatch")
        elif "wrist_rgb_degradation" in contract:
            raise RuntimeError("Unexpected Pulled-Drawer Student visual degradation contract")


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
    task = str(cfg.rma_task_id)
    if task not in GELSIGHT_PULLED_DRAWER_TEACHER_TASKS | GELSIGHT_PULLED_DRAWER_STUDENT_TASKS:
        raise RuntimeError(f"Unsupported Pulled-Drawer task: {task}")
    # Student-only visual/fusion choices are intentionally excluded from the
    # Teacher contract. All six comparison Students must match the existing
    # standard Large Drawer Teacher's physical MDP contract byte-for-byte.
    contract_task = GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK.get(task, task)
    four_tactile = contract_task in _FOUR_TACTILE_TASKS
    cylinder = contract_task in _CYLINDER_TASKS
    contract = {
        "profile": _environment_profile(contract_task),
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
            "rma_contact_state": 4 if four_tactile else 2,
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
        "geometry": _geometry_contract_for_task(contract_task),
        "geometry_seed": _cfg_seed(cfg),
        "gelpad_contact_filters": list(cfg.rma_cube_contact_sensor.filter_prim_paths_expr),
        "compliant_grasp": gelsight_compliant_grasp_contract(cfg),
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
    if contract_task in _PROGRESS_TASKS:
        contract["illegal_collision_terminates_episode"] = bool(
            cfg.illegal_collision_terminates_episode
        )
        if not bool(cfg.illegal_collision_terminates_episode):
            contract.pop("illegal_collision_termination_threshold_n", None)
        contract["success_reward_done_alignment"] = (
            "same_transition_after_committed_hold_counter"
        )
        contract["progress_reward"] = {
            "reach": str(cfg.reach_reward_mode),
            "reach_weight": float(cfg.reach_weight),
            "lift": str(cfg.lift_reward_mode),
            "lift_weight": float(cfg.lift_weight),
            "reach_holding_state_repeats_reward": True,
            "lift_holding_state_repeats_reward": True,
            "contact_holding_state_repeats_reward": cylinder,
            "contact": str(cfg.contact_reward_mode),
            "contact_weight": float(cfg.rma_contact_reward_weight),
            "success": str(cfg.success_reward_mode),
            "success_lift_delta_m": float(cfg.success_lift_delta),
            "success_hold_steps": int(cfg.success_hold_steps),
            "success_reward_weight": float(cfg.success_reward_weight),
            "action_rate_penalty_weight": float(cfg.rma_action_rate_penalty_weight),
            "action_rate_scales": str(cfg.rma_action_rate_penalty_scales),
            "action_magnitude_penalty_weight": float(
                cfg.rma_action_magnitude_penalty_weight
            ),
            "action_magnitude_scales": str(cfg.rma_action_magnitude_penalty_scales),
            "excess_contact_force_penalty": {
                "mode": str(cfg.rma_excess_contact_force_penalty_mode),
                "threshold_n": float(cfg.rma_excess_contact_force_threshold_n),
                "quadratic_weight": float(
                    cfg.rma_excess_contact_force_quadratic_weight
                ),
                "normalized_by_threshold": True,
            },
        }
        if four_tactile:
            contract["compliant_grasp"]["contact_threshold_comparison"] = "greater_than_or_equal"
            contract["progress_reward"]["contact_order"] = list(cfg.rma_contact_order)
            contract["progress_reward"]["contact_per_sensor_weights"] = list(cfg.rma_contact_reward_weights)
            if cylinder:
                contract["progress_reward"]["contact_absolute_scale"] = float(
                    cfg.cylinder_absolute_contact_reward_scale
                )
            contract["four_tactile_clearance"] = {
                "source": str(cfg.four_tactile_clearance_source),
                "points_hand_m": [list(point) for point in cfg.four_tactile_clearance_points_hand_m],
                "lowest_point_max_hand_z_m": float(cfg.gelsight_fingertip_bottom_offset_hand_m[2]),
                "world_reduction": "minimum_z",
            }
    if cylinder:
        contract.pop("cube_size_buckets_m", None)
        contract.pop("cube_size_assignment", None)
        contract.pop("cube_size_object_count_per_environment", None)
        contract["target_object"] = {
            "shape": str(cfg.cylinder_shape),
            "internal_compatibility_slot": str(
                cfg.cylinder_internal_compatibility_slot
            ),
            "nominal_height_m": float(cfg.cylinder_nominal_height_m),
            "nominal_radius_m": float(cfg.cylinder_nominal_radius_m),
            "density_kg_m3": float(cfg.cylinder_density_kg_m3),
            "scale_buckets": [float(value) for value in cfg.cylinder_scale_buckets],
            "scale_mode": "isotropic",
            "scale_assignment": str(cfg.cylinder_scale_assignment),
            "scale_lifetime": str(cfg.cylinder_scale_lifetime),
            "success_requires_upright": bool(cfg.success_requires_upright),
        }
        contract["progress_reward"]["success_requires_upright"] = bool(
            cfg.success_requires_upright
        )
    return contract


def student_environment_contract(cfg: Any) -> dict[str, Any]:
    task = str(cfg.rma_task_id)
    if task not in GELSIGHT_PULLED_DRAWER_STUDENT_TASKS:
        raise RuntimeError(f"Unsupported Pulled-Drawer Student task: {task}")
    contract = teacher_environment_contract(cfg)
    contract.update(
        {
            "profile": _environment_profile(task),
            "task": task,
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
    if _is_binary_student_task(task) or task in LARGE_DRAWER_FUSION_STUDENT_TASKS:
        contract["visual_domain_randomization"] = {
            "full_strength_from_first_step": not bool(cfg.dr_curriculum_enabled),
            "camera_pose_enabled": bool(cfg.camera_pose_randomization_enabled),
            "wrist_rgb_enabled": bool(cfg.wrist_visual_randomization_enabled),
            "dome_light_enabled": bool(cfg.light_randomization_enabled),
            "drawer_appearance_enabled": bool(
                cfg.pulled_drawer_appearance_randomization_enabled
            ),
        }
    if task in _DOWNSAMPLE_TASKS:
        contract["wrist_rgb_degradation"] = {
            "mode": "bilinear_downsample_then_bilinear_upsample",
            "input_output_size_hw": [224, 224],
            "downsample_size_hw": [
                int(cfg.wrist_downsample_size),
                int(cfg.wrist_downsample_size),
            ],
            "align_corners": False,
            "applied_after": "wrist_color_and_intrinsic_dr_before_sensor_noise",
            "random_gaussian_blur_enabled": bool(
                cfg.wrist_blur_randomization_enabled
            ),
        }
    return contract


def geometry_instance_sha256(base_env: Any) -> str:
    layout = base_env._pulled_drawer_layout
    cylinder = str(base_env.cfg.rma_task_id) in _CYLINDER_TASKS
    serializable = {
        "seed": _cfg_seed(base_env.cfg),
        "num_envs": int(base_env.num_envs),
        ("target_scale_bucket_ids" if cylinder else "cube_bucket_ids"): (
            base_env._active_cube_bucket_ids.detach().cpu().tolist()
        ),
        **{
            key: value.detach().cpu().tolist()
            for key, value in sorted(layout.items())
        },
    }
    encoded = json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _four_tactile_source_files(task: str) -> dict[str, Path]:
    source_dir = Path(__file__).resolve().parent
    large_drawer = task in _LARGE_DRAWER_TASKS
    downsample = task in _DOWNSAMPLE_TASKS
    cylinder = task in _CYLINDER_TASKS
    source_files = {
        "environment": source_dir
        / (
            "sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_downsample_env.py"
            if downsample
            else "sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_env.py"
            if large_drawer
            else "sim2real_cylinder_real_alignment_gelsight_pulled_drawer_four_tactile_env.py"
            if cylinder
            else "sim2real_cube_real_alignment_gelsight_pulled_drawer_four_tactile_env.py"
        ),
        "models": source_dir / "rma_gelsight_pulled_drawer_four_tactile_models.py",
        "asset": Path(GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_USD),
    }
    if cylinder:
        source_files["base_environment"] = (
            source_dir
            / "sim2real_cylinder_real_alignment_gelsight_pulled_drawer_four_tactile_env.py"
            if large_drawer
            else source_dir
            / "sim2real_cube_real_alignment_gelsight_pulled_drawer_four_tactile_env.py"
        )
    if large_drawer:
        source_files["four_tactile_base_environment"] = (
            source_dir
            / "sim2real_cube_real_alignment_gelsight_pulled_drawer_four_tactile_env.py"
        )
    if downsample:
        source_files["wrist_visual_degradation"] = (
            source_dir.parent
            / "sim2real_grasp"
            / "sim2real_cube_real_alignment_env.py"
        )
    return source_files


def write_teacher_manifest(
    base_env: Any,
    params_dir: str | Path,
    agent_cfg: Mapping[str, Any],
) -> Path:
    task = str(base_env.cfg.rma_task_id)
    kind, version = _teacher_artifact_identity(task)
    params = Path(params_dir)
    hashes = {}
    for name in ("agent.yaml", "env.yaml"):
        path = params / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Teacher run config: {path}")
        hashes[name] = sha256_file(path)
    four_tactile = task in _FOUR_TACTILE_TASKS
    cylinder = task in _CYLINDER_TASKS
    actor = (RMAGelSightPulledDrawerFourTactileActorCore() if four_tactile
             else RMAGelSightPulledDrawerActorCore())
    manifest = {
        "kind": kind,
        "version": version,
        "task": task,
        "model_version": FOUR_TACTILE_TEACHER_MODEL_VERSION if four_tactile else PULLED_DRAWER_TEACHER_MODEL_VERSION,
        "actor_contract": actor.contract(),
        "normalization": RMAGelSightPulledDrawerObservationNormalizer().contract(),
        "environment_contract": teacher_environment_contract(base_env.cfg),
        "geometry_instance_sha256": geometry_instance_sha256(base_env),
        "num_envs": int(base_env.num_envs),
        "curriculum_policy_step_offset": int(base_env.cfg.illegal_collision_curriculum_step_offset),
        "run_config_sha256": hashes,
        "trainer_timesteps": int(agent_cfg["trainer"]["timesteps"]),
    }
    if four_tactile:
        source_files = _four_tactile_source_files(task)
        manifest["source_sha256"] = {
            name: sha256_file(path) for name, path in source_files.items()
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
    task = str(manifest.get("task", ""))
    kind, version = _teacher_artifact_identity(task)
    if manifest.get("kind") != kind or manifest.get("version") != version:
        raise RuntimeError("Unsupported Pulled-Drawer Teacher manifest")
    four_tactile = task in _FOUR_TACTILE_TASKS
    cylinder = task in _CYLINDER_TASKS
    expected_model_version = FOUR_TACTILE_TEACHER_MODEL_VERSION if four_tactile else PULLED_DRAWER_TEACHER_MODEL_VERSION
    expected_actor = (RMAGelSightPulledDrawerFourTactileActorCore() if four_tactile
                      else RMAGelSightPulledDrawerActorCore())
    if manifest.get("model_version") != expected_model_version:
        raise RuntimeError("Pulled-Drawer Teacher model mismatch")
    if manifest.get("actor_contract") != expected_actor.contract():
        raise RuntimeError("Pulled-Drawer Teacher Actor contract mismatch")
    if manifest.get("normalization") != RMAGelSightPulledDrawerObservationNormalizer().contract():
        raise RuntimeError("Pulled-Drawer Teacher normalization mismatch")
    if four_tactile:
        expected_sources = _four_tactile_source_files(task)
        if manifest.get("source_sha256") != {
            name: sha256_file(path) for name, path in expected_sources.items()
        }:
            raise RuntimeError("Pulled-Drawer four-tactile source hash mismatch")
    _validate_environment_contract_for_task(
        task, manifest.get("environment_contract"), student=False
    )
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
    cfg_task = str(cfg.rma_task_id)
    expected_teacher_task = GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK.get(
        cfg_task, cfg_task
    )
    if manifest.get("task") != expected_teacher_task:
        raise RuntimeError("Pulled-Drawer Teacher task mismatch")
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
    student_task: str = GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
    model: torch.nn.Module,
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
    occlusion_predictor_checkpoint: str | Path | None = None,
    occlusion_predictor_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kind, version, model_version = _student_artifact_identity(student_task)
    expected_teacher_task = GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK[
        student_task
    ]
    if teacher_manifest.get("task") != expected_teacher_task:
        raise RuntimeError("Pulled-Drawer Student and Teacher tasks are not paired")
    expected_teacher_kind, expected_teacher_version = _teacher_artifact_identity(
        expected_teacher_task
    )
    if (
        teacher_manifest.get("kind") != expected_teacher_kind
        or teacher_manifest.get("version") != expected_teacher_version
    ):
        raise RuntimeError("Pulled-Drawer Student Teacher manifest is unsupported")
    expected_model_contract = _student_model_contract(student_task)
    if not hasattr(model, "contract") or model.contract() != expected_model_contract:
        raise RuntimeError("Pulled-Drawer Student model does not match its task")
    _validate_environment_contract_for_task(
        expected_teacher_task,
        teacher_manifest.get("environment_contract"),
        student=False,
    )
    _validate_environment_contract_for_task(
        student_task, student_env_contract, student=True
    )
    variant = LARGE_DRAWER_FUSION_VARIANT_BY_TASK.get(student_task)
    aux_variant = variant is not None and is_aux_variant(variant)
    if aux_variant and occlusion_predictor_checkpoint is None:
        raise RuntimeError("Aux Student checkpoint requires occlusion predictor provenance")
    if not aux_variant and occlusion_predictor_checkpoint is not None:
        raise RuntimeError("Only Aux Students may record an occlusion predictor")
    state = model.state_dict()
    component_hashes = {}
    component_specs = (
        [("", "model")]
        if student_task in LARGE_DRAWER_FUSION_STUDENT_TASKS
        else [
            ("vision_encoder.", "vision_encoder"),
            ("temporal_fusion.", "temporal_fusion"),
            ("position_head.", "position_head"),
            ("action_head.", "action_head"),
        ]
    )
    if student_task in _FOUR_TACTILE_TASKS and student_task not in LARGE_DRAWER_FUSION_STUDENT_TASKS:
        component_specs += [("inner_tactile_encoder.", "inner_tactile_encoder"),
                            ("down_tactile_encoder.", "down_tactile_encoder")]
    elif student_task not in LARGE_DRAWER_FUSION_STUDENT_TASKS:
        component_specs += [("tactile_encoder.", "tactile_encoder")]
    for prefix, name in component_specs:
        component = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        component_hashes[f"{name}_state_dict_sha256"] = state_dict_sha256(component)
    payload = {
        "kind": kind,
        "version": version,
        "model_version": model_version,
        "task": student_task,
        "global_step": int(global_step),
        "model": state,
        "optimizer": optimizer.state_dict(),
        "student_input_contract": student_input_contract(student_task),
        "student_model_contract": expected_model_contract,
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
    if aux_variant:
        predictor_path = Path(str(occlusion_predictor_checkpoint)).expanduser().resolve()
        validate_occlusion_predictor_metadata(occlusion_predictor_metadata or {})
        payload["occlusion_predictor_checkpoint"] = str(predictor_path)
        payload["occlusion_predictor_checkpoint_sha256"] = sha256_file(predictor_path)
        payload["occlusion_predictor_metadata"] = dict(occlusion_predictor_metadata or {})
    return payload


def load_student_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_teacher_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint).expanduser().resolve(), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Not a Pulled-Drawer Student checkpoint")
    task = str(payload.get("task", ""))
    kind, version, model_version = _student_artifact_identity(task)
    if payload.get("kind") != kind:
        raise RuntimeError("Not a Pulled-Drawer Student checkpoint")
    if payload.get("version") != version:
        raise RuntimeError("Pulled-Drawer Student checkpoint version mismatch")
    if payload.get("model_version") != model_version:
        raise RuntimeError("Pulled-Drawer Student model mismatch")
    if payload.get("student_input_contract") != student_input_contract(task):
        raise RuntimeError("Pulled-Drawer Student input contract mismatch")
    if payload.get("student_model_contract") != _student_model_contract(task):
        raise RuntimeError("Pulled-Drawer Student model contract mismatch")
    if payload.get("normalization") != RMAGelSightPulledDrawerObservationNormalizer().contract():
        raise RuntimeError("Pulled-Drawer Student normalization mismatch")
    environment_contract = payload.get("student_environment_contract")
    if not isinstance(environment_contract, Mapping) or environment_contract.get("task") != task:
        raise RuntimeError("Pulled-Drawer Student environment task mismatch")
    teacher_manifest = payload.get("teacher_manifest")
    if (
        not isinstance(teacher_manifest, Mapping)
        or teacher_manifest.get("task")
        != GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK[task]
    ):
        raise RuntimeError("Pulled-Drawer Student and Teacher tasks are not paired")
    expected_teacher_kind, expected_teacher_version = _teacher_artifact_identity(
        str(teacher_manifest["task"])
    )
    if (
        teacher_manifest.get("kind") != expected_teacher_kind
        or teacher_manifest.get("version") != expected_teacher_version
    ):
        raise RuntimeError("Pulled-Drawer Student Teacher manifest is unsupported")
    _validate_environment_contract_for_task(
        str(teacher_manifest["task"]),
        teacher_manifest.get("environment_contract"),
        student=False,
    )
    _validate_environment_contract_for_task(
        task, environment_contract, student=True
    )
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("Pulled-Drawer Student has no model state_dict")
    component_specs = (
        [("", "model")]
        if task in LARGE_DRAWER_FUSION_STUDENT_TASKS
        else [
            ("vision_encoder.", "vision_encoder"),
            ("temporal_fusion.", "temporal_fusion"),
            ("position_head.", "position_head"),
            ("action_head.", "action_head"),
        ]
    )
    if task in _FOUR_TACTILE_TASKS and task not in LARGE_DRAWER_FUSION_STUDENT_TASKS:
        component_specs += [("inner_tactile_encoder.", "inner_tactile_encoder"),
                            ("down_tactile_encoder.", "down_tactile_encoder")]
    elif task not in LARGE_DRAWER_FUSION_STUDENT_TASKS:
        component_specs += [("tactile_encoder.", "tactile_encoder")]
    for prefix, name in component_specs:
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
    variant = LARGE_DRAWER_FUSION_VARIANT_BY_TASK.get(task)
    if variant is not None and is_aux_variant(variant):
        if (
            not isinstance(payload.get("occlusion_predictor_checkpoint_sha256"), str)
            or not isinstance(payload.get("occlusion_predictor_metadata"), Mapping)
        ):
            raise RuntimeError("Aux Student occlusion predictor provenance is missing")
        validate_occlusion_predictor_metadata(payload["occlusion_predictor_metadata"])
    elif "occlusion_predictor_checkpoint_sha256" in payload:
        raise RuntimeError("Non-Aux Student contains unexpected occlusion predictor provenance")
    if expected_teacher_checkpoint is not None:
        if payload.get("teacher_checkpoint_sha256") != sha256_file(expected_teacher_checkpoint):
            raise RuntimeError("Student was distilled from a different Teacher checkpoint")
        expected_manifest = load_teacher_manifest(expected_teacher_checkpoint)
        if expected_manifest.get("task") != GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK[task]:
            raise RuntimeError("Pulled-Drawer Student and Teacher tasks are not paired")
        if payload.get("teacher_manifest") != expected_manifest:
            raise RuntimeError("Student Teacher manifest mismatch")
    return payload


def make_student_model_for_task(
    task: str, *, pretrained_backbone: bool = False
) -> torch.nn.Module:
    if task in LARGE_DRAWER_FUSION_VARIANT_BY_TASK:
        return make_large_drawer_fusion_student(
            LARGE_DRAWER_FUSION_VARIANT_BY_TASK[task],
            pretrained_backbone=pretrained_backbone,
        )
    if _is_binary_student_task(task):
        if task in _FOUR_TACTILE_TASKS:
            return RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent(
                pretrained_backbone=pretrained_backbone
            )
        return RMAGelSightPulledDrawerBinaryTactileThreeFrameStudent(
            pretrained_backbone=pretrained_backbone
        )
    return RMAGelSightPulledDrawerThreeFrameStudent(pretrained_backbone=pretrained_backbone)


def make_student_model_for_checkpoint(
    payload: Mapping[str, Any], *, pretrained_backbone: bool = False
) -> torch.nn.Module:
    task = str(payload.get("task", ""))
    kind, version, _ = _student_artifact_identity(task)
    if payload.get("kind") != kind or payload.get("version") != version:
        raise RuntimeError("Unsupported Pulled-Drawer Student checkpoint")
    return make_student_model_for_task(task, pretrained_backbone=pretrained_backbone)


def load_student_model_state(
    model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> None:
    model.load_state_dict(dict(state_dict), strict=True)


__all__ = (
    "CYLINDER_FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION",
    "CYLINDER_FOUR_TACTILE_STUDENT_KIND",
    "CYLINDER_FOUR_TACTILE_TEACHER_KIND",
    "CYLINDER_FOUR_TACTILE_TEACHER_MANIFEST_VERSION",
    "FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION",
    "FOUR_TACTILE_STUDENT_KIND",
    "FOUR_TACTILE_TEACHER_KIND",
    "FOUR_TACTILE_TEACHER_MANIFEST_VERSION",
    "GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK",
    "GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK",
    "GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK",
    "GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK",
    "GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK",
    "GELSIGHT_PULLED_DRAWER_PROGRESS_TEACHER_TASK",
    "GELSIGHT_PULLED_DRAWER_STUDENT_TASKS",
    "GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK",
    "GELSIGHT_PULLED_DRAWER_TEACHER_TASKS",
    "GELSIGHT_PULLED_DRAWER_TEACHER_TASK",
    "GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK",
    "GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK",
    "GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK",
    "GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK",
    "GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK",
    "LARGE_DRAWER_CYLINDER_FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION",
    "LARGE_DRAWER_CYLINDER_FOUR_TACTILE_STUDENT_KIND",
    "LARGE_DRAWER_CYLINDER_FOUR_TACTILE_TEACHER_KIND",
    "LARGE_DRAWER_CYLINDER_FOUR_TACTILE_TEACHER_MANIFEST_VERSION",
    "LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_CHECKPOINT_VERSION",
    "LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_KIND",
    "LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_TEACHER_KIND",
    "LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_TEACHER_MANIFEST_VERSION",
    "LARGE_DRAWER_FUSION_STUDENT_TASKS",
    "LARGE_DRAWER_FUSION_REGISTRY",
    "LARGE_DRAWER_FUSION_VARIANT_BY_TASK",
    "MANIFEST_FILENAME",
    "PROGRESS_BINARY_STUDENT_CHECKPOINT_VERSION",
    "PROGRESS_BINARY_STUDENT_KIND",
    "PROGRESS_TEACHER_KIND",
    "PROGRESS_TEACHER_MANIFEST_VERSION",
    "STUDENT_CHECKPOINT_VERSION",
    "geometry_instance_sha256",
    "infer_teacher_checkpoint_policy_step",
    "load_encoder_initialization_checkpoint",
    "load_student_checkpoint",
    "load_student_model_state",
    "load_teacher_manifest",
    "load_teacher_policy_state",
    "make_student_model_for_checkpoint",
    "make_student_model_for_task",
    "make_student_payload",
    "sha256_file",
    "state_dict_sha256",
    "student_environment_contract",
    "teacher_environment_contract",
    "validate_live_teacher_contract",
    "write_teacher_manifest",
)
