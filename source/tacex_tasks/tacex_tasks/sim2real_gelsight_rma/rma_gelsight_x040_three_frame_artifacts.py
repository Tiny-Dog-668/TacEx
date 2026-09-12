"""Fail-closed Teacher and three-frame Student artifacts for GelSight X040 DR."""

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
from tacex_tasks.sim2real_grasp.rma_direct_action_student.artifacts import (
    load_encoder_initialization_checkpoint,
)
from tacex_tasks.sim2real_grasp.rma_x040_wide_models import (
    RMAX040WideObservationNormalizer,
)
from tacex_tasks.sim2real_grasp.rma_xy_artifacts import (
    RMA_XY_STUDENT_HEATMAP_DR_TASK,
)

from .gelsight_geometry import geometry_contract
from .rma_gelsight_models import RMAGelSightActorCore
from .sim2real_cube_real_alignment_gelsight_rma_env import (
    gelsight_compliant_grasp_contract,
)
from .rma_gelsight_x040_three_frame_models import (
    GELSIGHT_X040_THREE_FRAME_LEGACY_MODEL_VERSION,
    GELSIGHT_X040_THREE_FRAME_MODEL_VERSION,
    RMAGelSightX040ThreeFrameStudent,
    gelsight_x040_three_frame_model_contract,
)
from .rma_gelsight_x040_binary_tactile_models import (
    GELSIGHT_X040_BINARY_TACTILE_MODEL_VERSION,
    RMAGelSightX040BinaryTactileThreeFrameStudent,
    gelsight_x040_binary_tactile_model_contract,
)
from .sim2real_cube_real_alignment_gelsight_x040_three_frame_env import (
    GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK,
    GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
)
from .sim2real_cube_real_alignment_gelsight_x040_progress_env import (
    GELSIGHT_X040_PROGRESS_TEACHER_TASK,
    GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK,
)
from .sim2real_cube_real_alignment_gelsight_x040_progress_binary_tactile_env import (
    GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
)


MANIFEST_FILENAME = "rma_gelsight_x040_dr_size_buckets_manifest.json"
TEACHER_MANIFEST_VERSION = 7
LEGACY_STUDENT_CHECKPOINT_VERSION = 3
# v4 was assigned to the withdrawn LED-DR experiment and must not be reused.
PRE_GREEN_BASE_LED_STUDENT_CHECKPOINT_VERSION = 5
# v6 used the pre-offset v5 robot asset; v8 used the 26 mm v6 asset. Both are
# intentionally rejected by the current shared 21 mm geometry contract.
STUDENT_CHECKPOINT_VERSION = 10
STUDENT_KIND = "tacex_rma_gelsight_x040_dr_three_frame_student"
TEACHER_KIND = "tacex_rma_gelsight_x040_dr_size_buckets_teacher"
GELSIGHT_X040_TEACHER_TASKS = frozenset(
    {
        GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK,
        GELSIGHT_X040_PROGRESS_TEACHER_TASK,
    }
)
GELSIGHT_X040_STUDENT_TO_TEACHER_TASK = {
    GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK: (
        GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK
    ),
    GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK: (
        GELSIGHT_X040_PROGRESS_TEACHER_TASK
    ),
    GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK: (
        GELSIGHT_X040_PROGRESS_TEACHER_TASK
    ),
}
GELSIGHT_X040_STUDENT_TASKS = frozenset(
    GELSIGHT_X040_STUDENT_TO_TEACHER_TASK
)


def _expected_environment_profile(task: str) -> str:
    if task in {
        GELSIGHT_X040_PROGRESS_TEACHER_TASK,
        GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK,
        GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
    }:
        return "rma_gelsight_x040_progress_three_frame_v5"
    if task in {
        GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK,
        GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
    }:
        return "rma_gelsight_x040_dr_static_size_buckets_v7"
    raise RuntimeError(f"Unsupported GelSight X040 task: {task}")


def _validate_environment_contract_for_task(
    task: str, contract: object, *, student: bool
) -> None:
    if not isinstance(contract, Mapping):
        raise RuntimeError("GelSight X040 environment contract is missing")
    if contract.get("profile") != _expected_environment_profile(task):
        raise RuntimeError("GelSight X040 environment profile does not match task")
    is_progress = task in {
        GELSIGHT_X040_PROGRESS_TEACHER_TASK,
        GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK,
        GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
    }
    if is_progress and (
        "progress_reward" not in contract
        or contract.get("success_terminates_episode") is not True
    ):
        raise RuntimeError("GelSight X040 Progress reward contract is incomplete")
    if student and is_progress and "visual_domain_randomization" not in contract:
        raise RuntimeError("GelSight X040 Progress Student DR contract is incomplete")


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
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def teacher_environment_contract(cfg: Any) -> dict[str, Any]:
    hand = cfg.robot.actuators["panda_hand"]
    task = str(cfg.rma_task_id)
    profile = _expected_environment_profile(task)
    contract = {
        "profile": profile,
        "robot_profile": str(cfg.rma_robot_profile),
        "robot_asset_filename": Path(str(cfg.robot.spawn.usd_path)).name,
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
        "teacher_actor_feature_dim": int(cfg.rma_actor_feature_dim),
        "teacher_actor_contact_input": "physics_contact_label[2]",
        "position_frame": str(cfg.rma_position_frame),
        "gelsight_geometry": geometry_contract(),
        "cube_nominal_position_root_m": [float(value) for value in cfg.cube.init_state.pos],
        "cube_reset_half_range_xy_m": [
            float(cfg.cube_x_pos_range),
            float(cfg.cube_y_pos_range),
        ],
        "cube_position_curriculum_enabled": bool(cfg.cube_position_curriculum_enabled),
        "cube_size_buckets_m": [float(value) for value in cfg.cube_size_buckets_m],
        "cube_size_sampling": str(cfg.cube_size_sampling),
        "cube_size_assignment": str(cfg.cube_size_assignment),
        "cube_size_object_count_per_environment": int(
            cfg.cube_size_object_count_per_environment
        ),
        "cube_size_requires_num_envs_multiple_of_bucket_count": bool(
            cfg.cube_size_requires_num_envs_multiple_of_bucket_count
        ),
        "cube_scale_authoring": "usd_before_physx_start",
        "physics_layout": {
            "gpu_dynamics": bool(cfg.enable_gpu_dynamics),
            "replicate_physics": bool(cfg.scene.replicate_physics),
            "env_spacing_m": float(cfg.scene.env_spacing),
        },
        "cube_center_z_buckets_root_m": [
            float(cfg.plate_top_height_m) + 0.5 * float(value)
            for value in cfg.cube_size_buckets_m
        ],
        "arm_joint_reset_noise": {
            "joint_count": 7,
            "distribution": str(cfg.arm_joint_reset_noise_distribution),
            "std_rad": float(cfg.arm_joint_reset_noise_std_rad),
            "clip_abs_rad": float(cfg.arm_joint_reset_noise_clip_rad),
            "finger_joint_noise": "none",
        },
        "gelpad_contact_filters": list(cfg.rma_cube_contact_sensor.filter_prim_paths_expr),
        "compliant_grasp": gelsight_compliant_grasp_contract(cfg),
        "illegal_collision_scope": str(cfg.illegal_collision_scope),
        "cube_illegal_filters": list(
            cfg.cube_illegal_contact_sensor.filter_prim_paths_expr
        ),
        "table_illegal_filters": list(
            cfg.table_contact_sensor.filter_prim_paths_expr
        ),
        "illegal_collision_penalty": float(cfg.illegal_collision_penalty),
        "illegal_collision_penalty_threshold_n": {
            "schedule": "linear_clamped_global_policy_step",
            "start_n": float(cfg.illegal_collision_penalty_threshold_start_n),
            "end_n": float(cfg.illegal_collision_penalty_threshold_end_n),
            "start_step": int(cfg.illegal_collision_curriculum_start_step),
            "end_step": int(cfg.illegal_collision_curriculum_end_step),
        },
        "illegal_collision_terminates_episode": bool(
            cfg.illegal_collision_terminates_episode
        ),
        "ground_collision": "terminated",
        "illegal_collision_threshold_comparison": "strictly_greater_than",
        "success_terminates_episode": bool(cfg.rma_success_terminates_episode),
        "timeout_semantics": "truncated_only",
    }
    lift_reward_mode = str(getattr(cfg, "lift_reward_mode", ""))
    if lift_reward_mode in {
        "signed_normalized_progress_delta",
        "absolute_normalized_progress_per_step",
    }:
        compliant_grasp = dict(contract["compliant_grasp"])
        compliant_grasp.pop("excess_contact_force_penalty_per_policy_step", None)
        compliant_grasp["excess_contact_force_penalty"] = {
            "mode": str(cfg.rma_excess_contact_force_penalty_mode),
            "quadratic_weight": float(cfg.rma_excess_contact_force_quadratic_weight),
            "normalized_by_threshold": True,
        }
        contract["compliant_grasp"] = compliant_grasp
        reach_reward_mode = str(cfg.reach_reward_mode)
        if reach_reward_mode == "signed_normalized_proximity_delta":
            reach_contract = "signed_normalized_proximity_delta_after_first_transition"
        elif reach_reward_mode == "absolute_normalized_proximity_per_step":
            reach_contract = reach_reward_mode
        else:
            raise RuntimeError(f"Unsupported reach reward mode: {reach_reward_mode}")
        contract["progress_reward"] = {
            "reach": reach_contract,
            "reach_weight": float(cfg.reach_weight),
            "lift": lift_reward_mode,
            "lift_weight": float(cfg.lift_weight),
            "reach_holding_state_repeats_reward": (
                reach_reward_mode == "absolute_normalized_proximity_per_step"
            ),
            "lift_holding_state_repeats_reward": (
                lift_reward_mode == "absolute_normalized_progress_per_step"
            ),
            "contact_holding_state_repeats_reward": False,
            "contact": "signed_contact_acquisition_delta",
            "contact_weight": float(cfg.rma_contact_reward_weight),
            "success": "once_on_confirmed_terminal_success",
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
    return contract


def student_environment_contract(cfg: Any) -> dict[str, Any]:
    contract = teacher_environment_contract(cfg)
    contract.update(
        {
            "camera_position_delta_max_m": [
                float(value) for value in cfg.camera_position_delta_max_m
            ],
            "camera_rotation_delta_max_deg": [
                float(value) for value in cfg.camera_rotation_delta_max_deg
            ],
            "dr_curriculum_enabled": bool(cfg.dr_curriculum_enabled),
            "visual_alignment": {
                "shared_ground_visible": bool(cfg.ground.spawn.visible),
                "shared_ground_color_rgb": [
                    float(value) for value in cfg.ground.spawn.color
                ],
                "franka_body_visual_profile": str(cfg.rma_franka_visual_profile),
                "base_status_led": {
                    "subset_path": str(cfg.rma_base_status_led_subset_path),
                    "color_rgb": [
                        float(value) for value in cfg.rma_base_status_led_color_rgb
                    ],
                    "material": "UsdPreviewSurface_emissive",
                },
                "gelsight_case_and_gelpad_visuals": "preserved_from_gelsight_asset",
                "floor_panel_size_m": [float(value) for value in cfg.plate.spawn.size],
                "backdrop_size_m": [float(value) for value in cfg.backdrop.spawn.size],
            },
            "wrist_rgb_history": {
                "shape": [3, 224, 224, 3],
                "dtype": "uint8",
                "order": str(cfg.wrist_rgb_history_order),
                "stride_policy_steps": int(cfg.wrist_rgb_history_stride_policy_steps),
                "policy_frequency_hz": 30,
                "reset_fill": str(cfg.wrist_rgb_history_reset_fill),
            },
            "render_timing": {
                "physics_decimation": int(cfg.decimation),
                "render_interval": int(cfg.sim.render_interval),
            },
            "gelsight_depth_camera": {
                "left_update_latest_camera_pose": bool(
                    cfg.gsmini_left.sensor_camera_cfg.update_latest_camera_pose
                ),
                "right_update_latest_camera_pose": bool(
                    cfg.gsmini_right.sensor_camera_cfg.update_latest_camera_pose
                ),
                "tactile_rgb_float_to_uint8_scale": float(
                    cfg.rma_gelsight_tactile_rgb_float_scale
                ),
            },
            "gelsight_reference": "first_post_reset_frame_per_environment",
        }
    )
    if str(cfg.rma_task_id) in {
        GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK,
        GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
    }:
        contract["visual_domain_randomization"] = {
            "full_strength_from_first_step": not bool(cfg.dr_curriculum_enabled),
            "camera_pose": {
                "enabled": bool(cfg.camera_pose_randomization_enabled),
                "position_delta_max_m": [
                    float(value) for value in cfg.camera_position_delta_max_m
                ],
                "rotation_delta_max_deg": [
                    float(value) for value in cfg.camera_rotation_delta_max_deg
                ],
            },
            "camera_intrinsic_warp": {
                "enabled": bool(cfg.camera_intrinsic_warp_enabled),
                "focal_scale_range": [
                    float(value) for value in cfg.camera_focal_scale_range
                ],
                "principal_point_shift_max_px": [
                    float(value) for value in cfg.camera_principal_point_shift_max_px
                ],
            },
            "wrist_rgb": {
                "enabled": bool(cfg.wrist_visual_randomization_enabled),
                "brightness_enabled": bool(cfg.wrist_brightness_randomization_enabled),
                "brightness_range": [float(value) for value in cfg.wrist_brightness_range],
                "gamma_enabled": bool(cfg.wrist_gamma_randomization_enabled),
                "gamma_range": [float(value) for value in cfg.wrist_gamma_range],
                "contrast_enabled": bool(cfg.wrist_contrast_randomization_enabled),
                "contrast_range": [float(value) for value in cfg.wrist_contrast_range],
                "saturation_enabled": bool(cfg.wrist_saturation_randomization_enabled),
                "saturation_range": [float(value) for value in cfg.wrist_saturation_range],
                "hue_enabled": bool(cfg.wrist_hue_randomization_enabled),
                "hue_max_deg": float(cfg.wrist_hue_max_deg),
                "white_balance_enabled": bool(
                    cfg.wrist_white_balance_randomization_enabled
                ),
                "white_balance_shift_max": float(cfg.wrist_white_balance_shift_max),
                "blur_enabled": bool(cfg.wrist_blur_randomization_enabled),
                "blur_probability": float(cfg.wrist_blur_probability),
                "blur_kernel_sizes": [int(value) for value in cfg.wrist_blur_kernel_sizes],
                "gaussian_noise_enabled": bool(
                    cfg.wrist_gaussian_noise_randomization_enabled
                ),
                "gaussian_noise_std_range": [
                    float(value) for value in cfg.wrist_gaussian_noise_std_range
                ],
            },
            "dome_light": {
                "enabled": bool(cfg.light_randomization_enabled),
                "nominal_intensity": float(cfg.light_nominal_intensity),
                "nominal_color_rgb": [float(value) for value in cfg.light_nominal_color],
                "intensity_range": [float(value) for value in cfg.light_intensity_range],
                "nominal_color_temperature": float(
                    cfg.light_nominal_color_temperature
                ),
                "color_temperature_range": [
                    float(value) for value in cfg.light_color_temperature_range
                ],
            },
            "materials": {
                "ground_color_enabled": bool(cfg.ground_color_randomization_enabled),
                "plate_color_enabled": bool(cfg.plate_color_randomization_enabled),
                "plate_color_center": [float(value) for value in cfg.plate_color_center],
                "plate_color_min": [float(value) for value in cfg.plate_color_min],
                "plate_color_max": [float(value) for value in cfg.plate_color_max],
                "backdrop_color_enabled": bool(cfg.backdrop_color_randomization_enabled),
                "backdrop_color_center": [
                    float(value) for value in cfg.backdrop_color_center
                ],
                "backdrop_color_min": [float(value) for value in cfg.backdrop_color_min],
                "backdrop_color_max": [float(value) for value in cfg.backdrop_color_max],
            },
        }
    return contract


def _is_binary_tactile_student_task(task: str | None) -> bool:
    return task == GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK


def student_input_contract(student_task: str | None = None) -> dict[str, Any]:
    contract = {
        "input_order": [
            "wrist_rgb_history",
            "proprio_obs",
            "action_history",
            "gsmini_left_rgb",
            "gsmini_right_rgb",
            "gsmini_left_reference_rgb",
            "gsmini_right_reference_rgb",
        ],
        "wrist_rgb_history": [3, 224, 224, 3],
        "proprio_obs": [15],
        "action_history": [4],
        "tactile_rgb": [96, 128, 3],
        "tactile_delta": "signed_float32_current_minus_reference_div_255",
        "runtime_output": {
            "action": [4],
            "left_right_contact_probability": [2],
            "cube_position_root_m": [3],
        },
        "training_only_labels": ["rma_cube_pos", "rma_contact_state"],
    }
    if _is_binary_tactile_student_task(student_task):
        contract["tactile_delta"] = (
            "float32_single_channel_max_abs_rgb_current_minus_reference_"
            "strict_gt_5_u8"
        )
        contract["tactile_network_input"] = [1, 96, 128]
        contract["tactile_binary_threshold_u8"] = 5.0
        contract["tactile_binary_values"] = [0.0, 1.0]
    return contract


def _student_model_version(student_task: str) -> int:
    if _is_binary_tactile_student_task(student_task):
        return GELSIGHT_X040_BINARY_TACTILE_MODEL_VERSION
    return GELSIGHT_X040_THREE_FRAME_MODEL_VERSION


def _student_model_contract(student_task: str) -> dict[str, object]:
    if _is_binary_tactile_student_task(student_task):
        return gelsight_x040_binary_tactile_model_contract()
    return gelsight_x040_three_frame_model_contract()


def write_teacher_manifest(base_env: Any, params_dir: str | Path, agent_cfg: Mapping[str, Any]) -> Path:
    task = str(base_env.cfg.rma_task_id)
    if task not in GELSIGHT_X040_TEACHER_TASKS:
        raise RuntimeError(f"Unsupported GelSight X040 Teacher task: {task}")
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
        "task": task,
        "model_version": RMA_MODEL_VERSION,
        "actor_inputs": {
            "proprio_obs": 15,
            "action_history": 4,
            "rma_cube_pos": 3,
            "rma_contact_state": 2,
        },
        "actor_contract": RMAGelSightActorCore().contract(),
        "normalization": RMAObservationNormalizer().contract(),
        "environment_contract": teacher_environment_contract(base_env.cfg),
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
        raise FileNotFoundError(f"GelSight X040 DR Teacher manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != TEACHER_KIND or manifest.get("version") != TEACHER_MANIFEST_VERSION:
        raise RuntimeError("Unsupported GelSight X040 DR Teacher manifest")
    if manifest.get("task") not in GELSIGHT_X040_TEACHER_TASKS:
        raise RuntimeError("GelSight X040 DR Teacher task mismatch")
    _validate_environment_contract_for_task(
        str(manifest["task"]), manifest.get("environment_contract"), student=False
    )
    if manifest.get("model_version") != RMA_MODEL_VERSION:
        raise RuntimeError("GelSight X040 DR Teacher model mismatch")
    if manifest.get("actor_contract") != RMAGelSightActorCore().contract():
        raise RuntimeError("GelSight X040 DR Teacher Actor contract mismatch")
    if manifest.get("normalization") != RMAObservationNormalizer().contract():
        raise RuntimeError("GelSight X040 DR Teacher normalization mismatch")
    offset = manifest.get("curriculum_policy_step_offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise RuntimeError("GelSight X040 DR Teacher curriculum offset is invalid")
    for name, expected_hash in manifest.get("run_config_sha256", {}).items():
        path = _run_dir(checkpoint_path) / "params" / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Teacher run config hash mismatch: {path}")
    return manifest


def validate_live_teacher_contract(cfg: Any, manifest: Mapping[str, Any]) -> None:
    live_task = str(cfg.rma_task_id)
    expected_teacher_task = GELSIGHT_X040_STUDENT_TO_TEACHER_TASK.get(
        live_task, live_task
    )
    if manifest.get("task") != expected_teacher_task:
        raise RuntimeError("Live GelSight X040 task is not paired with this Teacher")
    if manifest.get("environment_contract") != teacher_environment_contract(cfg):
        raise RuntimeError("Live GelSight X040 DR environment differs from Teacher contract")


def load_teacher_policy_state(checkpoint: str | Path, device: str | torch.device) -> dict[str, torch.Tensor]:
    load_teacher_manifest(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = payload.get("policy") if isinstance(payload, Mapping) else None
    if not isinstance(policy, Mapping):
        raise RuntimeError("Teacher checkpoint has no policy state_dict")
    return dict(policy)


def infer_teacher_checkpoint_policy_step(checkpoint: str | Path) -> int:
    path = Path(checkpoint).expanduser().resolve()
    suffix = path.stem.removeprefix("agent_")
    if not suffix.isdigit():
        raise RuntimeError("Teacher checkpoint must be named agent_<step>.pt")
    return int(load_teacher_manifest(path)["curriculum_policy_step_offset"]) + int(suffix)


def make_student_payload(
    *,
    student_task: str = GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    teacher_checkpoint: str | Path,
    teacher_manifest: Mapping[str, Any],
    teacher_actor_state_dict: Mapping[str, torch.Tensor],
    encoder_init_checkpoint: str | Path,
    encoder_init_payload: Mapping[str, Any],
    student_env_contract: Mapping[str, Any],
    loss: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
) -> dict[str, Any]:
    if student_task not in GELSIGHT_X040_STUDENT_TASKS:
        raise RuntimeError(f"Unsupported GelSight X040 Student task: {student_task}")
    if teacher_manifest.get("task") != GELSIGHT_X040_STUDENT_TO_TEACHER_TASK[student_task]:
        raise RuntimeError("GelSight X040 Student and Teacher tasks are not paired")
    expected_model_contract = _student_model_contract(student_task)
    if not hasattr(model, "contract") or model.contract() != expected_model_contract:
        raise RuntimeError("GelSight X040 Student model does not match its task")
    _validate_environment_contract_for_task(
        str(teacher_manifest["task"]),
        teacher_manifest.get("environment_contract"),
        student=False,
    )
    _validate_environment_contract_for_task(
        student_task, student_env_contract, student=True
    )
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
        "model_version": _student_model_version(student_task),
        "task": student_task,
        "global_step": int(global_step),
        "model": state,
        "optimizer": optimizer.state_dict(),
        "student_input_contract": student_input_contract(student_task),
        "student_model_contract": expected_model_contract,
        "normalization": model.normalizer.contract(),
        "student_environment_contract": dict(student_env_contract),
        "teacher_checkpoint": str(Path(teacher_checkpoint).expanduser().resolve()),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_manifest": dict(teacher_manifest),
        "teacher_actor_state_dict_sha256": state_dict_sha256(teacher_actor_state_dict),
        "encoder_init_checkpoint": str(
            Path(encoder_init_checkpoint).expanduser().resolve()
        ),
        "encoder_init_checkpoint_sha256": sha256_file(encoder_init_checkpoint),
        "encoder_init_task": encoder_init_payload.get("task"),
        "encoder_init_state_dict_sha256": encoder_init_payload.get(
            "vision_encoder_state_dict_sha256"
        ),
        "loss": dict(loss),
        "optimizer_config": dict(optimizer_config),
        "training_privileged_inputs": {
            "rma_cube_pos": [3],
            "rma_contact_state": [2],
        },
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
        raise RuntimeError("Not a GelSight X040 DR three-frame Student checkpoint")
    version = payload.get("version")
    legacy = False
    if version != STUDENT_CHECKPOINT_VERSION:
        raise RuntimeError("GelSight X040 DR Student version mismatch")
    task = payload.get("task")
    if task not in GELSIGHT_X040_STUDENT_TASKS:
        raise RuntimeError("GelSight X040 DR Student task mismatch")
    expected_model_version = (
        GELSIGHT_X040_THREE_FRAME_LEGACY_MODEL_VERSION
        if legacy
        else _student_model_version(str(task))
    )
    if payload.get("model_version") != expected_model_version:
        raise RuntimeError("GelSight X040 DR Student model version mismatch")
    teacher_manifest = payload.get("teacher_manifest")
    if (
        not isinstance(teacher_manifest, Mapping)
        or teacher_manifest.get("task")
        != GELSIGHT_X040_STUDENT_TO_TEACHER_TASK[task]
    ):
        raise RuntimeError("GelSight X040 Student embedded Teacher task mismatch")
    _validate_environment_contract_for_task(
        str(teacher_manifest["task"]),
        teacher_manifest.get("environment_contract"),
        student=False,
    )
    _validate_environment_contract_for_task(
        str(task), payload.get("student_environment_contract"), student=True
    )
    if payload.get("student_input_contract") != student_input_contract(str(task)):
        raise RuntimeError("GelSight X040 DR Student input contract mismatch")
    expected_model_contract = (
        gelsight_x040_three_frame_model_contract(legacy_position_normalization=True)
        if legacy
        else _student_model_contract(str(task))
    )
    if payload.get("student_model_contract") != expected_model_contract:
        raise RuntimeError("GelSight X040 DR Student model contract mismatch")
    expected_normalization = (
        RMAObservationNormalizer().contract()
        if legacy
        else RMAX040WideObservationNormalizer().contract()
    )
    if payload.get("normalization") != expected_normalization:
        raise RuntimeError("GelSight X040 DR Student normalization mismatch")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("GelSight X040 DR Student has no model state_dict")
    for prefix, name in (
        ("vision_encoder.", "vision_encoder"),
        ("temporal_fusion.", "temporal_fusion"),
        ("tactile_encoder.", "tactile_encoder"),
        ("position_head.", "position_head"),
        ("action_head.", "action_head"),
    ):
        component = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not component or payload.get(f"{name}_state_dict_sha256") != state_dict_sha256(component):
            raise RuntimeError(f"GelSight X040 DR Student component hash mismatch: {name}")
    if not isinstance(payload.get("teacher_actor_state_dict_sha256"), str):
        raise RuntimeError("GelSight X040 DR Student Teacher Actor provenance is missing")
    if not legacy and (
        not isinstance(payload.get("encoder_init_checkpoint_sha256"), str)
        or not isinstance(payload.get("encoder_init_state_dict_sha256"), str)
        or payload.get("encoder_init_task") != RMA_XY_STUDENT_HEATMAP_DR_TASK
    ):
        raise RuntimeError("GelSight X040 DR Student encoder initialization provenance is missing")
    if expected_teacher_checkpoint is not None:
        if payload.get("teacher_checkpoint_sha256") != sha256_file(expected_teacher_checkpoint):
            raise RuntimeError("Student was distilled from a different Teacher checkpoint")
        if payload.get("teacher_manifest") != load_teacher_manifest(expected_teacher_checkpoint):
            raise RuntimeError("Student Teacher manifest mismatch")
    return payload


def make_student_model_for_checkpoint(
    payload: Mapping[str, Any], *, pretrained_backbone: bool = False
) -> torch.nn.Module:
    """Construct the matching normalization profile for a validated checkpoint."""
    version = payload.get("version")
    if version != STUDENT_CHECKPOINT_VERSION:
        raise RuntimeError("GelSight X040 DR Student version mismatch")
    task = str(payload.get("task"))
    if _is_binary_tactile_student_task(task):
        return RMAGelSightX040BinaryTactileThreeFrameStudent(
            pretrained_backbone=pretrained_backbone
        )
    return RMAGelSightX040ThreeFrameStudent(
        pretrained_backbone=pretrained_backbone,
        legacy_position_normalization=version == LEGACY_STUDENT_CHECKPOINT_VERSION,
    )


def load_student_model_state(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict(dict(state_dict), strict=True)


__all__ = (
    "GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK",
    "GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK",
    "GELSIGHT_X040_PROGRESS_TEACHER_TASK",
    "GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK",
    "GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK",
    "GELSIGHT_X040_STUDENT_TASKS",
    "GELSIGHT_X040_STUDENT_TO_TEACHER_TASK",
    "GELSIGHT_X040_TEACHER_TASKS",
    "LEGACY_STUDENT_CHECKPOINT_VERSION",
    "PRE_GREEN_BASE_LED_STUDENT_CHECKPOINT_VERSION",
    "STUDENT_CHECKPOINT_VERSION",
    "TEACHER_MANIFEST_VERSION",
    "MANIFEST_FILENAME",
    "load_student_checkpoint",
    "load_encoder_initialization_checkpoint",
    "load_student_model_state",
    "load_teacher_manifest",
    "load_teacher_policy_state",
    "infer_teacher_checkpoint_policy_step",
    "make_student_payload",
    "make_student_model_for_checkpoint",
    "sha256_file",
    "state_dict_sha256",
    "student_environment_contract",
    "student_input_contract",
    "teacher_environment_contract",
    "validate_live_teacher_contract",
    "write_teacher_manifest",
)
