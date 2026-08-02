"""Persist and verify frozen vision encoders used by sim-to-real policies."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


ENCODER_ARTIFACT_FILENAME = "vision_encoder_resnet18.pt"
ENCODER_MANIFEST_FILENAME = "vision_encoder_resnet18.json"
STATE_DICT_HASH_ALGORITHM = "sorted_key_dtype_shape_raw_bytes_v1"
POLICY_CONTRACT_VERSION = 9
SUPPORTED_POLICY_CONTRACT_VERSIONS = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, POLICY_CONTRACT_VERSION}
)
REAL_ALIGNMENT_V7_ACTION_SCALES = [0.01, 0.01, 0.01, 0.002]
REAL_ALIGNMENT_V8_ACTION_SCALES = [0.025, 0.025, 0.025, 0.002]
REAL_ALIGNMENT_V9_ACTION_SCALES = [0.025, 0.025, 0.025, 0.005]
ACTION_HISTORY_CONTRACT_V1 = "processed_action_pre_privileged_dz_gate_v1"
ACTION_HISTORY_CONTRACT_V2 = "per_dimension_processed_action_pre_privileged_dz_gate_v2"
ACTION_HISTORY_CONTRACT_V3 = "per_dimension_processed_action_no_privileged_gate_v3"
REAL_ALIGNMENT_CUBE_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-v0"
REAL_ALIGNMENT_CUBE_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-DR-v0"
REAL_ALIGNMENT_CUBE_TASKS = frozenset({REAL_ALIGNMENT_CUBE_TASK, REAL_ALIGNMENT_CUBE_DR_TASK})
STRICT_SIM2REAL_CUBE_TASKS = frozenset(
    {
        "TacEx-Sim2Real-Cube-Grasp-v0",
        *REAL_ALIGNMENT_CUBE_TASKS,
    }
)
_RUN_CONFIG_FILENAMES = ("agent.yaml", "agent.pkl", "env.yaml", "env.pkl")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in a deterministic order."""
    digest = hashlib.sha256()
    digest.update(f"TacEx:{STATE_DICT_HASH_ALGORITHM}\0".encode("utf-8"))
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Encoder state entry {name!r} is not a tensor: {type(tensor).__name__}")
        tensor = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        # Reshape scalar buffers such as BatchNorm.num_batches_tracked before
        # viewing them as bytes; PyTorch disallows dtype views on 0-D tensors.
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, byteorder="little", signed=False))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, byteorder="little", signed=False))
        digest.update(raw)
    return digest.hexdigest()


def module_state_dict_sha256(module: torch.nn.Module) -> str:
    """Return the canonical state-dict hash for a module."""
    return state_dict_sha256(module.state_dict())


def checkpoint_run_dir(checkpoint_path: str | Path) -> Path:
    """Resolve a checkpoint stored under the required <run>/checkpoints layout."""
    checkpoint_path = Path(checkpoint_path).expanduser().absolute()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError(
            "Checkpoint must use the run layout <run>/checkpoints/<file>: "
            f"{checkpoint_path}"
        )
    return checkpoint_path.parent.parent


def normalized_policy_output_expression(value: Any) -> str:
    """Normalize a skrl model output expression for contract comparison."""
    return "".join(str(value or "").split()).lower()


def load_saved_agent_config(params_dir: str | Path) -> tuple[dict[str, Any], Path]:
    """Load the immutable-by-convention agent config stored with one run."""
    params_dir = Path(params_dir)
    yaml_path = params_dir / "agent.yaml"
    if yaml_path.is_file():
        import yaml

        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TypeError(f"Saved agent config must be a dict: {yaml_path}")
        return config, yaml_path

    pickle_path = params_dir / "agent.pkl"
    if pickle_path.is_file():
        import pickle

        with pickle_path.open("rb") as file:
            config = pickle.load(file)
        if not isinstance(config, dict):
            raise TypeError(f"Saved agent config must be a dict: {pickle_path}")
        return config, pickle_path
    raise FileNotFoundError(f"Run has no params/agent.yaml or params/agent.pkl: {params_dir}")


def load_saved_env_config(params_dir: str | Path) -> tuple[Any, Path]:
    """Load the exact pickled environment config saved by the training entry point."""
    import pickle

    pickle_path = Path(params_dir) / "env.pkl"
    if not pickle_path.is_file():
        raise FileNotFoundError(f"Run has no params/env.pkl: {pickle_path}")
    with pickle_path.open("rb") as file:
        config = pickle.load(file)
    return config, pickle_path


def validate_live_vision_contract(base_env: Any, manifest: Mapping[str, Any]) -> None:
    """Validate live encoder identity and ImageNet normalization against a run manifest."""
    expected_architecture = getattr(base_env, "_resnet18_architecture", "unknown")
    expected_weights_id = getattr(base_env, "_resnet18_weights_id", "unknown")
    if manifest.get("architecture") != expected_architecture:
        raise RuntimeError(
            "Vision encoder architecture mismatch: "
            f"expected {expected_architecture!r}, got {manifest.get('architecture')!r}"
        )
    if manifest.get("weights_id") != expected_weights_id:
        raise RuntimeError(
            "Vision encoder weights identifier mismatch: "
            f"expected {expected_weights_id!r}, got {manifest.get('weights_id')!r}"
        )

    expected = manifest.get("normalization", {})
    mean = getattr(base_env, "_imgnet_mean", None)
    std = getattr(base_env, "_imgnet_std", None)
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise RuntimeError("Live environment has no ImageNet normalization tensors.")
    actual_mean = mean.detach().cpu().reshape(-1)
    actual_std = std.detach().cpu().reshape(-1)
    expected_mean = torch.tensor(expected.get("mean", []), dtype=actual_mean.dtype)
    expected_std = torch.tensor(expected.get("std", []), dtype=actual_std.dtype)
    if not torch.equal(actual_mean, expected_mean) or not torch.equal(actual_std, expected_std):
        raise RuntimeError(
            "ImageNet normalization does not match the training manifest: "
            f"expected {expected}, got mean={actual_mean.tolist()}, std={actual_std.tolist()}"
        )


def validate_resume_agent_config(current: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    """Require checkpoint-bearing model and preprocessing semantics to stay unchanged."""
    if current.get("models") != saved.get("models"):
        raise RuntimeError("Current model config differs from the strict checkpoint run model config.")
    current_agent = current.get("agent", {})
    saved_agent = saved.get("agent", {})
    for key in (
        "state_preprocessor",
        "state_preprocessor_kwargs",
        "value_preprocessor",
        "value_preprocessor_kwargs",
    ):
        if current_agent.get(key) != saved_agent.get(key):
            raise RuntimeError(
                f"Current agent preprocessing config differs from the strict checkpoint for {key}."
            )


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary_path, path)


def _normalization_from_env(base_env: Any) -> dict[str, list[float]]:
    mean = getattr(base_env, "_imgnet_mean", None)
    std = getattr(base_env, "_imgnet_std", None)
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise RuntimeError("Vision encoder normalization tensors are unavailable on the environment.")
    return {
        "mean": mean.detach().cpu().reshape(-1).tolist(),
        "std": std.detach().cpu().reshape(-1).tolist(),
    }


def _observation_dim(base_env: Any, key: str) -> int | None:
    observation_space = getattr(base_env.cfg, "observation_space", None)
    if isinstance(observation_space, Mapping):
        value = observation_space.get(key)
        if isinstance(value, int):
            return int(value)
        shape = getattr(value, "shape", None)
        if shape is not None:
            total = 1
            for dimension in shape:
                total *= int(dimension)
            return total
    return None


def _gripper_control_mode(cfg: Any) -> str:
    return str(getattr(cfg, "gripper_control_mode", "legacy_per_finger_delta_per_physics_application"))


def _action_scales(cfg: Any) -> list[float]:
    action_scale = float(cfg.action_scale)
    if _gripper_control_mode(cfg) == "total_width_delta_cached_target":
        return [action_scale, action_scale, action_scale, float(cfg.gripper_width_delta_scale)]
    return [action_scale] * int(cfg.action_space)


def _float_vector(value: Any) -> list[float]:
    """Convert a configuration vector to a JSON-stable list of floats."""
    return [float(component) for component in value]


def _build_policy_contract(
    base_env: Any,
    params_dir: Path,
    task: str,
    policy_output: Any,
) -> dict[str, Any]:
    output_expression = normalized_policy_output_expression(policy_output)
    actor_mean_transform = "tanh" if output_expression == "tanh(actions)" else "identity"
    if task in STRICT_SIM2REAL_CUBE_TASKS and actor_mean_transform != "tanh":
        raise RuntimeError(f"Strict Cube task {task} requires policy output tanh(ACTIONS).")

    cfg = base_env.cfg
    camera_cfg = getattr(cfg, "wrist_camera", None)
    action_scale = float(cfg.action_scale)
    action_scales = _action_scales(cfg)
    gripper_control_mode = _gripper_control_mode(cfg)
    total_width_delta_control = gripper_control_mode == "total_width_delta_cached_target"
    gripper_width_delta_scale = (
        float(cfg.gripper_width_delta_scale) if total_width_delta_control else None
    )
    max_gripper_opening_width = float(getattr(cfg, "max_gripper_opening_width", 0.08))
    sim_dt = float(cfg.sim.dt)
    decimation = int(cfg.decimation)
    privileged_dz_gate_enabled = bool(getattr(cfg, "privileged_dz_gate_enabled", True))
    config_hashes = {}
    for filename in _RUN_CONFIG_FILENAMES:
        path = params_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required run config artifact is missing: {path}")
        config_hashes[filename] = sha256_file(path)

    contract = {
        "version": POLICY_CONTRACT_VERSION,
        "task": task,
        "actor_mean_transform": actor_mean_transform,
        "policy_output_expression": output_expression,
        "action_history": (
            ACTION_HISTORY_CONTRACT_V2
            if privileged_dz_gate_enabled
            else ACTION_HISTORY_CONTRACT_V3
        ),
        "action_history_units": (
            ["m_relative_x", "m_relative_y", "m_relative_z", "m_total_width_delta"]
            if total_width_delta_control
            else ["environment_processed_action"] * int(cfg.action_space)
        ),
        "deployment_history_source": "clipped_action",
        "deployment_history_scales": action_scales,
        "deployment_history_delay_steps": 1,
        "action_dim": int(cfg.action_space),
        "action_scale": action_scale,
        "action_scales": action_scales,
        "action_noise_scale": float(getattr(cfg, "action_noise_scale", 0.0)),
        "xyz_command_frame": "robot_root",
        "privileged_dz_gate": (
            "ground_truth_object_xy_applied_after_action_history"
            if privileged_dz_gate_enabled
            else "disabled"
        ),
        "gripper_control_mode": gripper_control_mode,
        "gripper_width_delta_scale": gripper_width_delta_scale,
        "gripper_width_bounds": [0.0, max_gripper_opening_width],
        "gripper_width_delta_updates_per_policy_step": 1 if total_width_delta_control else None,
        "gripper_joint_target_mapping": (
            "symmetric_half_total_width"
            if total_width_delta_control
            else "legacy_per_finger_increment_from_measured_joint_position"
        ),
        "gripper_finger_target_increment_per_physics_application_per_processed_unit": (
            None if total_width_delta_control else 0.2
        ),
        "gripper_target_reapplications_per_policy_step": decimation,
        "gripper_realized_motion": (
            "cached_total_width_target_is_updated_once_per_policy_step_and_reapplied_each_physics_step"
            if total_width_delta_control
            else "depends_on_PD_tracking_and_is_not_a_fixed_per_policy_step_delta"
        ),
        "sim_dt": sim_dt,
        "decimation": decimation,
        "nominal_policy_frequency_hz": 1.0 / (sim_dt * decimation),
        "camera_height": int(camera_cfg.height) if camera_cfg is not None else None,
        "camera_width": int(camera_cfg.width) if camera_cfg is not None else None,
        "actor_observation_dims": {
            "action_history": _observation_dim(base_env, "action_history"),
            "proprio_obs": _observation_dim(base_env, "proprio_obs"),
            "wrist_resnet": _observation_dim(base_env, "wrist_resnet"),
        },
        "run_config_sha256": config_hashes,
    }
    if task in STRICT_SIM2REAL_CUBE_TASKS:
        contract.update(
            {
                "lift_reference_mode": str(cfg.lift_reference_mode),
                "lift_reward_start_delta_m": float(cfg.lift_reward_start_delta),
                "success_lift_delta_m": float(cfg.success_lift_delta),
                "lift_reward_requires_upright": bool(cfg.lift_reward_requires_upright),
                "success_requires_upright": bool(
                    getattr(cfg, "success_requires_upright", True)
                ),
                "lift_tilt_curriculum_enabled": bool(cfg.lift_tilt_curriculum_enabled),
                "lift_tilt_curriculum_start_deg": float(cfg.lift_tilt_curriculum_start_deg),
                "lift_tilt_curriculum_end_deg": float(cfg.lift_tilt_curriculum_end_deg),
                "lift_tilt_curriculum_start_step": int(cfg.lift_tilt_curriculum_start_step),
                "lift_tilt_curriculum_end_step": int(cfg.lift_tilt_curriculum_end_step),
                "lift_tilt_curriculum_step_offset": int(cfg.lift_tilt_curriculum_step_offset),
            }
        )
    if task in REAL_ALIGNMENT_CUBE_TASKS:
        contract.update(
            {
                "arm_ik_tcp_source": str(cfg.arm_ik_tcp_source),
                "arm_ik_tcp_offset_m": _float_vector(cfg.arm_ik_tcp_offset_m),
                "reach_center_source": str(cfg.reach_center_source),
                "fingertip_local_offset_m": _float_vector(cfg.fingertip_local_offset_m),
                "critic_gripper_position_source": str(cfg.reach_center_source),
                "action_gate_center_source": str(cfg.reach_center_source),
                "camera_update_period_s": float(camera_cfg.update_period),
                "nominal_camera_frequency_hz": 1.0 / float(camera_cfg.update_period),
                "camera_pose_position_m": _float_vector(camera_cfg.offset.pos),
                "camera_pose_quaternion_wxyz": _float_vector(camera_cfg.offset.rot),
                "camera_pose_convention": str(camera_cfg.offset.convention),
                "camera_raw_resolution_wh": [
                    int(value) for value in cfg.camera_raw_resolution
                ],
                "camera_raw_intrinsic_matrix": _float_vector(
                    cfg.camera_raw_intrinsic_matrix
                ),
                "camera_crop_roi_xywh": [
                    int(value) for value in cfg.camera_crop_roi_xywh
                ],
                "camera_model_intrinsic_matrix": _float_vector(
                    cfg.camera_model_intrinsic_matrix
                ),
                "camera_native_render_intrinsic_matrix": _float_vector(
                    cfg.camera_native_render_intrinsic_matrix
                ),
                "camera_nominal_intrinsic_compensation_enabled": bool(
                    cfg.camera_nominal_intrinsic_compensation_enabled
                ),
                "camera_nominal_intrinsic_compensation_mode": str(
                    cfg.camera_nominal_intrinsic_compensation_mode
                ),
                "camera_resize_interpolation": str(cfg.camera_resize_interpolation),
                "deployment_camera_serial": str(cfg.deployment_camera_serial),
                "cube_nominal_xy_m": _float_vector(cfg.cube.init_state.pos[:2]),
                "cube_full_xy_bounds_m": [
                    float(cfg.cube.init_state.pos[0]) - float(cfg.cube_x_pos_range),
                    float(cfg.cube.init_state.pos[0]) + float(cfg.cube_x_pos_range),
                    float(cfg.cube.init_state.pos[1]) - float(cfg.cube_y_pos_range),
                    float(cfg.cube.init_state.pos[1]) + float(cfg.cube_y_pos_range),
                ],
                "cube_position_curriculum_enabled": bool(
                    cfg.cube_position_curriculum_enabled
                ),
                "cube_position_curriculum_initial_xy_range_m": [
                    float(cfg.cube_position_curriculum_initial_x_range),
                    float(cfg.cube_position_curriculum_initial_y_range),
                ],
                "cube_position_curriculum_start_step": int(
                    cfg.cube_position_curriculum_start_step
                ),
                "cube_position_curriculum_end_step": int(
                    cfg.cube_position_curriculum_end_step
                ),
                "cube_position_curriculum_step_offset": int(
                    cfg.cube_position_curriculum_step_offset
                ),
                "plate_thickness_m": float(cfg.plate_thickness_m),
                "plate_top_height_m": float(cfg.plate_top_height_m),
                "table_collision_force_threshold_n": float(
                    cfg.table_collision_force_threshold_n
                ),
                "table_collision_penalty": float(cfg.table_collision_penalty),
                "table_collision_robot_body_names": list(
                    cfg.table_collision_robot_body_names
                ),
                "table_contact_sensor_history_length": int(
                    cfg.table_contact_sensor.history_length
                ),
                "episode_length_s": float(cfg.episode_length_s),
                "max_episode_length_steps": int(base_env.max_episode_length),
            }
        )
    return contract


def validate_sim2real_policy_contract(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    task: str,
    params_dir: str | Path,
) -> dict[str, Any]:
    """Validate task identity, policy semantics, and saved run-config hashes."""
    if isinstance(manifest_or_path, Mapping):
        manifest = dict(manifest_or_path)
    else:
        manifest = json.loads(Path(manifest_or_path).read_text(encoding="utf-8"))
    contract = manifest.get("policy_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Vision encoder manifest has no sim2real policy_contract.")
    contract_version = contract.get("version")
    if contract_version not in SUPPORTED_POLICY_CONTRACT_VERSIONS:
        raise RuntimeError(f"Unsupported sim2real policy contract version: {contract.get('version')!r}")
    if contract.get("task") != task:
        raise RuntimeError(
            f"Checkpoint task mismatch: requested {task!r}, run contract declares {contract.get('task')!r}."
        )
    if contract_version == 1:
        if task in REAL_ALIGNMENT_CUBE_TASKS:
            raise RuntimeError(
                "Real-Alignment Cube contract v1 uses the obsolete per-finger gripper increment semantics. "
                "Retrain with the total-width gripper contract."
            )
        if contract.get("action_history") != ACTION_HISTORY_CONTRACT_V1:
            raise RuntimeError(f"Unsupported action-history contract: {contract.get('action_history')!r}")
        expected_history_deployment = {
            "deployment_history_source": "clipped_action",
            "deployment_history_scale": contract.get("action_scale"),
            "deployment_history_delay_steps": 1,
        }
    elif contract_version in {2, 3}:
        if contract.get("action_history") != ACTION_HISTORY_CONTRACT_V2:
            raise RuntimeError(f"Unsupported action-history contract: {contract.get('action_history')!r}")
        expected_history_deployment = {
            "deployment_history_source": "clipped_action",
            "deployment_history_scales": contract.get("action_scales"),
            "deployment_history_delay_steps": 1,
        }
    else:
        privileged_dz_gate = contract.get("privileged_dz_gate")
        expected_action_history = (
            ACTION_HISTORY_CONTRACT_V3
            if privileged_dz_gate == "disabled"
            else ACTION_HISTORY_CONTRACT_V2
        )
        if contract.get("action_history") != expected_action_history:
            raise RuntimeError(
                "Unsupported action-history/dz-gate contract: "
                f"expected {expected_action_history!r}, got {contract.get('action_history')!r}."
            )
        expected_history_deployment = {
            "deployment_history_source": "clipped_action",
            "deployment_history_scales": contract.get("action_scales"),
            "deployment_history_delay_steps": 1,
        }
    if task in STRICT_SIM2REAL_CUBE_TASKS and contract_version == 1:
        raise RuntimeError(
            "Strict Cube contract v1 uses obsolete action-history and privileged dz-gate semantics. "
            "Retrain without the ground-truth object-XY dz gate."
        )
    if task in STRICT_SIM2REAL_CUBE_TASKS and contract_version == 2:
        raise RuntimeError(
            "Strict Cube contract v2 uses obsolete lowest-corner/non-curriculum lift semantics. "
            "Retrain with the center-of-mass lift curriculum contract."
        )
    if task in STRICT_SIM2REAL_CUBE_TASKS and contract_version == 3:
        raise RuntimeError(
            "Strict Cube contract v3 uses the obsolete ground-truth object-XY dz gate. "
            "Retrain with privileged_dz_gate_enabled=false."
        )
    if task in REAL_ALIGNMENT_CUBE_TASKS and contract_version == 4:
        raise RuntimeError(
            "Real-Alignment Cube contract v4 predates the 5/2 mm action scales, "
            "new calibrated camera pose, full-range object curriculum, table-collision "
            "penalty, and tilt-independent success. Retrain from scratch with the current contract."
        )
    if task in REAL_ALIGNMENT_CUBE_TASKS and contract_version == 5:
        raise RuntimeError(
            "Real-Alignment Cube contract v5 uses the obsolete 5 mm XYZ action scale. "
            "Retrain from scratch with the 10 mm XYZ / 2 mm total-width contract v6."
        )
    if task in REAL_ALIGNMENT_CUBE_TASKS and contract_version == 6:
        raise RuntimeError(
            "Real-Alignment Cube contract v6 uses the obsolete 480x480 center crop. "
            "Retrain from scratch with the calibrated 400x398 crop and intrinsic "
            "compensation contract v7."
        )
    for key, expected_value in expected_history_deployment.items():
        if contract.get(key) != expected_value:
            raise RuntimeError(
                f"Unsupported deployment history contract for {key}: "
                f"expected {expected_value!r}, got {contract.get(key)!r}"
            )
    if task in STRICT_SIM2REAL_CUBE_TASKS and (
        contract.get("actor_mean_transform") != "tanh"
        or contract.get("policy_output_expression") != "tanh(actions)"
    ):
        raise RuntimeError("Strict Cube checkpoint was not trained with the tanh Actor-mean contract.")
    if task in STRICT_SIM2REAL_CUBE_TASKS and contract_version in {7, 8, POLICY_CONTRACT_VERSION}:
        expected_lift_contract = {
            "privileged_dz_gate": "disabled",
            "lift_reference_mode": "center_of_mass",
            "lift_reward_start_delta_m": 0.0,
            "success_lift_delta_m": 0.035,
            "lift_reward_requires_upright": False,
            "success_requires_upright": task not in REAL_ALIGNMENT_CUBE_TASKS,
            "lift_tilt_curriculum_enabled": task not in REAL_ALIGNMENT_CUBE_TASKS,
        }
        if task not in REAL_ALIGNMENT_CUBE_TASKS:
            expected_lift_contract.update(
                {
                    "lift_tilt_curriculum_start_deg": 40.0,
                    "lift_tilt_curriculum_end_deg": 10.0,
                    "lift_tilt_curriculum_start_step": 0,
                    "lift_tilt_curriculum_end_step": 120_000,
                    "lift_tilt_curriculum_step_offset": 0,
                }
            )
        for key, expected_value in expected_lift_contract.items():
            if contract.get(key) != expected_value:
                raise RuntimeError(
                    f"Strict Cube checkpoint has obsolete lift-reward semantics for {key}: "
                    f"expected {expected_value!r}, got {contract.get(key)!r}."
                )
    if task in REAL_ALIGNMENT_CUBE_TASKS:
        expected_action_scales = {
            7: REAL_ALIGNMENT_V7_ACTION_SCALES,
            8: REAL_ALIGNMENT_V8_ACTION_SCALES,
            POLICY_CONTRACT_VERSION: REAL_ALIGNMENT_V9_ACTION_SCALES,
        }[contract_version]
        expected_cube_full_xy_bounds = (
            [0.45, 0.55, -0.05, 0.05]
            if contract_version == POLICY_CONTRACT_VERSION
            else [0.4, 0.6, -0.1, 0.1]
        )
        expected_center_contract = {
            "arm_ik_tcp_source": "panda_hand_fixed_offset",
            "arm_ik_tcp_offset_m": [0.0, 0.0, 0.1034],
            "reach_center_source": "mean_of_left_and_right_fingertip_centers",
            "fingertip_local_offset_m": [0.0, 0.0, 0.045],
            "critic_gripper_position_source": "mean_of_left_and_right_fingertip_centers",
            "action_gate_center_source": "mean_of_left_and_right_fingertip_centers",
            "action_scales": expected_action_scales,
            "deployment_history_scales": expected_action_scales,
            "camera_pose_position_m": [
                1.172904219177,
                0.031653013416,
                0.512212537004,
            ],
            "camera_pose_quaternion_wxyz": [
                -0.375287920087,
                0.607013774710,
                0.582420741286,
                -0.389203461533,
            ],
            "camera_pose_convention": "ros",
            "camera_raw_resolution_wh": [640, 480],
            "camera_raw_intrinsic_matrix": [
                604.8974,
                0.0,
                320.980103,
                0.0,
                605.085815,
                247.913223,
                0.0,
                0.0,
                1.0,
            ],
            "camera_crop_roi_xywh": [100, 34, 400, 398],
            "camera_model_intrinsic_matrix": [
                338.742544,
                0.0,
                123.748857,
                0.0,
                340.550811,
                120.393372,
                0.0,
                0.0,
                1.0,
            ],
            "camera_native_render_intrinsic_matrix": [
                300.0,
                0.0,
                112.0,
                0.0,
                300.0,
                112.0,
                0.0,
                0.0,
                1.0,
            ],
            "camera_nominal_intrinsic_compensation_enabled": True,
            "camera_nominal_intrinsic_compensation_mode": (
                "gpu_affine_grid_centered_square_pixel_render_to_calibrated_k"
            ),
            "camera_resize_interpolation": "bilinear",
            "deployment_camera_serial": "215322076207",
            "cube_nominal_xy_m": [0.5, 0.0],
            "cube_full_xy_bounds_m": expected_cube_full_xy_bounds,
            "cube_position_curriculum_enabled": True,
            "cube_position_curriculum_initial_xy_range_m": [0.02, 0.02],
            "cube_position_curriculum_start_step": 20_000,
            "cube_position_curriculum_end_step": 100_000,
            "cube_position_curriculum_step_offset": 0,
            "plate_thickness_m": 0.001,
            "plate_top_height_m": 0.001,
            "table_collision_force_threshold_n": 1.0,
            "table_collision_penalty": -10.0,
            "table_collision_robot_body_names": [
                "panda_link1",
                "panda_link2",
                "panda_link3",
                "panda_link4",
                "panda_link5",
                "panda_link6",
                "panda_link7",
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
            ],
            "table_contact_sensor_history_length": 2,
        }
        for key, expected_value in expected_center_contract.items():
            if contract.get(key) != expected_value:
                raise RuntimeError(
                    f"Real-Alignment checkpoint has obsolete grasp-center semantics for {key}: "
                    f"expected {expected_value!r}, got {contract.get(key)!r}."
                )
        expected_timing_contract = {
            "sim_dt": 1.0 / 60.0,
            "decimation": 2,
            "nominal_policy_frequency_hz": 30.0,
            "camera_update_period_s": 1.0 / 30.0,
            "nominal_camera_frequency_hz": 30.0,
            "episode_length_s": 5.0,
            "max_episode_length_steps": 150,
        }
        for key, expected_value in expected_timing_contract.items():
            if contract.get(key) != expected_value:
                raise RuntimeError(
                    f"Real-Alignment checkpoint has obsolete timing semantics for {key}: "
                    f"expected {expected_value!r}, got {contract.get(key)!r}."
                )

    expected_hashes = contract.get("run_config_sha256")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("Policy contract has no run-config hashes.")
    params_dir = Path(params_dir)
    for filename in _RUN_CONFIG_FILENAMES:
        path = params_dir / filename
        expected_hash = expected_hashes.get(filename)
        if not path.is_file() or not expected_hash:
            raise RuntimeError(f"Policy contract artifact is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Run config SHA-256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
    return contract


def validate_live_env_against_policy_contract(base_env: Any, contract: Mapping[str, Any]) -> None:
    """Fail if a live environment changes deployment-relevant saved semantics."""
    cfg = base_env.cfg
    camera_cfg = getattr(cfg, "wrist_camera", None)
    actual = {
        "action_dim": int(cfg.action_space),
        "action_scale": float(cfg.action_scale),
        "action_noise_scale": float(getattr(cfg, "action_noise_scale", 0.0)),
        "sim_dt": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "camera_height": int(camera_cfg.height) if camera_cfg is not None else None,
        "camera_width": int(camera_cfg.width) if camera_cfg is not None else None,
        "actor_observation_dims": {
            "action_history": _observation_dim(base_env, "action_history"),
            "proprio_obs": _observation_dim(base_env, "proprio_obs"),
            "wrist_resnet": _observation_dim(base_env, "wrist_resnet"),
        },
    }
    if contract.get("version") in {7, 8, POLICY_CONTRACT_VERSION}:
        gripper_control_mode = _gripper_control_mode(cfg)
        total_width_delta_control = gripper_control_mode == "total_width_delta_cached_target"
        actual.update(
            {
                "action_scales": _action_scales(cfg),
                "privileged_dz_gate": (
                    "ground_truth_object_xy_applied_after_action_history"
                    if bool(getattr(cfg, "privileged_dz_gate_enabled", True))
                    else "disabled"
                ),
                "gripper_control_mode": gripper_control_mode,
                "gripper_width_delta_scale": (
                    float(cfg.gripper_width_delta_scale) if total_width_delta_control else None
                ),
                "gripper_width_bounds": [
                    0.0,
                    float(getattr(cfg, "max_gripper_opening_width", 0.08)),
                ],
                "gripper_width_delta_updates_per_policy_step": (
                    1 if total_width_delta_control else None
                ),
                "gripper_joint_target_mapping": (
                    "symmetric_half_total_width"
                    if total_width_delta_control
                    else "legacy_per_finger_increment_from_measured_joint_position"
                ),
            }
        )
        if contract.get("task") in STRICT_SIM2REAL_CUBE_TASKS:
            actual.update(
                {
                    "lift_reference_mode": str(cfg.lift_reference_mode),
                    "lift_reward_start_delta_m": float(cfg.lift_reward_start_delta),
                    "success_lift_delta_m": float(cfg.success_lift_delta),
                    "lift_reward_requires_upright": bool(cfg.lift_reward_requires_upright),
                    "success_requires_upright": bool(
                        getattr(cfg, "success_requires_upright", True)
                    ),
                    "lift_tilt_curriculum_enabled": bool(cfg.lift_tilt_curriculum_enabled),
                    "lift_tilt_curriculum_start_deg": float(cfg.lift_tilt_curriculum_start_deg),
                    "lift_tilt_curriculum_end_deg": float(cfg.lift_tilt_curriculum_end_deg),
                    "lift_tilt_curriculum_start_step": int(cfg.lift_tilt_curriculum_start_step),
                    "lift_tilt_curriculum_end_step": int(cfg.lift_tilt_curriculum_end_step),
                    "lift_tilt_curriculum_step_offset": int(cfg.lift_tilt_curriculum_step_offset),
                }
            )
        if contract.get("task") in REAL_ALIGNMENT_CUBE_TASKS:
            actual.update(
                {
                    "arm_ik_tcp_source": str(cfg.arm_ik_tcp_source),
                    "arm_ik_tcp_offset_m": _float_vector(cfg.arm_ik_tcp_offset_m),
                    "reach_center_source": str(cfg.reach_center_source),
                    "fingertip_local_offset_m": _float_vector(cfg.fingertip_local_offset_m),
                    "critic_gripper_position_source": str(cfg.reach_center_source),
                    "action_gate_center_source": str(cfg.reach_center_source),
                    "camera_update_period_s": float(camera_cfg.update_period),
                    "nominal_camera_frequency_hz": 1.0 / float(camera_cfg.update_period),
                    "camera_pose_position_m": _float_vector(camera_cfg.offset.pos),
                    "camera_pose_quaternion_wxyz": _float_vector(camera_cfg.offset.rot),
                    "camera_pose_convention": str(camera_cfg.offset.convention),
                    "camera_raw_resolution_wh": [
                        int(value) for value in cfg.camera_raw_resolution
                    ],
                    "camera_raw_intrinsic_matrix": _float_vector(
                        cfg.camera_raw_intrinsic_matrix
                    ),
                    "camera_crop_roi_xywh": [
                        int(value) for value in cfg.camera_crop_roi_xywh
                    ],
                    "camera_model_intrinsic_matrix": _float_vector(
                        cfg.camera_model_intrinsic_matrix
                    ),
                    "camera_native_render_intrinsic_matrix": _float_vector(
                        cfg.camera_native_render_intrinsic_matrix
                    ),
                    "camera_nominal_intrinsic_compensation_enabled": bool(
                        cfg.camera_nominal_intrinsic_compensation_enabled
                    ),
                    "camera_nominal_intrinsic_compensation_mode": str(
                        cfg.camera_nominal_intrinsic_compensation_mode
                    ),
                    "camera_resize_interpolation": str(
                        cfg.camera_resize_interpolation
                    ),
                    "deployment_camera_serial": str(cfg.deployment_camera_serial),
                    "cube_nominal_xy_m": _float_vector(cfg.cube.init_state.pos[:2]),
                    "cube_full_xy_bounds_m": [
                        float(cfg.cube.init_state.pos[0]) - float(cfg.cube_x_pos_range),
                        float(cfg.cube.init_state.pos[0]) + float(cfg.cube_x_pos_range),
                        float(cfg.cube.init_state.pos[1]) - float(cfg.cube_y_pos_range),
                        float(cfg.cube.init_state.pos[1]) + float(cfg.cube_y_pos_range),
                    ],
                    "cube_position_curriculum_enabled": bool(
                        cfg.cube_position_curriculum_enabled
                    ),
                    "cube_position_curriculum_initial_xy_range_m": [
                        float(cfg.cube_position_curriculum_initial_x_range),
                        float(cfg.cube_position_curriculum_initial_y_range),
                    ],
                    "cube_position_curriculum_start_step": int(
                        cfg.cube_position_curriculum_start_step
                    ),
                    "cube_position_curriculum_end_step": int(
                        cfg.cube_position_curriculum_end_step
                    ),
                    "cube_position_curriculum_step_offset": int(
                        cfg.cube_position_curriculum_step_offset
                    ),
                    "plate_thickness_m": float(cfg.plate_thickness_m),
                    "plate_top_height_m": float(cfg.plate_top_height_m),
                    "table_collision_force_threshold_n": float(
                        cfg.table_collision_force_threshold_n
                    ),
                    "table_collision_penalty": float(cfg.table_collision_penalty),
                    "table_collision_robot_body_names": list(
                        cfg.table_collision_robot_body_names
                    ),
                    "table_contact_sensor_history_length": int(
                        cfg.table_contact_sensor.history_length
                    ),
                    "episode_length_s": float(cfg.episode_length_s),
                    "max_episode_length_steps": int(base_env.max_episode_length),
                }
            )
    for key, actual_value in actual.items():
        expected_value = contract.get(key)
        if actual_value != expected_value:
            raise RuntimeError(
                f"Live environment violates saved policy contract for {key}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )


def save_training_vision_encoder(
    base_env: Any,
    params_dir: str | Path,
    *,
    task: str,
    policy_output: Any,
) -> dict[str, Any] | None:
    """Save one exact frozen encoder artifact and manifest for a training run."""
    encoder = getattr(base_env, "_resnet18", None)
    if encoder is None:
        return None
    if encoder.training:
        raise RuntimeError("Refusing to save a vision encoder in train mode. BatchNorm must remain in eval mode.")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Refusing to save a vision encoder with trainable parameters.")

    params_dir = Path(params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = params_dir / ENCODER_ARTIFACT_FILENAME
    manifest_path = params_dir / ENCODER_MANIFEST_FILENAME
    policy_contract = _build_policy_contract(base_env, params_dir, task, policy_output)

    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in encoder.state_dict().items()
    }
    state_hash = state_dict_sha256(state_dict)
    _atomic_torch_save(state_dict, artifact_path)

    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unknown"

    manifest = {
        "format_version": 1,
        "architecture": getattr(base_env, "_resnet18_architecture", "unknown"),
        "weights_id": getattr(base_env, "_resnet18_weights_id", "unknown"),
        "feature_dim": 512,
        "mode": "eval",
        "all_parameters_frozen": True,
        "state_dict_sha256": state_hash,
        "artifact_sha256": sha256_file(artifact_path),
        "hash_canonicalization": STATE_DICT_HASH_ALGORITHM,
        "artifact": artifact_path.name,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "normalization": _normalization_from_env(base_env),
        "policy_contract": policy_contract,
    }
    _atomic_json_dump(manifest, manifest_path)
    return manifest


def load_verified_vision_encoder(
    encoder: torch.nn.Module,
    artifact_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Strictly restore an encoder artifact after verifying its manifest hashes."""
    artifact_path = Path(artifact_path)
    manifest_path = Path(manifest_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Training vision encoder artifact not found: {artifact_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training vision encoder manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise RuntimeError(f"Unsupported vision encoder manifest format: {manifest.get('format_version')!r}")
    if manifest.get("hash_canonicalization") != STATE_DICT_HASH_ALGORITHM:
        raise RuntimeError(
            "Unsupported vision encoder hash canonicalization: "
            f"{manifest.get('hash_canonicalization')!r}"
        )
    if manifest.get("mode") != "eval" or manifest.get("all_parameters_frozen") is not True:
        raise RuntimeError("Vision encoder manifest does not declare an eval-mode, fully frozen encoder.")

    artifact_hash = sha256_file(artifact_path)
    if artifact_hash != manifest.get("artifact_sha256"):
        raise RuntimeError(
            "Vision encoder artifact SHA-256 mismatch: "
            f"expected {manifest.get('artifact_sha256')}, got {artifact_hash}"
        )

    state_dict = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Vision encoder artifact must contain a state dict, got {type(state_dict).__name__}")
    state_hash = state_dict_sha256(state_dict)
    if state_hash != manifest.get("state_dict_sha256"):
        raise RuntimeError(
            "Vision encoder state-dict SHA-256 mismatch: "
            f"expected {manifest.get('state_dict_sha256')}, got {state_hash}"
        )

    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    loaded_hash = module_state_dict_sha256(encoder)
    if loaded_hash != state_hash:
        raise RuntimeError(f"Loaded vision encoder hash mismatch: expected {state_hash}, got {loaded_hash}")

    verified_manifest = dict(manifest)
    verified_manifest["verified"] = True
    verified_manifest["source_artifact"] = str(artifact_path.resolve())
    verified_manifest["source_manifest"] = str(manifest_path.resolve())
    return verified_manifest
