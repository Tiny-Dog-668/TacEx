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
POLICY_CONTRACT_VERSION = 1
ACTION_HISTORY_CONTRACT = "processed_action_pre_privileged_dz_gate_v1"
STRICT_SIM2REAL_CUBE_TASKS = frozenset(
    {
        "TacEx-Sim2Real-Cube-Grasp-v0",
        "TacEx-Sim2Real-Cube-Real-Alignment-v0",
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
    sim_dt = float(cfg.sim.dt)
    decimation = int(cfg.decimation)
    config_hashes = {}
    for filename in _RUN_CONFIG_FILENAMES:
        path = params_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required run config artifact is missing: {path}")
        config_hashes[filename] = sha256_file(path)

    return {
        "version": POLICY_CONTRACT_VERSION,
        "task": task,
        "actor_mean_transform": actor_mean_transform,
        "policy_output_expression": output_expression,
        "action_history": ACTION_HISTORY_CONTRACT,
        "action_history_units": "environment_processed_action",
        "deployment_history_source": "clipped_action",
        "deployment_history_scale": action_scale,
        "deployment_history_delay_steps": 1,
        "action_dim": int(cfg.action_space),
        "action_scale": action_scale,
        "action_noise_scale": float(getattr(cfg, "action_noise_scale", 0.0)),
        "xyz_command_frame": "robot_root",
        "privileged_dz_gate": "applied_after_action_history_and_not_available_on_the_real_robot",
        "gripper_finger_target_increment_per_physics_application_per_processed_unit": 0.2,
        "gripper_target_reapplications_per_policy_step": decimation,
        "gripper_realized_motion": "depends_on_PD_tracking_and_is_not_a_fixed_per_policy_step_delta",
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
    if contract.get("version") != POLICY_CONTRACT_VERSION:
        raise RuntimeError(f"Unsupported sim2real policy contract version: {contract.get('version')!r}")
    if contract.get("task") != task:
        raise RuntimeError(
            f"Checkpoint task mismatch: requested {task!r}, run contract declares {contract.get('task')!r}."
        )
    if contract.get("action_history") != ACTION_HISTORY_CONTRACT:
        raise RuntimeError(f"Unsupported action-history contract: {contract.get('action_history')!r}")
    expected_history_deployment = {
        "deployment_history_source": "clipped_action",
        "deployment_history_scale": contract.get("action_scale"),
        "deployment_history_delay_steps": 1,
    }
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
