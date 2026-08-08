"""Fail-closed artifacts for the PandaHand RMA XY/force contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .rma_xy_models import RMA_XY_MODEL_VERSION, RMAXYActorCore, RMAXYObservationNormalizer


RMA_XY_TEACHER_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0"
RMA_XY_STUDENT_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-v0"
RMA_XY_STUDENT_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0"
RMA_XY_STUDENT_HEATMAP_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-v0"
RMA_XY_STUDENT_HEATMAP_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0"
RMA_XY_STUDENT_TASKS = frozenset((RMA_XY_STUDENT_TASK, RMA_XY_STUDENT_DR_TASK, RMA_XY_STUDENT_HEATMAP_TASK, RMA_XY_STUDENT_HEATMAP_DR_TASK))
RMA_XY_TEACHER_MANIFEST_VERSION = 2
RMA_XY_STUDENT_CHECKPOINT_VERSION = 2
RMA_XY_MANIFEST_FILENAME = "rma_xy_manifest.json"


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
        digest.update(key.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii")); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_run_dir(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.parent.name != "checkpoints":
        raise RuntimeError(f"Expected checkpoint under checkpoints/: {checkpoint}")
    return checkpoint.parent.parent


def _env_contract(cfg: Any) -> dict[str, Any]:
    hand = cfg.robot.actuators["panda_hand"]
    return {
        "robot_profile": "franka_panda_hand",
        "action_dim": int(cfg.action_space),
        "action_scales": [float(cfg.action_scale)] * 3 + [float(cfg.gripper_width_delta_scale)],
        "gripper_control_mode": str(cfg.gripper_control_mode),
        "gripper_actuator": {"effort_limit_sim": float(hand.effort_limit_sim), "stiffness": float(hand.stiffness), "damping": float(hand.damping)},
        "contact_filter_prim_paths": list(cfg.rma_cube_contact_sensor.filter_prim_paths_expr),
        "contact_force_threshold_n": float(cfg.rma_contact_force_threshold_n),
        "contact_reward_weight": float(cfg.rma_contact_reward_weight),
        "position_frame": str(cfg.rma_position_frame),
        "object_pose_components": str(cfg.rma_object_pose_components),
        "contact_components": str(cfg.rma_contact_components),
        "actor_feature_dim": int(cfg.rma_actor_feature_dim),
        "policy_frequency_hz": 1.0 / (float(cfg.sim.dt) * int(cfg.decimation)),
        "episode_length_s": float(cfg.episode_length_s),
        "success_terminates_episode": bool(cfg.rma_success_terminates_episode),
    }


def write_teacher_manifest(base_env: Any, params_dir: str | Path, agent_cfg: Mapping[str, Any]) -> Path:
    params = Path(params_dir)
    hashes = {name: sha256_file(params / name) for name in ("agent.yaml", "env.yaml")}
    manifest = {
        "kind": "tacex_rma_xy_teacher", "version": RMA_XY_TEACHER_MANIFEST_VERSION,
        "task": RMA_XY_TEACHER_TASK, "model_version": RMA_XY_MODEL_VERSION,
        "actor_inputs": {"proprio_obs": 15, "action_history": 4, "rma_cube_xy": 2, "rma_contact_force": 2},
        "actor_contract": RMAXYActorCore().contract(), "normalization": RMAXYObservationNormalizer().contract(),
        "environment_contract": _env_contract(base_env.cfg), "run_config_sha256": hashes,
        "trainer_timesteps": int(agent_cfg["trainer"]["timesteps"]),
    }
    output = params / RMA_XY_MANIFEST_FILENAME
    _atomic_json_dump(manifest, output)
    return output


def load_teacher_manifest(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = _checkpoint_run_dir(checkpoint_path) / "params" / RMA_XY_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"PandaHand RMA XY manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "tacex_rma_xy_teacher" or manifest.get("version") != RMA_XY_TEACHER_MANIFEST_VERSION:
        raise RuntimeError("Unsupported PandaHand RMA XY teacher manifest")
    if manifest.get("task") != RMA_XY_TEACHER_TASK or manifest.get("model_version") != RMA_XY_MODEL_VERSION:
        raise RuntimeError("Teacher manifest model/task contract differs from PandaHand RMA XY")
    if manifest.get("normalization") != RMAXYObservationNormalizer().contract() or manifest.get("actor_contract") != RMAXYActorCore().contract():
        raise RuntimeError("Teacher manifest model contract differs from current PandaHand RMA XY")
    run_dir = _checkpoint_run_dir(checkpoint_path)
    for name, expected_hash in manifest.get("run_config_sha256", {}).items():
        path = run_dir / "params" / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Teacher run config hash mismatch: {path}")
    return manifest


def validate_live_env_contract(cfg: Any, manifest: Mapping[str, Any]) -> None:
    if manifest.get("environment_contract") != _env_contract(cfg):
        raise RuntimeError("Live PandaHand RMA environment differs from the Teacher contract")


def load_teacher_policy_state(checkpoint: str | Path, device: str | torch.device) -> dict[str, torch.Tensor]:
    load_teacher_manifest(checkpoint)
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    policy = payload.get("policy") if isinstance(payload, Mapping) else None
    if not isinstance(policy, Mapping):
        raise RuntimeError("Teacher checkpoint has no policy state_dict")
    return dict(policy)


def load_student_checkpoint(checkpoint: str | Path, *, device: str | torch.device = "cpu", expected_teacher_checkpoint: str | Path | None = None, expected_task: str | None = None) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint).expanduser().resolve(), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "tacex_rma_xy_student":
        raise RuntimeError("Checkpoint is not a PandaHand RMA XY Student artifact")
    if payload.get("version") != RMA_XY_STUDENT_CHECKPOINT_VERSION or payload.get("model_version") != RMA_XY_MODEL_VERSION:
        raise RuntimeError("Unsupported PandaHand RMA XY Student checkpoint version")
    if payload.get("task") not in RMA_XY_STUDENT_TASKS or (expected_task is not None and payload.get("task") != expected_task):
        raise RuntimeError("RMA XY Student task mismatch")
    if payload.get("normalization") != RMAXYObservationNormalizer().contract() or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("Student checkpoint contract differs from current PandaHand RMA XY")
    if expected_teacher_checkpoint is not None and payload.get("teacher_checkpoint_sha256") != sha256_file(expected_teacher_checkpoint):
        raise RuntimeError("Student checkpoint was distilled from a different Teacher")
    return payload


def load_student_model_state(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state_dict, strict=False)
    allowed = {key for key in model.state_dict() if key.startswith("heatmap_head.")}
    if set(result.missing_keys) - allowed or result.unexpected_keys:
        raise RuntimeError(f"RMA XY Student model state mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
