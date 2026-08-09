"""Fail-closed artifacts for the RMA direct-action visual Student."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..rma_xy_artifacts import (
    RMA_XY_STUDENT_HEATMAP_DR_TASK,
    load_student_checkpoint as load_xy_student_checkpoint,
    load_teacher_manifest,
    sha256_file,
    state_dict_sha256,
    validate_live_env_contract,
)
from .models import (
    RMA_DIRECT_ACTION_STUDENT_MODEL_VERSION,
    RMADirectActionVisualStudent,
    RMADirectActionObservationNormalizer,
)


RMA_DIRECT_ACTION_STUDENT_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Direct-Action-Student-DR-v0"
RMA_DIRECT_ACTION_STUDENT_TASKS = frozenset((RMA_DIRECT_ACTION_STUDENT_DR_TASK,))
RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_KIND = "tacex_rma_direct_action_student"
RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_VERSION = 1


def direct_action_input_contract() -> dict[str, object]:
    return {
        "wrist_rgb": [224, 224, 3],
        "proprio_obs": [15],
        "action_history": [4],
    }


def load_encoder_initialization_checkpoint(
    checkpoint: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Extract only the visual encoder from the approved Heatmap-DR Student."""
    payload = load_xy_student_checkpoint(checkpoint, device=device)
    if payload.get("task") != RMA_XY_STUDENT_HEATMAP_DR_TASK:
        raise RuntimeError(
            "Direct-action encoder initialization must be an RMA XY Heatmap-DR Student checkpoint"
        )
    source = payload.get("model")
    if not isinstance(source, Mapping):
        raise RuntimeError("Encoder initialization checkpoint has no model state")
    encoder_state = {
        key[len("vision_encoder."):]: value
        for key, value in source.items()
        if key.startswith("vision_encoder.")
    }
    if not encoder_state:
        raise RuntimeError("Encoder initialization checkpoint has no vision_encoder state")
    expected = payload.get("vision_encoder_state_dict_sha256")
    actual = state_dict_sha256(encoder_state)
    if expected != actual:
        raise RuntimeError("Encoder initialization checkpoint vision encoder hash mismatch")
    return encoder_state, dict(payload)


def make_student_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    task: str,
    teacher_checkpoint: str | Path,
    teacher_manifest: Mapping[str, Any],
    encoder_init_checkpoint: str | Path,
    encoder_init_payload: Mapping[str, Any],
    loss: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_KIND,
        "version": RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_VERSION,
        "model_version": RMA_DIRECT_ACTION_STUDENT_MODEL_VERSION,
        "task": task,
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "student_input_contract": direct_action_input_contract(),
        "student_model_contract": model.contract(),
        "normalization": model.normalizer.contract(),
        "teacher_checkpoint": str(Path(teacher_checkpoint).expanduser().resolve()),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_manifest": dict(teacher_manifest),
        "encoder_init_checkpoint": str(Path(encoder_init_checkpoint).expanduser().resolve()),
        "encoder_init_checkpoint_sha256": sha256_file(encoder_init_checkpoint),
        "encoder_init_task": encoder_init_payload.get("task"),
        "encoder_init_state_dict_sha256": encoder_init_payload.get("vision_encoder_state_dict_sha256"),
        "vision_encoder_state_dict_sha256": state_dict_sha256(model.vision_encoder.state_dict()),
        "loss": dict(loss),
        "optimizer_config": dict(optimizer_config),
        "training_privileged_inputs": {
            "rma_cube_xy": [2],
            "rma_contact_force": [2],
            "purpose": "Teacher action target only; never passed to Student",
        },
    }


def load_student_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_teacher_checkpoint: str | Path | None = None,
    expected_task: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint).expanduser().resolve(), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_KIND:
        raise RuntimeError("Checkpoint is not an RMA direct-action Student artifact")
    if (
        payload.get("version") != RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_VERSION
        or payload.get("model_version") != RMA_DIRECT_ACTION_STUDENT_MODEL_VERSION
    ):
        raise RuntimeError("Unsupported RMA direct-action Student checkpoint version")
    if payload.get("task") not in RMA_DIRECT_ACTION_STUDENT_TASKS or (
        expected_task is not None and payload.get("task") != expected_task
    ):
        raise RuntimeError("RMA direct-action Student task mismatch")
    if payload.get("student_input_contract") != direct_action_input_contract():
        raise RuntimeError("Direct-action Student runtime input contract differs")
    if payload.get("normalization") != RMADirectActionObservationNormalizer().contract():
        raise RuntimeError("Direct-action Student normalization differs")
    if payload.get("student_model_contract") != RMADirectActionVisualStudent().contract():
        raise RuntimeError("Direct-action Student model contract differs")
    if not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("Direct-action Student checkpoint has no model state")
    if (
        payload.get("encoder_init_task") != RMA_XY_STUDENT_HEATMAP_DR_TASK
        or not isinstance(payload.get("encoder_init_checkpoint_sha256"), str)
        or not isinstance(payload.get("encoder_init_state_dict_sha256"), str)
    ):
        raise RuntimeError("Direct-action Student encoder initialization provenance differs")
    vision_state = {
        key[len("vision_encoder."):]: value
        for key, value in payload["model"].items()
        if key.startswith("vision_encoder.")
    }
    if not vision_state or payload.get("vision_encoder_state_dict_sha256") != state_dict_sha256(vision_state):
        raise RuntimeError("Direct-action Student vision encoder state hash differs")
    if expected_teacher_checkpoint is not None:
        if payload.get("teacher_checkpoint_sha256") != sha256_file(expected_teacher_checkpoint):
            raise RuntimeError("Direct-action Student was distilled from a different Teacher")
        if payload.get("teacher_manifest") != load_teacher_manifest(expected_teacher_checkpoint):
            raise RuntimeError("Direct-action Student Teacher manifest differs from the requested Teacher")
    return payload


def load_student_model_state(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    result = model.load_state_dict(dict(state_dict), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Direct-action Student model state mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


__all__ = (
    "RMA_DIRECT_ACTION_STUDENT_DR_TASK",
    "RMA_DIRECT_ACTION_STUDENT_TASKS",
    "RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_KIND",
    "RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_VERSION",
    "direct_action_input_contract",
    "load_encoder_initialization_checkpoint",
    "load_student_checkpoint",
    "load_student_model_state",
    "load_teacher_manifest",
    "make_student_payload",
    "sha256_file",
    "state_dict_sha256",
    "validate_live_env_contract",
)
