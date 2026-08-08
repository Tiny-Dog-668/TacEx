"""Fail-closed replay support for the archived PandaHand RMA v5 Student."""

from __future__ import annotations

from typing import Any, Mapping


RMA_LEGACY_STUDENT_HEATMAP_DR_REPLAY_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Legacy-Student-Heatmap-DR-v0"
)
RMA_LEGACY_V5_SOURCE_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0"
)
RMA_LEGACY_V5_MODEL_VERSION = 3
RMA_LEGACY_V5_STUDENT_CHECKPOINT_VERSION = 5


def validate_legacy_v5_student_payload(payload: Mapping[str, Any]) -> None:
    """Reject every artifact except the archived RGB/XYZ/contact v5 contract."""
    required = {
        "kind": "tacex_rma_student",
        "version": RMA_LEGACY_V5_STUDENT_CHECKPOINT_VERSION,
        "task": RMA_LEGACY_V5_SOURCE_TASK,
        "model_version": RMA_LEGACY_V5_MODEL_VERSION,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                "Legacy replay requires the archived PandaHand RMA v5 Heatmap-DR "
                f"Student ({key}={expected!r})"
            )

    manifest = payload.get("teacher_manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("Legacy replay checkpoint has no Teacher manifest")
    actor_contract = manifest.get("actor_contract")
    if not isinstance(actor_contract, Mapping) or actor_contract.get("feature_dim") != 30:
        raise RuntimeError("Legacy replay checkpoint is not a 30-feature RMA Teacher artifact")
    if manifest.get("model_version") != RMA_LEGACY_V5_MODEL_VERSION:
        raise RuntimeError("Legacy replay Teacher model version differs from Student contract")
    if manifest.get("actor_inputs") != {
        "proprio_obs": 15,
        "action_history": 4,
        "rma_cube_pos": 3,
        "rma_contact_state": 2,
    }:
        raise RuntimeError("Legacy replay Teacher observation contract differs from RGB/XYZ/contact v5")
