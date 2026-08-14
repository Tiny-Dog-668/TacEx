"""Shared geometric contract for the rigid dual-GelSight Panda gripper."""

from __future__ import annotations

import torch


# Offsets are expressed in the panda_hand local frame.  The hand +Z axis is
# collinear with the centered fingertip reference despite the hand yaw.
GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M = 0.1182
GELSIGHT_HAND_TO_FINGERTIP_BOTTOM_M = 0.1353
GELSIGHT_TABLE_CLEARANCE_MIN_M = 0.0100
GELSIGHT_TABLE_CLEARANCE_PENALTY = -10.0
GELSIGHT_GEOMETRY_CONTRACT_VERSION = 1


def geometry_contract() -> dict[str, object]:
    """Return JSON-safe geometry and near-table reward semantics."""
    return {
        "version": GELSIGHT_GEOMETRY_CONTRACT_VERSION,
        "center_source": "panda_hand_plus_z_gelpad_left_right_midpoint",
        "center_offset_hand_m": [0.0, 0.0, GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M],
        "lowest_point_source": "panda_hand_plus_z_panda_fingertip_centered",
        "lowest_point_offset_hand_m": [0.0, 0.0, GELSIGHT_HAND_TO_FINGERTIP_BOTTOM_M],
        "table_clearance_m": GELSIGHT_TABLE_CLEARANCE_MIN_M,
        "table_clearance_penalty": GELSIGHT_TABLE_CLEARANCE_PENALTY,
        "table_clearance_condition": "strictly_less_than",
    }


def table_clearance_penalty(
    clearance_m: torch.Tensor,
    *,
    min_clearance_m: float = GELSIGHT_TABLE_CLEARANCE_MIN_M,
    penalty_value: float = GELSIGHT_TABLE_CLEARANCE_PENALTY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return strict near-table mask and its per-policy-step reward penalty."""
    below_clearance = clearance_m < min_clearance_m
    return below_clearance, below_clearance.to(clearance_m.dtype) * penalty_value
