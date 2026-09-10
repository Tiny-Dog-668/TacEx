"""Shared geometric contract for the rigid dual-GelSight Panda gripper."""

from __future__ import annotations

from pathlib import Path

import torch

from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_FINGER_EXTENSION_M,
    GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_PROFILE,
    GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_USD,
)


# Offsets are expressed in the panda_hand local frame. The shared RMA GelSight
# v7 asset extends both fingers and the complete GelSight assembly by 21 mm
# along hand +Z while panda_hand remains coincident with panda_link8.
GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M = 0.1392
GELSIGHT_HAND_TO_FINGERTIP_BOTTOM_M = 0.1563
GELSIGHT_TABLE_CLEARANCE_MIN_M = 0.0100
GELSIGHT_TABLE_CLEARANCE_PENALTY = -10.0
GELSIGHT_GEOMETRY_CONTRACT_VERSION = 3


def geometry_contract() -> dict[str, object]:
    """Return JSON-safe geometry and near-table reward semantics."""
    return {
        "version": GELSIGHT_GEOMETRY_CONTRACT_VERSION,
        "asset_geometry": "panda_fingers_and_gelsight_plus_z_21mm",
        "asset_profile": GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_PROFILE,
        "asset_filename": Path(GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_USD).name,
        "finger_and_gelsight_extension_m": GELSIGHT_FINGER_EXTENSION_M,
        "panda_link8_to_hand_fixed_joint_local_pos0_m": [0.0, 0.0, 0.0],
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
