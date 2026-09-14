"""Deployable 32-feature Teacher and 11-input four-tactile Student."""

from __future__ import annotations

import torch
import torch.nn as nn
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_PROFILE,
)

from .rma_gelsight_pulled_drawer_models import (
    RMAGelSightPulledDrawerActorCore,
    RMAGelSightPulledDrawerPrivilegedTeacherPolicy,
)
from .rma_gelsight_pulled_drawer_binary_tactile_models import (
    RMAGelSightPulledDrawerBinaryTactileThreeFrameStudent,
)
from .rma_gelsight_x040_binary_tactile_models import BinaryReferenceDeltaTactileEncoder


FOUR_TACTILE_TEACHER_MODEL_VERSION = 1
FOUR_TACTILE_STUDENT_MODEL_VERSION = 1
FOUR_TACTILE_ACTOR_FEATURE_DIM = 32
FOUR_TACTILE_FUSION_DIM = 1555


class RMAGelSightPulledDrawerFourTactileActorCore(RMAGelSightPulledDrawerActorCore):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FOUR_TACTILE_ACTOR_FEATURE_DIM, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 4),
        )

    def forward(self, proprio_obs, action_history, cube_position, contact_state,
                position_is_normalized: bool = False):
        if not torch.jit.is_scripting() and contact_state.shape[-1] != 4:
            raise ValueError(f"Expected contact_state[...,4], got {tuple(contact_state.shape)}")
        if position_is_normalized:
            normalized_cube = cube_position
            cube_root = self.normalizer.denormalize_position(cube_position)
        else:
            cube_root = cube_position
            normalized_cube = self.normalizer.normalize_position(cube_position)
        gripper_root = self.kinematics(proprio_obs[..., :7])
        features = torch.cat((
            self.normalizer.normalize_proprio(proprio_obs),
            self.normalizer.normalize_history(action_history), normalized_cube,
            self.normalizer.normalize_gripper_position(gripper_root),
            self.normalizer.normalize_target_position(cube_root - gripper_root),
            contact_state,
        ), dim=-1)
        return torch.tanh(self.network(features))

    def contract(self) -> dict[str, object]:
        contract = super().contract()
        contract.update({
            "model_version": FOUR_TACTILE_TEACHER_MODEL_VERSION,
            "feature_dim": 32,
            "contact_components": "left_inner_right_inner_left_down_right_down_binary",
            "four_tactile_asset_profile": GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_PROFILE,
            "tcp_geometry": "unchanged_v7_inner_gelpad_midpoint",
        })
        contract["feature_order"][-1] = "four_tactile_cube_gelpad_contact[4]"
        return contract


class RMAGelSightPulledDrawerFourTactilePrivilegedTeacherPolicy(
    RMAGelSightPulledDrawerPrivilegedTeacherPolicy
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.actor_core = RMAGelSightPulledDrawerFourTactileActorCore().to(self.device)


def four_tactile_student_model_contract() -> dict[str, object]:
    return {
        "model_version": FOUR_TACTILE_STUDENT_MODEL_VERSION,
        "runtime_input_order": [
            "wrist_rgb_history", "proprio_obs", "action_history",
            "gsmini_left_rgb", "gsmini_right_rgb", "gsmini_left_down_rgb", "gsmini_right_down_rgb",
            "gsmini_left_reference_rgb", "gsmini_right_reference_rgb",
            "gsmini_left_down_reference_rgb", "gsmini_right_down_reference_rgb",
        ],
        "runtime_output": {"action": [4], "contact_probability": [4], "cube_position_root_m": [3]},
        "contact_order": ["left_inner", "right_inner", "left_down", "right_down"],
        "tactile_preprocessing": "max_abs_rgb_current_minus_reference_strict_gt_5_u8",
        "tactile_encoder_sharing": "inner_left_right_shared;down_left_right_shared;directions_independent",
        "actor_feature_dim": FOUR_TACTILE_FUSION_DIM,
        "action_head": [1555, 512, 256, 128, 64, 4],
        "position_normalization": "pulled_drawer_contiguous_rigid_tray_robot_root_xyz_v3",
        "runtime_privileged_inputs": [],
    }


class RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent(
    RMAGelSightPulledDrawerBinaryTactileThreeFrameStudent
):
    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone)
        self.inner_tactile_encoder = BinaryReferenceDeltaTactileEncoder()
        self.down_tactile_encoder = BinaryReferenceDeltaTactileEncoder()
        del self.tactile_encoder
        self.action_head = nn.Sequential(
            nn.Linear(FOUR_TACTILE_FUSION_DIM, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 4),
        )

    def forward_with_training_outputs(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ):
        visual = self.encode_visual_features(wrist_rgb_history)
        left, right, inner_logits = self.inner_tactile_encoder(
            left_current, right_current, left_reference, right_reference
        )
        left_down, right_down, down_logits = self.down_tactile_encoder(
            left_down_current, right_down_current, left_down_reference, right_down_reference
        )
        normalized_position = self.position_head(visual)
        features = torch.cat((visual, left, right, left_down, right_down,
                              self.normalizer.normalize_proprio(proprio_obs),
                              self.normalizer.normalize_history(action_history)), dim=-1)
        return torch.tanh(self.action_head(features)), normalized_position, torch.cat((inner_logits, down_logits), dim=-1)

    def forward(self, wrist_rgb_history, proprio_obs, action_history,
                left_current, right_current, left_down_current, right_down_current,
                left_reference, right_reference, left_down_reference, right_down_reference):
        action, position, logits = self.forward_with_training_outputs(
            wrist_rgb_history, proprio_obs, action_history, left_current, right_current,
            left_down_current, right_down_current, left_reference, right_reference,
            left_down_reference, right_down_reference,
        )
        return action, torch.sigmoid(logits), self.normalizer.denormalize_position(position)

    @torch.jit.unused
    def contract(self) -> dict[str, object]:
        return four_tactile_student_model_contract()


__all__ = tuple(name for name in globals() if name.startswith(("FOUR_", "RMA", "four_")))
