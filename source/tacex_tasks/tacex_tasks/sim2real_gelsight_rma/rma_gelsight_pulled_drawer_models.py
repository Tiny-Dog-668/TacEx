"""Drawer-specific normalization profiles for dual-GelSight Teacher and Student."""

from __future__ import annotations

import torch

from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_PULLED_DRAWER_FRANKA_ASSET_PROFILE,
    GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M,
)
from tacex_tasks.sim2real_grasp.rma_models import RMAObservationNormalizer

from .gelsight_geometry import GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M
from .rma_gelsight_models import (
    GelSightPandaHandKinematics,
    RMAGelSightActorCore,
    RMAGelSightPrivilegedTeacherPolicy,
)
from .rma_gelsight_x040_three_frame_models import RMAGelSightX040ThreeFrameStudent


PULLED_DRAWER_TEACHER_MODEL_VERSION = 5
PULLED_DRAWER_STUDENT_MODEL_VERSION = 4
PULLED_DRAWER_LINK7_TO_GELPAD_MIDPOINT_M = (
    0.107
    + GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M
)


class RMAGelSightPulledDrawerHandKinematics(GelSightPandaHandKinematics):
    """Panda FK ending at the extended Pulled-Drawer GelSight midpoint."""

    def __init__(self) -> None:
        super().__init__()
        # Input/output: Panda arm joint positions [N,7] -> robot-root XYZ [N,3].
        # The shared GelSight v7 asset keeps panda_hand coincident with link8
        # and extends the finger/GelSight assembly by 21 mm along hand local +Z.
        self.link7_to_fingertip_midpoint = torch.tensor(
            [0.0, 0.0, PULLED_DRAWER_LINK7_TO_GELPAD_MIDPOINT_M, 1.0]
        )

    def contract(self) -> dict[str, object]:
        contract = super().contract()
        contract.update(
            {
                "geometry_profile": GELSIGHT_PULLED_DRAWER_FRANKA_ASSET_PROFILE,
                "panda_hand_to_finger_gelsight_extension_m": (
                    GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M
                ),
                "panda_hand_to_fingertip_midpoint_m": (
                    GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M
                ),
                "panda_hand_to_gelpad_midpoint_m": (
                    GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M
                ),
                "panda_link7_to_gelpad_midpoint_m": (
                    PULLED_DRAWER_LINK7_TO_GELPAD_MIDPOINT_M
                ),
            }
        )
        return contract


class RMAGelSightPulledDrawerObservationNormalizer(RMAObservationNormalizer):
    """Normalize robot-root Cube XYZ around the open pulled-drawer tray."""

    def __init__(self) -> None:
        super().__init__()
        # The contiguous nominal tray center is x=0.5275 m. The X scale covers
        # sampled cabinet/tray depths plus the +/-6 cm episodic Cube reset;
        # Y includes tray placement DR plus the same episodic reset.
        self.cube_position_center = torch.tensor([0.5275, 0.00, 0.033])
        self.cube_position_scale = torch.tensor([0.11, 0.08, 0.10])

    def contract(self) -> dict[str, list[float] | str]:
        contract = super().contract()
        contract["profile"] = "pulled_drawer_contiguous_rigid_tray_robot_root_xyz_v3"
        return contract


class RMAGelSightPulledDrawerActorCore(RMAGelSightActorCore):
    """Thirty-feature GelSight Actor with the pulled-drawer XYZ normalizer."""

    def __init__(self) -> None:
        super().__init__()
        self.kinematics = RMAGelSightPulledDrawerHandKinematics()
        self.normalizer = RMAGelSightPulledDrawerObservationNormalizer()

    def contract(self) -> dict[str, object]:
        contract = super().contract()
        contract["model_version"] = PULLED_DRAWER_TEACHER_MODEL_VERSION
        contract["normalization_profile"] = (
            "pulled_drawer_contiguous_rigid_tray_robot_root_xyz_v3"
        )
        return contract


class RMAGelSightPulledDrawerPrivilegedTeacherPolicy(RMAGelSightPrivilegedTeacherPolicy):
    """skrl Teacher wrapper that installs the drawer-specific Actor core."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.actor_core = RMAGelSightPulledDrawerActorCore().to(self.device)


def pulled_drawer_student_model_contract() -> dict[str, object]:
    return {
        "model_version": PULLED_DRAWER_STUDENT_MODEL_VERSION,
        "runtime_input_order": [
            "wrist_rgb_history",
            "proprio_obs",
            "action_history",
            "gsmini_left_rgb",
            "gsmini_right_rgb",
            "gsmini_left_reference_rgb",
            "gsmini_right_reference_rgb",
        ],
        "runtime_output": {
            "action": [4],
            "left_right_contact_probability": [2],
            "cube_position_root_m": [3],
        },
        "visual_encoder": "shared_resnet18_layer4_global_average_pool_per_frame",
        "frame_order": "oldest_to_newest",
        "temporal_fusion": [1536, 512],
        "tactile_feature": "shared_cnn_signed_current_minus_reference_256_per_side",
        "actor_feature_dim": 1043,
        "action_head": [1043, 512, 256, 128, 64, 4],
        "position_head": [512, 256, 128, 3],
        "position_normalization": "pulled_drawer_contiguous_rigid_tray_robot_root_xyz_v3",
        "runtime_privileged_inputs": [],
    }


class RMAGelSightPulledDrawerThreeFrameStudent(RMAGelSightX040ThreeFrameStudent):
    """Seven-input Student with drawer-specific auxiliary XYZ normalization."""

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone)
        self.normalizer = RMAGelSightPulledDrawerObservationNormalizer()

    @torch.jit.unused
    def contract(self) -> dict[str, object]:
        return pulled_drawer_student_model_contract()


__all__ = (
    "PULLED_DRAWER_STUDENT_MODEL_VERSION",
    "PULLED_DRAWER_TEACHER_MODEL_VERSION",
    "RMAGelSightPulledDrawerActorCore",
    "RMAGelSightPulledDrawerHandKinematics",
    "RMAGelSightPulledDrawerObservationNormalizer",
    "RMAGelSightPulledDrawerPrivilegedTeacherPolicy",
    "RMAGelSightPulledDrawerThreeFrameStudent",
    "pulled_drawer_student_model_contract",
)
