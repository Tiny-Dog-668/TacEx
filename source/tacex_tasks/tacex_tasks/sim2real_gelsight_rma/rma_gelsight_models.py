"""GelSight-specific deployable Teacher FK and skrl policy wrapper."""

from __future__ import annotations

import torch

from tacex_tasks.sim2real_grasp.rma_models import (
    PandaFingertipKinematics,
    RMAActorCore,
    RMAPrivilegedTeacherPolicy,
)

from .gelsight_geometry import GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M


class GelSightPandaHandKinematics(PandaFingertipKinematics):
    """Panda FK ending at the midpoint of the two GelSight sensing faces."""

    def __init__(self) -> None:
        super().__init__()
        # The inherited link7-to-link8 transform is 0.107 m.  From panda_hand,
        # the GelSight midpoint is the measured +Z 0.1182 m offset.
        self.link7_to_fingertip_midpoint = torch.tensor(
            [0.0, 0.0, 0.107 + GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M, 1.0]
        )

    def contract(self) -> dict[str, object]:
        contract = super().contract()
        contract.update(
            {
                "output": "mean_of_left_and_right_gelpad_centers",
                "panda_hand_to_fingertip_midpoint_m": GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M,
                "geometry_profile": "gelsight_rigid_dual_pad_v1",
            }
        )
        return contract


class RMAGelSightActorCore(RMAActorCore):
    """RMA Actor Core whose privileged FK agrees with GelSight environment geometry."""

    def __init__(self) -> None:
        super().__init__()
        self.kinematics = GelSightPandaHandKinematics()


class RMAGelSightPrivilegedTeacherPolicy(RMAPrivilegedTeacherPolicy):
    """skrl Teacher policy backed by :class:`RMAGelSightActorCore`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.actor_core = RMAGelSightActorCore().to(self.device)


__all__ = (
    "GelSightPandaHandKinematics",
    "RMAGelSightActorCore",
    "RMAGelSightPrivilegedTeacherPolicy",
)
