"""Binary-tactile Student model for the Pulled-Drawer Progress task."""

from __future__ import annotations

import torch

from .rma_gelsight_pulled_drawer_models import (
    RMAGelSightPulledDrawerThreeFrameStudent,
    pulled_drawer_student_model_contract,
)
from .rma_gelsight_x040_binary_tactile_models import (
    BinaryReferenceDeltaTactileEncoder,
    GELSIGHT_X040_BINARY_TACTILE_THRESHOLD_U8,
)


PULLED_DRAWER_BINARY_TACTILE_STUDENT_MODEL_VERSION = 1


def pulled_drawer_binary_tactile_student_model_contract() -> dict[str, object]:
    contract = pulled_drawer_student_model_contract()
    contract.update(
        {
            "model_version": PULLED_DRAWER_BINARY_TACTILE_STUDENT_MODEL_VERSION,
            "tactile_feature": (
                "shared_cnn_single_channel_binary_max_abs_rgb_current_minus_"
                "reference_strict_gt_5_u8_256_per_side"
            ),
            "tactile_preprocessing": {
                "difference": "absolute_current_minus_reference_in_uint8_value_domain",
                "rgb_reduction": "maximum_across_rgb_channels_per_pixel",
                "threshold_u8": GELSIGHT_X040_BINARY_TACTILE_THRESHOLD_U8,
                "comparison": "strictly_greater_than",
                "network_input_shape_chw": [1, 96, 128],
                "network_input_dtype": "float32",
                "network_input_values": [0.0, 1.0],
            },
        }
    )
    return contract


class RMAGelSightPulledDrawerBinaryTactileThreeFrameStudent(
    RMAGelSightPulledDrawerThreeFrameStudent
):
    """Drawer-normalized Student whose tactile branch sees binary masks only."""

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone)
        self.tactile_encoder = BinaryReferenceDeltaTactileEncoder()

    @torch.jit.unused
    def contract(self) -> dict[str, object]:
        return pulled_drawer_binary_tactile_student_model_contract()


__all__ = (
    "PULLED_DRAWER_BINARY_TACTILE_STUDENT_MODEL_VERSION",
    "RMAGelSightPulledDrawerBinaryTactileThreeFrameStudent",
    "pulled_drawer_binary_tactile_student_model_contract",
)
