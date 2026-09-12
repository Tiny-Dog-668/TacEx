"""Single-channel binary tactile Student model for the X040 Progress task."""

from __future__ import annotations

import torch
import torch.nn as nn

from .rma_gelsight_x040_three_frame_models import (
    GELSIGHT_X040_THREE_FRAME_TACTILE_SIDE_DIM,
    RMAGelSightX040ThreeFrameStudent,
    gelsight_x040_three_frame_model_contract,
)


GELSIGHT_X040_BINARY_TACTILE_MODEL_VERSION = 1
GELSIGHT_X040_BINARY_TACTILE_THRESHOLD_U8 = 5.0


def gelsight_x040_binary_tactile_model_contract() -> dict[str, object]:
    contract = gelsight_x040_three_frame_model_contract()
    contract.update(
        {
            "model_version": GELSIGHT_X040_BINARY_TACTILE_MODEL_VERSION,
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


class BinaryReferenceDeltaTactileEncoder(nn.Module):
    """Encode a one-channel binary mask derived from each RGB tactile pair.

    Raw inputs are current/reference RGB ``uint8`` tensors ``[N,H,W,3]``.
    The network input is ``[N,1,H,W]`` float32 and is one exactly when the
    maximum absolute RGB-channel difference at that pixel is strictly above 5.
    Left and right sides share all CNN and contact-head weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.side_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                128,
                GELSIGHT_X040_THREE_FRAME_TACTILE_SIDE_DIM,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.contact_output = nn.Sequential(
            nn.Linear(GELSIGHT_X040_THREE_FRAME_TACTILE_SIDE_DIM, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        final = self.contact_output[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, -2.0)

    @staticmethod
    def binary_delta(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if current.ndim != 4 or current.shape[-1] != 3:
                raise ValueError(
                    f"Expected current tactile [N,H,W,3], got {tuple(current.shape)}"
                )
            if current.shape != reference.shape:
                raise ValueError("Current/reference tactile shapes differ")
        # [N,H,W,3] -> [N,1,H,W]. Thresholding in the original uint8 value
        # domain makes the deployment contract independent of normalization.
        absolute_rgb_delta = torch.abs(
            current.to(torch.float32) - reference.to(torch.float32)
        )
        pixel_delta = torch.amax(absolute_rgb_delta, dim=-1)
        return (
            (pixel_delta > 5.0)
            .to(torch.float32)
            .unsqueeze(1)
            .contiguous()
        )

    def _one_side(
        self, current: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.side_encoder(self.binary_delta(current, reference))
        return feature, self.contact_output(feature)

    def forward(
        self,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left_feature, left_logit = self._one_side(left_current, left_reference)
        right_feature, right_logit = self._one_side(right_current, right_reference)
        return left_feature, right_feature, torch.cat((left_logit, right_logit), dim=-1)


class RMAGelSightX040BinaryTactileThreeFrameStudent(
    RMAGelSightX040ThreeFrameStudent
):
    """X040 three-frame Student whose tactile branch sees binary masks only."""

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone)
        self.tactile_encoder = BinaryReferenceDeltaTactileEncoder()

    @torch.jit.unused
    def contract(self) -> dict[str, object]:
        return gelsight_x040_binary_tactile_model_contract()


__all__ = (
    "BinaryReferenceDeltaTactileEncoder",
    "GELSIGHT_X040_BINARY_TACTILE_MODEL_VERSION",
    "GELSIGHT_X040_BINARY_TACTILE_THRESHOLD_U8",
    "RMAGelSightX040BinaryTactileThreeFrameStudent",
    "gelsight_x040_binary_tactile_model_contract",
)
