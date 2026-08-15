"""Three-frame visual and reference-delta tactile Student for GelSight X040 DR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from tacex_tasks.sim2real_grasp.rma_models import RMAObservationNormalizer
from tacex_tasks.sim2real_grasp.rma_x040_wide_models import (
    RMAX040WideObservationNormalizer,
)

from .rma_gelsight_size_buckets_models import ReferenceDeltaTactileEncoder


GELSIGHT_X040_THREE_FRAME_LEGACY_MODEL_VERSION = 2
GELSIGHT_X040_THREE_FRAME_MODEL_VERSION = 3
GELSIGHT_X040_THREE_FRAME_VISUAL_DIM = 512
GELSIGHT_X040_THREE_FRAME_TACTILE_SIDE_DIM = 256
GELSIGHT_X040_THREE_FRAME_PROPRIO_DIM = 15
GELSIGHT_X040_THREE_FRAME_HISTORY_DIM = 4
GELSIGHT_X040_THREE_FRAME_ACTION_DIM = 4
GELSIGHT_X040_THREE_FRAME_FUSION_DIM = 1043


def gelsight_x040_three_frame_model_contract(
    *, legacy_position_normalization: bool = False
) -> dict[str, object]:
    contract = {
        "model_version": (
            GELSIGHT_X040_THREE_FRAME_LEGACY_MODEL_VERSION
            if legacy_position_normalization
            else GELSIGHT_X040_THREE_FRAME_MODEL_VERSION
        ),
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
        "actor_feature_order": [
            "temporal_visual[512]",
            "left_tactile[256]",
            "right_tactile[256]",
            "normalized_proprio[15]",
            "normalized_action_history[4]",
        ],
        "actor_feature_dim": GELSIGHT_X040_THREE_FRAME_FUSION_DIM,
        "action_head": [1043, 512, 256, 128, 64, 4],
        "position_head": [512, 256, 128, 3],
        "position_head_output": "unbounded_normalized_robot_root_xyz",
        "contact_head": "shared_tactile_feature_256_to_logit_per_side",
        "heatmap": "absent",
        "runtime_privileged_inputs": [],
    }
    if not legacy_position_normalization:
        contract["position_normalization"] = "x040_wide_robot_root_xyz"
    return contract


class RMAGelSightX040ThreeFrameStudent(nn.Module):
    """Output action, contact probability, and robot-root Cube position in metres."""

    def __init__(
        self,
        *,
        pretrained_backbone: bool = True,
        legacy_position_normalization: bool = False,
    ) -> None:
        super().__init__()
        self._legacy_position_normalization = legacy_position_normalization
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        self.vision_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.temporal_fusion = nn.Sequential(nn.Linear(3 * 512, 512), nn.ELU())
        self.tactile_encoder = ReferenceDeltaTactileEncoder()
        self.normalizer = (
            RMAObservationNormalizer()
            if legacy_position_normalization
            else RMAX040WideObservationNormalizer()
        )
        self.position_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 3),
        )
        self.action_head = nn.Sequential(
            nn.Linear(GELSIGHT_X040_THREE_FRAME_FUSION_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, GELSIGHT_X040_THREE_FRAME_ACTION_DIM),
        )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.freeze_vision_backbone()

    def freeze_vision_backbone(self) -> None:
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad_(False)
        self.vision_encoder.eval()

    @torch.jit.unused
    def unfreeze_backbone_after_layer2(self) -> None:
        for index, module in enumerate(self.vision_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(index >= 6)
        self.vision_encoder.eval()

    @torch.jit.unused
    def load_vision_encoder_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        result = self.vision_encoder.load_state_dict(state_dict, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "GelSight X040 vision encoder state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        return self

    def encode_visual_features(self, wrist_rgb_history: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if wrist_rgb_history.ndim != 5 or wrist_rgb_history.shape[1:] != (3, 224, 224, 3):
                raise ValueError(
                    "Expected wrist_rgb_history [N,3,224,224,3], got "
                    f"{tuple(wrist_rgb_history.shape)}"
                )
        batch_size = wrist_rgb_history.shape[0]
        frames = wrist_rgb_history.reshape(-1, 224, 224, 3)
        image = frames.to(torch.float32).permute(0, 3, 1, 2).div(255.0)
        feature_map = self.vision_encoder((image - self.image_mean) / self.image_std)
        features = torch.flatten(F.adaptive_avg_pool2d(feature_map, (1, 1)), 1)
        return self.temporal_fusion(features.reshape(batch_size, 3 * 512))

    def forward_with_training_outputs(
        self,
        wrist_rgb_history: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.jit.is_scripting():
            if proprio_obs.shape[-1] != GELSIGHT_X040_THREE_FRAME_PROPRIO_DIM:
                raise ValueError(f"Expected proprio_obs[...,15], got {tuple(proprio_obs.shape)}")
            if action_history.shape[-1] != GELSIGHT_X040_THREE_FRAME_HISTORY_DIM:
                raise ValueError(f"Expected action_history[...,4], got {tuple(action_history.shape)}")
        visual_feature = self.encode_visual_features(wrist_rgb_history)
        left_feature, right_feature, contact_logits = self.tactile_encoder(
            left_current, right_current, left_reference, right_reference
        )
        # Input [N,512] -> normalized robot-root Cube XYZ [N,3]. Keep this
        # auxiliary output unbounded so evaluation outside the reset box remains
        # representable; the X040-Wide training labels themselves lie near [-1,1].
        normalized_position = self.position_head(visual_feature)
        actor_features = torch.cat(
            (
                visual_feature,
                left_feature,
                right_feature,
                self.normalizer.normalize_proprio(proprio_obs),
                self.normalizer.normalize_history(action_history),
            ),
            dim=-1,
        )
        actions = torch.tanh(self.action_head(actor_features))
        return actions, normalized_position, contact_logits

    def forward(
        self,
        wrist_rgb_history: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, normalized_position, contact_logits = self.forward_with_training_outputs(
            wrist_rgb_history,
            proprio_obs,
            action_history,
            left_current,
            right_current,
            left_reference,
            right_reference,
        )
        return (
            actions,
            torch.sigmoid(contact_logits),
            self.normalizer.denormalize_position(normalized_position),
        )

    @torch.jit.unused
    def contract(self) -> dict[str, object]:
        return gelsight_x040_three_frame_model_contract(
            legacy_position_normalization=self._legacy_position_normalization
        )


__all__ = (
    "GELSIGHT_X040_THREE_FRAME_FUSION_DIM",
    "GELSIGHT_X040_THREE_FRAME_LEGACY_MODEL_VERSION",
    "GELSIGHT_X040_THREE_FRAME_MODEL_VERSION",
    "RMAGelSightX040ThreeFrameStudent",
    "gelsight_x040_three_frame_model_contract",
)
