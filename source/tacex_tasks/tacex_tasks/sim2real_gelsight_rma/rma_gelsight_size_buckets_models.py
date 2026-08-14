"""Direct-action visual/tactile Student for the fixed-size GelSight task."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from tacex_tasks.sim2real_grasp.rma_models import (
    CubeCenterHeatmapHead,
    RMAObservationNormalizer,
)


GELSIGHT_STUDENT_VISUAL_DIM = 512
GELSIGHT_STUDENT_TACTILE_SIDE_DIM = 256
GELSIGHT_STUDENT_PROPRIO_DIM = 15
GELSIGHT_STUDENT_HISTORY_DIM = 4
GELSIGHT_STUDENT_ACTION_DIM = 4
GELSIGHT_STUDENT_FUSION_DIM = (
    GELSIGHT_STUDENT_VISUAL_DIM
    + 2 * GELSIGHT_STUDENT_TACTILE_SIDE_DIM
    + GELSIGHT_STUDENT_PROPRIO_DIM
    + GELSIGHT_STUDENT_HISTORY_DIM
)


def gelsight_student_model_contract() -> dict[str, object]:
    return {
        "runtime_input_order": [
            "wrist_rgb",
            "proprio_obs",
            "action_history",
            "gsmini_left_rgb",
            "gsmini_right_rgb",
            "gsmini_left_reference_rgb",
            "gsmini_right_reference_rgb",
        ],
        "visual_feature": "resnet18_layer4_global_average_pool[512]",
        "tactile_feature": "shared_cnn_signed_current_minus_reference[256]_per_side",
        "actor_feature_order": [
            "visual[512]",
            "left_tactile[256]",
            "right_tactile[256]",
            "normalized_proprio[15]",
            "normalized_action_history[4]",
        ],
        "actor_feature_dim": GELSIGHT_STUDENT_FUSION_DIM,
        "action_head": [1043, 512, 256, 128, 64, 4],
        "output": "tanh_mean_action[4]",
        "auxiliary_outputs": {
            "normalized_cube_position_root": [3],
            "cube_center_heatmap": [1, 14, 14],
            "left_right_contact_logits": [2],
        },
        "runtime_privileged_inputs": [],
    }


class ReferenceDeltaTactileEncoder(nn.Module):
    """Encode each signed GelSight current-reference image into 256 features.

    The left and right images use the same CNN weights. Inputs are RGB tensors
    ``[N, 96, 128, 3]`` and outputs are independent ``[N, 256]`` features.
    A shared scalar classifier provides auxiliary left/right contact logits;
    the continuous features, rather than thresholded contacts, enter the Actor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.side_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, GELSIGHT_STUDENT_TACTILE_SIDE_DIM, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.contact_output = nn.Sequential(
            nn.Linear(GELSIGHT_STUDENT_TACTILE_SIDE_DIM, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        final = self.contact_output[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, -2.0)

    @staticmethod
    def signed_delta(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if current.ndim != 4 or current.shape[-1] != 3:
                raise ValueError(
                    f"Expected current tactile [N,H,W,3], got {tuple(current.shape)}"
                )
            if current.shape != reference.shape:
                raise ValueError("Current/reference tactile shapes differ")
        return (
            (current.to(torch.float32) - reference.to(torch.float32))
            .div(255.0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _one_side(
        self, current: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.side_encoder(self.signed_delta(current, reference))
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


class RMAGelSightReferenceStudent(nn.Module):
    """Fuse 512-D vision and two 256-D tactile features into four actions.

    Cube XYZ, projected heatmaps and physics contact states are training-only
    auxiliary labels. They are never runtime inputs to this module.
    """

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        self.vision_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.tactile_encoder = ReferenceDeltaTactileEncoder()
        self.normalizer = RMAObservationNormalizer()
        self.position_head = nn.Sequential(
            nn.Linear(GELSIGHT_STUDENT_VISUAL_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 3),
        )
        self.heatmap_head = CubeCenterHeatmapHead()
        self.action_head = nn.Sequential(
            nn.Linear(GELSIGHT_STUDENT_FUSION_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, GELSIGHT_STUDENT_ACTION_DIM),
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
        """Train ResNet18 layer3/layer4 while all BatchNorm stays in eval mode."""
        for index, module in enumerate(self.vision_encoder):
            trainable = index >= 6
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
        self.vision_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        return self

    def encode_visual(
        self, wrist_rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_scripting():
            if wrist_rgb.ndim != 4 or wrist_rgb.shape[-1] != 3:
                raise ValueError(
                    f"Expected wrist_rgb [N,H,W,3], got {tuple(wrist_rgb.shape)}"
                )
        image = wrist_rgb.to(torch.float32).permute(0, 3, 1, 2) / 255.0
        x = (image - self.image_mean) / self.image_std
        layer3_feature = x
        for index, module in enumerate(self.vision_encoder):
            x = module(x)
            if index == 6:
                layer3_feature = x
        visual_feature = torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)
        return layer3_feature, visual_feature

    def forward_with_auxiliary(
        self,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.jit.is_scripting():
            if proprio_obs.shape[-1] != GELSIGHT_STUDENT_PROPRIO_DIM:
                raise ValueError(
                    f"Expected proprio_obs[...,15], got {tuple(proprio_obs.shape)}"
                )
            if action_history.shape[-1] != GELSIGHT_STUDENT_HISTORY_DIM:
                raise ValueError(
                    f"Expected action_history[...,4], got {tuple(action_history.shape)}"
                )
        layer3_feature, visual_feature = self.encode_visual(wrist_rgb)
        left_feature, right_feature, contact_logits = self.tactile_encoder(
            left_current, right_current, left_reference, right_reference
        )
        normalized_position = torch.tanh(self.position_head(visual_feature))
        heatmap = self.heatmap_head(layer3_feature)
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
        return actions, normalized_position, contact_logits, heatmap

    def forward(
        self,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> torch.Tensor:
        actions, _, _, _ = self.forward_with_auxiliary(
            wrist_rgb,
            proprio_obs,
            action_history,
            left_current,
            right_current,
            left_reference,
            right_reference,
        )
        return actions

    @torch.jit.unused
    def contract(self) -> dict[str, object]:
        return gelsight_student_model_contract()


__all__ = (
    "GELSIGHT_STUDENT_ACTION_DIM",
    "GELSIGHT_STUDENT_FUSION_DIM",
    "GELSIGHT_STUDENT_TACTILE_SIDE_DIM",
    "GELSIGHT_STUDENT_VISUAL_DIM",
    "RMAGelSightReferenceStudent",
    "ReferenceDeltaTactileEncoder",
    "gelsight_student_model_contract",
)
