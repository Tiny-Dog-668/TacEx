"""Three-input direct-action visual Student for the PandaHand RMA-XY route."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from torchvision.models import resnet18


RMA_DIRECT_ACTION_STUDENT_MODEL_VERSION = 1
RMA_DIRECT_ACTION_PROPRIO_DIM = 15
RMA_DIRECT_ACTION_HISTORY_DIM = 4
RMA_DIRECT_ACTION_DIM = 4
RMA_DIRECT_ACTION_VISUAL_DIM = 512
RMA_DIRECT_ACTION_FEATURE_DIM = (
    RMA_DIRECT_ACTION_VISUAL_DIM
    + RMA_DIRECT_ACTION_PROPRIO_DIM
    + RMA_DIRECT_ACTION_HISTORY_DIM
)


class RMADirectActionObservationNormalizer(nn.Module):
    """Normalization for only the deployable proprioception/history inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("joint_lower", torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]))
        self.register_buffer("joint_upper", torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]))
        self.register_buffer("joint_velocity_scale", torch.tensor([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]))
        # This is the RMA Teacher's learned history scale, not the environment
        # action scale. Both are saved in the direct-action checkpoint contract.
        self.register_buffer("history_scale", torch.tensor([0.025, 0.025, 0.025, 0.005]))

    def normalize_proprio(self, proprio_obs: torch.Tensor) -> torch.Tensor:
        joint_mid = 0.5 * (self.joint_lower + self.joint_upper)
        joint_half_range = 0.5 * (self.joint_upper - self.joint_lower)
        return torch.cat((
            (proprio_obs[..., :7] - joint_mid) / joint_half_range,
            proprio_obs[..., 7:14] / self.joint_velocity_scale,
            (proprio_obs[..., 14:15] - 0.04) / 0.04,
        ), dim=-1)

    def normalize_history(self, action_history: torch.Tensor) -> torch.Tensor:
        return action_history / self.history_scale

    def contract(self) -> dict[str, object]:
        return {
            "joint_lower": self.joint_lower.detach().cpu().tolist(),
            "joint_upper": self.joint_upper.detach().cpu().tolist(),
            "joint_velocity_scale": self.joint_velocity_scale.detach().cpu().tolist(),
            "gripper_width_center_scale": [0.04, 0.04],
            "history_scale": self.history_scale.detach().cpu().tolist(),
        }


class RMADirectActionVisualStudent(nn.Module):
    """Map RGB, proprioception and action history directly to four actions.

    Runtime signature is exactly ``(wrist_rgb, proprio_obs, action_history)``;
    no cube pose or contact measurement can enter this module.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.vision_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.normalizer = RMADirectActionObservationNormalizer()
        self.action_head = nn.Sequential(
            nn.Linear(RMA_DIRECT_ACTION_FEATURE_DIM, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, RMA_DIRECT_ACTION_DIM),
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.freeze_backbone_after_layer2()

    def freeze_backbone_after_layer2(self) -> None:
        """Freeze conv1 through layer2 and keep all encoder BN in eval mode."""
        for index, module in enumerate(self.vision_encoder):
            requires_grad = index >= 6
            for parameter in module.parameters():
                parameter.requires_grad_(requires_grad)
        self.vision_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # BatchNorm statistics must remain frozen even while layer3/layer4
        # weights are fine-tuned on the direct action loss.
        self.vision_encoder.eval()
        return self

    def load_vision_encoder_state(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        result = self.vision_encoder.load_state_dict(dict(state_dict), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "Direct-action vision encoder state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )

    def _encode(self, wrist_rgb: torch.Tensor) -> torch.Tensor:
        image = wrist_rgb.to(torch.float32).permute(0, 3, 1, 2) / 255.0
        feature_map = self.vision_encoder((image - self.image_mean) / self.image_std)
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)), 1)

    def forward(
        self,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if proprio_obs.shape[-1] != RMA_DIRECT_ACTION_PROPRIO_DIM:
                raise ValueError(f"Expected proprio_obs[...,15], got {tuple(proprio_obs.shape)}")
            if action_history.shape[-1] != RMA_DIRECT_ACTION_HISTORY_DIM:
                raise ValueError(f"Expected action_history[...,4], got {tuple(action_history.shape)}")
        features = torch.cat((
            self._encode(wrist_rgb),
            self.normalizer.normalize_proprio(proprio_obs),
            self.normalizer.normalize_history(action_history),
        ), dim=-1)
        return torch.tanh(self.action_head(features))

    def contract(self) -> dict[str, object]:
        return {
            "model_version": RMA_DIRECT_ACTION_STUDENT_MODEL_VERSION,
            "input_order": ["wrist_rgb", "proprio_obs", "action_history"],
            "input_signature": {
                "wrist_rgb": [224, 224, 3],
                "proprio_obs": [RMA_DIRECT_ACTION_PROPRIO_DIM],
                "action_history": [RMA_DIRECT_ACTION_HISTORY_DIM],
            },
            "output_signature": {"mean_actions": [RMA_DIRECT_ACTION_DIM]},
            "feature_order": [
                "resnet18_global_average_pool_visual[512]",
                "normalized_proprio_obs[15]",
                "normalized_action_history[4]",
            ],
            "feature_dim": RMA_DIRECT_ACTION_FEATURE_DIM,
            "runtime_privileged_inputs": [],
            "vision_encoder": "resnet18_layer4_global_average_pool",
            "action_head": [531, 512, 256, 128, 64, 4],
            "activation": "ELU_then_tanh",
            "batch_norm": "all_resnet18_batch_norm_eval",
        }
