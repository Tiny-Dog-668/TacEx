"""Occluded grasping env exposing vision/tactile tokens for policy-token Transformer fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from .vt_box import OccludedGraspingVisionFourTactileBoxCfg, OccludedGraspingVisionFourTactileBoxEnv

try:
    import torchvision

    _HAS_TORCHVISION = True
except Exception:
    torchvision = None
    _HAS_TORCHVISION = False


@configclass
class OccludedGraspingVTPolicyTokenTransformerBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for env-side token extraction and policy-token Transformer fusion."""

    vision_image_hw = (224, 224)
    tactile_policy_img_hw = (96, 128)
    tactile_policy_img_channels = 3
    vision_token_count = 49
    vision_token_dim = 512
    tactile_token_count = 16
    tactile_token_dim = 128

    def __post_init__(self):
        super().__post_init__()
        critic_obs = {
            key: value
            for key, value in self.observation_space.items()
            if str(key).startswith("critic_")
        }
        self.observation_space = {
            "proprio_obs": 18,
            "vision_tokens": [int(self.vision_token_count), int(self.vision_token_dim)],
            "tactile_tokens": [int(self.tactile_token_count), int(self.tactile_token_dim)],
            **critic_obs,
        }


class OccludedGraspingVTPolicyTokenTransformerBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Extract frozen visual/tactile tokens in the env; policy only fuses tokens."""

    cfg: OccludedGraspingVTPolicyTokenTransformerBoxCfg

    def __init__(
        self,
        cfg: OccludedGraspingVTPolicyTokenTransformerBoxCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        if hasattr(self, "_resnet18"):
            del self._resnet18
        if hasattr(self, "_tactile_encoder"):
            del self._tactile_encoder

        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for Policy-Token-Transformer env-side visual tokens")

        self._vision_token_encoder = self._build_resnet18_layer4_encoder()
        tactile_channels = int(getattr(self.cfg, "tactile_policy_img_channels", 3))
        self._tactile_token_encoder = nn.Sequential(
            nn.Conv2d(tactile_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
        ).to(self.device)
        self._tactile_token_encoder.eval()
        for parameter in self._tactile_token_encoder.parameters():
            parameter.requires_grad_(False)

        self._vision_token_hw = (7, 7)
        self._tactile_token_hw = (2, 2)
        self.register_buffer_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.register_buffer_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _build_resnet18_layer4_encoder(self) -> nn.Module:
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            backbone = torchvision.models.resnet18(weights=weights)
        except Exception:
            backbone = torchvision.models.resnet18(pretrained=True)
        encoder = nn.Sequential(*list(backbone.children())[:-2]).to(self.device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        return encoder

    def _device_autocast_kwargs(self) -> tuple[str, bool]:
        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        return dev_type, dev_type == "cuda"

    def _prep_rgb_nchw(self, rgb_tensor: torch.Tensor | None, target_hw: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if rgb_tensor is None:
            return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)

        rgb = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = rgb.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            rgb = rgb / 255.0
        rgb = rgb.clamp(0.0, 1.0)

        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(-1)
        rgb = rgb.permute(0, 3, 1, 2).contiguous()
        if rgb.shape[1] == 1:
            rgb = rgb.repeat(1, 3, 1, 1)
        elif rgb.shape[1] != 3:
            rgb = rgb[:, :3]

        if rgb.shape[2] != target_h or rgb.shape[3] != target_w:
            rgb = F.interpolate(rgb, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return rgb

    def _get_base_policy_observations_without_images(self) -> dict[str, torch.Tensor]:
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        can_pos = self._can.data.root_pos_w
        can_quat = self._can.data.root_quat_w
        can_lin_vel = self._can.data.root_lin_vel_w
        can_ang_vel = self._can.data.root_ang_vel_w

        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        gripper_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        gripper_lin_vel = self._robot.data.body_link_lin_vel_w[:, self._body_idx]
        gripper_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._body_idx]

        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        hand_quat = hand_quat / torch.linalg.norm(hand_quat, dim=-1, keepdim=True).clamp(min=1e-9)
        offset_rot = self._offset_rot / torch.linalg.norm(self._offset_rot, dim=-1, keepdim=True).clamp(min=1e-9)
        ee_pos, _ = math_utils.combine_frame_transforms(hand_pos, hand_quat, self._offset_pos, offset_rot)
        target_pos_relative = can_pos - ee_pos
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        return {
            "proprio_obs": proprio_obs,
            "critic_can_pos": can_pos,
            "critic_can_quat": can_quat,
            "critic_can_lin_vel": can_lin_vel,
            "critic_can_ang_vel": can_ang_vel,
            "critic_gripper_pos": gripper_pos,
            "critic_gripper_quat": gripper_quat,
            "critic_gripper_lin_vel": gripper_lin_vel,
            "critic_gripper_ang_vel": gripper_ang_vel,
            "critic_target_pos": target_pos_relative,
            "critic_target_distance": target_distance,
        }

    def _encode_vision_tokens(self) -> torch.Tensor:
        third_rgb = self.third_person_camera.data.output.get("rgb")
        if third_rgb is not None:
            third_rgb = self._degrade_third_person_rgb(third_rgb.to(device=self.device, dtype=torch.float32) / 255.0)
        vision_image = self._prep_rgb_nchw(third_rgb, tuple(getattr(self.cfg, "vision_image_hw", (224, 224))))
        vision_image = (vision_image - self.register_buffer_mean) / self.register_buffer_std
        dev_type, use_amp = self._device_autocast_kwargs()
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
            fmap = self._vision_token_encoder(vision_image).float()
        fmap = F.adaptive_avg_pool2d(fmap, self._vision_token_hw)
        return fmap.flatten(start_dim=2).transpose(1, 2).contiguous()

    def _encode_tactile_tokens(self) -> torch.Tensor:
        tactile_hw = tuple(getattr(self.cfg, "tactile_policy_img_hw", (96, 128)))
        # Requested order: bottom-left, bottom-right, inner-left, inner-right.
        tactile_rgbs = [
            self.gsmini_left_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left_down") else None,
            self.gsmini_right_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right_down") else None,
            self.gsmini_left.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left") else None,
            self.gsmini_right.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right") else None,
        ]
        tactile_images = [self._prep_rgb_nchw(rgb, tactile_hw) for rgb in tactile_rgbs]
        tactile_batch = torch.cat(tactile_images, dim=0)
        dev_type, use_amp = self._device_autocast_kwargs()
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
            fmap = self._tactile_token_encoder(tactile_batch).float()
        fmap = F.adaptive_avg_pool2d(fmap, self._tactile_token_hw)
        tokens = fmap.flatten(start_dim=2).transpose(1, 2).contiguous()
        return tokens.reshape(4, self.num_envs, 4, -1).permute(1, 0, 2, 3).reshape(self.num_envs, 16, -1)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs = self._get_base_policy_observations_without_images()
        obs["vision_tokens"] = self._encode_vision_tokens()
        obs["tactile_tokens"] = self._encode_tactile_tokens()

        return {"policy": obs}
