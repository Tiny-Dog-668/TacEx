"""Occluded grasping with tokenized visual/tactile features for cross-attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

try:
    import torchvision

    _HAS_TORCHVISION = True
except Exception:
    torchvision = None
    _HAS_TORCHVISION = False

from .vt_box import (
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
)


def _maybe_call_parent_post_init(instance) -> None:
    parent_post_init = getattr(super(instance.__class__, instance), "__post_init__", None)
    if callable(parent_post_init):
        parent_post_init()


def _downsample_stride2(size: int, repeats: int) -> int:
    size = int(size)
    for _ in range(int(repeats)):
        size = (size + 1) // 2
    return size


def _resnet18_spatial_hw(img_size_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = (int(img_size_hw[0]), int(img_size_hw[1]))
    return _downsample_stride2(height, 5), _downsample_stride2(width, 5)


def _cnn_spatial_hw(img_size_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = (int(img_size_hw[0]), int(img_size_hw[1]))
    return _downsample_stride2(height, 3), _downsample_stride2(width, 3)


def _reduce_resnet_channels_to_256(fmap: torch.Tensor) -> torch.Tensor:
    if fmap.shape[1] != 512:
        return fmap
    return fmap.view(fmap.shape[0], 256, 2, fmap.shape[2], fmap.shape[3]).mean(dim=2)


@configclass
class OccludedGraspingVisionFourTactileCrossAttentionBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Token-level VT config for cross-attention fusion on the occluded box task."""

    third_camera_token_hw = (4, 5)
    third_camera_token_dim = 256
    tactile_token_hw = (3, 4)
    tactile_token_dim = 256

    def __post_init__(self):
        _maybe_call_parent_post_init(self)

        full_visual_hw = _resnet18_spatial_hw((int(self.third_person_camera.height), int(self.third_person_camera.width)))
        visual_hw = tuple(int(v) for v in getattr(self, "third_camera_token_hw", full_visual_hw))
        if len(visual_hw) != 2 or any(v <= 0 for v in visual_hw):
            raise ValueError(f"third_camera_token_hw must be two positive integers, got {visual_hw}")
        if visual_hw[0] > full_visual_hw[0] or visual_hw[1] > full_visual_hw[1]:
            raise ValueError(f"third_camera_token_hw={visual_hw} exceeds full ResNet token grid {full_visual_hw}")
        self.third_camera_token_hw = visual_hw
        self.third_camera_token_dim = 256

        tactile_encoder_type = str(getattr(self, "tactile_encoder_type", "cnn")).strip().lower()
        if tactile_encoder_type not in {"resnet", "cnn"}:
            raise ValueError(
                f"Unsupported tactile_encoder_type='{tactile_encoder_type}'. Use 'resnet' or 'cnn'."
            )
        if tactile_encoder_type == "resnet" and not _HAS_TORCHVISION:
            tactile_encoder_type = "cnn"
        self.tactile_encoder_type = tactile_encoder_type

        tactile_img_res_hw = tuple(getattr(self, "tactile_img_res_hw", (96, 128)))
        if tactile_encoder_type == "resnet":
            full_tactile_hw = _resnet18_spatial_hw(tactile_img_res_hw)
            tactile_token_dim = 512
        else:
            full_tactile_hw = _cnn_spatial_hw(tactile_img_res_hw)
            tactile_token_dim = 256

        tactile_hw = tuple(int(v) for v in getattr(self, "tactile_token_hw", (3, 4)))
        if len(tactile_hw) != 2 or any(v <= 0 for v in tactile_hw):
            raise ValueError(f"tactile_token_hw must be two positive integers, got {tactile_hw}")
        if tactile_hw[0] > full_tactile_hw[0] or tactile_hw[1] > full_tactile_hw[1]:
            raise ValueError(f"tactile_token_hw={tactile_hw} exceeds encoder token grid {full_tactile_hw}")
        self.tactile_token_hw = tactile_hw
        self.tactile_token_dim = tactile_token_dim

        critic_obs = {
            key: value
            for key, value in self.observation_space.items()
            if str(key).startswith("critic_")
        }
        token_count_visual = int(self.third_camera_token_hw[0] * self.third_camera_token_hw[1])
        token_count_tactile = int(self.tactile_token_hw[0] * self.tactile_token_hw[1])
        self.observation_space = {
            "proprio_obs": 18,
            "third_resnet_tokens": [token_count_visual, int(self.third_camera_token_dim)],
            "tactile_left_tokens": [token_count_tactile, int(self.tactile_token_dim)],
            "tactile_right_tokens": [token_count_tactile, int(self.tactile_token_dim)],
            "tactile_left_down_tokens": [token_count_tactile, int(self.tactile_token_dim)],
            "tactile_right_down_tokens": [token_count_tactile, int(self.tactile_token_dim)],
            **critic_obs,
        }


class OccludedGraspingVisionFourTactileCrossAttentionBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Occluded grasping env exposing visual/tactile token sequences for cross-attention."""

    cfg: OccludedGraspingVisionFourTactileCrossAttentionBoxCfg

    def __init__(
        self,
        cfg: OccludedGraspingVisionFourTactileCrossAttentionBoxCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._visual_token_hw = tuple(getattr(self.cfg, "third_camera_token_hw", (4, 5)))
        self._visual_token_count = int(self._visual_token_hw[0] * self._visual_token_hw[1])
        self._visual_token_dim = int(getattr(self.cfg, "third_camera_token_dim", 256))
        self._tactile_token_hw = tuple(getattr(self.cfg, "tactile_token_hw", (3, 4)))
        self._tactile_token_count = int(self._tactile_token_hw[0] * self._tactile_token_hw[1])
        self._tactile_token_dim = int(getattr(self.cfg, "tactile_token_dim", 256))

        if hasattr(self, "_resnet18"):
            del self._resnet18
        if hasattr(self, "_tactile_encoder"):
            del self._tactile_encoder

        self._visual_token_encoder = None
        if getattr(self, "_use_resnet18", False) and _HAS_TORCHVISION:
            self._visual_token_encoder = self._build_resnet18_spatial_encoder()
        else:
            print("[WARN] torchvision is unavailable; visual cross-attention tokens will be zeros.")

        if self._tactile_encoder_type == "resnet":
            self._tactile_token_encoder = self._build_resnet18_spatial_encoder()
            self._tactile_uses_imagenet_norm = True
        else:
            self._tactile_token_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            ).to(self.device)
            self._tactile_token_encoder.eval()
            for parameter in self._tactile_token_encoder.parameters():
                parameter.requires_grad_(False)
            self._tactile_uses_imagenet_norm = False

        self.extras.setdefault("meta", {})
        self.extras["meta"]["tactile_encoder_type"] = self._tactile_encoder_type
        self.extras["meta"]["vision_token_hw"] = list(self._visual_token_hw)
        self.extras["meta"]["tactile_token_hw"] = list(self._tactile_token_hw)
        print(
            f"[INFO] Cross-attention token env: vision_tokens={self._visual_token_hw}x{self._visual_token_dim}, "
            f"tactile_tokens={self._tactile_token_hw}x{self._tactile_token_dim}, "
            f"tactile_encoder={self._tactile_encoder_type}"
        )

    def _build_resnet18_spatial_encoder(self) -> nn.Module:
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

    def _prep_rgb_01(self, rgb_tensor: torch.Tensor | None, target_hw: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if rgb_tensor is None:
            return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
        x = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = x.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x

    def _get_base_policy_observations_without_tactile(self) -> dict[str, torch.Tensor]:
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

        third_rgb = self.third_person_camera.data.output.get("rgb")
        if third_rgb is None:
            third_rgb = torch.zeros(
                (self.num_envs, self.cfg.third_person_camera.height, self.cfg.third_person_camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            third_rgb = third_rgb.to(device=self.device, dtype=torch.float32) / 255.0

        third_tokens = torch.zeros(
            (self.num_envs, self._visual_token_count, self._visual_token_dim),
            device=self.device,
            dtype=torch.float32,
        )
        if self._visual_token_encoder is not None:
            xt = third_rgb.permute(0, 3, 1, 2).contiguous()
            if hasattr(self, "_imagenet_mean"):
                xt = (xt - self._imagenet_mean) / self._imagenet_std
            dev_type, use_amp = self._device_autocast_kwargs()
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                fmap = self._visual_token_encoder(xt).float()
            if tuple(fmap.shape[-2:]) != self._visual_token_hw:
                fmap = F.adaptive_avg_pool2d(fmap, self._visual_token_hw)
            if self._visual_token_dim == 256:
                fmap = _reduce_resnet_channels_to_256(fmap)
            third_tokens = fmap.flatten(start_dim=2).transpose(1, 2).contiguous()

        return {
            "proprio_obs": proprio_obs,
            "third_resnet_tokens": third_tokens,
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

    def _encode_tactile_token_batch(
        self,
        tactile_rgb_map: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        sensor_keys = ("left", "right", "left_down", "right_down")
        prepared = [
            self._prep_rgb_01(tactile_rgb_map.get(key), tuple(getattr(self.cfg, "tactile_img_res_hw", (96, 128))))
            for key in sensor_keys
        ]
        tactile_batch = torch.cat(prepared, dim=0)
        if self._tactile_uses_imagenet_norm and hasattr(self, "_imagenet_mean"):
            tactile_batch = (tactile_batch - self._imagenet_mean) / self._imagenet_std

        dev_type, use_amp = self._device_autocast_kwargs()
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
            fmap = self._tactile_token_encoder(tactile_batch).float()
        if tuple(fmap.shape[-2:]) != self._tactile_token_hw:
            fmap = F.adaptive_avg_pool2d(fmap, self._tactile_token_hw)
        tokens = fmap.flatten(start_dim=2).transpose(1, 2).contiguous()
        return dict(zip(sensor_keys, tokens.chunk(len(sensor_keys), dim=0)))

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs = self._get_base_policy_observations_without_tactile()
        tactile_tokens = self._encode_tactile_token_batch(
            {
                "left": self.gsmini_left.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left") else None,
                "right": self.gsmini_right.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right") else None,
                "left_down": (
                    self.gsmini_left_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left_down") else None
                ),
                "right_down": (
                    self.gsmini_right_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right_down") else None
                ),
            }
        )

        obs.update(
            {
                "tactile_left_tokens": tactile_tokens["left"],
                "tactile_right_tokens": tactile_tokens["right"],
                "tactile_left_down_tokens": tactile_tokens["left_down"],
                "tactile_right_down_tokens": tactile_tokens["right_down"],
            }
        )
        return {"policy": obs}
