"""VT task exposing raw tactile images for policy-side CNN reconstruction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from .vt_box import OccludedGraspingVisionFourTactileBoxCfg, OccludedGraspingVisionFourTactileBoxEnv


@configclass
class OccludedGraspingVTCNNReconBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for trainable policy-side tactile CNN reconstruction."""

    tactile_recon_img_hw = (32, 32)

    def __post_init__(self):
        super().__post_init__()
        recon_h, recon_w = tuple(int(v) for v in self.tactile_recon_img_hw)
        raw_dim = 3 * recon_h * recon_w
        self.observation_space = {
            "proprio_obs": 18,
            "third_resnet": 256,
            "tactile_left_rgb_raw": raw_dim,
            "tactile_right_rgb_raw": raw_dim,
            "tactile_left_down_rgb_raw": raw_dim,
            "tactile_right_down_rgb_raw": raw_dim,
            "critic_can_pos": 3,
            "critic_can_quat": 4,
            "critic_can_lin_vel": 3,
            "critic_can_ang_vel": 3,
            "critic_gripper_pos": 3,
            "critic_gripper_quat": 4,
            "critic_gripper_lin_vel": 3,
            "critic_gripper_ang_vel": 3,
            "critic_target_pos": 3,
            "critic_target_distance": 1,
        }


class OccludedGraspingVTCNNReconBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Expose four downsampled tactile RGB images and keep CNN trainable in policy."""

    cfg: OccludedGraspingVTCNNReconBoxCfg

    def __init__(self, cfg: OccludedGraspingVTCNNReconBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if hasattr(self, "_tactile_encoder"):
            del self._tactile_encoder

    def _prep_raw_tactile_rgb(self, rgb_tensor: torch.Tensor | None) -> torch.Tensor:
        recon_h, recon_w = tuple(int(v) for v in getattr(self.cfg, "tactile_recon_img_hw", (32, 32)))
        if rgb_tensor is None:
            return torch.zeros((self.num_envs, 3, recon_h, recon_w), dtype=torch.float32, device=self.device)
        rgb = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = rgb.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            rgb = rgb / 255.0
        rgb = rgb.clamp(0.0, 1.0)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(-1)
        rgb_nchw = rgb.permute(0, 3, 1, 2).contiguous()
        if rgb_nchw.shape[1] == 1:
            rgb_nchw = rgb_nchw.repeat(1, 3, 1, 1)
        elif rgb_nchw.shape[1] != 3:
            rgb_nchw = rgb_nchw[:, :3, :, :]
        if rgb_nchw.shape[2] != recon_h or rgb_nchw.shape[3] != recon_w:
            rgb_nchw = F.interpolate(rgb_nchw, size=(recon_h, recon_w), mode="bilinear", align_corners=False)
        return rgb_nchw

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
        third_rgb = self._degrade_third_person_rgb(third_rgb)

        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        use_amp = dev_type == "cuda"

        if hasattr(self, "_use_resnet18") and self._use_resnet18:
            xt = third_rgb.permute(0, 3, 1, 2).contiguous().to(self.device)
            if hasattr(self, "_imagenet_mean"):
                xt = (xt - self._imagenet_mean) / self._imagenet_std
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                third_feat = self._resnet18(xt).view(self.num_envs, 256)
        else:
            third_feat = torch.zeros((self.num_envs, 256), device=self.device)

        return {
            "proprio_obs": proprio_obs,
            "third_resnet": third_feat,
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

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs = self._get_base_policy_observations_without_tactile()

        tact_l_raw = self.gsmini_left.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left") else None
        tact_r_raw = self.gsmini_right.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right") else None
        tact_ld_raw = self.gsmini_left_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left_down") else None
        tact_rd_raw = self.gsmini_right_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right_down") else None

        tact_l = self._prep_raw_tactile_rgb(tact_l_raw)
        tact_r = self._prep_raw_tactile_rgb(tact_r_raw)
        tact_ld = self._prep_raw_tactile_rgb(tact_ld_raw)
        tact_rd = self._prep_raw_tactile_rgb(tact_rd_raw)

        obs["tactile_left_rgb_raw"] = tact_l.reshape(self.num_envs, -1)
        obs["tactile_right_rgb_raw"] = tact_r.reshape(self.num_envs, -1)
        obs["tactile_left_down_rgb_raw"] = tact_ld.reshape(self.num_envs, -1)
        obs["tactile_right_down_rgb_raw"] = tact_rd.reshape(self.num_envs, -1)
        return {"policy": obs}


@configclass
class OccludedGraspingVTTactileCrossAlphaReconDownsampleBoxCfg(OccludedGraspingVTCNNReconBoxCfg):
    """Policy-side tactile reconstruction task with downsampled third-person RGB."""

    visual_degradation_mode = "downsample"
    visual_downsample_size = 32


class OccludedGraspingVTTactileCrossAlphaReconBoxEnv(OccludedGraspingVTCNNReconBoxEnv):
    """Alias env for tactile-cross-alpha reconstruction policy."""

    cfg: OccludedGraspingVTTactileCrossAlphaReconDownsampleBoxCfg
