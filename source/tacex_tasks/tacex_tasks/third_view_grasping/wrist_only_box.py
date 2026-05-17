"""Wrist-camera-only variant of the third-view grasping task."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from .privileged_wrist_box import (
    OccludedGraspingPrivilegedWristBoxCfg,
    OccludedGraspingPrivilegedWristBoxEnv,
)


@configclass
class OccludedGraspingWristOnlyBoxCfg(OccludedGraspingPrivilegedWristBoxCfg):
    """Configuration for wrist-camera policy with privileged critic observations."""

    observation_space = {
        "proprio_obs": 18,
        "wrist_resnet": 512,
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


class OccludedGraspingWristOnlyBoxEnv(OccludedGraspingPrivilegedWristBoxEnv):
    """Occluded grasping environment with wrist-camera-only observations."""

    cfg: OccludedGraspingWristOnlyBoxCfg

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        wrist_rgb = self.wrist_camera.data.output.get("rgb")
        if wrist_rgb is None:
            wrist_feat = torch.zeros((self.num_envs, self._wrist_feature_dim), device=self.device, dtype=torch.float32)
        else:
            x = wrist_rgb.to(device=self.device, dtype=torch.float32)
            max_val = x.max()
            if torch.isfinite(max_val) and max_val > 1.5:
                x = x / 255.0
            x = x.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()
            target_h, target_w = int(self.cfg.wrist_camera.height), int(self.cfg.wrist_camera.width)
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            with torch.no_grad():
                wrist_feat = self._wrist_encoder(x).view(self.num_envs, self._wrist_feature_dim)

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
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )
        target_pos_relative = can_pos - ee_pos
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        obs = {
            "proprio_obs": proprio_obs,
            "wrist_resnet": wrist_feat,
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
        return {"policy": obs}
