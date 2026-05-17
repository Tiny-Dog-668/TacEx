# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cylinder grasping environment with privileged state observations and 4 GelSight RGB sensors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from torchvision.models import ResNet18_Weights, resnet18

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from .cylinder_grasping_tactile_depth import (
    CylinderGraspingFourTactileDepthCfg,
    CylinderGraspingFourTactileDepthEnv,
)


@configclass
class CylinderGraspingFourTactileRGBCfg(CylinderGraspingFourTactileDepthCfg):
    """Configuration for the cylinder grasping environment with 4 tactile RGB sensors (no vision cameras)."""

    tactile_img_res_hw = (96, 128)

    # Cylinder reset randomization (XY translation range and yaw range)
    cylinder_pos_range = 0.0
    cylinder_rot_range = 0.0

    episode_length_s = 1.67
    max_episode_length = 100

    reach_weight = 5.0
    reward_print_interval = 200


    # Override robot initial joint angles for this RGB task (and use the 4-pad gripper USD with *_down pads).
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.down.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # "panda_joint1": -0.372,
                # "panda_joint2": 0.260,
                # "panda_joint3": 0.448,
                # "panda_joint4": -2.393,
                # "panda_joint5": -0.336,
                # "panda_joint6": 2.644,
                # "panda_joint7": 1.144,


                "panda_joint1": -0.485,
                "panda_joint2": 0.253,
                "panda_joint3": 0.468,
                "panda_joint4": -2.383,
                "panda_joint5": -0.220,
                "panda_joint6": 2.599,
                "panda_joint7": 0.948,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Override tactile sensors to output tactile RGB (and keep camera_depth for rewards/debug).
    gsmini_left: GelSightMiniCfg = GelSightMiniCfg(prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left")
    gsmini_left.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_left.marker_motion_sim_cfg = None
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_right: GelSightMiniCfg = GelSightMiniCfg(prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right")
    gsmini_right.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_right.marker_motion_sim_cfg = None
    gsmini_right.optical_sim_cfg = gsmini_right.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_left_down: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_down"
    )
    gsmini_left_down.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left_down.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_left_down.marker_motion_sim_cfg = None
    gsmini_left_down.optical_sim_cfg = gsmini_left_down.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_right_down: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_down"
    )
    gsmini_right_down.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right_down.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_right_down.marker_motion_sim_cfg = None
    gsmini_right_down.optical_sim_cfg = gsmini_right_down.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    observation_space = {
        "proprio_obs": 18,
        "tactile_left_resnet": 128,
        "tactile_right_resnet": 128,
        "tactile_left_down_resnet": 128,
        "tactile_right_down_resnet": 128,
        "critic_cylinder_pos": 3,
        "critic_cylinder_quat": 4,
        "critic_cylinder_lin_vel": 3,
        "critic_cylinder_ang_vel": 3,
        "critic_gripper_pos": 3,
        "critic_gripper_quat": 4,
        "critic_gripper_lin_vel": 3,
        "critic_gripper_ang_vel": 3,
        "critic_target_pos": 3,
        "critic_target_distance": 1,
        "rnn_reset": 1,
    }


class CylinderGraspingFourTactileRGBEnv(CylinderGraspingFourTactileDepthEnv):
    """Cylinder grasping environment with privileged state and 4 tactile RGB sensors."""

    cfg: CylinderGraspingFourTactileRGBCfg

    def __init__(self, cfg: CylinderGraspingFourTactileRGBCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Replace the 1-channel encoder from the depth env with an ImageNet-pretrained ResNet18 (3-ch RGB).
        self._tactile_feature_dim = 128
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, self._tactile_feature_dim)
        self._tactile_encoder = resnet.to(self.device)
        self._tactile_encoder.eval()
        for p in self._tactile_encoder.parameters():
            p.requires_grad_(False)
        # ImageNet normalization constants for tactile RGB
        self._imgnet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._imgnet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        # One-shot reset pulse for LSTM state clearing after env reset
        self._rnn_reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if hasattr(self, "_rnn_reset_buf"):
            self._rnn_reset_buf[env_ids] = True

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        # Proprioceptive observations
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        # Privileged observations - cylinder
        cylinder_pos = self._cylinder.data.root_pos_w
        cylinder_quat = self._cylinder.data.root_quat_w
        cylinder_lin_vel = self._cylinder.data.root_lin_vel_w
        cylinder_ang_vel = self._cylinder.data.root_ang_vel_w

        # Privileged observations - gripper
        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        gripper_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        gripper_lin_vel = self._robot.data.body_link_lin_vel_w[:, self._body_idx]
        gripper_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._body_idx]

        # Target information (relative to end-effector)
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]

        def _norm_quat(q: torch.Tensor) -> torch.Tensor:
            n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
            return q / n

        hand_quat = _norm_quat(hand_quat)
        offset_rot = _norm_quat(self._offset_rot)
        ee_pos, _ = math_utils.combine_frame_transforms(hand_pos, hand_quat, self._offset_pos, offset_rot)
        target_pos_relative = cylinder_pos - ee_pos
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        # Tactile RGB from GelSight sensors (left/right + left_down/right_down).
        tact_l_raw = self.gsmini_left.data.output.get("tactile_rgb")
        tact_r_raw = self.gsmini_right.data.output.get("tactile_rgb")
        tact_ld_raw = self.gsmini_left_down.data.output.get("tactile_rgb")
        tact_rd_raw = self.gsmini_right_down.data.output.get("tactile_rgb")

        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (240, 320))

        def _prep_rgb_3ch(rgb_tensor):
            # expected NHWC with C=3; convert to float [0,1], resize if needed; return NCHW with C=3
            if rgb_tensor is None:
                return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
            x = rgb_tensor.to(device=self.device, dtype=torch.float32)
            # Taxim outputs are typically in [0,1], but be robust to [0,255] pipelines.
            max_val = x.max()
            if torch.isfinite(max_val) and max_val > 1.5:
                x = x / 255.0
            x = x.clamp(0.0, 1.0)
            xn = x.permute(0, 3, 1, 2).contiguous()  # NCHW (C=3)
            if xn.shape[2] != target_h or xn.shape[3] != target_w:
                xn = F.interpolate(xn, size=(target_h, target_w), mode="bilinear", align_corners=False)
            # ImageNet normalization for ResNet encoder
            xn = (xn - self._imgnet_mean) / self._imgnet_std
            return xn

        tact_l = _prep_rgb_3ch(tact_l_raw)
        tact_r = _prep_rgb_3ch(tact_r_raw)
        tact_ld = _prep_rgb_3ch(tact_ld_raw)
        tact_rd = _prep_rgb_3ch(tact_rd_raw)

        # Tactile features (FP16 autocast on CUDA if available)
        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        use_amp = dev_type == "cuda"

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            fl = self._tactile_encoder(tact_l).view(self.num_envs, self._tactile_feature_dim).float()
            fr = self._tactile_encoder(tact_r).view(self.num_envs, self._tactile_feature_dim).float()
            fld = self._tactile_encoder(tact_ld).view(self.num_envs, self._tactile_feature_dim).float()
            frd = self._tactile_encoder(tact_rd).view(self.num_envs, self._tactile_feature_dim).float()

        obs = {
            "proprio_obs": proprio_obs,
            "tactile_left_resnet": fl,
            "tactile_right_resnet": fr,
            "tactile_left_down_resnet": fld,
            "tactile_right_down_resnet": frd,
            "critic_cylinder_pos": cylinder_pos,
            "critic_cylinder_quat": cylinder_quat,
            "critic_cylinder_lin_vel": cylinder_lin_vel,
            "critic_cylinder_ang_vel": cylinder_ang_vel,
            "critic_gripper_pos": gripper_pos,
            "critic_gripper_quat": gripper_quat,
            "critic_gripper_lin_vel": gripper_lin_vel,
            "critic_gripper_ang_vel": gripper_ang_vel,
            "critic_target_pos": target_pos_relative,
            "critic_target_distance": target_distance,
        }

        # 标记每个 env 是否刚刚 reset，用于上游 LSTM 状态清零
        # 使用一次性 reset 标志，确保 reset 后首帧必定触发清零
        if hasattr(self, "_rnn_reset_buf"):
            rnn_reset = self._rnn_reset_buf.to(torch.float32).unsqueeze(-1)
            self._rnn_reset_buf[:] = False
        elif hasattr(self, "reset_buf"):
            rnn_reset = self.reset_buf.to(torch.float32).unsqueeze(-1)
        elif hasattr(self, "episode_length_buf"):
            rnn_reset = (self.episode_length_buf == 0).to(torch.float32).unsqueeze(-1)
        else:
            rnn_reset = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        obs["rnn_reset"] = rnn_reset

        return {"policy": obs}
