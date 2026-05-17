# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cube Grasping Environment with Wrist Camera and Two Tactile Sensors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

try:
    import torchvision
    from torchvision.models import ResNet18_Weights
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from .cube_grasping_vision_only import (
    CubeGraspingVisionOnlyCfg,
    CubeGraspingVisionOnlyEnv,
)


@configclass
class CubeGraspingVisionTwoTactileCfg(CubeGraspingVisionOnlyCfg):
    """Configuration for cube grasping with wrist-camera vision and 2 tactile RGB sensors."""

    tactile_img_res_hw = (96, 128)

    # Use the 2-pad GelSight Mini asset (inner left/right pads).
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.3136,
                "panda_joint2": -0.4448,
                "panda_joint3": 0.3858,
                "panda_joint4": -3.0236,
                "panda_joint5": 0.2319,
                "panda_joint6": 2.5776,
                "panda_joint7": 0.6249,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Two tactile RGB sensors (inner left/right).
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

    observation_space = {
        "proprio_obs": 18,
        "gripper_state": 1,
        "action_history": 5,
        "wrist_resnet": 512,
        "tactile_left_resnet": 128,
        "tactile_right_resnet": 128,
        "critic_cube_pos": 3,
        "critic_cube_quat": 4,
        "critic_cube_lin_vel": 3,
        "critic_cube_ang_vel": 3,
        "critic_gripper_pos": 3,
        "critic_gripper_quat": 4,
        "critic_gripper_lin_vel": 3,
        "critic_gripper_ang_vel": 3,
        "critic_target_pos": 3,
        "critic_target_distance": 1,
    }


class CubeGraspingVisionTwoTactileEnv(CubeGraspingVisionOnlyEnv):
    """Cube grasping environment with wrist-camera vision and 2 tactile RGB sensors."""

    cfg: CubeGraspingVisionTwoTactileCfg

    def __init__(self, cfg: CubeGraspingVisionTwoTactileCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._tactile_feature_dim = 128
        if _HAS_TORCHVISION:
            resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            resnet.fc = nn.Linear(resnet.fc.in_features, self._tactile_feature_dim)
            self._tactile_encoder = resnet.to(self.device)
            trainable = not bool(self.cfg.resnet18_frozen)
            self._tactile_encoder.train(trainable)
            for p in self._tactile_encoder.parameters():
                p.requires_grad_(trainable)
            self._tactile_imgnet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self._tactile_imgnet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:
            self._tactile_encoder = None
            self._tactile_imgnet_mean = None
            self._tactile_imgnet_std = None

    def _setup_scene(self):
        super()._setup_scene()

        # tactile sensors
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs_dict = super()._get_observations()
        obs = obs_dict["policy"]

        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))

        def _prep_rgb_3ch(rgb_tensor: torch.Tensor | None) -> torch.Tensor:
            if rgb_tensor is None or self._tactile_encoder is None:
                return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
            x = rgb_tensor.to(device=self.device, dtype=torch.float32)
            max_val = x.max()
            if torch.isfinite(max_val) and max_val > 1.5:
                x = x / 255.0
            x = x.clamp(0.0, 1.0)
            xn = x.permute(0, 3, 1, 2).contiguous()
            if xn.shape[2] != target_h or xn.shape[3] != target_w:
                xn = F.interpolate(xn, size=(target_h, target_w), mode="bilinear", align_corners=False)
            xn = (xn - self._tactile_imgnet_mean) / self._tactile_imgnet_std
            return xn

        tact_l_raw = getattr(self, "gsmini_left", None)
        tact_l_raw = tact_l_raw.data.output.get("tactile_rgb") if tact_l_raw is not None else None
        tact_r_raw = getattr(self, "gsmini_right", None)
        tact_r_raw = tact_r_raw.data.output.get("tactile_rgb") if tact_r_raw is not None else None

        tact_l = _prep_rgb_3ch(tact_l_raw)
        tact_r = _prep_rgb_3ch(tact_r_raw)

        if self._tactile_encoder is None:
            zeros = torch.zeros((self.num_envs, self._tactile_feature_dim), device=self.device, dtype=torch.float32)
            fl = fr = zeros
        else:
            dev = self.device
            dev_type = getattr(dev, "type", None)
            if dev_type is None:
                dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
            use_amp = dev_type == "cuda"
            if self.cfg.resnet18_frozen:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    fl = self._tactile_encoder(tact_l).view(self.num_envs, self._tactile_feature_dim).float()
                    fr = self._tactile_encoder(tact_r).view(self.num_envs, self._tactile_feature_dim).float()
            else:
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    fl = self._tactile_encoder(tact_l).view(self.num_envs, self._tactile_feature_dim).float()
                    fr = self._tactile_encoder(tact_r).view(self.num_envs, self._tactile_feature_dim).float()

        obs.update(
            {
                "tactile_left_resnet": fl,
                "tactile_right_resnet": fr,
            }
        )

        return {"policy": obs}
