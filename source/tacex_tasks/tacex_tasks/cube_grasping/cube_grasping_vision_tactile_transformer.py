# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cube Grasping Environment with Wrist Camera and Two Tactile Sensors (Transformer)."""

from __future__ import annotations

import importlib

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


def _patch_skrl_runner_for_custom_models() -> None:
    """Allow skrl Runner to instantiate custom models via module:Class strings."""
    try:
        from skrl.utils.runner.torch import Runner as TorchRunner
    except Exception:
        return

    if getattr(TorchRunner, "_tacex_custom_model_patch", False):
        return

    original_component = TorchRunner._component

    def _component(self, name: str):
        if isinstance(name, str) and ":" in name:
            module_path, class_name = name.split(":", 1)

            def _instantiator(*args, **kwargs):
                return_source = kwargs.pop("return_source", False)
                if return_source:
                    return f"CustomModel({module_path}:{class_name})"
                cls = getattr(importlib.import_module(module_path), class_name)
                return cls(*args, **kwargs)

            return _instantiator
        return original_component(self, name)

    TorchRunner._component = _component
    TorchRunner._tacex_custom_model_patch = True


_patch_skrl_runner_for_custom_models()


@configclass
class CubeGraspingVisionTwoTactileCfg(CubeGraspingVisionOnlyCfg):
    """Configuration for cube grasping with wrist-camera vision and 2 tactile RGB sensors."""

    wrist_img_res_hw = (64, 64)
    tactile_img_res_hw = (64, 64)

    # Override wrist camera render size for this task.
    wrist_camera = CubeGraspingVisionOnlyCfg().wrist_camera.replace(height=64, width=64)

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
    gsmini_left.data_types = ["tactile_rgb"]
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
    gsmini_right.data_types = ["tactile_rgb"]
    gsmini_right.marker_motion_sim_cfg = None
    gsmini_right.optical_sim_cfg = gsmini_right.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    observation_space = {
        "proprio_obs": 18,
        "gripper_state": 1,
        "action_history": 5,
        "wrist_rgb": [3, wrist_img_res_hw[0], wrist_img_res_hw[1]],
        "tactile_left_rgb": [3, tactile_img_res_hw[0], tactile_img_res_hw[1]],
        "tactile_right_rgb": [3, tactile_img_res_hw[0], tactile_img_res_hw[1]],
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

        obs_space = getattr(self.cfg, "observation_space", {}) or {}
        self._want_wrist_rgb = "wrist_rgb" in obs_space
        self._want_wrist_resnet = "wrist_resnet" in obs_space
        if not self._want_wrist_resnet:
            self._use_resnet18 = False

        self._want_tactile_left_rgb = "tactile_left_rgb" in obs_space
        self._want_tactile_right_rgb = "tactile_right_rgb" in obs_space
        self._want_tactile_left_resnet = "tactile_left_resnet" in obs_space
        self._want_tactile_right_resnet = "tactile_right_resnet" in obs_space
        self._want_tactile_resnet = self._want_tactile_left_resnet or self._want_tactile_right_resnet

        self._tactile_feature_dim = 128
        if self._want_tactile_resnet and _HAS_TORCHVISION:
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
            if self._want_tactile_resnet and not _HAS_TORCHVISION:
                print("[WARN] tactile_resnet requested but torchvision is unavailable; using zeros.")

    def _setup_scene(self):
        super()._setup_scene()

        # tactile sensors
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

    def _prep_rgb_nchw(self, rgb_tensor: torch.Tensor | None, target_h: int, target_w: int) -> torch.Tensor:
        if rgb_tensor is None:
            return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
        x = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = x.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)
        xn = x.permute(0, 3, 1, 2).contiguous()
        if xn.shape[2] != target_h or xn.shape[3] != target_w:
            xn = F.interpolate(xn, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return xn

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs_dict = super()._get_observations()
        obs = obs_dict["policy"]
        obs_space = getattr(self.cfg, "observation_space", {}) or {}

        # remove unused resnet keys
        if "wrist_resnet" not in obs_space and "wrist_resnet" in obs:
            obs.pop("wrist_resnet", None)

        # wrist rgb (raw, downsampled)
        if self._want_wrist_rgb:
            target_h, target_w = self.cfg.wrist_img_res_hw
            wrist_rgb = self.wrist_camera.data.output.get("rgb")
            obs["wrist_rgb"] = self._prep_rgb_nchw(wrist_rgb, target_h, target_w)
        else:
            obs.pop("wrist_rgb", None)

        # tactile rgb
        target_h, target_w = self.cfg.tactile_img_res_hw
        tact_l_raw = getattr(self, "gsmini_left", None)
        tact_l_raw = tact_l_raw.data.output.get("tactile_rgb") if tact_l_raw is not None else None
        tact_r_raw = getattr(self, "gsmini_right", None)
        tact_r_raw = tact_r_raw.data.output.get("tactile_rgb") if tact_r_raw is not None else None

        tact_l_img = self._prep_rgb_nchw(tact_l_raw, target_h, target_w)
        tact_r_img = self._prep_rgb_nchw(tact_r_raw, target_h, target_w)

        if self._want_tactile_left_rgb:
            obs["tactile_left_rgb"] = tact_l_img
        else:
            obs.pop("tactile_left_rgb", None)
        if self._want_tactile_right_rgb:
            obs["tactile_right_rgb"] = tact_r_img
        else:
            obs.pop("tactile_right_rgb", None)

        # tactile resnet (optional)
        if self._want_tactile_resnet:
            if self._tactile_encoder is None:
                zeros = torch.zeros((self.num_envs, self._tactile_feature_dim), device=self.device, dtype=torch.float32)
                fl = fr = zeros
            else:
                x_l = (tact_l_img - self._tactile_imgnet_mean) / self._tactile_imgnet_std
                x_r = (tact_r_img - self._tactile_imgnet_mean) / self._tactile_imgnet_std
                if self.cfg.resnet18_frozen:
                    with torch.no_grad():
                        fl = self._tactile_encoder(x_l).view(self.num_envs, self._tactile_feature_dim)
                        fr = self._tactile_encoder(x_r).view(self.num_envs, self._tactile_feature_dim)
                else:
                    fl = self._tactile_encoder(x_l).view(self.num_envs, self._tactile_feature_dim)
                    fr = self._tactile_encoder(x_r).view(self.num_envs, self._tactile_feature_dim)

            if self._want_tactile_left_resnet:
                obs["tactile_left_resnet"] = fl
            else:
                obs.pop("tactile_left_resnet", None)
            if self._want_tactile_right_resnet:
                obs["tactile_right_resnet"] = fr
            else:
                obs.pop("tactile_right_resnet", None)

        return {"policy": obs}


#
