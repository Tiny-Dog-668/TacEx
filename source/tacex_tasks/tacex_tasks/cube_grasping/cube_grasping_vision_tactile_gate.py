# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cube Grasping Environment with Wrist Camera and Tactile Gate."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.assets import ArticulationCfg
import isaaclab.utils.math as math_utils
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
class CubeGraspingVisionTwoTactileGateCfg(CubeGraspingVisionOnlyCfg):
    """Configuration for cube grasping with wrist-camera vision and 2 tactile RGB sensors + gate."""

    tactile_img_res_hw = (96, 128)
    tactile_gate_enable = True
    tactile_gate_diff_threshold = 0.02

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


class CubeGraspingVisionTwoTactileGateEnv(CubeGraspingVisionOnlyEnv):
    """Cube grasping environment with wrist-camera vision, 2 tactile RGB sensors, and gate."""

    cfg: CubeGraspingVisionTwoTactileGateCfg

    def __init__(self, cfg: CubeGraspingVisionTwoTactileGateCfg, render_mode: str | None = None, **kwargs):
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

        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        self._tactile_ref_left = torch.zeros(
            (self.num_envs, 3, target_h, target_w), device=self.device, dtype=torch.float32
        )
        self._tactile_ref_right = torch.zeros(
            (self.num_envs, 3, target_h, target_w), device=self.device, dtype=torch.float32
        )
        self._tactile_ref_pending_left = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self._tactile_ref_pending_right = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)

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

        def _prep_rgb_01(rgb_tensor: torch.Tensor | None) -> torch.Tensor:
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

        def _prep_rgb_for_encoder(x_01: torch.Tensor) -> torch.Tensor:
            if self._tactile_encoder is None:
                return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
            return (x_01 - self._tactile_imgnet_mean) / self._tactile_imgnet_std

        tact_l_raw = getattr(self, "gsmini_left", None)
        tact_l_raw = tact_l_raw.data.output.get("tactile_rgb") if tact_l_raw is not None else None
        tact_r_raw = getattr(self, "gsmini_right", None)
        tact_r_raw = tact_r_raw.data.output.get("tactile_rgb") if tact_r_raw is not None else None

        has_tact_l = tact_l_raw is not None
        has_tact_r = tact_r_raw is not None

        tact_l_01 = _prep_rgb_01(tact_l_raw)
        tact_r_01 = _prep_rgb_01(tact_r_raw)

        if self._tactile_encoder is None or not has_tact_l:
            tact_l = torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
        else:
            tact_l = _prep_rgb_for_encoder(tact_l_01)
        if self._tactile_encoder is None or not has_tact_r:
            tact_r = torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
        else:
            tact_r = _prep_rgb_for_encoder(tact_r_01)

        gate_l = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        gate_r = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if bool(getattr(self.cfg, "tactile_gate_enable", True)):
            diff_thresh = float(getattr(self.cfg, "tactile_gate_diff_threshold", 0.02))
            if has_tact_l and torch.any(self._tactile_ref_pending_left):
                pending = self._tactile_ref_pending_left
                self._tactile_ref_left[pending] = tact_l_01[pending].detach()
                self._tactile_ref_pending_left[pending] = False
            if has_tact_r and torch.any(self._tactile_ref_pending_right):
                pending = self._tactile_ref_pending_right
                self._tactile_ref_right[pending] = tact_r_01[pending].detach()
                self._tactile_ref_pending_right[pending] = False

            if has_tact_l:
                diff_l = (tact_l_01 - self._tactile_ref_left).abs().mean(dim=(1, 2, 3))
                gate_l = (~self._tactile_ref_pending_left) & (diff_l >= diff_thresh)
            if has_tact_r:
                diff_r = (tact_r_01 - self._tactile_ref_right).abs().mean(dim=(1, 2, 3))
                gate_r = (~self._tactile_ref_pending_right) & (diff_r >= diff_thresh)
        else:
            if has_tact_l:
                gate_l = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
            if has_tact_r:
                gate_r = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)

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

        if fl is not None:
            fl = fl * gate_l.to(dtype=fl.dtype).unsqueeze(-1)
        if fr is not None:
            fr = fr * gate_r.to(dtype=fr.dtype).unsqueeze(-1)

        self._tactile_gate_any = gate_l | gate_r

        obs.update(
            {
                "tactile_left_resnet": fl,
                "tactile_right_resnet": fr,
            }
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards and print tactile gate statistics with reward logs."""
        cube_pos = self._cube.data.root_com_pos_w
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )

        # reaching_object: r_reach = 1 - tanh(||p_obj - p_ee|| / sigma)
        sigma = max(self.cfg.reach_sigma, 1e-6)
        reach_distance = torch.norm(cube_pos - ee_pos, dim=-1)
        reach_reward = 1.0 - torch.tanh(reach_distance / sigma)

        # lifting_object: base reward above h_min + linear reward for height above h_min
        h_min = self.cfg.minimal_lift_height
        height = cube_pos[:, 2]
        base_reward = (height > h_min).float()
        denom = max(self.cfg.success_height - h_min, 1e-6)
        linear_reward = torch.clamp((height - h_min) / denom, 0.0, 1.0)
        lift_reward = base_reward * 0.5 + linear_reward

        # success bonus: give once when height stays above threshold for N consecutive steps
        hold_steps = max(1, int(getattr(self.cfg, "success_hold_steps", 1)))
        success_now = height > self.cfg.success_height
        if hold_steps > 1:
            self._success_hold_buf = torch.where(
                success_now,
                torch.clamp(self._success_hold_buf + 1, max=hold_steps),
                torch.zeros_like(self._success_hold_buf),
            )
            success_hold = self._success_hold_buf >= hold_steps
        else:
            success_hold = success_now
        success_first = success_hold & ~self._success_achieved_buf
        self._success_achieved_buf |= success_first
        success_reward = success_first.float()

        rewards = (
            self.cfg.reach_weight * reach_reward
            + self.cfg.lift_weight * lift_reward
            + self.cfg.success_reward_weight * success_reward
        )

        # drop penalty: if lifted before and now falls below lift threshold
        lifted = height > self.cfg.minimal_lift_height
        self._ever_lifted_buf |= lifted
        dropped_now = self._ever_lifted_buf & ~self._dropped_buf & (height <= self.cfg.minimal_lift_height)
        drop_penalty = (-self.cfg.success_reward_weight) * dropped_now.float()
        if not self.cfg.drop_penalty_enabled:
            drop_penalty = torch.zeros_like(drop_penalty)
        if dropped_now.any() and self.cfg.drop_penalty_enabled:
            rewards = rewards + drop_penalty
            self._dropped_buf |= dropped_now

        self._ep_return = self._ep_return + rewards
        self._ep_len = self._ep_len + 1
        if success_first.any():
            success_ids = success_first.nonzero(as_tuple=False).squeeze(-1)
            success_steps = torch.clamp(self._ep_len[success_ids], min=1)
            self._success_steps_buf[success_ids] = success_steps

        if (
            self.reward_print_interval > 0
            and (self.step_count + 1) % self.reward_print_interval == 0
        ):
            if self._success_steps_window_count > 0:
                window_avg = self._success_steps_window_sum / self._success_steps_window_count
                window_min = self._success_steps_window_min
                window_max = self._success_steps_window_max
            else:
                window_avg = 0.0
                window_min = 0
                window_max = 0
            if self._success_window_done > 0:
                window_success_rate = self._success_window_success / self._success_window_done
            else:
                window_success_rate = 0.0
            self.extras.setdefault("log", {})
            self.extras["log"]["success_rate"] = torch.tensor(window_success_rate, device=self.device)
            self.extras["log"]["info/success_steps_mean_window"] = torch.tensor(window_avg, device=self.device)
            self.extras["log"]["info/success_steps_min_window"] = torch.tensor(float(window_min), device=self.device)
            self.extras["log"]["info/success_steps_max_window"] = torch.tensor(float(window_max), device=self.device)

            gate_any = getattr(self, "_tactile_gate_any", None)
            if gate_any is None:
                gate_pct = 0.0
            else:
                gate_pct = float(gate_any.to(dtype=torch.float32).mean().item() * 100.0)
            self.extras["log"]["info/tactile_gate_pct"] = torch.tensor(gate_pct, device=self.device)

            cube_height_mean = float(cube_pos[:, 2].mean().item())
            success_step = float(success_reward.mean().item())
            success_term = float(success_step * self.cfg.success_reward_weight)
            print(
                f"[奖励] step {self.step_count + 1}: "
                f"reach={reach_reward.mean().item():.3f} "
                f"(w={self.cfg.reach_weight}), lift={lift_reward.mean().item():.3f} "
                f"(w={self.cfg.lift_weight}), success={window_success_rate:.3f} "
                f"(w={self.cfg.success_reward_weight}), "
                f"success_step={success_step:.3f} "
                f"(term={success_term:.3f}), "
                f"cube_h={cube_height_mean:.4f}, "
                f"drop_penalty={drop_penalty.mean().item():.3f} "
                f"(count={int(dropped_now.sum().item())}), "
                f"tactile_gate={gate_pct:.1f}%, "
                f"success_steps(avg={window_avg:.1f},"
                f"min={window_min},max={window_max}), "
                f"total={rewards.mean().item():.3f}"
            )
            # reset windowed success-steps stats after printing
            self._success_steps_window_sum = 0.0
            self._success_steps_window_count = 0
            self._success_steps_window_min = None
            self._success_steps_window_max = None
            # reset windowed success counts after printing
            self._success_window_done = 0
            self._success_window_success = 0
            self._success_window_timeout = 0
            self._success_window_collision = 0

        self.step_count += 1
        self.prev_actions = self.processed_actions.detach().clone()

        return rewards

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if hasattr(self, "_tactile_ref_pending_left"):
            self._tactile_ref_pending_left[env_ids] = True
            self._tactile_ref_pending_right[env_ids] = True
            self._tactile_ref_left[env_ids] = 0.0
            self._tactile_ref_right[env_ids] = 0.0
