# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can grasping environment with privileged state observations and 6 GelSight RGB sensors."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from .can_grasping_vision_only_resnet18 import (
    CanGraspingVisionOnlyCfg,
    CanGraspingVisionOnlyEnv,
)


@configclass
class CanGraspingSixTactileRGBCfg(CanGraspingVisionOnlyCfg):
    """Configuration for can grasping with 6 tactile RGB sensors (no vision cameras)."""

    tactile_img_res_hw = (96, 128)

    can_pos_range = 0.02
    # tactile contact reward (inner sensors only: left/right)
    tactile_contact_weight_inner = 15.0
    tactile_contact_threshold = 0.8  # depth_norm threshold in [0,1] to count a pixel as contact
    tactile_ratio_scale = 0.01  # contact ratio scale before tanh
    # reset-search parameters (scripted exploration before episode starts)
    reset_settle_steps = 20  # reset 后物理 settle
    reset_search_max_steps = 150  # scripted search 最大步数
    reset_search_steps_per_check = 2  # 每次动作后推进多少个 sim step 再检查触觉 (>=2 对齐 update_period=2)
    reset_search_pattern = "privileged_reach"  # privileged_reach
    # reset-search privileged reach (hover-then-descend)
    reset_priv_hover_z: float = 0.06  # hover height above can (m)
    reset_priv_xy_tol: float = 0.01  # XY alignment tolerance (m)
    reset_priv_xy_kp: float = 6.0  # XY proportional gain
    reset_priv_z_kp: float = 4.0  # Z proportional gain
    reset_priv_max_delta_world: float = 0.01  # max per-step delta in world (m)
    reset_priv_xy_noise: float = 0.0  # XY target noise for contact diversity (m)
    reset_priv_z_bias: float = 0.0  # Z bias when descending toward can (m)
    reset_contact_depth_thr = 0.8  # 与 tactile_contact_threshold 对齐
    reset_contact_ratio_min = 0.002  # 触觉接触像素比例阈值
    reset_contact_confirm_steps = 1  # 连续满足多少次才认为接触
    reset_search_retries = 1  # 没接触到是否重试 reset+search

    # Use the 6-pad GelSight Mini asset (adds *_down and *_out pads expected by this env).
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.six.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # "panda_joint1": -0.3136,
                # "panda_joint2": -0.4448,
                # "panda_joint3": 0.3858,
                # "panda_joint4": -3.0236,
                # "panda_joint5": 0.2319,
                # "panda_joint6": 2.5776,
                # "panda_joint7": 0.6249,

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

    # Six tactile RGB sensors (left/right + left_down/right_down + left_out/right_out).
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

    gsmini_left_out: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_out"
    )
    gsmini_left_out.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left_out.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_left_out.marker_motion_sim_cfg = None
    gsmini_left_out.optical_sim_cfg = gsmini_left_out.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_right_out: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_out"
    )
    gsmini_right_out.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right_out.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_right_out.marker_motion_sim_cfg = None
    gsmini_right_out.optical_sim_cfg = gsmini_right_out.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    observation_space = {
        "proprio_obs": 18,
        "gripper_state": 1,
        "action_history": 5,
        "tactile_left_resnet": 128,
        "tactile_right_resnet": 128,
        "tactile_left_down_resnet": 128,
        "tactile_right_down_resnet": 128,
        "tactile_left_out_resnet": 128,
        "tactile_right_out_resnet": 128,
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
        "rnn_reset": 1,
    }


class CanGraspingSixTactileRGBEnv(CanGraspingVisionOnlyEnv):
    """Can grasping environment with privileged state and 6 tactile RGB sensors."""

    cfg: CanGraspingSixTactileRGBCfg

    def __init__(self, cfg: CanGraspingSixTactileRGBCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ImageNet-pretrained ResNet18 to encode 3-ch tactile RGB into 128-D features.
        self._tactile_feature_dim = 128
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, self._tactile_feature_dim)
        self._tactile_encoder = resnet.to(self.device)
        self._tactile_encoder.eval()
        for p in self._tactile_encoder.parameters():
            p.requires_grad_(False)
        self._imgnet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._imgnet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self._rnn_reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._reset_search_total = 0
        self._reset_search_fail = 0
        self._reset_search_print_interval = 100
        self._reset_priv_xy_noise_buf = torch.zeros((self.num_envs, 2), device=self.device)
        self._reset_priv_reach_steps_sum = 0
        self._reset_priv_reach_ratio_sum = 0.0
        self._reset_priv_reach_contacts = 0
        self._tactile_contacted_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _setup_scene(self):
        """Setup the scene."""
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # can
        self._can = RigidObject(self.cfg.can)
        self.scene.rigid_objects["can"] = self._can

        # plate
        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        # clone environments first
        self.scene.clone_environments(copy_from_source=False)

        # tactile sensors
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right
        self.gsmini_left_down = GelSightSensor(self.cfg.gsmini_left_down)
        self.scene.sensors["gsmini_left_down"] = self.gsmini_left_down
        self.gsmini_right_down = GelSightSensor(self.cfg.gsmini_right_down)
        self.scene.sensors["gsmini_right_down"] = self.gsmini_right_down
        self.gsmini_left_out = GelSightSensor(self.cfg.gsmini_left_out)
        self.scene.sensors["gsmini_left_out"] = self.gsmini_left_out
        self.gsmini_right_out = GelSightSensor(self.cfg.gsmini_right_out)
        self.scene.sensors["gsmini_right_out"] = self.gsmini_right_out

        # add frame transformer for end-effector
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.11841),
                    ),
                ),
            ],
        )

        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _step_sim_n(self, n: int):
        """Step the simulator for n physics steps (no rewards/dones/rollout updates)."""
        if n <= 0:
            return
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(n):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

    def _compute_any_contact_ratio(self) -> torch.Tensor:
        """Compute tactile contact ratio from depth (any GelSight pad).

        Reset-search gating should start the RL episode as soon as *any* tactile sensor reports contact,
        not only the inner left/right pads.
        """
        depth_tensors = []
        for name in (
            "gsmini_left",
            "gsmini_right",
            "gsmini_left_down",
            "gsmini_right_down",
            "gsmini_left_out",
            "gsmini_right_out",
        ):
            sensor = getattr(self, name, None)
            depth_tensors.append(sensor.data.output.get("camera_depth") if sensor is not None else None)

        def _to_01(depth_tensor: torch.Tensor) -> torch.Tensor:
            d = depth_tensor.to(device=self.device, dtype=torch.float32)
            if d.dtype == torch.uint8 or (d.numel() > 0 and d.max() > 2.0):
                d = d / 255.0
            return d

        def _contact_ratio_from_depth(depth_tensor, thr: float) -> torch.Tensor:
            if depth_tensor is None:
                return torch.zeros((self.num_envs,), device=self.device)
            d = _to_01(depth_tensor)
            contact_px = (d < thr).float()
            # robust to depth shapes: [N,H,W] or [N,H,W,C]
            return contact_px.reshape(contact_px.shape[0], -1).mean(dim=1)

        thr = float(getattr(self.cfg, "reset_contact_depth_thr", self.cfg.tactile_contact_threshold))
        ratios = [_contact_ratio_from_depth(d, thr) for d in depth_tensors]
        # "any sensor contacts" => take max ratio across all sensors
        return torch.stack(ratios, dim=0).amax(dim=0)

    def _compute_ee_pos_w(self) -> torch.Tensor:
        """Compute end-effector position in world frame (same path as observations)."""
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]

        def _norm_quat(q: torch.Tensor) -> torch.Tensor:
            n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
            return q / n

        hand_quat = _norm_quat(hand_quat)
        offset_rot = _norm_quat(self._offset_rot)
        ee_pos, _ = math_utils.combine_frame_transforms(hand_pos, hand_quat, self._offset_pos, offset_rot)
        return ee_pos

    def _make_privileged_reach_delta_xyz(self, active_env_ids: torch.Tensor) -> torch.Tensor:
        """Compute privileged reach delta xyz in world frame (hover-then-descend)."""
        if active_env_ids.numel() == 0:
            return torch.zeros((0, 3), device=self.device, dtype=torch.float32)
        active_env_ids = active_env_ids.to(device=self.device)

        ee_pos = self._compute_ee_pos_w()[active_env_ids]
        can_pos = self._can.data.root_com_pos_w[active_env_ids]
        xy_noise = self._reset_priv_xy_noise_buf[active_env_ids]

        target_xy = can_pos[:, :2] + xy_noise
        xy_err = target_xy - ee_pos[:, :2]
        xy_dist = torch.norm(xy_err, dim=-1, keepdim=True)

        hover_z = float(getattr(self.cfg, "reset_priv_hover_z", 0.06))
        z_bias = float(getattr(self.cfg, "reset_priv_z_bias", 0.0))
        xy_tol = float(getattr(self.cfg, "reset_priv_xy_tol", 0.01))
        hover_z_target = can_pos[:, 2:3] + hover_z
        descend_z_target = can_pos[:, 2:3] + z_bias
        target_z = torch.where(xy_dist < xy_tol, descend_z_target, hover_z_target)

        kp_xy = float(getattr(self.cfg, "reset_priv_xy_kp", 6.0))
        kp_z = float(getattr(self.cfg, "reset_priv_z_kp", 4.0))
        delta_xy = xy_err * kp_xy
        delta_z = (target_z - ee_pos[:, 2:3]) * kp_z
        delta_xyz = torch.cat([delta_xy, delta_z], dim=-1)

        max_delta = max(float(getattr(self.cfg, "reset_priv_max_delta_world", 0.01)), 0.0)
        delta_xyz = torch.clamp(delta_xyz, min=-max_delta, max=max_delta)
        return delta_xyz

    def _make_search_actions(self, k: int, active_env_ids: torch.Tensor) -> torch.Tensor:
        """Create scripted 3D search actions in the 5D action space (dx, dy, dz, dyaw, gripper)."""
        action_dim = int(self.cfg.action_space)
        actions = torch.zeros((len(active_env_ids), action_dim), device=self.device, dtype=torch.float32)
        pattern = getattr(self.cfg, "reset_search_pattern", "privileged_reach")
        if pattern != "privileged_reach":
            return actions
        delta_xyz = self._make_privileged_reach_delta_xyz(active_env_ids)
        action_scale = float(getattr(self.cfg, "action_scale", 1.0))
        action_scale = max(action_scale, 1e-6)
        delta_xyz = torch.clamp(delta_xyz / action_scale, min=-1.0, max=1.0)
        if action_dim > 0:
            actions[:, 0] = delta_xyz[:, 0]
        if action_dim > 1:
            actions[:, 1] = delta_xyz[:, 1]
        if action_dim > 2:
            actions[:, 2] = delta_xyz[:, 2]
        # dyaw/gripper left at zero to avoid large orientation/gripper changes
        return actions

    def _scripted_search_to_contact(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Scripted search until tactile contact is confirmed or max steps reached."""
        if env_ids.numel() == 0:
            return torch.zeros((0,), device=self.device, dtype=torch.bool)
        env_ids = env_ids.to(device=self.device)
        pattern = getattr(self.cfg, "reset_search_pattern", "privileged_reach")
        use_privileged = pattern == "privileged_reach"
        if use_privileged:
            xy_noise = float(getattr(self.cfg, "reset_priv_xy_noise", 0.0))
            if xy_noise > 0.0:
                noise = (torch.rand((env_ids.numel(), 2), device=self.device) * 2.0 - 1.0) * xy_noise
            else:
                noise = torch.zeros((env_ids.numel(), 2), device=self.device)
            self._reset_priv_xy_noise_buf[env_ids] = noise
            contact_steps = torch.full((env_ids.numel(),), -1, device=self.device, dtype=torch.int32)
            contact_ratios = torch.zeros((env_ids.numel(),), device=self.device, dtype=torch.float32)
        # active_mask/confirm_count are per-env (parallel) to keep vectorized search in sync.
        active_mask = torch.ones((env_ids.numel(),), device=self.device, dtype=torch.bool)
        confirm_count = torch.zeros((env_ids.numel(),), device=self.device, dtype=torch.int32)

        ratio_min = float(getattr(self.cfg, "reset_contact_ratio_min", 0.0))
        confirm_steps = max(1, int(getattr(self.cfg, "reset_contact_confirm_steps", 1)))
        steps_per_check = max(1, int(getattr(self.cfg, "reset_search_steps_per_check", 2)))
        max_steps = max(0, int(getattr(self.cfg, "reset_search_max_steps", 0)))

        for k in range(max_steps):
            if not active_mask.any():
                break
            active_env_ids = env_ids[active_mask]
            search_actions = self._make_search_actions(k, active_env_ids)

            # Apply scripted actions through the same action pipeline as training.
            if hasattr(self, "actions"):
                actions_full = self.actions.clone()
            else:
                actions_full = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
            actions_full[active_env_ids] = search_actions
            self._pre_physics_step(actions_full)

            # Step physics a few times before checking (sensor update_period=2).
            self._step_sim_n(steps_per_check)
            ratio = self._compute_any_contact_ratio()[env_ids]

            hit = ratio > ratio_min
            confirm_count = torch.where(
                active_mask,
                torch.where(hit, confirm_count + 1, torch.zeros_like(confirm_count)),
                confirm_count,
            )
            newly_contacted = active_mask & (confirm_count >= confirm_steps)
            if newly_contacted.any():
                contacted_env_ids = env_ids[newly_contacted]
                self._tactile_contacted_buf[contacted_env_ids] = True
                ratio_mean = ratio[newly_contacted].mean().item()
                print(
                    f"[tactile_contact] step={k + 1} envs={contacted_env_ids.numel()} "
                    f"ratio_mean={ratio_mean:.4f}"
                )
            if use_privileged and newly_contacted.any():
                contact_steps[newly_contacted] = k + 1
                contact_ratios[newly_contacted] = ratio[newly_contacted]
            active_mask = active_mask & ~newly_contacted

        if use_privileged:
            contacted = confirm_count >= confirm_steps
            if contacted.any():
                self._reset_priv_reach_steps_sum += int(contact_steps[contacted].sum().item())
                self._reset_priv_reach_ratio_sum += float(contact_ratios[contacted].sum().item())
                self._reset_priv_reach_contacts += int(contacted.sum().item())

        return confirm_count >= confirm_steps

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        self._tactile_contacted_buf[env_ids] = False

        # settle after reset
        settle_steps = max(0, int(getattr(self.cfg, "reset_settle_steps", 0)))
        if settle_steps > 0:
            if hasattr(self, "actions"):
                actions_full = self.actions.clone()
            else:
                actions_full = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
            actions_full[env_ids] = 0.0
            self._pre_physics_step(actions_full)
            self._step_sim_n(settle_steps)

        # scripted search until contact (with retries)
        contacted_mask = self._scripted_search_to_contact(env_ids)
        retries = max(0, int(getattr(self.cfg, "reset_search_retries", 0)))
        for _ in range(retries):
            if contacted_mask.all():
                break
            failed_env_ids = env_ids[~contacted_mask]
            super()._reset_idx(failed_env_ids)
            if settle_steps > 0:
                if hasattr(self, "actions"):
                    actions_full = self.actions.clone()
                else:
                    actions_full = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
                actions_full[failed_env_ids] = 0.0
                self._pre_physics_step(actions_full)
                self._step_sim_n(settle_steps)
            retry_mask = self._scripted_search_to_contact(failed_env_ids)
            contacted_mask[~contacted_mask] = retry_mask

        # start episode at contact (do not let search steps count into rollout)
        if hasattr(self, "episode_length_buf"):
            self.episode_length_buf[env_ids] = 0
        if hasattr(self, "_rnn_reset_buf"):
            self._rnn_reset_buf[env_ids] = True

        failed = env_ids[~contacted_mask]
        self._reset_search_total += int(env_ids.numel())
        self._reset_search_fail += int(failed.numel())
        if self._reset_search_print_interval > 0 and self._reset_search_total % self._reset_search_print_interval == 0:
            fail_rate = self._reset_search_fail / max(1, self._reset_search_total)
            msg = (
                f"[reset_search] total={self._reset_search_total} "
                f"failed={self._reset_search_fail} rate={fail_rate:.3f}"
            )
            if getattr(self.cfg, "reset_search_pattern", "") == "privileged_reach":
                if self._reset_priv_reach_contacts > 0:
                    avg_steps = self._reset_priv_reach_steps_sum / max(1, self._reset_priv_reach_contacts)
                    avg_ratio = self._reset_priv_reach_ratio_sum / max(1, self._reset_priv_reach_contacts)
                    msg += f" privileged_reach_avg_steps={avg_steps:.2f} avg_ratio={avg_ratio:.4f}"
            print(msg)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        # Proprioceptive observations
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        # Privileged observations - can
        can_pos = self._can.data.root_com_pos_w
        can_quat = self._can.data.root_quat_w
        can_lin_vel = self._can.data.root_com_lin_vel_w
        can_ang_vel = self._can.data.root_com_ang_vel_w

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
        target_pos_relative = can_pos - ee_pos
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        # Gripper state
        gripper_state = self._robot.data.joint_pos[:, self._finger_joint_ids[0:1]]

        # Tactile RGB from GelSight sensors (left/right + left_down/right_down + left_out/right_out).
        tact_l_raw = self.gsmini_left.data.output.get("tactile_rgb")
        tact_r_raw = self.gsmini_right.data.output.get("tactile_rgb")
        tact_ld_raw = self.gsmini_left_down.data.output.get("tactile_rgb")
        tact_rd_raw = self.gsmini_right_down.data.output.get("tactile_rgb")
        tact_lo_raw = self.gsmini_left_out.data.output.get("tactile_rgb")
        tact_ro_raw = self.gsmini_right_out.data.output.get("tactile_rgb")

        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (240, 320))

        def _prep_rgb_3ch(rgb_tensor):
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
            xn = (xn - self._imgnet_mean) / self._imgnet_std
            return xn

        tact_l = _prep_rgb_3ch(tact_l_raw)
        tact_r = _prep_rgb_3ch(tact_r_raw)
        tact_ld = _prep_rgb_3ch(tact_ld_raw)
        tact_rd = _prep_rgb_3ch(tact_rd_raw)
        tact_lo = _prep_rgb_3ch(tact_lo_raw)
        tact_ro = _prep_rgb_3ch(tact_ro_raw)

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
            flo = self._tactile_encoder(tact_lo).view(self.num_envs, self._tactile_feature_dim).float()
            fro = self._tactile_encoder(tact_ro).view(self.num_envs, self._tactile_feature_dim).float()

        obs = {
            "proprio_obs": proprio_obs,
            "gripper_state": gripper_state,
            "action_history": self.action_history,
            "tactile_left_resnet": fl,
            "tactile_right_resnet": fr,
            "tactile_left_down_resnet": fld,
            "tactile_right_down_resnet": frd,
            "tactile_left_out_resnet": flo,
            "tactile_right_out_resnet": fro,
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

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards based on reaching, lifting, action rate, and inner tactile contact."""
        can_pos = self._can.data.root_com_pos_w
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )

        # reaching_object: r_reach = 1 - tanh(||p_obj - p_ee|| / sigma)
        sigma = max(self.cfg.reach_sigma, 1e-6)
        reach_distance = torch.norm(can_pos - ee_pos, dim=-1)
        reach_reward = 1.0 - torch.tanh(reach_distance / sigma)

        # lifting_object: base reward above h_min + linear reward for height above h_min
        h_min = self.cfg.minimal_lift_height
        height = can_pos[:, 2]
        base_reward = (height > h_min).float()
        denom = max(self.cfg.success_height - h_min, 1e-6)
        linear_reward = torch.clamp((height - h_min) / denom, 0.0, 1.0)
        lift_reward = base_reward * 0.5 + linear_reward

        # success bonus when object height exceeds success_height
        success_reward = (can_pos[:, 2] > self.cfg.success_height).float()

        # action_rate penalty: use sanitized, scaled actions to avoid explosion
        action_curr = torch.where(
            torch.isfinite(self.processed_actions), self.processed_actions, torch.zeros_like(self.processed_actions)
        )
        action_prev = torch.where(
            torch.isfinite(self.prev_actions), self.prev_actions, torch.zeros_like(self.prev_actions)
        )
        action_diff = action_curr - action_prev
        action_rate_penalty = torch.sum(action_diff * action_diff, dim=-1)

        # joint_vel penalty: sum_j omega_j^2
        joint_vel = self._robot.data.joint_vel
        joint_vel = torch.where(torch.isfinite(joint_vel), joint_vel, torch.zeros_like(joint_vel))
        joint_vel_penalty = torch.sum(joint_vel * joint_vel, dim=-1)

        # schedule for penalty weights
        schedule_reached = self.step_count >= self.cfg.penalty_schedule_steps
        action_weight = (
            self.cfg.action_rate_weight_final if schedule_reached else self.cfg.action_rate_weight_initial
        )
        joint_weight = (
            self.cfg.joint_vel_weight_final if schedule_reached else self.cfg.joint_vel_weight_initial
        )

        # tactile contact reward (inner tactile sensors: left/right)
        tact_l = self.gsmini_left.data.output.get("camera_depth") if hasattr(self, "gsmini_left") else None
        tact_r = self.gsmini_right.data.output.get("camera_depth") if hasattr(self, "gsmini_right") else None

        def _to_01(depth_tensor: torch.Tensor) -> torch.Tensor:
            d = depth_tensor.to(device=self.device, dtype=torch.float32)
            if d.dtype == torch.uint8 or (d.numel() > 0 and d.max() > 2.0):
                d = d / 255.0
            return d

        def _contact_ratio_from_depth(depth_tensor, thr: float) -> torch.Tensor:
            if depth_tensor is None:
                return torch.zeros((self.num_envs,), device=self.device)
            d = _to_01(depth_tensor)
            contact_px = (d < thr).float()
            return contact_px.mean(dim=(1, 2, 3), keepdim=False)

        thr = float(getattr(self.cfg, "tactile_contact_threshold", 0.8))
        ratio_l = _contact_ratio_from_depth(tact_l, thr)
        ratio_r = _contact_ratio_from_depth(tact_r, thr)
        ratio_scale = float(getattr(self.cfg, "tactile_ratio_scale", 0.01))
        s_l = torch.tanh(ratio_l / (ratio_scale + 1e-6))
        s_r = torch.tanh(ratio_r / (ratio_scale + 1e-6))
        tactile_contact_inner = 0.5 * (s_l + s_r)
        tactile_weight = float(getattr(self.cfg, "tactile_contact_weight_inner", 0.0))

        rewards = (
            self.cfg.reach_weight * reach_reward
            + self.cfg.lift_weight * lift_reward
            + self.cfg.success_reward_weight * success_reward
            + tactile_weight * tactile_contact_inner
            + action_weight * action_rate_penalty
            + joint_weight * joint_vel_penalty
        )

        if (
            self.reward_print_interval > 0
            and (self.step_count + 1) % self.reward_print_interval == 0
        ):
            print(
                f"[奖励] step {self.step_count + 1}: "
                f"reach={reach_reward.mean().item():.3f} "
                f"(w={self.cfg.reach_weight}), lift={lift_reward.mean().item():.3f} "
                f"(w={self.cfg.lift_weight}), success={success_reward.mean().item():.3f} "
                f"(w={self.cfg.success_reward_weight}), tactile_inner={tactile_contact_inner.mean().item():.3f} "
                f"(w={tactile_weight}), "
                f"action_penalty={action_rate_penalty.mean().item():.6f} "
                f"(w={action_weight}), joint_penalty={joint_vel_penalty.mean().item():.6f} "
                f"(w={joint_weight}), total={rewards.mean().item():.3f}"
            )

        self.step_count += 1
        self.prev_actions = action_curr.detach().clone()

        return rewards
