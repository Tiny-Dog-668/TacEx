"""Privileged-only variant of the third-view grasping task."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from .vt_box import (
    CAN_RESET_ROOT_Z,
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
    sample_uniform,
)


@configclass
class OccludedGraspingPrivilegedBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for privileged-only observations (no vision/tactile inputs to policy)."""

    observation_space = {
        "proprio_obs": 18,
        "can_pos": 3,
        "can_quat": 4,
        "can_lin_vel": 3,
        "can_ang_vel": 3,
        "gripper_pos": 3,
        "gripper_quat": 4,
        "gripper_lin_vel": 3,
        "gripper_ang_vel": 3,
        "target_pos": 3,
        "target_distance": 1,
    }


class OccludedGraspingPrivilegedBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Occluded grasping environment with privileged-only observations."""

    cfg: OccludedGraspingPrivilegedBoxCfg

    def __init__(self, cfg: OccludedGraspingPrivilegedBoxCfg, render_mode: str | None = None, **kwargs):
        # Bypass vision/tactile encoder initialization from parent for a clean privileged-only env.
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.step_count = 0
        self.reward_print_interval = int(getattr(self.cfg, "reward_print_interval", 200))

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1

        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.processed_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self._tactile_encoder_type = "none"
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])
        self._left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self._right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        self._ep_return = torch.zeros((self.num_envs,), device=self.device)
        self._ep_len = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._ep_avg_accum_sum = torch.tensor(0.0, device=self.device)
        self._ep_avg_accum_count = torch.tensor(0, dtype=torch.long, device=self.device)
        self._episode_start_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._episode_count = 0
        self._success_count = 0
        self._timeout_count = 0
        self._collision_count = 0
        self._recent_success_window = max(1, int(getattr(self.cfg, "recent_success_rate_window", 100)))
        self._recent_success_buf = torch.zeros((self._recent_success_window,), device=self.device, dtype=torch.float32)
        self._recent_success_count = torch.tensor(0, dtype=torch.long, device=self.device)
        self._recent_success_write_idx = torch.tensor(0, dtype=torch.long, device=self.device)
        self._success_hold_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._success_achieved_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._gate_checkpoints = (40, 80, 120, 160, 200)
        self._ep_gate_running_sum = torch.zeros((self.num_envs,), device=self.device)
        self._ep_gate_running_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._ep_gate_checkpoint_values = {
            step: torch.zeros((self.num_envs,), device=self.device) for step in self._gate_checkpoints
        }
        self._ep_gate_checkpoint_recorded = {
            step: torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device) for step in self._gate_checkpoints
        }

        self.set_debug_vis(self.cfg.debug_vis)
        self._initialize_can_positions()

    def _setup_scene(self):
        """Setup a clean privileged scene without cameras and tactile sensors."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._can = RigidObject(self.cfg.can)
        self.scene.rigid_objects["can"] = self._can

        self._box_floor = RigidObject(self.cfg.box_floor)
        self.scene.rigid_objects["box_floor"] = self._box_floor
        self._box_wall_front = RigidObject(self.cfg.box_wall_front)
        self.scene.rigid_objects["box_wall_front"] = self._box_wall_front
        self._box_wall_back = RigidObject(self.cfg.box_wall_back)
        self.scene.rigid_objects["box_wall_back"] = self._box_wall_back
        self._box_wall_left = RigidObject(self.cfg.box_wall_left)
        self.scene.rigid_objects["box_wall_left"] = self._box_wall_left
        self._box_wall_right = RigidObject(self.cfg.box_wall_right)
        self.scene.rigid_objects["box_wall_right"] = self._box_wall_right

        self.scene.clone_environments(copy_from_source=False)

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
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.11841)),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _initialize_can_positions(self):
        """Initialize can positions inside configured reset bounds."""
        can_state = self._can.data.default_root_state.clone()
        x_min, x_max, y_min, y_max = self._get_can_reset_xy_bounds()
        can_state[:, 0] = sample_uniform(x_min, x_max, (self.num_envs,), self.device)
        can_state[:, 1] = sample_uniform(y_min, y_max, (self.num_envs,), self.device)
        can_state[:, 2] = CAN_RESET_ROOT_Z
        z_jitter = max(float(getattr(self.cfg, "can_reset_z_jitter", 0.0)), 0.0)
        can_state[:, 2] += sample_uniform(-z_jitter, z_jitter, (self.num_envs,), self.device)
        rand_yaw = sample_uniform(-self.cfg.can_rot_range, self.cfg.can_rot_range, (self.num_envs,), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        can_state[:, 3:7] = rand_quat
        env_ids = torch.arange(self.num_envs, device=self.device)
        can_state[:, :3] += self.scene.env_origins
        self._can.write_root_state_to_sim(can_state, env_ids)
        self._can.write_root_velocity_to_sim(torch.zeros((self.num_envs, 6), device=self.device), env_ids)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        # Proprio
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        # Object state
        can_pos = self._can.data.root_pos_w
        can_quat = self._can.data.root_quat_w
        can_lin_vel = self._can.data.root_lin_vel_w
        can_ang_vel = self._can.data.root_ang_vel_w

        # Gripper state
        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        gripper_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        gripper_lin_vel = self._robot.data.body_link_lin_vel_w[:, self._body_idx]
        gripper_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._body_idx]

        # Relative target
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )
        target_pos_relative = can_pos - ee_pos
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        obs = {
            "proprio_obs": proprio_obs,
            "can_pos": can_pos,
            "can_quat": can_quat,
            "can_lin_vel": can_lin_vel,
            "can_ang_vel": can_ang_vel,
            "gripper_pos": gripper_pos,
            "gripper_quat": gripper_quat,
            "gripper_lin_vel": gripper_lin_vel,
            "gripper_ang_vel": gripper_ang_vel,
            "target_pos": target_pos_relative,
            "target_distance": target_distance,
        }
        return {"policy": obs}
