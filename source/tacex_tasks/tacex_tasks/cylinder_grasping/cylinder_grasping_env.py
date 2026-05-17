# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the cylinder grasping environment."""

import torch
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, TiledCamera
from isaaclab.utils.math import quat_from_euler_xyz

from tacex import GelSightSensor

from .cylinder_grasping_env_cfg import CylinderGraspingEnvCfg


class CylinderGraspingEnv(DirectRLEnv):
    """Environment for training a robot to grasp a cylinder using vision and tactile feedback."""

    cfg: CylinderGraspingEnvCfg

    def __init__(self, cfg: CylinderGraspingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize tensors
        self._init_tensors()

        # Get robot and object references
        self._robot: Articulation = self.scene["robot"]
        self._cylinder: RigidObject = self.scene["cylinder"]
        self._wrist_camera: Camera = self.scene["wrist_camera"]
        self._gelsight_left: GelSightSensor = self.scene["gelsight_left"]
        self._gelsight_right: GelSightSensor = self.scene["gelsight_right"]

        # Task-specific variables
        self._cylinder_initial_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._cylinder_current_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._gripper_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._gripper_quat = torch.zeros(self.num_envs, 4, device=self.device)

    def _init_tensors(self):
        """Initialize tensors for the environment."""
        # Initialize cylinder position tracking
        self._cylinder_initial_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._cylinder_current_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Initialize gripper pose tracking
        self._gripper_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._gripper_quat = torch.zeros(self.num_envs, 4, device=self.device)

    def _setup_scene(self):
        """Setup the scene with robot, sensors, and objects."""
        # This method is called by the parent class
        pass

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step."""
        # Convert actions to joint commands
        self._apply_action(actions)

    def _apply_action(self, actions: torch.Tensor):
        """Apply actions to the robot."""
        # Scale actions
        action_scale = getattr(self.cfg, 'action_scale', 1.0)
        actions = actions * action_scale
        
        # Apply arm actions (first 6 dimensions)
        arm_actions = actions[:, :6]
        arm_joint_ids = list(range(7))  # First 7 joints are arm joints
        self._robot.set_joint_position_target(arm_actions, joint_ids=arm_joint_ids)
        
        # Apply gripper actions (last dimension)
        gripper_actions = actions[:, 6:7]
        gripper_joint_ids = [7, 8]  # Last 2 joints are gripper joints
        self._robot.set_joint_position_target(gripper_actions, joint_ids=gripper_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get observations from the environment."""
        # Proprioceptive observations
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)
        
        # Vision observations
        wrist_rgb = self._wrist_camera.data.rgb
        tactile_left = self._gelsight_left.data.tactile_rgb
        tactile_right = self._gelsight_right.data.tactile_rgb
        
        return {
            "proprio_obs": proprio_obs,
            "wrist_rgb": wrist_rgb,
            "tactile_left": tactile_left,
            "tactile_right": tactile_right,
        }

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards for the current state."""
        # Get current positions
        self._cylinder_current_pos = self._cylinder.data.root_pos_w
        self._gripper_pos = self._robot.data.ee_pos_w
        self._gripper_quat = self._robot.data.ee_quat_w
        
        # Calculate individual reward components
        grasp_reward = self._get_grasp_reward()
        lift_reward = self._get_lift_reward()
        approach_reward = self._get_approach_reward()
        gripper_penalty = self._get_gripper_penalty()
        action_penalty = self._get_action_penalty()
        
        # Combine rewards
        total_reward = (
            self.cfg.reward_terms["grasp_reward"]["weight"] * grasp_reward +
            self.cfg.reward_terms["lift_reward"]["weight"] * lift_reward +
            self.cfg.reward_terms["approach_reward"]["weight"] * approach_reward +
            self.cfg.reward_terms["gripper_penalty"]["weight"] * gripper_penalty +
            self.cfg.reward_terms["action_penalty"]["weight"] * action_penalty
        )
        
        return total_reward

    def _get_grasp_reward(self) -> torch.Tensor:
        """Calculate grasp reward based on distance between gripper and cylinder."""
        distance = torch.norm(self._gripper_pos - self._cylinder_current_pos, dim=-1)
        threshold = self.cfg.reward_terms["grasp_reward"]["threshold"]
        
        # Reward for being close to the cylinder
        grasp_reward = torch.exp(-distance / threshold)
        return grasp_reward

    def _get_lift_reward(self) -> torch.Tensor:
        """Calculate lift reward based on cylinder height."""
        cylinder_height = self._cylinder_current_pos[:, 2]
        target_height = self.cfg.reward_terms["lift_reward"]["target_height"]
        
        # Reward for lifting the cylinder
        lift_reward = torch.clamp(cylinder_height / target_height, 0.0, 1.0)
        return lift_reward

    def _get_approach_reward(self) -> torch.Tensor:
        """Calculate approach reward for moving towards the cylinder."""
        distance = torch.norm(self._gripper_pos - self._cylinder_current_pos, dim=-1)
        threshold = self.cfg.reward_terms["approach_reward"]["threshold"]
        
        # Reward for approaching the cylinder
        approach_reward = torch.exp(-distance / threshold)
        return approach_reward

    def _get_gripper_penalty(self) -> torch.Tensor:
        """Calculate penalty for excessive gripper opening."""
        gripper_pos = self._robot.data.joint_pos[:, self._robot.gripper_joint_ids]
        gripper_penalty = torch.sum(torch.abs(gripper_pos), dim=-1)
        return gripper_penalty

    def _get_action_penalty(self) -> torch.Tensor:
        """Calculate penalty for large actions."""
        action_penalty = torch.sum(torch.square(self._actions), dim=-1)
        return action_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate done flags."""
        # Check if episode length exceeded
        time_out = self.episode_length_buf >= self.max_episode_length
        
        # Check if cylinder fell too low
        cylinder_height = self._cylinder_current_pos[:, 2]
        cylinder_fell = cylinder_height < -0.1
        
        # Check if cylinder was successfully lifted
        success = cylinder_height > self.cfg.lift_height
        
        # Done if time out, cylinder fell, or success
        done = time_out | cylinder_fell | success
        
        return done, success

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at the given indices."""
        # Reset robot
        self._robot.reset(env_ids)
        
        # Reset cylinder position with randomization
        cylinder_pos = torch.zeros(len(env_ids), 3, device=self.device)
        cylinder_pos[:, 0] = 0.5 + torch.randn(len(env_ids), device=self.device) * self.cfg.cylinder_pos_range
        cylinder_pos[:, 1] = torch.randn(len(env_ids), device=self.device) * self.cfg.cylinder_pos_range
        cylinder_pos[:, 2] = 0.05
        
        cylinder_rot = quat_from_euler_xyz(
            torch.randn(len(env_ids), device=self.device) * self.cfg.cylinder_rot_range,
            torch.randn(len(env_ids), device=self.device) * self.cfg.cylinder_rot_range,
            torch.randn(len(env_ids), device=self.device) * self.cfg.cylinder_rot_range,
        )
        
        self._cylinder.write_root_pose_to_sim(cylinder_pos, cylinder_rot, env_ids)
        self._cylinder.write_root_velocity_to_sim(
            torch.zeros_like(cylinder_pos), torch.zeros_like(cylinder_pos), env_ids
        )
        
        # Store initial cylinder position
        self._cylinder_initial_pos[env_ids] = cylinder_pos

    def _reset_scene(self, env_ids: torch.Tensor):
        """Reset the scene at the given indices."""
        # Reset robot
        self._robot.reset(env_ids)
        
        # Reset cylinder
        self._cylinder.reset(env_ids)
        
        # Reset sensors
        self._wrist_camera.reset(env_ids)
        self._gelsight_left.reset(env_ids)
        self._gelsight_right.reset(env_ids)

    def _get_info(self) -> dict[str, Any]:
        """Get additional information about the environment."""
        return {
            "cylinder_pos": self._cylinder_current_pos,
            "gripper_pos": self._gripper_pos,
            "gripper_quat": self._gripper_quat,
            "cylinder_height": self._cylinder_current_pos[:, 2],
        }
