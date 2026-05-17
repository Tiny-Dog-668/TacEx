# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Franka robot in cylinder grasping task."""

from tacex_tasks.cylinder_grasping.cylinder_grasping_env_cfg import CylinderGraspingEnvCfg


class CylinderGraspingFrankaCfg(CylinderGraspingEnvCfg):
    """Configuration for Franka robot in cylinder grasping task."""

    # Override specific parameters for Franka
    action_scale = 0.1
    episode_length_s = 15.0
    
    # More aggressive reward weights for better learning
    reward_terms = {
        "grasp_reward": {"weight": 2.0, "threshold": 0.02},
        "lift_reward": {"weight": 1.0, "target_height": 0.2},
        "approach_reward": {"weight": 0.5, "threshold": 0.1},
        "gripper_penalty": {"weight": -0.2},
        "action_penalty": {"weight": -0.05},
    }
    
    # Task specific parameters
    cylinder_height = 0.1
    cylinder_radius = 0.02
    grasp_threshold = 0.02
    lift_height = 0.2
    approach_threshold = 0.1
    
    # Randomization parameters
    cylinder_pos_range = 0.15
    cylinder_rot_range = 0.3
