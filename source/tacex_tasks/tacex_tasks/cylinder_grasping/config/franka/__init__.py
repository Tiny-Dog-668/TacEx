# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Franka robot in cylinder grasping task."""

import gymnasium as gym

from .cylinder_grasping_franka_cfg import CylinderGraspingFrankaCfg

# Register the environment
gym.register(
    id="Isaac-CylinderGrasping-Franka-v0",
    entry_point="tacex_tasks.cylinder_grasping:CylinderGraspingEnv",
    kwargs={"cfg": CylinderGraspingFrankaCfg()},
)
