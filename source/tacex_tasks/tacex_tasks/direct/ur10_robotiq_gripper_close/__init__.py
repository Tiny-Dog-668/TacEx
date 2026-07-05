# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal direct RL task for closing the Robotiq finger joint."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-UR10-Robotiq-Gripper-Close-Direct-v0",
    entry_point=f"{__name__}.ur10_robotiq_gripper_close_env:UR10RobotiqGripperCloseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_robotiq_gripper_close_env:UR10RobotiqGripperCloseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10RobotiqGripperClosePPORunnerCfg",
    },
)
