# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal direct RL task for UR10 + Robotiq cube pick and place."""

import gymnasium as gym

from . import agents


# Run:
# ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-UR10-Robotiq-Pick-Place-Direct-v0 --num_envs 64 --headless
gym.register(
    id="Isaac-UR10-Robotiq-Pick-Place-Direct-v0",
    entry_point=f"{__name__}.ur10_robotiq_pick_place_env:UR10RobotiqPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_robotiq_pick_place_env:UR10RobotiqPickPlaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Run:
# ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0 --num_envs 64 --headless
gym.register(
    id="Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0",
    entry_point=f"{__name__}.ur10_robotiq_pick_place_env:UR10RobotiqPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_robotiq_2f85_pick_place_env_cfg:UR10Robotiq2F85PickPlaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Run:
# ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-UR10-Robotiq-2F85-Third-Person-Pick-Place-Direct-v0 --num_envs 64 --enable_cameras --headless
gym.register(
    id="Isaac-UR10-Robotiq-2F85-Third-Person-Pick-Place-Direct-v0",
    entry_point=(
        f"{__name__}.ur10_robotiq_2f85_third_person_pick_place_env:"
        "UR10Robotiq2F85ThirdPersonPickPlaceEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ur10_robotiq_2f85_third_person_pick_place_env:"
            "UR10Robotiq2F85ThirdPersonPickPlaceEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Run:
# ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-UR10-Robotiq-2F85-Grasp-Direct-v0 --num_envs 64 --headless
gym.register(
    id="Isaac-UR10-Robotiq-2F85-Grasp-Direct-v0",
    entry_point=f"{__name__}.ur10_robotiq_2f85_grasp_env:UR10Robotiq2F85GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_robotiq_2f85_grasp_env_cfg:UR10Robotiq2F85GraspEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
