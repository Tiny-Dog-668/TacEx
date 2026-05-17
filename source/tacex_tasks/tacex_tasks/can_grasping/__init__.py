"""
Can Grasping Environments:
Goal is to grasp and lift a can using a wrist camera.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

from .can_grasping_vision_only_resnet18 import (
    CanGraspingVisionOnlyCfg as CanGraspingVisionOnlyResNet18Cfg,
    CanGraspingVisionOnlyEnv as CanGraspingVisionOnlyResNet18Env,
)
from .can_grasping_vision_six_tactile import (
    CanGraspingVisionSixTactileCfg,
    CanGraspingVisionSixTactileEnv,
)
from .can_grasping_vision_two_tactile import (
    CanGraspingVisionTwoTactileCfg,
    CanGraspingVisionTwoTactileEnv,
)
from .can_grasping_vision_four_tactile import (
    CanGraspingVisionFourTactileCfg,
    CanGraspingVisionFourTactileEnv,
)
from .can_grasping_vision_two_tactile_gate import (
    CanGraspingVisionTwoTactileCfg as CanGraspingVisionTwoTactileGateCfg,
    CanGraspingVisionTwoTactileEnv as CanGraspingVisionTwoTactileGateEnv,
)
from .can_grasping_vision_two_tactile_latent import (
    CanGraspingVisionTwoTactileCfg as CanGraspingVisionTwoTactileLatentCfg,
    CanGraspingVisionTwoTactileEnv as CanGraspingVisionTwoTactileLatentEnv,
)
from .can_grasping_four_tactile_rgb import (
    CanGraspingSixTactileRGBCfg,
    CanGraspingSixTactileRGBEnv,
)
from .can_grasping_two_tactile_two_force import (
    CanGraspingSixTactileRGBCfg as CanGraspingTwoTactileTwoForceCfg,
    CanGraspingSixTactileRGBEnv as CanGraspingTwoTactileTwoForceEnv,
)
from .can_grasping_six_tactile_rgb_explore import (
    CanGraspingSixTactileRGBCfg as CanGraspingSixTactileRGBExploreCfg,
    CanGraspingSixTactileRGBEnv as CanGraspingSixTactileRGBExploreEnv,
)


# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Vision-Only-ResNet18-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Vision-Only-ResNet18-v0",
    entry_point=f"{__name__}.can_grasping_vision_only_resnet18:CanGraspingVisionOnlyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingVisionOnlyResNet18Cfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_only_cfg_resnet18.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Privileged-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Privileged-v0",
    entry_point=f"{__name__}.can_grasping_vision_only_resnet18:CanGraspingVisionOnlyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingVisionOnlyResNet18Cfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_privileged_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Vision-Six-Tactile-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Vision-Six-Tactile-v0",
    entry_point=f"{__name__}.can_grasping_vision_six_tactile:CanGraspingVisionSixTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingVisionSixTactileCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_six_tactile_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Vision-Two-Tactile-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Vision-Two-Tactile-v0",
    entry_point=f"{__name__}.can_grasping_vision_two_tactile:CanGraspingVisionTwoTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingVisionTwoTactileCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_two_tactile_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Vision-Four-Tactile-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Vision-Four-Tactile-v0",
    entry_point=f"{__name__}.can_grasping_vision_four_tactile:CanGraspingVisionFourTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingVisionFourTactileCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_four_tactile_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Vision-Two-Tactile-Gate-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Vision-Two-Tactile-Gate-v0",
    entry_point=f"{__name__}.can_grasping_vision_two_tactile_gate:CanGraspingVisionTwoTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingVisionTwoTactileGateCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_two_tactile_gate_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Vision-Two-Tactile-Latent-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Vision-Two-Tactile-Latent-v0",
    entry_point=f"{__name__}.can_grasping_vision_two_tactile_latent:CanGraspingVisionTwoTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingVisionTwoTactileLatentCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_two_tactile_latent_align_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Six-Tactile-RGB-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Six-Tactile-RGB-v0",
    entry_point=f"{__name__}.can_grasping_four_tactile_rgb:CanGraspingSixTactileRGBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingSixTactileRGBCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_six_tactile_rgb_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Two-Tactile-Two-Force-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Two-Tactile-Two-Force-v0",
    entry_point=f"{__name__}.can_grasping_two_tactile_two_force:CanGraspingSixTactileRGBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingTwoTactileTwoForceCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_two_tactile_two_force_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Can-Grasping-Six-Tactile-RGB-Explore-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Can-Grasping-Six-Tactile-RGB-Explore-v0",
    entry_point=f"{__name__}.can_grasping_six_tactile_rgb_explore:CanGraspingSixTactileRGBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CanGraspingSixTactileRGBExploreCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_six_tactile_rgb_cfg_explore.yaml",
    },
)
