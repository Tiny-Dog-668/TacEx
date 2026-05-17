"""
Cylinder Grasping Environments:
Goal is to grasp and lift a cylinder using tactile sensors.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

from .cylinder_grasping_four_tactile_rgb import (
    CylinderGraspingFourTactileRGBCfg,
    CylinderGraspingFourTactileRGBEnv,
)

from .cylinder_grasping_privileged import (
    CylinderGraspingPrivilegedCfg,
    CylinderGraspingPrivilegedEnv,
)

from .cylinder_grasping_vision_only_resnet18 import (
    CylinderGraspingVisionOnlyCfg as CylinderGraspingVisionOnlyResNet18Cfg,
    CylinderGraspingVisionOnlyEnv as CylinderGraspingVisionOnlyResNet18Env,
)

from .cylinder_grasping_vision_tactile import (
    CylinderGraspingVisionOnlyCfg as CylinderGraspingVisionTactileCfg,
    CylinderGraspingVisionOnlyEnv as CylinderGraspingVisionTactileEnv,
)

from .cylinder_grasping_one_vision_two_tactile import (
    CylinderGraspingVisionOnlyCfg as CylinderGraspingOneVisionTwoTactileCfg,
    CylinderGraspingVisionOnlyEnv as CylinderGraspingOneVisionTwoTactileEnv,
)

from .cylinder_grasping_two_vision_tactile import (
    CylinderGraspingVisionOnlyCfg as CylinderGraspingTwoVisionTactileCfg,
    CylinderGraspingVisionOnlyEnv as CylinderGraspingTwoVisionTactileEnv,
)

from .cylinder_grasping_four_vision_tactile import (
    CylinderGraspingFourVisionTactileCfg,
    CylinderGraspingFourVisionTactileEnv,
)

from .cylinder_grasping_four_vision_tactile_depth import (
    CylinderGraspingFourVisionTactileCfg as CylinderGraspingFourVisionTactileDepthCfg,
    CylinderGraspingFourVisionTactileEnv as CylinderGraspingFourVisionTactileDepthEnv,
)

# two-vision (wrist + third) with inner tactile depth (no outer sensors)
from .cylinder_grasping_two_vision_tactile_depth import (
    CylinderGraspingFourVisionTactileCfg as CylinderGraspingTwoVisionTactileDepthCfg,
    CylinderGraspingFourVisionTactileEnv as CylinderGraspingTwoVisionTactileDepthEnv,
)

# wrist camera + four tactile sensors (no third-person camera, no outer pads)
from .cylinder_grasping_four_tactile_one_vision import (
    CylinderGraspingFourTactileOneVisionCfg,
    CylinderGraspingFourTactileOneVisionEnv,
)

# four tactile depth sensors only (no vision cameras)
from .cylinder_grasping_tactile_depth import (
    CylinderGraspingFourTactileDepthCfg,
    CylinderGraspingFourTactileDepthEnv,
)


# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Four-Tactile-RGB-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-Four-Tactile-RGB-v0",
    entry_point=f"{__name__}.cylinder_grasping_four_tactile_rgb:CylinderGraspingFourTactileRGBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingFourTactileRGBCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_four_tactile_rgb_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Privileged-v0 --num_envs 64
gym.register(
    id="TacEx-Cylinder-Grasping-Privileged-v0",
    entry_point=f"{__name__}.cylinder_grasping_privileged:CylinderGraspingPrivilegedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingPrivilegedCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_privileged_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Vision-Only-ResNet18-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-Vision-Only-ResNet18-v0",
    entry_point=f"{__name__}.cylinder_grasping_vision_only_resnet18:CylinderGraspingVisionOnlyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingVisionOnlyResNet18Cfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_only_cfg_resnet18.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Vision-Tactile-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-Vision-Tactile-v0",
    entry_point=f"{__name__}.cylinder_grasping_vision_tactile:CylinderGraspingVisionOnlyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingVisionTactileCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_tactile_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-One-Vision-Two-Tactile-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-One-Vision-Two-Tactile-v0",
    entry_point=f"{__name__}.cylinder_grasping_one_vision_two_tactile:CylinderGraspingVisionOnlyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingOneVisionTwoTactileCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_one_vision_two_tactile.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Two-Vision-Tactile-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-Two-Vision-Tactile-v0",
    entry_point=f"{__name__}.cylinder_grasping_two_vision_tactile:CylinderGraspingVisionOnlyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingTwoVisionTactileCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_two_vision_tactile_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Four-Vision-Tactile-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-Four-Vision-Tactile-v0",
    entry_point=f"{__name__}.cylinder_grasping_four_vision_tactile:CylinderGraspingFourVisionTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingFourVisionTactileCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_four_vision_tactile_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Four-Vision-Tactile-Depth-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-Four-Vision-Tactile-Depth-v0",
    entry_point=f"{__name__}.cylinder_grasping_four_vision_tactile_depth:CylinderGraspingFourVisionTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingFourVisionTactileDepthCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_four_vision_tactile_depth_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Two-Vision-Tactile-Depth-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-Two-Vision-Tactile-Depth-v0",
    entry_point=f"{__name__}.cylinder_grasping_two_vision_tactile_depth:CylinderGraspingFourVisionTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingTwoVisionTactileDepthCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_two_vision_tactile_depth_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Four-Tactile-One-Vision-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-Four-Tactile-One-Vision-v0",
    entry_point=f"{__name__}.cylinder_grasping_four_tactile_one_vision:CylinderGraspingFourTactileOneVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingFourTactileOneVisionCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_four_tactile_one_vision.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Four-Tactile-Depth-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cylinder-Grasping-Four-Tactile-Depth-v0",
    entry_point=f"{__name__}.cylinder_grasping_tactile_depth:CylinderGraspingFourTactileDepthEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingFourTactileDepthCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_tactile_depth_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# Complex env with extra objects and visual randomization
