"""
Cube Grasping Environments:
Goal is to grasp and lift a cube using a wrist camera.
"""

import gymnasium as gym

from . import agents

from .cube_grasping_vision_only import (
    CubeGraspingVisionOnlyCfg as CubeGraspingVisionOnlyResNet18Cfg,
    CubeGraspingVisionOnlyEnv as CubeGraspingVisionOnlyResNet18Env,
)
from .cube_grasping_vision_tactile_fusion import (
    CubeGraspingVisionTwoTactileCfg,
    CubeGraspingVisionTwoTactileEnv,
)
from .cube_grasping_vision_tactile_gate import (
    CubeGraspingVisionTwoTactileGateCfg,
    CubeGraspingVisionTwoTactileGateEnv,
)
from .cube_grasping_vision_degratation_tactile_fusion import (
    CubeGraspingVisionDegradationTactileFusionCfg,
    CubeGraspingVisionDegradationTactileFusionEnv,
)
from .cube_grasping_vision_tactile_transformer import (
    CubeGraspingVisionTwoTactileCfg as CubeGraspingVisionTwoTactileTransformerCfg,
    CubeGraspingVisionTwoTactileEnv as CubeGraspingVisionTwoTactileTransformerEnv,
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cube-Grasping-Vision-Only-ResNet18-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cube-Grasping-Vision-Only-ResNet18-v0",
    entry_point=f"{__name__}.cube_grasping_vision_only:CubeGraspingVisionOnlyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeGraspingVisionOnlyResNet18Cfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_only_cfg_resnet18.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cube-Grasping-Vision-Tactile-Fusion-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cube-Grasping-Vision-Tactile-Fusion-v0",
    entry_point=f"{__name__}.cube_grasping_vision_tactile_fusion:CubeGraspingVisionTwoTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeGraspingVisionTwoTactileCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_tactile_fusion_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cube-Grasping-Vision-Tactile-Gate-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cube-Grasping-Vision-Tactile-Gate-v0",
    entry_point=f"{__name__}.cube_grasping_vision_tactile_gate:CubeGraspingVisionTwoTactileGateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeGraspingVisionTwoTactileGateCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_tactile_gate_cfg.yaml",
    },
)


# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cube-Grasping-Vision-Degradation-Tactile-Fusion-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cube-Grasping-Vision-Degradation-Tactile-Fusion-v0",
    entry_point=f"{__name__}.cube_grasping_vision_degratation_tactile_fusion:CubeGraspingVisionDegradationTactileFusionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeGraspingVisionDegradationTactileFusionCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_degradation_tactile_fusion_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Cube-Grasping-Vision-Tactile-Transformer-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Cube-Grasping-Vision-Tactile-Transformer-v0",
    entry_point=f"{__name__}.cube_grasping_vision_tactile_transformer:CubeGraspingVisionTwoTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeGraspingVisionTwoTactileTransformerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_tactile_transformer_cfg.yaml",
    },
)
