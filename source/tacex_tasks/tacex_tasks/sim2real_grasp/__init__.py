"""Minimal sim-to-real Franka grasping task."""

import gymnasium as gym

from . import agents
from ..cylinder_grasping import agents as cylinder_agents
from .sim2real_grasp_env import Sim2RealGraspEnv, Sim2RealGraspEnvCfg

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Sim2Real-Grasp-v0 --num_envs 4 --enable_cameras
# isaaclab -p ./scripts/reinforcement_learning/skrl/play.py --task TacEx-Sim2Real-Grasp-v0 --num_envs 1 --enable_cameras
gym.register(
    id="TacEx-Sim2Real-Grasp-v0",
    entry_point=f"{__name__}.sim2real_grasp_env:Sim2RealGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealGraspEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_only_cfg_resnet18.yaml",
        "skrl_sac_cfg_entry_point": f"{cylinder_agents.__name__}:skrl_sac_cfg.yaml",
    },
)
