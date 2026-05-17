import gymnasium as gym

from .cylinder_grasping_complex_env import (
    CylinderGraspingComplexEnv,
    CylinderGraspingComplexEnvCfg,
)

__all__ = [
    "CylinderGraspingComplexEnv",
    "CylinderGraspingComplexEnvCfg",
]
# python  scripts/reinforcement_learning/skrl/train.py --task TacEx-Cylinder-Grasping-Complex-v0 --num_envs 64 --enable_cameras --headless --video
# Register Gym environment for complex grasping (extra objects + visual randomization)
gym.register(
    id="TacEx-Cylinder-Grasping-Complex-v0",
    entry_point="tacex_tasks.grasping_complex_env.cylinder_grasping_complex_env:CylinderGraspingComplexEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CylinderGraspingComplexEnvCfg,
        "skrl_cfg_entry_point": "tacex_tasks.grasping_complex_env.agents:skrl_ppo_complex_env.yaml",
        "skrl_sac_cfg_entry_point": "tacex_tasks.cylinder_grasping.agents:skrl_sac_cfg.yaml",
    },
)
