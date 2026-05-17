"""
Sparch grasping environments.
"""

import gymnasium as gym

from . import agents
from .vision_two_tactile_ijepa_env import (
    SparchGraspVisionTwoTactileIJEPAEnv,
    SparchGraspVisionTwoTactileIJEPACfg,
)


# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Sparch-Grasp-Vision-Two-Tactile-IJEPA-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Sparch-Grasp-Vision-Two-Tactile-IJEPA-v0",
    entry_point=f"{__name__}.vision_two_tactile_ijepa_env:SparchGraspVisionTwoTactileIJEPAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SparchGraspVisionTwoTactileIJEPACfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_two_tactile_sparsh_ijepa_cfg.yaml",
    },
)
