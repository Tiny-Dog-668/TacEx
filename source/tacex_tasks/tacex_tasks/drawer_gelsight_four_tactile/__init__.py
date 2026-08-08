"""D435-calibrated four-GelSight drawer grasping task."""

from __future__ import annotations

import gymnasium as gym

from . import agents
from .drawer_gelsight_four_tactile_d435_env import (
    DrawerGelSightFourTactileD435Cfg,
    DrawerGelSightFourTactileD435Env,
)


DRAWER_GELSIGHT_FOUR_TACTILE_D435_TASK = "TacEx-Drawer-GelSight-Four-Tactile-D435-v0"


gym.register(
    id=DRAWER_GELSIGHT_FOUR_TACTILE_D435_TASK,
    entry_point=(
        f"{__name__}.drawer_gelsight_four_tactile_d435_env:"
        "DrawerGelSightFourTactileD435Env"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DrawerGelSightFourTactileD435Cfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_d435.yaml",
    },
)
