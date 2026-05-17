"""UR5 robot configurations."""

from .ur5_rg2 import UR5_RG2_CFG, UR5_RG2_URDF_PATH, make_ur5_rg2_cfg
from .ur5_robotiq import (
    UR5_ROBOTIQ_CFG,
    UR5_ROBOTIQ_GELSIGHT_URDF_PATH,
    UR5_ROBOTIQ_URDF_PATH,
    make_ur5_robotiq_cfg,
)
from .ur5e_gripper_usd import UR5E_GRIPPER_USD_CFG, UR5E_GRIPPER_USD_PATH

__all__ = [
    "UR5_RG2_CFG",
    "UR5_RG2_URDF_PATH",
    "UR5_ROBOTIQ_CFG",
    "UR5_ROBOTIQ_GELSIGHT_URDF_PATH",
    "UR5_ROBOTIQ_URDF_PATH",
    "UR5E_GRIPPER_USD_CFG",
    "UR5E_GRIPPER_USD_PATH",
    "make_ur5_rg2_cfg",
    "make_ur5_robotiq_cfg",
]
