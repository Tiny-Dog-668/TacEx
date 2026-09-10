# Copyright (c) 2022-2023, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Modified version of the original FRANKA_PANDA_CFG of Isaac Lab
#
"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG`: Franka Emika Panda robot with Panda hand

Reference: https://github.com/frankaemika/franka_ros
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR


GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_PROFILE = (
    "gelsight_physics_isaaclab_panda_arm_link0_7_visuals_green_base_led_"
    "finger_offset_21mm_usd_v7"
)
GELSIGHT_STANDARD_FRANKA_BASE_LED_SUBSET_PATH = (
    "panda_link0/standard_visuals/panda_link0/subset_5"
)
GELSIGHT_STANDARD_FRANKA_BASE_LED_COLOR_RGB = (0.0, 1.0, 0.0)
GELSIGHT_LEGACY_STANDARD_FRANKA_ARM_VISUAL_USD = str(
    Path(TACEX_ASSETS_DATA_DIR)
    / "Robots/Franka/GelSight_Mini/Gripper/franka_gsmini_standard_arm_visuals_v5.usd"
)
GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_USD = str(
    Path(TACEX_ASSETS_DATA_DIR)
    / "Robots/Franka/GelSight_Mini/Gripper/franka_gsmini_standard_arm_visuals_v7.usd"
)
GELSIGHT_FINGER_EXTENSION_M = 0.021
# Backward-compatible names for the Pulled-Drawer profile. All RMA GelSight
# tasks use the same shifted v7 asset and physical finger geometry.
GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M = GELSIGHT_FINGER_EXTENSION_M
GELSIGHT_PULLED_DRAWER_FRANKA_ASSET_PROFILE = GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_PROFILE
GELSIGHT_PULLED_DRAWER_FRANKA_USD = GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_USD


def create_gelsight_standard_franka_arm_visual_usd() -> str:
    """Return the persistent shifted-finger GelSight/standard-Panda USD asset."""
    output = Path(GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_USD)
    if not output.is_file():
        raise FileNotFoundError(f"Persistent GelSight Franka USD is missing: {output}")
    return str(output)

# todo find a good way to save the prim path of the sensor for the user?
# -> currently, we need to look into the asset to figure out the prim name (in this case its /gelsight_mini_case)
FRANKA_PANDA_ARM_GSMINI_GRIPPER_RIGID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.down.usda",
        usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=40.0,
            velocity_limit_sim=0.2,
            stiffness=400,
            damping=40,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot with a Gripper and two GelSight Mini sensors.

The gelpads are simulated via PhysX and rigid.

Sensor case prim names:
- `gelsight_mini_case_left`
- `gelsight_mini_case_right`

Gelpad prim names:
- `gelpad_left`
- `gelpad_right`
"""


FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG = FRANKA_PANDA_ARM_GSMINI_GRIPPER_RIGID_CFG.copy()
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.

Sensor case prim names:
- `gelsight_mini_case_left`
- `gelsight_mini_case_right`

Gelpad prim names:
- `gelpad_left`
- `gelpad_right`
"""
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.actuators["panda_forearm"].damping = 80.0
