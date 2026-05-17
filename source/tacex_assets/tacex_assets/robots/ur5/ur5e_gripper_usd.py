# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a downloaded UR5e + gripper USD asset."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

UR5E_GRIPPER_USD_PATH = "/home/tinydog/Projects/TacEx/ur5e_gripper/ur5e_gripper.usd"


UR5E_GRIPPER_USD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR5E_GRIPPER_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for the downloaded UR5e + gripper USD asset."""
