# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for UR5 robots with Robotiq grippers used in TacEx."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

UR5_VISUALTACTILE_ASSET_DIR = Path("/home/tinydog/Projects/VisualTactile/assets")

# Plain UR5 + Robotiq 2F-140 without DIGIT/GelSight connectors.
UR5_ROBOTIQ_URDF_PATH = str(UR5_VISUALTACTILE_ASSET_DIR / "ur5RQ.urdf")
UR5_ROBOTIQ_GELSIGHT_URDF_PATH = str(UR5_VISUALTACTILE_ASSET_DIR / "ur5RQGS.urdf")


def make_ur5_robotiq_cfg(asset_path: str = UR5_ROBOTIQ_URDF_PATH) -> ArticulationCfg:
    """Create a reusable UR5 + Robotiq articulation config."""

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=asset_path,
            fix_base=True,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            force_usd_conversion=False,
            make_instanceable=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
            ),
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
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                "finger_joint": 0.0,
            },
        ),
        actuators={
            "arm_main": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                effort_limit_sim=150.0,
                velocity_limit_sim=3.2,
                stiffness=800.0,
                damping=40.0,
            ),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_[1-3]_joint"],
                effort_limit_sim=28.0,
                velocity_limit_sim=3.2,
                stiffness=400.0,
                damping=20.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=1000.0,
                velocity_limit_sim=2.0,
                stiffness=40.0,
                damping=4.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


UR5_ROBOTIQ_CFG = make_ur5_robotiq_cfg()
"""Configuration of a UR5 arm with a Robotiq gripper imported from a local URDF."""
