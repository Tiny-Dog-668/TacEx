# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .ur10_robotiq_pick_place_env import (
    GRIPPER_ACTUATOR_DAMPING,
    GRIPPER_ACTUATOR_EFFORT_LIMIT_SIM,
    GRIPPER_ACTUATOR_STIFFNESS,
    GRIPPER_ACTUATOR_VELOCITY_LIMIT_SIM,
    GRIPPER_LINKAGE_DAMPING,
    GRIPPER_LINKAGE_EFFORT_LIMIT_SIM,
    GRIPPER_LINKAGE_STIFFNESS,
    GRIPPER_LINKAGE_VELOCITY_LIMIT_SIM,
    UR10RobotiqPickPlaceEnvCfg,
)

ROBOTIQ_2F85_GRIPPER_COUPLING_RULES = (
    ("right_outer_knuckle_joint", 1.0, 0.0),
    ("right_inner_finger_joint", -1.0, 0.0),
    ("right_inner_finger_knuckle_joint", -1.0, 0.0),
    ("left_inner_finger_knuckle_joint", -1.0, 0.0),
    ("left_inner_finger_joint", -1.0, 0.0),
)
ROBOTIQ_2F85_GRIP_BODY_CANDIDATE_GROUPS = (
    ("left_inner_finger", "right_inner_finger"),
    ("left_outer_finger", "right_outer_finger"),
)
ROBOTIQ_2F85_GRIP_BODY_FALLBACKS = ("base_link", "ee_link", "wrist_3_link")
ROBOTIQ_2F85_GRIPPER_LINKAGE_JOINT_NAMES = (
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
)

_TACEX_ROOT = Path(__file__).resolve().parents[5]
_UR_ROBOTIQ_ASSET_DIR = _TACEX_ROOT / "source" / "tacex_assets" / "tacex_assets" / "data" / "Robots" / "URRobotiq"
DEFAULT_UR10_ROBOTIQ_2F85_USD_PATH = str(_UR_ROBOTIQ_ASSET_DIR / "ur10_robotiq_2f85.usda")
UR10_ROBOTIQ_2F85_USD_PATH = os.environ.get("UR10_ROBOTIQ_2F85_USD_PATH", DEFAULT_UR10_ROBOTIQ_2F85_USD_PATH)


@configclass
class UR10Robotiq2F85PickPlaceEnvCfg(UR10RobotiqPickPlaceEnvCfg):
    robot_usd_path = UR10_ROBOTIQ_2F85_USD_PATH
    gripper_coupling_rules = ROBOTIQ_2F85_GRIPPER_COUPLING_RULES
    grip_body_candidate_groups = ROBOTIQ_2F85_GRIP_BODY_CANDIDATE_GROUPS
    grip_body_fallbacks = ROBOTIQ_2F85_GRIP_BODY_FALLBACKS
    grip_center_world_offset = (0.0, 0.0, -0.1)
    gripper_linkage_joint_names = ROBOTIQ_2F85_GRIPPER_LINKAGE_JOINT_NAMES
    ee_action_noise_std = 0.05
    gripper_action_noise_std = 0.02
    close_target_deg = 35.0

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": -0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
                "finger_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm_main": IdealPDActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                effort_limit=150.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "arm_wrist": IdealPDActuatorCfg(
                joint_names_expr=["wrist_[1-3]_joint"],
                effort_limit=28.0,
                stiffness=400.0,
                damping=20.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=GRIPPER_ACTUATOR_EFFORT_LIMIT_SIM,
                velocity_limit_sim=GRIPPER_ACTUATOR_VELOCITY_LIMIT_SIM,
                stiffness=GRIPPER_ACTUATOR_STIFFNESS,
                damping=GRIPPER_ACTUATOR_DAMPING,
            ),
            "gripper_linkage": ImplicitActuatorCfg(
                joint_names_expr=list(gripper_linkage_joint_names),
                effort_limit_sim=GRIPPER_LINKAGE_EFFORT_LIMIT_SIM,
                velocity_limit_sim=GRIPPER_LINKAGE_VELOCITY_LIMIT_SIM,
                stiffness=GRIPPER_LINKAGE_STIFFNESS,
                damping=GRIPPER_LINKAGE_DAMPING,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
