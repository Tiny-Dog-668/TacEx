# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .ur10_robotiq_2f85_pick_place_env_cfg import UR10Robotiq2F85PickPlaceEnvCfg


@configclass
class UR10Robotiq2F85GraspEnvCfg(UR10Robotiq2F85PickPlaceEnvCfg):
    bottle_height = 0.10
    bottle_radius = 0.025
    object_height = bottle_height

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.09)),
        spawn=sim_utils.CylinderCfg(
            radius=bottle_radius,
            height=bottle_height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                rest_offset=0.0005,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.95, 0.94),
                metallic=0.0,
                roughness=0.88,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    reach_sigma = 0.10
    reward_reach_weight = 5.0
    reward_lift_weight = 15.0
    reward_success_weight = 100.0
    lift_upright_tilt_threshold_deg = 5
    lift_reward_start_center_height = 0.095
    success_center_height = 0.130
    success_hold_steps = 5
    fall_height = 0.0
