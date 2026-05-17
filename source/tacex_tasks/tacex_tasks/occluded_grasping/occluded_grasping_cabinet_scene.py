"""Shared cabinet-scene constants for occluded grasping tasks."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

CAN_RADIUS = 0.03
CAN_HEIGHT = 0.07
CAN_COLLISION_REST_OFFSET = 0.0005

CABINET_CENTER_X = 0.65
CABINET_CENTER_Y = 0.0
CABINET_WIDTH_Y = 0.6
CABINET_DEPTH_X = 0.3
CABINET_HEIGHT_Z = 0.6

ROBOT_SUPPORT_CENTER_X = 0.0
ROBOT_SUPPORT_CENTER_Y = 0.0
ROBOT_SUPPORT_WIDTH_X = 0.32
ROBOT_SUPPORT_WIDTH_Y = 0.34
ROBOT_SUPPORT_HEIGHT = 0.15
ROBOT_BASE_Z = ROBOT_SUPPORT_HEIGHT

CABINET_SUPPORT_WIDTH_X = CABINET_DEPTH_X + 0.06
CABINET_SUPPORT_WIDTH_Y = CABINET_WIDTH_Y + 0.06
CABINET_SUPPORT_HEIGHT = 0.25
CABINET_BASE_Z = CABINET_SUPPORT_HEIGHT

CABINET_PANEL_THICKNESS = 0.015
CABINET_BACK_THICKNESS = 0.012
CABINET_INTERNAL_SHELF_Z_LOCAL = 0.3

CABINET_BACK_X = CABINET_CENTER_X + CABINET_DEPTH_X * 0.5
CABINET_FRONT_X = CABINET_CENTER_X - CABINET_DEPTH_X * 0.5
CABINET_CENTER_Z = CABINET_BASE_Z + CABINET_HEIGHT_Z * 0.5
CABINET_INTERNAL_SHELF_Z = CABINET_BASE_Z + CABINET_INTERNAL_SHELF_Z_LOCAL
CABINET_SHELF_TOP_Z = CABINET_INTERNAL_SHELF_Z + CABINET_PANEL_THICKNESS * 0.5

TARGET_OBJECT_X = CABINET_CENTER_X + 0.065
TARGET_OBJECT_Y = -0.06
CABINET_OBJECT_Z = CABINET_SHELF_TOP_Z + 0.5 * CAN_HEIGHT

THIRD_PERSON_CAMERA_POSE = (
    (-1.2, -0.55, 0.30),
    (0.561, 0.561, -0.43, -0.431),
)

FRANKA_HOME_JOINTS = {
    "panda_joint1": -0.0520,
    "panda_joint2": -0.8608,
    "panda_joint3": -0.0263,
    "panda_joint4": -2.9234,
    "panda_joint5": 0.3194,
    "panda_joint6": 3.0801,
    "panda_joint7": 0.4257,
    "panda_finger_joint1": 0.02,
    "panda_finger_joint2": 0.02,
}

CABINET_RESET_X_HALF_RANGE = 0.05
CABINET_RESET_Y_HALF_RANGE = 0.10
CABINET_RESET_MARGIN_X = 0.035
CABINET_RESET_MARGIN_Y = 0.05


def make_fixed_panel_cfg(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> sim_utils.CuboidCfg:
    """Create a kinematic wooden panel used for cabinet/support geometry."""
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            max_angular_velocity=100.0,
            max_linear_velocity=10.0,
            max_depenetration_velocity=1.0,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0005,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=0.65,
            metallic=0.0,
        ),
    )
