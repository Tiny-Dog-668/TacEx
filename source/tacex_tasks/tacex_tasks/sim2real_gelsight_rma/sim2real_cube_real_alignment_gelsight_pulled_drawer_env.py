"""Pulled-drawer Real-Alignment tasks for the dual-GelSight Franka."""

from __future__ import annotations

import colorsys
from collections.abc import Mapping

import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

import isaaclab.sim as sim_utils
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_PULLED_DRAWER_FRANKA_ASSET_PROFILE,
    GELSIGHT_PULLED_DRAWER_FRANKA_USD,
    GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M,
)

from .gelsight_geometry import (
    GELSIGHT_HAND_TO_FINGERTIP_BOTTOM_M,
    GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M,
)
from .sim2real_cube_real_alignment_gelsight_size_buckets_env import (
    GELSIGHT_SIZE_BUCKETS_M,
    _CUBE_ILLEGAL_ROBOT_BODY_NAMES,
    _TABLE_ILLEGAL_ROBOT_BODY_NAMES,
)
from .sim2real_cube_real_alignment_gelsight_x040_three_frame_env import (
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg,
)


GELSIGHT_PULLED_DRAWER_TEACHER_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-Teacher-v0"
)
GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-"
    "Three-Frame-Direct-Action-Student-DR-v0"
)

PULLED_DRAWER_GEOMETRY_VERSION = 9
PULLED_DRAWER_PANDA_FINGER_JOINT_LOCAL_POS0_Z_M = (
    0.0584 + GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M
)
PULLED_DRAWER_GELSIGHT_CENTER_OFFSET_HAND_M = (
    GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M
)
PULLED_DRAWER_GELSIGHT_BOTTOM_OFFSET_HAND_M = (
    GELSIGHT_HAND_TO_FINGERTIP_BOTTOM_M
)
PULLED_DRAWER_CABINET_NOMINAL_SIZE_M = (0.30, 0.25, 0.135)
PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M = (0.79, -0.01)
PULLED_DRAWER_TRAY_NOMINAL_SIZE_M = (0.225, 0.225, 0.115)
PULLED_DRAWER_TRAY_REFERENCE_CENTER_XY_M = (0.49, 0.0)
# A zero seam makes the nominal tray center x=0.5275 m. Sampled layouts derive
# tray X from the two sampled depths and cabinet front face in the same way.
PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M = (
    PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M[0]
    - 0.5 * PULLED_DRAWER_CABINET_NOMINAL_SIZE_M[0]
    - 0.5 * PULLED_DRAWER_TRAY_NOMINAL_SIZE_M[0],
    PULLED_DRAWER_TRAY_REFERENCE_CENTER_XY_M[1],
)
PULLED_DRAWER_PANEL_THICKNESS_M = 0.008
PULLED_DRAWER_PANEL_CONTACT_OFFSET_M = 0.001
PULLED_DRAWER_PANEL_REST_OFFSET_M = 0.0005
PULLED_DRAWER_PANEL_SOLVER_POSITION_ITERATION_COUNT = 32
PULLED_DRAWER_PANEL_SOLVER_VELOCITY_ITERATION_COUNT = 4
PULLED_DRAWER_ROBOT_SOLVER_POSITION_ITERATION_COUNT = 32
PULLED_DRAWER_ROBOT_SOLVER_VELOCITY_ITERATION_COUNT = 4
PULLED_DRAWER_ILLEGAL_COLLISION_CURRICULUM_START_STEP = 0
PULLED_DRAWER_ILLEGAL_COLLISION_CURRICULUM_END_STEP = 100_000
PULLED_DRAWER_ILLEGAL_COLLISION_PENALTY_THRESHOLD_START_N = 20.0
PULLED_DRAWER_ILLEGAL_COLLISION_PENALTY_THRESHOLD_END_N = 5.0
PULLED_DRAWER_ILLEGAL_COLLISION_TERMINATION_THRESHOLD_START_N = 200.0
PULLED_DRAWER_ILLEGAL_COLLISION_TERMINATION_THRESHOLD_END_N = 20.0
PULLED_DRAWER_ILLEGAL_COLLISION_PENALTY = -10.0
PULLED_DRAWER_SIZE_SCALE_RANGE = (0.90, 1.10)
PULLED_DRAWER_POSITION_DELTA_XY_M = (0.02, 0.02)
PULLED_DRAWER_SEAM_GAP_M = 0.0
PULLED_DRAWER_MIN_SIDE_CLEARANCE_M = 0.001
PULLED_DRAWER_CUBE_RESET_HALF_RANGE_XY_M = (0.03, 0.03)
PULLED_DRAWER_TRAY_OPACITY = 1.0
PULLED_DRAWER_TRAY_NOMINAL_COLOR_RGB = (0.34, 0.40, 0.43)
PULLED_DRAWER_TRAY_ROUGHNESS = 0.18
PULLED_DRAWER_TRAY_HUE_RANGE = (0.52, 0.62)
PULLED_DRAWER_TRAY_SATURATION_RANGE = (0.08, 0.20)
PULLED_DRAWER_TRAY_VALUE_RANGE = (0.30, 0.55)

_CABINET_PANEL_NAMES = (
    "pulled_drawer_cabinet_left",
    "pulled_drawer_cabinet_right",
    "pulled_drawer_cabinet_back",
    "pulled_drawer_cabinet_top",
)
_TRAY_PANEL_NAMES = (
    "pulled_drawer_tray_floor",
    "pulled_drawer_tray_front",
    "pulled_drawer_tray_back",
    "pulled_drawer_tray_left",
    "pulled_drawer_tray_right",
)


def _kinematic_cuboid_cfg(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    opacity: float = 1.0,
    roughness: float = 0.72,
) -> sim_utils.CuboidCfg:
    """Create one static contact-reporting cuboid."""
    return sim_utils.CuboidCfg(
        size=size,
        activate_contact_sensors=True,
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=(
                PULLED_DRAWER_PANEL_SOLVER_POSITION_ITERATION_COUNT
            ),
            solver_velocity_iteration_count=(
                PULLED_DRAWER_PANEL_SOLVER_VELOCITY_ITERATION_COUNT
            ),
            max_depenetration_velocity=1.0,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=PULLED_DRAWER_PANEL_CONTACT_OFFSET_M,
            rest_offset=PULLED_DRAWER_PANEL_REST_OFFSET_M,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            opacity=opacity,
            roughness=roughness,
            metallic=0.0,
        ),
    )


def _cabinet_nominal_panels() -> dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
]:
    depth, width, height = PULLED_DRAWER_CABINET_NOMINAL_SIZE_M
    center_x, center_y = PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M
    thickness = PULLED_DRAWER_PANEL_THICKNESS_M
    inner_width = width - 2.0 * thickness
    return {
        "pulled_drawer_cabinet_left": (
            (depth, thickness, height),
            (center_x, center_y - 0.5 * width + 0.5 * thickness, 0.5 * height),
        ),
        "pulled_drawer_cabinet_right": (
            (depth, thickness, height),
            (center_x, center_y + 0.5 * width - 0.5 * thickness, 0.5 * height),
        ),
        "pulled_drawer_cabinet_back": (
            (thickness, inner_width, height),
            (center_x + 0.5 * depth - 0.5 * thickness, center_y, 0.5 * height),
        ),
        "pulled_drawer_cabinet_top": (
            (depth, width, thickness),
            (center_x, center_y, height - 0.5 * thickness),
        ),
    }


def _tray_nominal_panels() -> dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
]:
    depth, width, height = PULLED_DRAWER_TRAY_NOMINAL_SIZE_M
    center_x, center_y = PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M
    thickness = PULLED_DRAWER_PANEL_THICKNESS_M
    wall_height = height - thickness
    wall_center_z = thickness + 0.5 * wall_height
    return {
        "pulled_drawer_tray_floor": (
            (depth, width, thickness),
            (center_x, center_y, 0.5 * thickness),
        ),
        "pulled_drawer_tray_front": (
            (thickness, width, wall_height),
            (center_x - 0.5 * depth + 0.5 * thickness, center_y, wall_center_z),
        ),
        "pulled_drawer_tray_back": (
            (thickness, width, wall_height),
            (center_x + 0.5 * depth - 0.5 * thickness, center_y, wall_center_z),
        ),
        "pulled_drawer_tray_left": (
            (depth - 2.0 * thickness, thickness, wall_height),
            (center_x, center_y - 0.5 * width + 0.5 * thickness, wall_center_z),
        ),
        "pulled_drawer_tray_right": (
            (depth - 2.0 * thickness, thickness, wall_height),
            (center_x, center_y + 0.5 * width - 0.5 * thickness, wall_center_z),
        ),
    }


def sample_pulled_drawer_layouts(num_envs: int, seed: int) -> dict[str, torch.Tensor]:
    """Sample deterministic per-env geometry; tensors are CPU float64 [N,*]."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    result = {
        "shared_size_scale": torch.empty((num_envs,), dtype=torch.float64),
        "cabinet_size_m": torch.empty((num_envs, 3), dtype=torch.float64),
        "cabinet_center_xy_m": torch.empty((num_envs, 2), dtype=torch.float64),
        "tray_size_m": torch.empty((num_envs, 3), dtype=torch.float64),
        "tray_center_xy_m": torch.empty((num_envs, 2), dtype=torch.float64),
        "gap_m": torch.empty((num_envs,), dtype=torch.float64),
        "side_clearance_m": torch.empty((num_envs,), dtype=torch.float64),
    }
    cabinet_size0 = torch.tensor(PULLED_DRAWER_CABINET_NOMINAL_SIZE_M, dtype=torch.float64)
    tray_size0 = torch.tensor(PULLED_DRAWER_TRAY_NOMINAL_SIZE_M, dtype=torch.float64)
    cabinet_xy0 = torch.tensor(PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M, dtype=torch.float64)
    tray_xy0 = torch.tensor(PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M, dtype=torch.float64)
    position_delta = torch.tensor(PULLED_DRAWER_POSITION_DELTA_XY_M, dtype=torch.float64)
    scale_low, scale_high = PULLED_DRAWER_SIZE_SCALE_RANGE

    for env_id in range(num_envs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 104_729 * env_id + 17_171)
        for _ in range(1024):
            # One isotropic scale is shared by the cabinet and tray so their
            # relative proportions cannot drift independently across axes.
            shared_scale = scale_low + (scale_high - scale_low) * torch.rand(
                1, generator=generator, dtype=torch.float64
            )
            cabinet_size = cabinet_size0 * shared_scale
            tray_size = tray_size0 * shared_scale
            cabinet_xy = cabinet_xy0 + (
                2.0 * torch.rand(2, generator=generator, dtype=torch.float64) - 1.0
            ) * position_delta
            tray_y = tray_xy0[1] + (
                2.0 * torch.rand(1, generator=generator, dtype=torch.float64)[0] - 1.0
            ) * position_delta[1]
            # The tray must fit laterally inside the cabinet opening. This is
            # the per-side free space after accounting for the sampled Y shift.
            side_clearance = (
                0.5 * (cabinet_size[1] - 2.0 * PULLED_DRAWER_PANEL_THICKNESS_M)
                - 0.5 * tray_size[1]
                - torch.abs(tray_y - cabinet_xy[1])
            )
            if float(side_clearance) >= PULLED_DRAWER_MIN_SIDE_CLEARANCE_M:
                # Join the two physical bodies at one exact X seam:
                # cabinet front face == tray back face.
                cabinet_front_x = cabinet_xy[0] - 0.5 * cabinet_size[0]
                tray_x = cabinet_front_x - 0.5 * tray_size[0]
                tray_xy = torch.stack((tray_x, tray_y))
                gap = cabinet_front_x - (tray_x + 0.5 * tray_size[0])
                result["shared_size_scale"][env_id] = shared_scale[0]
                result["cabinet_size_m"][env_id] = cabinet_size
                result["cabinet_center_xy_m"][env_id] = cabinet_xy
                result["tray_size_m"][env_id] = tray_size
                result["tray_center_xy_m"][env_id] = tray_xy
                result["gap_m"][env_id] = gap
                result["side_clearance_m"][env_id] = side_clearance
                break
        else:
            raise RuntimeError(f"Unable to sample a contiguous fitted layout for env {env_id}")
    return result


def balanced_cube_bucket_ids(num_envs: int, seed: int) -> torch.Tensor:
    """Return seeded balanced ids [N]; every consecutive group contains 0..7."""
    bucket_count = len(GELSIGHT_SIZE_BUCKETS_M)
    if num_envs <= 0 or num_envs % bucket_count != 0:
        raise ValueError("Pulled-Drawer num_envs must be a positive multiple of 8")
    result = torch.empty((num_envs,), dtype=torch.long)
    for group_start in range(0, num_envs, bucket_count):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 65_537 * (group_start // bucket_count) + 29_791)
        result[group_start : group_start + bucket_count] = torch.randperm(
            bucket_count, generator=generator
        )
    return result


def pulled_drawer_geometry_contract() -> dict[str, object]:
    return {
        "version": PULLED_DRAWER_GEOMETRY_VERSION,
        "coordinate_frame": "robot_root_aligned_world_m",
        "robot_end_effector": {
            "asset_profile": GELSIGHT_PULLED_DRAWER_FRANKA_ASSET_PROFILE,
            "asset_filename": GELSIGHT_PULLED_DRAWER_FRANKA_USD.rsplit("/", 1)[-1],
            "panda_link8_to_hand_fixed_joint_local_pos0_m": [
                0.0,
                0.0,
                0.0,
            ],
            "panda_finger_joint_local_pos0_m": [
                0.0,
                0.0,
                PULLED_DRAWER_PANDA_FINGER_JOINT_LOCAL_POS0_Z_M,
            ],
            "finger_and_gelsight_extension_from_base_asset_m": (
                GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M
            ),
            "panda_hand_to_gelpad_midpoint_m": (
                PULLED_DRAWER_GELSIGHT_CENTER_OFFSET_HAND_M
            ),
            "panda_hand_to_fingertip_bottom_m": (
                PULLED_DRAWER_GELSIGHT_BOTTOM_OFFSET_HAND_M
            ),
            "direction": "panda_hand_local_positive_z",
        },
        "cabinet": {
            "shape": "open_minus_x_four_panel_shell_without_bottom",
            "nominal_outer_size_m": list(PULLED_DRAWER_CABINET_NOMINAL_SIZE_M),
            "nominal_center_xy_m": list(PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M),
            "bottom_z_m": 0.0,
            "panel_thickness_m": PULLED_DRAWER_PANEL_THICKNESS_M,
        },
        "tray": {
            "shape": "open_top_five_panel_pulled_drawer",
            "nominal_outer_size_m": list(PULLED_DRAWER_TRAY_NOMINAL_SIZE_M),
            "nominal_center_xy_m": list(PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M),
            "measured_reference_center_xy_m": list(
                PULLED_DRAWER_TRAY_REFERENCE_CENTER_XY_M
            ),
            "bottom_z_m": 0.0,
            "panel_thickness_m": PULLED_DRAWER_PANEL_THICKNESS_M,
            "floor_top_z_m": PULLED_DRAWER_PANEL_THICKNESS_M,
        },
        "fixed_per_environment": True,
        "sampling": {
            "shared_isotropic_size_scale_uniform": list(PULLED_DRAWER_SIZE_SCALE_RANGE),
            "size_scale_coupling": "same_scalar_for_cabinet_and_tray_xyz",
            "cabinet_position_delta_xy_uniform_m": list(
                PULLED_DRAWER_POSITION_DELTA_XY_M
            ),
            "tray_position_delta_y_uniform_m": PULLED_DRAWER_POSITION_DELTA_XY_M[1],
            "tray_center_x_constraint": "cabinet_front_minus_half_tray_depth",
            "seam_gap_m": PULLED_DRAWER_SEAM_GAP_M,
            "minimum_side_clearance_m": PULLED_DRAWER_MIN_SIDE_CLEARANCE_M,
            "seed_rule": "base_seed_plus_104729_env_id_plus_17171",
        },
        "collision": {
            "panel_type": "kinematic_rigid_body",
            "collision_enabled": True,
            "contact_reporting_enabled": True,
            "robot_contact_scope": "all_cabinet_and_tray_panels_except_robot_base",
            "panel_contact_offset_m": PULLED_DRAWER_PANEL_CONTACT_OFFSET_M,
            "panel_rest_offset_m": PULLED_DRAWER_PANEL_REST_OFFSET_M,
            "panel_solver_position_iteration_count": (
                PULLED_DRAWER_PANEL_SOLVER_POSITION_ITERATION_COUNT
            ),
            "panel_solver_velocity_iteration_count": (
                PULLED_DRAWER_PANEL_SOLVER_VELOCITY_ITERATION_COUNT
            ),
            "robot_articulation_solver_position_iteration_count": (
                PULLED_DRAWER_ROBOT_SOLVER_POSITION_ITERATION_COUNT
            ),
            "robot_articulation_solver_velocity_iteration_count": (
                PULLED_DRAWER_ROBOT_SOLVER_VELOCITY_ITERATION_COUNT
            ),
            "static_friction": 1.0,
            "dynamic_friction": 0.8,
            "restitution": 0.0,
        },
        "cube": {
            "size_buckets_m": list(GELSIGHT_SIZE_BUCKETS_M),
            "assignment": "seeded_balanced_permutation_per_group_of_8",
            "reset_half_range_xy_m": list(PULLED_DRAWER_CUBE_RESET_HALF_RANGE_XY_M),
            "spawn_z": "tray_floor_top_plus_half_cube_size",
        },
        "appearance": {
            "frequency": "per_environment_episode_reset",
            "cabinet_hsv": {"h": [0.0, 1.0], "s": [0.0, 0.15], "v": [0.65, 1.0]},
            "tray_nominal_rgb": list(PULLED_DRAWER_TRAY_NOMINAL_COLOR_RGB),
            "tray_hsv": {
                "h": list(PULLED_DRAWER_TRAY_HUE_RANGE),
                "s": list(PULLED_DRAWER_TRAY_SATURATION_RANGE),
                "v": list(PULLED_DRAWER_TRAY_VALUE_RANGE),
            },
            "tray_opacity": PULLED_DRAWER_TRAY_OPACITY,
            "tray_roughness": PULLED_DRAWER_TRAY_ROUGHNESS,
            "opacity_randomization_enabled": False,
        },
    }


def _cube_illegal_sensor_cfg() -> ContactSensorCfg:
    filters = [
        f"/World/envs/env_.*/Robot/{name}" for name in _CUBE_ILLEGAL_ROBOT_BODY_NAMES
    ]
    filters.extend(f"/World/envs/env_.*/{name}" for name in _CABINET_PANEL_NAMES)
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/cube",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        filter_prim_paths_expr=filters,
    )


def _surface_robot_sensor_cfg(name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        filter_prim_paths_expr=[
            f"/World/envs/env_.*/Robot/{body_name}"
            for body_name in _TABLE_ILLEGAL_ROBOT_BODY_NAMES
        ],
    )


def _pulled_drawer_robot_cfg():
    """Copy the GelSight robot and apply Pulled-Drawer-only solver settings."""
    robot = (
        Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg()
        .robot.copy()
    )
    robot.spawn = robot.spawn.copy()
    robot.spawn.usd_path = GELSIGHT_PULLED_DRAWER_FRANKA_USD
    robot.spawn.articulation_props = robot.spawn.articulation_props.copy()
    robot.spawn.articulation_props.solver_position_iteration_count = (
        PULLED_DRAWER_ROBOT_SOLVER_POSITION_ITERATION_COUNT
    )
    robot.spawn.articulation_props.solver_velocity_iteration_count = (
        PULLED_DRAWER_ROBOT_SOLVER_VELOCITY_ITERATION_COUNT
    )
    return robot


_NOMINAL_PANELS = _cabinet_nominal_panels()


@configclass
class _PulledDrawerCfgMixin:
    # These values are intentionally local instead of inherited from X040.
    # Teacher and Student share one explicit Pulled-Drawer physics/safety
    # contract without changing any other GelSight environment.
    robot = _pulled_drawer_robot_cfg()
    pulled_drawer_robot_asset_profile = GELSIGHT_PULLED_DRAWER_FRANKA_ASSET_PROFILE
    pulled_drawer_finger_extension_local_z_m = (
        GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M
    )
    arm_ik_tcp_offset_m = (0.0, 0.0, PULLED_DRAWER_GELSIGHT_CENTER_OFFSET_HAND_M)
    gelsight_center_offset_hand_m = (
        0.0,
        0.0,
        PULLED_DRAWER_GELSIGHT_CENTER_OFFSET_HAND_M,
    )
    gelsight_fingertip_bottom_offset_hand_m = (
        0.0,
        0.0,
        PULLED_DRAWER_GELSIGHT_BOTTOM_OFFSET_HAND_M,
    )
    rma_contact_force_threshold_n = 1.0
    illegal_collision_curriculum_start_step = (
        PULLED_DRAWER_ILLEGAL_COLLISION_CURRICULUM_START_STEP
    )
    illegal_collision_curriculum_end_step = (
        PULLED_DRAWER_ILLEGAL_COLLISION_CURRICULUM_END_STEP
    )
    illegal_collision_penalty_threshold_start_n = (
        PULLED_DRAWER_ILLEGAL_COLLISION_PENALTY_THRESHOLD_START_N
    )
    illegal_collision_penalty_threshold_end_n = (
        PULLED_DRAWER_ILLEGAL_COLLISION_PENALTY_THRESHOLD_END_N
    )
    illegal_collision_termination_threshold_start_n = (
        PULLED_DRAWER_ILLEGAL_COLLISION_TERMINATION_THRESHOLD_START_N
    )
    illegal_collision_termination_threshold_end_n = (
        PULLED_DRAWER_ILLEGAL_COLLISION_TERMINATION_THRESHOLD_END_N
    )
    illegal_collision_terminates_episode = True
    illegal_collision_penalty = PULLED_DRAWER_ILLEGAL_COLLISION_PENALTY
    cube_x_pos_range = PULLED_DRAWER_CUBE_RESET_HALF_RANGE_XY_M[0]
    cube_y_pos_range = PULLED_DRAWER_CUBE_RESET_HALF_RANGE_XY_M[1]
    cube_position_curriculum_enabled = False
    cube_position_curriculum_force_full_range = True
    cube_illegal_contact_sensor = _cube_illegal_sensor_cfg()
    pulled_drawer_geometry_seed_source = "cfg.seed"
    pulled_drawer_appearance_randomization_enabled = False
    pulled_drawer_opacity_randomization_enabled = False
    pulled_drawer_tray_opacity = PULLED_DRAWER_TRAY_OPACITY
    pulled_drawer_collision_scope = (
        "cube_non_gelpad_robot_or_cabinet_and_nonbase_robot_floor_cabinet_tray"
    )

    for _panel_name, (_panel_size, _panel_pos) in _NOMINAL_PANELS.items():
        locals()[_panel_name] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{_panel_name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_panel_pos),
            spawn=_kinematic_cuboid_cfg(_panel_size, (0.92, 0.92, 0.90)),
        )
    del _panel_name, _panel_size, _panel_pos

    for _panel_name, (_panel_size, _panel_pos) in _tray_nominal_panels().items():
        locals()[_panel_name] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{_panel_name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_panel_pos),
            spawn=_kinematic_cuboid_cfg(
                _panel_size,
                PULLED_DRAWER_TRAY_NOMINAL_COLOR_RGB,
                opacity=PULLED_DRAWER_TRAY_OPACITY,
                roughness=PULLED_DRAWER_TRAY_ROUGHNESS,
            ),
        )
    del _panel_name, _panel_size, _panel_pos

    pulled_drawer_surface_contact_sensors = {
        name: _surface_robot_sensor_cfg(name)
        for name in (*_CABINET_PANEL_NAMES, *_TRAY_PANEL_NAMES)
    }


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg(
    _PulledDrawerCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg,
):
    rma_task_id = GELSIGHT_PULLED_DRAWER_TEACHER_TASK
    cube = Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg().cube.copy()
    cube.init_state = cube.init_state.copy()
    cube.init_state.pos = (*PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M, 0.033)


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg(
    _PulledDrawerCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg,
):
    rma_task_id = GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK
    pulled_drawer_appearance_randomization_enabled = True
    cube = Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg().cube.copy()
    cube.init_state = cube.init_state.copy()
    cube.init_state.pos = (*PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M, 0.033)


class _PulledDrawerSceneMixin:
    """Author heterogeneous geometry before PhysX starts and keep it fixed."""

    cfg: _PulledDrawerCfgMixin

    def _setup_scene(self) -> None:
        for name in (*_CABINET_PANEL_NAMES, *_TRAY_PANEL_NAMES):
            rigid_object = RigidObject(getattr(self.cfg, name))
            setattr(self, f"_{name}", rigid_object)
            self.scene.rigid_objects[name] = rigid_object
        super()._setup_scene()
        seed = int(getattr(self.cfg, "seed", 42) or 42)
        sampled = sample_pulled_drawer_layouts(self.num_envs, seed)
        self._pulled_drawer_layout = {
            key: value.to(device=self.device, dtype=torch.float32)
            for key, value in sampled.items()
        }
        self._author_pulled_drawer_geometry(sampled)
        self._register_pulled_drawer_contact_sensors()
        if bool(self.cfg.pulled_drawer_appearance_randomization_enabled):
            self._setup_pulled_drawer_materials()

    def _author_fixed_cube_scales(self) -> None:
        seed = int(getattr(self.cfg, "seed", 42) or 42)
        bucket_ids = balanced_cube_bucket_ids(self.num_envs, seed)
        self._active_cube_bucket_ids = bucket_ids.to(self.device)
        prim_paths = sim_utils.find_matching_prim_paths(self.cfg.cube.prim_path)
        if len(prim_paths) != self.num_envs:
            raise RuntimeError(f"Expected one Cube per env, found {len(prim_paths)}")
        nominal_size = float(self.cfg.cube.spawn.size[0])
        stage = sim_utils.stage_utils.get_current_stage()
        with Sdf.ChangeBlock():
            for env_id, prim_path in enumerate(prim_paths):
                bucket_id = int(bucket_ids[env_id].item())
                scale = float(self.cfg.cube_size_buckets_m[bucket_id]) / nominal_size
                self._author_transform_spec(stage, prim_path, None, (scale, scale, scale))

    @staticmethod
    def _author_transform_spec(
        stage,
        prim_path: str,
        translation: tuple[float, float, float] | None,
        scale: tuple[float, float, float],
    ) -> None:
        prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
        if translation is not None:
            translate_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            if translate_spec is None:
                translate_spec = Sdf.AttributeSpec(
                    prim_spec, prim_path + ".xformOp:translate", Sdf.ValueTypeNames.Double3
                )
            translate_spec.default = Gf.Vec3d(*translation)
        scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
        if scale_spec is None:
            scale_spec = Sdf.AttributeSpec(
                prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3
            )
        scale_spec.default = Gf.Vec3d(*scale)
        order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
        if order_spec is None:
            order_spec = Sdf.AttributeSpec(
                prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
            )
        order_spec.default = Vt.TokenArray(
            ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        )

    def _author_pulled_drawer_geometry(self, sampled: Mapping[str, torch.Tensor]) -> None:
        stage = sim_utils.stage_utils.get_current_stage()
        nominal_panels = _cabinet_nominal_panels()
        nominal_tray_panels = _tray_nominal_panels()
        with Sdf.ChangeBlock():
            for env_id in range(self.num_envs):
                depth, width, height = sampled["cabinet_size_m"][env_id].tolist()
                center_x, center_y = sampled["cabinet_center_xy_m"][env_id].tolist()
                thickness = PULLED_DRAWER_PANEL_THICKNESS_M
                inner_width = width - 2.0 * thickness
                targets = {
                    "pulled_drawer_cabinet_left": ((depth, thickness, height), (center_x, center_y - 0.5 * width + 0.5 * thickness, 0.5 * height)),
                    "pulled_drawer_cabinet_right": ((depth, thickness, height), (center_x, center_y + 0.5 * width - 0.5 * thickness, 0.5 * height)),
                    "pulled_drawer_cabinet_back": ((thickness, inner_width, height), (center_x + 0.5 * depth - 0.5 * thickness, center_y, 0.5 * height)),
                    "pulled_drawer_cabinet_top": ((depth, width, thickness), (center_x, center_y, height - 0.5 * thickness)),
                }
                for name, (target_size, target_pos) in targets.items():
                    nominal_size = nominal_panels[name][0]
                    scale = tuple(target_size[index] / nominal_size[index] for index in range(3))
                    self._author_transform_spec(stage, f"/World/envs/env_{env_id}/{name}", target_pos, scale)
                tray_depth, tray_width, tray_height = sampled["tray_size_m"][env_id].tolist()
                tray_x, tray_y = sampled["tray_center_xy_m"][env_id].tolist()
                tray_wall_height = tray_height - thickness
                tray_wall_center_z = thickness + 0.5 * tray_wall_height
                tray_targets = {
                    "pulled_drawer_tray_floor": (
                        (tray_depth, tray_width, thickness),
                        (tray_x, tray_y, 0.5 * thickness),
                    ),
                    "pulled_drawer_tray_front": (
                        (thickness, tray_width, tray_wall_height),
                        (tray_x - 0.5 * tray_depth + 0.5 * thickness, tray_y, tray_wall_center_z),
                    ),
                    "pulled_drawer_tray_back": (
                        (thickness, tray_width, tray_wall_height),
                        (tray_x + 0.5 * tray_depth - 0.5 * thickness, tray_y, tray_wall_center_z),
                    ),
                    "pulled_drawer_tray_left": (
                        (tray_depth - 2.0 * thickness, thickness, tray_wall_height),
                        (tray_x, tray_y - 0.5 * tray_width + 0.5 * thickness, tray_wall_center_z),
                    ),
                    "pulled_drawer_tray_right": (
                        (tray_depth - 2.0 * thickness, thickness, tray_wall_height),
                        (tray_x, tray_y + 0.5 * tray_width - 0.5 * thickness, tray_wall_center_z),
                    ),
                }
                for name, (target_size, target_pos) in tray_targets.items():
                    nominal_size = nominal_tray_panels[name][0]
                    scale = tuple(target_size[index] / nominal_size[index] for index in range(3))
                    self._author_transform_spec(
                        stage,
                        f"/World/envs/env_{env_id}/{name}",
                        target_pos,
                        scale,
                    )

    def _register_pulled_drawer_contact_sensors(self) -> None:
        self._pulled_drawer_surface_sensors: dict[str, ContactSensor] = {}
        for name, cfg in self.cfg.pulled_drawer_surface_contact_sensors.items():
            sensor = ContactSensor(cfg)
            self._pulled_drawer_surface_sensors[name] = sensor
            self.scene.sensors[f"{name}_robot_contact"] = sensor

    def _compute_illegal_collision_force(self) -> torch.Tensor:
        force = super()._compute_illegal_collision_force()
        for sensor in self._pulled_drawer_surface_sensors.values():
            force = torch.maximum(force, self._max_filtered_contact_force(sensor))
        return force

    def _validate_pulled_drawer_physics(self) -> None:
        """Fail closed if any visible drawer panel lacks a PhysX rigid collider."""
        if getattr(self, "_pulled_drawer_physics_validated", False):
            return
        stage = sim_utils.stage_utils.get_current_stage()
        failures: list[str] = []
        for name in (*_CABINET_PANEL_NAMES, *_TRAY_PANEL_NAMES):
            asset = self.scene.rigid_objects[name]
            view = asset.root_physx_view
            if view is None or int(view.count) != self.num_envs:
                count = 0 if view is None else int(view.count)
                failures.append(f"{name}: PhysX rigid-body count {count} != {self.num_envs}")
                continue
            mesh = stage.GetPrimAtPath(f"/World/envs/env_0/{name}/geometry/mesh")
            if not mesh.IsValid() or not mesh.HasAPI(UsdPhysics.CollisionAPI):
                failures.append(f"{name}: missing UsdPhysics.CollisionAPI")
                continue
            collision_api = UsdPhysics.CollisionAPI(mesh)
            collision_enabled = collision_api.GetCollisionEnabledAttr().Get()
            if collision_enabled is not True:
                failures.append(f"{name}: physics:collisionEnabled={collision_enabled!r}")
        if failures:
            raise RuntimeError(
                "Pulled-Drawer panels must be physical rigid colliders; " + "; ".join(failures)
            )
        self._pulled_drawer_physics_validated = True

    def _set_fixed_cube_spawn_height(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        tray_xy = self._pulled_drawer_layout["tray_center_xy_m"][env_ids]
        self._cube.data.default_root_state[env_ids, :2] = tray_xy
        self._cube.data.default_root_state[env_ids, 2] = (
            PULLED_DRAWER_PANEL_THICKNESS_M
            + 0.5 * self.active_cube_size_m[env_ids]
        )

    def _grasp_support_height_m(self) -> float:
        return PULLED_DRAWER_PANEL_THICKNESS_M

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        self._validate_pulled_drawer_physics()
        super()._reset_idx(env_ids)
        for sensor in self._pulled_drawer_surface_sensors.values():
            sensor.reset(env_ids)

    def _setup_pulled_drawer_materials(self) -> None:
        self._pulled_drawer_material_paths = {"cabinet": [], "tray": []}
        for env_id in range(self.num_envs):
            cabinet_path = f"/World/Looks/PulledDrawer/env_{env_id}/cabinet"
            tray_path = f"/World/Looks/PulledDrawer/env_{env_id}/tray"
            sim_utils.spawn_preview_surface(
                cabinet_path,
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.92, 0.92, 0.90), roughness=0.72, metallic=0.0
                ),
            )
            sim_utils.spawn_preview_surface(
                tray_path,
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=PULLED_DRAWER_TRAY_NOMINAL_COLOR_RGB,
                    opacity=float(self.cfg.pulled_drawer_tray_opacity),
                    roughness=PULLED_DRAWER_TRAY_ROUGHNESS,
                    metallic=0.0,
                ),
            )
            self._pulled_drawer_material_paths["cabinet"].append(cabinet_path)
            self._pulled_drawer_material_paths["tray"].append(tray_path)
            for name in _CABINET_PANEL_NAMES:
                sim_utils.bind_visual_material(
                    self._resolve_visual_target_prim(f"/World/envs/env_{env_id}/{name}"),
                    cabinet_path,
                )
            for name in _TRAY_PANEL_NAMES:
                sim_utils.bind_visual_material(
                    self._resolve_visual_target_prim(f"/World/envs/env_{env_id}/{name}"),
                    tray_path,
                )

    @staticmethod
    def _hsv_samples_to_rgb(samples: torch.Tensor) -> list[tuple[float, float, float]]:
        return [
            tuple(float(value) for value in colorsys.hsv_to_rgb(*row))
            for row in samples.detach().cpu().tolist()
        ]

    def _randomize_scene_visuals(self, env_ids: torch.Tensor) -> None:
        super()._randomize_scene_visuals(env_ids)
        if not bool(self.cfg.pulled_drawer_appearance_randomization_enabled):
            return
        env_ids = self._normalize_dr_env_ids(env_ids)
        count = int(env_ids.numel())
        if count == 0:
            return
        cabinet_hsv = torch.empty((count, 3), device=self.device)
        cabinet_hsv[:, 0].uniform_(0.0, 1.0)
        cabinet_hsv[:, 1].uniform_(0.0, 0.15)
        cabinet_hsv[:, 2].uniform_(0.65, 1.0)
        tray_hsv = torch.empty((count, 3), device=self.device)
        tray_hsv[:, 0].uniform_(*PULLED_DRAWER_TRAY_HUE_RANGE)
        tray_hsv[:, 1].uniform_(*PULLED_DRAWER_TRAY_SATURATION_RANGE)
        tray_hsv[:, 2].uniform_(*PULLED_DRAWER_TRAY_VALUE_RANGE)
        cabinet_rgb = self._hsv_samples_to_rgb(cabinet_hsv)
        tray_rgb = self._hsv_samples_to_rgb(tray_hsv)
        stage = sim_utils.stage_utils.get_current_stage()
        for local_id, env_id in enumerate(env_ids.detach().cpu().tolist()):
            cabinet_shader = stage.GetPrimAtPath(
                f"{self._pulled_drawer_material_paths['cabinet'][env_id]}/Shader"
            )
            tray_shader = stage.GetPrimAtPath(
                f"{self._pulled_drawer_material_paths['tray'][env_id]}/Shader"
            )
            sim_utils.safe_set_attribute_on_usd_prim(
                cabinet_shader, "inputs:diffuse_color", cabinet_rgb[local_id], camel_case=True
            )
            sim_utils.safe_set_attribute_on_usd_prim(
                tray_shader, "inputs:diffuse_color", tray_rgb[local_id], camel_case=True
            )


class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnv(
    _PulledDrawerSceneMixin,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg


class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnv(
    _PulledDrawerSceneMixin,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg


__all__ = (
    "GELSIGHT_PULLED_DRAWER_TEACHER_TASK",
    "GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK",
    "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnv",
    "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnv",
    "balanced_cube_bucket_ids",
    "pulled_drawer_geometry_contract",
    "sample_pulled_drawer_layouts",
)
