"""Standalone Franka cabinet scene with partial third-person occlusion.

Usage:
    isaaclab -p scripts/cylinder_grasping/franka_cabinet_partial_scene.py
"""

from __future__ import annotations

import argparse
import math
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Build a standalone Franka + cabinet scene where the third-person camera only sees part of the object."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.input as carb_input
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.appwindow
import torch
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from pxr import UsdGeom

from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)

YCB_CAN_USD_PATH = "/home/tinydog/Projects/ycb-tools/models/ycb/002_master_chef_can/can.usda"
YCB_CAN_SCALE = (0.5, 0.5, 0.5)

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
CABINET_CENTER_Z = CABINET_BASE_Z + CABINET_HEIGHT_Z * 0.5
CABINET_INTERNAL_SHELF_Z = CABINET_BASE_Z + CABINET_INTERNAL_SHELF_Z_LOCAL
UPPER_SHELF_TOP_Z = CABINET_INTERNAL_SHELF_Z + CABINET_PANEL_THICKNESS * 0.5

# Put the only target object on the upper layer, slightly deep inside the cabinet.
TARGET_OBJECT_X = CABINET_CENTER_X + 0.065
TARGET_OBJECT_Y = -0.06
TARGET_OBJECT_Z = UPPER_SHELF_TOP_Z + 0.0008

FRANKA_HOME_JOINTS = {
    "panda_joint1": -0.4693,
    "panda_joint2": -0.1410,
    "panda_joint3": 0.4929,
    "panda_joint4": -2.1960,
    "panda_joint5": 0.0312,
    "panda_joint6": 2.0935,
    "panda_joint7": 0.7937,
    "panda_finger_joint1": 0.02,
    "panda_finger_joint2": 0.02,
}

THIRD_PERSON_CAMERA_POSE = (
    (-1.2, -0.55, 0.30),
    (0.561, 0.561, -0.43, -0.431),
)

EE_TRANSLATION_STEP = 0.01
EE_ROTATION_STEP_DEG = 5.0
EE_ROTATION_STEP_RAD = math.radians(EE_ROTATION_STEP_DEG)
IK_SETTLE_STEPS = 24


@sim_utils.clone
def spawn_ycb_can(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn the YCB can and explicitly attach rigid-body and collision properties."""
    spawn_cfg = cfg.replace(rigid_props=None, collision_props=None, mass_props=None)
    prim = _spawn_from_usd_file(prim_path, spawn_cfg.usd_path, spawn_cfg, translation, orientation)

    if cfg.rigid_props is not None:
        sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)
    else:
        sim_utils.define_rigid_body_properties(prim_path, sim_utils.RigidBodyPropertiesCfg())

    if cfg.collision_props is not None:
        mesh_prims = sim_utils.get_all_matching_child_prims(
            prim_path, predicate=lambda prim: prim.IsA(UsdGeom.Mesh)
        )
        if not mesh_prims:
            raise RuntimeError(f"No mesh prims found under '{prim_path}' for collision.")
        for mesh_prim in mesh_prims:
            sim_utils.define_collision_properties(mesh_prim.GetPath().pathString, cfg.collision_props)

    if cfg.mass_props is not None:
        sim_utils.define_mass_properties(prim_path, cfg.mass_props)

    return prim


def make_fixed_panel_cfg(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> sim_utils.CuboidCfg:
    """Create a kinematic wooden panel used for the cabinet."""
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


def spawn_cabinet() -> None:
    """Spawn a simple two-layer cabinet."""
    body_color = (0.48, 0.35, 0.24)
    shelf_color = (0.43, 0.31, 0.22)

    bottom_cfg = make_fixed_panel_cfg(
        size=(CABINET_DEPTH_X, CABINET_WIDTH_Y, CABINET_PANEL_THICKNESS),
        color=body_color,
    )
    bottom_cfg.func(
        "/World/Cabinet/bottom",
        bottom_cfg,
        translation=(CABINET_CENTER_X, CABINET_CENTER_Y, CABINET_BASE_Z + CABINET_PANEL_THICKNESS * 0.5),
    )

    top_cfg = make_fixed_panel_cfg(
        size=(CABINET_DEPTH_X, CABINET_WIDTH_Y, CABINET_PANEL_THICKNESS),
        color=body_color,
    )
    top_cfg.func(
        "/World/Cabinet/top",
        top_cfg,
        translation=(CABINET_CENTER_X, CABINET_CENTER_Y, CABINET_BASE_Z + CABINET_HEIGHT_Z - CABINET_PANEL_THICKNESS * 0.5),
    )

    left_side_cfg = make_fixed_panel_cfg(
        size=(CABINET_DEPTH_X, CABINET_PANEL_THICKNESS, CABINET_HEIGHT_Z),
        color=body_color,
    )
    left_side_cfg.func(
        "/World/Cabinet/left_side",
        left_side_cfg,
        translation=(
            CABINET_CENTER_X,
            CABINET_CENTER_Y - CABINET_WIDTH_Y * 0.5 + CABINET_PANEL_THICKNESS * 0.5,
            CABINET_CENTER_Z,
        ),
    )
    left_side_cfg.func(
        "/World/Cabinet/right_side",
        left_side_cfg,
        translation=(
            CABINET_CENTER_X,
            CABINET_CENTER_Y + CABINET_WIDTH_Y * 0.5 - CABINET_PANEL_THICKNESS * 0.5,
            CABINET_CENTER_Z,
        ),
    )

    back_cfg = make_fixed_panel_cfg(
        size=(CABINET_BACK_THICKNESS, CABINET_WIDTH_Y - 2 * CABINET_PANEL_THICKNESS, CABINET_HEIGHT_Z - 2 * CABINET_PANEL_THICKNESS),
        color=body_color,
    )
    back_cfg.func(
        "/World/Cabinet/back",
        back_cfg,
        translation=(
            CABINET_BACK_X - CABINET_BACK_THICKNESS * 0.5,
            CABINET_CENTER_Y,
            CABINET_CENTER_Z,
        ),
    )

    # One internal shelf splits the cabinet into upper and lower layers.
    shelf_cfg = make_fixed_panel_cfg(
        size=(CABINET_DEPTH_X - CABINET_PANEL_THICKNESS, CABINET_WIDTH_Y - 2 * CABINET_PANEL_THICKNESS, CABINET_PANEL_THICKNESS),
        color=shelf_color,
    )
    shelf_cfg.func(
        "/World/Cabinet/shelf",
        shelf_cfg,
        translation=(
            CABINET_CENTER_X,
            CABINET_CENTER_Y,
            CABINET_INTERNAL_SHELF_Z,
        ),
    )


def spawn_support_boxes() -> None:
    """Spawn simple support boxes below Franka and the cabinet."""
    support_color = (0.58, 0.58, 0.60)

    robot_support_cfg = make_fixed_panel_cfg(
        size=(ROBOT_SUPPORT_WIDTH_X, ROBOT_SUPPORT_WIDTH_Y, ROBOT_SUPPORT_HEIGHT),
        color=support_color,
    )
    robot_support_cfg.func(
        "/World/RobotSupportBox",
        robot_support_cfg,
        translation=(ROBOT_SUPPORT_CENTER_X, ROBOT_SUPPORT_CENTER_Y, ROBOT_SUPPORT_HEIGHT * 0.5),
    )

    cabinet_support_cfg = make_fixed_panel_cfg(
        size=(CABINET_SUPPORT_WIDTH_X, CABINET_SUPPORT_WIDTH_Y, CABINET_SUPPORT_HEIGHT),
        color=support_color,
    )
    cabinet_support_cfg.func(
        "/World/CabinetSupportBox",
        cabinet_support_cfg,
        translation=(CABINET_CENTER_X, CABINET_CENTER_Y, CABINET_SUPPORT_HEIGHT * 0.5),
    )


def spawn_ground_and_light() -> None:
    """Spawn the base scene objects."""
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)


def spawn_robot() -> Articulation:
    """Spawn Franka with the GelSight rigid gripper asset."""
    robot_cfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(ROBOT_SUPPORT_CENTER_X, ROBOT_SUPPORT_CENTER_Y, ROBOT_BASE_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": FRANKA_HOME_JOINTS["panda_joint1"],
                "panda_joint2": FRANKA_HOME_JOINTS["panda_joint2"],
                "panda_joint3": FRANKA_HOME_JOINTS["panda_joint3"],
                "panda_joint4": FRANKA_HOME_JOINTS["panda_joint4"],
                "panda_joint5": FRANKA_HOME_JOINTS["panda_joint5"],
                "panda_joint6": FRANKA_HOME_JOINTS["panda_joint6"],
                "panda_joint7": FRANKA_HOME_JOINTS["panda_joint7"],
                "panda_finger_joint.*": 0.02,
            },
        ),
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            rigid_props=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.rigid_props.replace(
                disable_gravity=True,
            ),
        ),
    )
    return Articulation(robot_cfg)


def spawn_target_object() -> RigidObject:
    """Spawn the object on the upper layer inside the cabinet."""
    target_cfg = RigidObjectCfg(
        prim_path="/World/TargetObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(TARGET_OBJECT_X, TARGET_OBJECT_Y, TARGET_OBJECT_Z)),
        spawn=sim_utils.UsdFileCfg(
            func=spawn_ycb_can,
            usd_path=YCB_CAN_USD_PATH,
            scale=YCB_CAN_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=1.0,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0005,
            ),
        ),
    )
    return RigidObject(target_cfg)


def spawn_third_person_camera() -> Camera:
    """Spawn the fixed third-person camera used for the partial-visibility view."""
    camera_cfg = CameraCfg(
        prim_path="/World/third_person_camera",
        update_period=0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=200.0,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=THIRD_PERSON_CAMERA_POSE[0],
            rot=THIRD_PERSON_CAMERA_POSE[1],
            convention="opengl",
        ),
    )
    return Camera(camera_cfg)


def apply_franka_home_pose(robot: Articulation) -> None:
    """Write the desired Franka joint pose into simulation and keep it as the position target."""
    robot.reset()

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = joint_pos.new_zeros(joint_pos.shape)
    name_to_index = {name: index for index, name in enumerate(robot.joint_names)}

    for joint_name, joint_value in FRANKA_HOME_JOINTS.items():
        joint_pos[:, name_to_index[joint_name]] = joint_value

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()


def step_scene(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    target_object: RigidObject,
    third_person_camera: Camera,
    num_steps: int = 1,
) -> None:
    """Advance the scene and refresh asset buffers."""
    for _ in range(num_steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)
        target_object.update(sim.cfg.dt)
        third_person_camera.update(sim.cfg.dt)


def compute_frame_pose(
    robot: Articulation,
    body_idx: int,
    offset_pos: torch.Tensor,
    offset_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the end-effector frame pose in the robot root frame."""
    ee_pos_w = robot.data.body_link_pos_w[:, body_idx]
    ee_quat_w = robot.data.body_link_quat_w[:, body_idx]
    root_pos_w = robot.data.root_link_pos_w
    root_quat_w = robot.data.root_link_quat_w

    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(ee_pose_b, ee_quat_b, offset_pos, offset_rot)
    return ee_pose_b, ee_quat_b


def compute_frame_jacobian(
    robot: Articulation,
    jacobi_body_idx: int,
    offset_pos: torch.Tensor,
    offset_rot: torch.Tensor,
) -> torch.Tensor:
    """Compute the geometric Jacobian for the offset end-effector frame in the robot root frame."""
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, :]
    base_rot = robot.data.root_link_quat_w
    base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
    jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_pos), jacobian[:, 3:, :])
    jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(offset_rot), jacobian[:, 3:, :])
    return jacobian


def print_joint_angles(robot: Articulation) -> None:
    """Print all current robot joint angles after a keyboard move."""
    joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
    print("[JOINT] Updated joint angles:")
    for joint_name, joint_angle in zip(robot.joint_names, joint_pos):
        print(f"  {joint_name}: {joint_angle:.4f} rad ({math.degrees(joint_angle):.2f} deg)")


def move_end_effector_by_delta(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    target_object: RigidObject,
    third_person_camera: Camera,
    ik_controller: DifferentialIKController,
    body_idx: int,
    jacobi_body_idx: int,
    arm_joint_ids: list[int],
    finger_joint_ids: list[int],
    finger_joint_targets: torch.Tensor,
    offset_pos: torch.Tensor,
    offset_rot: torch.Tensor,
    delta_xyz: tuple[float, float, float],
    delta_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    settle_steps: int = IK_SETTLE_STEPS,
) -> None:
    """Move the Franka end-effector by a small relative pose delta and print the resulting joint angles."""
    device = robot.data.joint_pos.device
    delta = torch.zeros((1, 6), device=device)
    delta[0, 0] = delta_xyz[0]
    delta[0, 1] = delta_xyz[1]
    delta[0, 2] = delta_xyz[2]
    delta[0, 3] = delta_rpy[0]
    delta[0, 4] = delta_rpy[1]
    delta[0, 5] = delta_rpy[2]

    ee_pos_curr_b, ee_quat_curr_b = compute_frame_pose(robot, body_idx, offset_pos, offset_rot)
    ik_controller.set_command(delta, ee_pos_curr_b, ee_quat_curr_b)

    joint_pos = robot.data.joint_pos.clone()
    jacobian = compute_frame_jacobian(robot, jacobi_body_idx, offset_pos, offset_rot)
    arm_joint_pos_des = ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
    joint_pos[:, arm_joint_ids] = arm_joint_pos_des[:, : len(arm_joint_ids)]
    joint_pos[:, finger_joint_ids] = finger_joint_targets

    robot.set_joint_position_target(joint_pos)
    step_scene(sim, robot, target_object, third_person_camera, num_steps=settle_steps)

    print(
        "[EE] Applied delta "
        f"dx={delta_xyz[0]:+.3f} m, dy={delta_xyz[1]:+.3f} m, dz={delta_xyz[2]:+.3f} m, "
        f"droll={math.degrees(delta_rpy[0]):+.1f} deg, "
        f"dpitch={math.degrees(delta_rpy[1]):+.1f} deg, "
        f"dyaw={math.degrees(delta_rpy[2]):+.1f} deg"
    )
    print_joint_angles(robot)


def main() -> None:
    """Build the scene and keep simulation running."""
    sim_cfg = SimulationCfg(
        dt=1 / 60,
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.55, 1.00, 0.92], [CABINET_CENTER_X, CABINET_CENTER_Y, 0.28])

    spawn_ground_and_light()
    spawn_support_boxes()
    spawn_cabinet()
    robot = spawn_robot()
    target_object = spawn_target_object()
    third_person_camera = spawn_third_person_camera()

    sim.reset()
    apply_franka_home_pose(robot)
    step_scene(sim, robot, target_object, third_person_camera, num_steps=10)

    body_ids, _ = robot.find_bodies("panda_hand")
    body_idx = body_ids[0]
    jacobi_body_idx = body_idx - 1
    joint_name_to_index = {name: index for index, name in enumerate(robot.joint_names)}
    arm_joint_ids = [joint_name_to_index[f"panda_joint{i}"] for i in range(1, 8)]
    finger_joint_ids = [
        joint_name_to_index["panda_finger_joint1"],
        joint_name_to_index["panda_finger_joint2"],
    ]
    finger_joint_targets = robot.data.joint_pos[:, finger_joint_ids].clone()

    device = robot.data.joint_pos.device
    offset_pos = torch.tensor([[0.0, 0.0, 0.11841]], device=device)
    offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )
    ik_controller = DifferentialIKController(cfg=ik_controller_cfg, num_envs=1, device=device)

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window is not None else None
    input_iface = carb_input.acquire_input_interface() if keyboard is not None else None
    kb_sub = None
    command_queue = deque()

    def _on_kb_event(event, *args, **kwargs):
        if event.type != carb_input.KeyboardEventType.KEY_PRESS:
            return True

        if event.input == carb_input.KeyboardInput.I:
            command_queue.append(("move +X", (EE_TRANSLATION_STEP, 0.0, 0.0), (0.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.K:
            command_queue.append(("move -X", (-EE_TRANSLATION_STEP, 0.0, 0.0), (0.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.J:
            command_queue.append(("move +Y", (0.0, EE_TRANSLATION_STEP, 0.0), (0.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.L:
            command_queue.append(("move -Y", (0.0, -EE_TRANSLATION_STEP, 0.0), (0.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.U:
            command_queue.append(("move +Z", (0.0, 0.0, EE_TRANSLATION_STEP), (0.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.O:
            command_queue.append(("move -Z", (0.0, 0.0, -EE_TRANSLATION_STEP), (0.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.Z:
            command_queue.append(("roll +", (0.0, 0.0, 0.0), (EE_ROTATION_STEP_RAD, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.X:
            command_queue.append(("roll -", (0.0, 0.0, 0.0), (-EE_ROTATION_STEP_RAD, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.C:
            command_queue.append(("pitch +", (0.0, 0.0, 0.0), (0.0, EE_ROTATION_STEP_RAD, 0.0)))
        elif event.input == carb_input.KeyboardInput.V:
            command_queue.append(("pitch -", (0.0, 0.0, 0.0), (0.0, -EE_ROTATION_STEP_RAD, 0.0)))
        elif event.input == carb_input.KeyboardInput.B:
            command_queue.append(("yaw +", (0.0, 0.0, 0.0), (0.0, 0.0, EE_ROTATION_STEP_RAD)))
        elif event.input == carb_input.KeyboardInput.N:
            command_queue.append(("yaw -", (0.0, 0.0, 0.0), (0.0, 0.0, -EE_ROTATION_STEP_RAD)))
        return True

    if keyboard is not None and input_iface is not None:
        kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)
        print("[INFO] Scene ready.")
        print("[INFO] Third-person camera prim: /World/third_person_camera")
        print("[INFO] Click the viewport first, then use keyboard:")
        print("       I/K: +X/-X, J/L: +Y/-Y, U/O: +Z/-Z")
        print("       Z/X: +Roll/-Roll, C/V: +Pitch/-Pitch, B/N: +Yaw/-Yaw")
        print(f"[INFO] Each key press moves the end-effector by {EE_TRANSLATION_STEP:.3f} m.")
        print(f"[INFO] Each rotation key press rotates the end-effector by {EE_ROTATION_STEP_DEG:.1f} deg.")
        print_joint_angles(robot)
    else:
        print("[WARN] Keyboard unavailable. End-effector keyboard control disabled.")

    try:
        while simulation_app.is_running():
            if command_queue:
                command_name, delta_xyz, delta_rpy = command_queue.popleft()
                print(f"[KEY] {command_name}")
                move_end_effector_by_delta(
                    sim=sim,
                    robot=robot,
                    target_object=target_object,
                    third_person_camera=third_person_camera,
                    ik_controller=ik_controller,
                    body_idx=body_idx,
                    jacobi_body_idx=jacobi_body_idx,
                    arm_joint_ids=arm_joint_ids,
                    finger_joint_ids=finger_joint_ids,
                    finger_joint_targets=finger_joint_targets,
                    offset_pos=offset_pos,
                    offset_rot=offset_rot,
                    delta_xyz=delta_xyz,
                    delta_rpy=delta_rpy,
                )
            else:
                step_scene(sim, robot, target_object, third_person_camera, num_steps=1)
    finally:
        if kb_sub is not None and keyboard is not None and input_iface is not None:
            try:
                input_iface.unsubscribe_from_keyboard_events(keyboard, kb_sub)
            except Exception:
                pass


if __name__ == "__main__":
    main()
    simulation_app.close()
