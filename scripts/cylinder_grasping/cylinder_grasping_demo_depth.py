from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Cylinder Grasping Demo with Franka robot equipped with GelSight sensors and wrist camera"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="Whether to render tactile images in the GUI",
)
parser.add_argument(
    "--tactile_render",
    choices=["depth", "rgb"],
    default="rgb",
    help="Tactile rendering mode: depth=camera_depth, rgb=tactile_rgb (press V to toggle at runtime).",
)
parser.add_argument(
    "--terminal_input",
    action="store_true",
    default=False,
    help="Enable terminal keyboard fallback (not used now, kept for compatibility).",
)
parser.add_argument(
    "--fsm_test",
    action="store_true",
    default=False,
    help="Run the 4-stage FSM test environment (0=approach, 1=probe, 2=grasp, 3=lift).",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import traceback
from contextlib import suppress

import carb
import omni.ui
import omni.appwindow
import carb.input as carb_input

from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import pynvml

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

YCB_CAN_USD_PATH = "/home/tinydog/Projects/ycb-tools/models/ycb/002_master_chef_can/can.usda"
YCB_CAN_SCALE = (0.4, 0.4, 0.4)


@sim_utils.clone
def spawn_ycb_can(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
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


class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)
                    # add simple manual control buttons (UI fallback)
                    import omni.ui as ui

                    with ui.CollapsableFrame("Manual Control (XY)", collapsed=False):
                        with ui.HStack(height=0):
                            ui.Label("Step: 0.004m", style={"color": 0xFFAAAAAA})

                        def _mk_cb(dx, dy):
                            def _cb():
                                try:
                                    action_scale = getattr(self.env, "action_scale", 0.1)
                                    step_units = 0.004 / max(action_scale, 1e-6)
                                    if hasattr(self.env, "enqueue_manual_move"):
                                        self.env.enqueue_manual_move(dx * step_units, dy * step_units)
                                except Exception:
                                    pass

                            return _cb

                        with ui.HStack():
                            ui.Spacer(width=8)
                            ui.Button("⬅", clicked_fn=_mk_cb(-1.0, 0.0))
                            ui.Button("⬆", clicked_fn=_mk_cb(0.0, 1.0))
                            ui.Button("⬇", clicked_fn=_mk_cb(0.0, -1.0))
                            ui.Button("➡", clicked_fn=_mk_cb(1.0, 0.0))
                            ui.Spacer(width=8)


@configclass
class CylinderGraspingDemoCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    debug_vis = True
    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
    )

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # plate under cylinder
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
            scale=(10, 10, 0.01),
            visible=False,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=1.0,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0005,
            ),
        ),
    )

    cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.05]),
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

    # robot configuration
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.down.usda"
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": -0.6199,
                "panda_joint2": -0.2263,
                "panda_joint3": 0.1508,
                "panda_joint4": -2.8275,
                "panda_joint5": 0.0273,
                "panda_joint6": 2.6023,
                "panda_joint7": 0.3497,
                "panda_finger_joint.*": 0.02,
            },
        ),
        # spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
        #     rigid_props=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.rigid_props.replace(
        #         disable_gravity=True,
        #     ),
        # ),
    )

    # wrist camera
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        update_period=0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, 0, -0.12),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )

    # third-person camera
    third_person_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=40.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.25, 0.0, 0.5),
            rot=(0.5, -0.5, -0.5, 0.5),
            convention="ros",
        ),
    )

    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 24),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,
        marker_motion_sim_cfg=None,
        # 输出深度与触觉 RGB（用于运行时切换显示）
        data_types=["camera_depth", "tactile_rgb"],
    )
    # settings for optical sim
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(320, 240),
    )
    gsmini_right = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )

    # additional down-facing tactile sensors (suffix _down)
    gsmini_left_down = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_down",
    )
    gsmini_right_down = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_down",
    )

    # IK controller
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )
    episode_length_s = 0
    action_space = 5  # xyz + rz + gripper
    observation_space = 0
    state_space = 0


class CylinderGraspingDemo(DirectRLEnv):
    """Cylinder grasping demo environment."""

    cfg: CylinderGraspingDemoCfg

    def __init__(self, cfg: CylinderGraspingDemoCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        # end-effector body index
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # finger joints
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])

        # For a fixed base robot, the frame index is one less than the body index.
        self._jacobi_body_idx = self._body_idx - 1

        # ee offset w.r.t panda hand
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # IK command buffers
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.action_scale = 0.1
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)

        # 当前动作 [dx, dy, dz, d_yaw, gripper]
        self.current_actions = torch.zeros((self.num_envs, 5), device=self.device)

        self.step_count = 0
        self.manual_move_units = torch.zeros((self.num_envs, 2), device=self.device)

        self.joint_angles = {


            'joint1': -0.456,
            'joint2': 0.227,
            'joint3': 0.445,
            'joint4': -2.412,
            'joint5': -0.277,
            'joint6': 2.625,
            'joint7': 1.008,

            
            # 'joint1': -0.372,
            # 'joint2': 0.260,
            # 'joint3': 0.448,
            # 'joint4': -2.393,
            # 'joint5': -0.336,
            # 'joint6': 2.644,
            # 'joint7': 1.144,

            # "joint1": -0.3136,
            # "joint2": -0.4448,
            # "joint3": 0.3858,
            # "joint4": -3.0236,
            # "joint5": 0.2319,
            # "joint6": 2.5776,
            # "joint7": 0.6249,
            "finger_left": 0.02,
            "finger_right": 0.02,


        }

        self.set_debug_vis(self.cfg.debug_vis)

    def enqueue_manual_move(self, dx_units: float, dy_units: float):
        try:
            self.manual_move_units[:, 0] += float(dx_units)
            self.manual_move_units[:, 1] += float(dy_units)
        except Exception:
            pass

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self._cylinder

        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        self.wrist_camera = Camera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera

        self.third_person_camera = Camera(self.cfg.third_person_camera)
        self.scene.sensors["third_person_camera"] = self.third_person_camera

        self.scene.clone_environments(copy_from_source=False)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.11841),
                    ),
                ),
            ],
        )

        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left

        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

        # down-facing sensors
        try:
            self.gsmini_left_down = GelSightSensor(self.cfg.gsmini_left_down)
            self.scene.sensors["gsmini_left_down"] = self.gsmini_left_down
        except Exception as e:
            print(f"[WARN] Skipping gsmini_left_down: {e}")
            self.gsmini_left_down = None
        try:
            self.gsmini_right_down = GelSightSensor(self.cfg.gsmini_right_down)
            self.scene.sensors["gsmini_right_down"] = self.gsmini_right_down
        except Exception as e:
            print(f"[WARN] Skipping gsmini_right_down: {e}")
            self.gsmini_right_down = None

        # ground
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step."""
        self.current_actions = actions.clone()

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()

        # 5维动作 [dx, dy, dz, d_yaw, gripper] -> 6维 [dx, dy, dz, droll, dpitch, dyaw]
        self.processed_actions[:, :3] = self.current_actions[:, :3] * self.action_scale
        self.processed_actions[:, 3] = 0.0
        self.processed_actions[:, 4] = 0.0
        self.processed_actions[:, 5] = self.current_actions[:, 3] * self.action_scale * 10.0

        self._ik_controller.set_command(self.processed_actions, ee_pos_curr_b, ee_quat_curr_b)
        self._apply_joint_control()

    def _apply_joint_control(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
            arm_joint_pos_des = arm_joint_pos_des[:, :7]
        else:
            arm_joint_pos_des = joint_pos[:, :7].clone()

        gripper_action = self.current_actions[:, -1]
        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]

        gripper_pos_des = gripper_action.unsqueeze(-1) * 0.5
        gripper_pos_des = gripper_pos_des.expand(-1, len(self._finger_joint_ids))

        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_pos_des], dim=1)
        self._robot.set_joint_position_target(joint_pos_des)

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w

        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        jacobian[:, 0:3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(self._offset_pos),
            jacobian[:, 3:, :],
        )
        jacobian[:, 3:, :] = torch.bmm(
            math_utils.matrix_from_quat(self._offset_rot),
            jacobian[:, 3:, :],
        )

        return jacobian

    def set_joint_angle(self, joint_name: str, angle: float):
        if joint_name in self.joint_angles:
            self.joint_angles[joint_name] = angle
            print(f"设置 {joint_name} 角度为 {angle} 弧度 ({angle * 180 / 3.14159:.1f} 度)")
        else:
            print(f"错误: 未知的关节名称 '{joint_name}'")
            print(f"可用的关节: {list(self.joint_angles.keys())}")

    def set_all_joint_angles(self, angles: dict):
        for joint_name, angle in angles.items():
            if joint_name in self.joint_angles:
                self.joint_angles[joint_name] = angle
            else:
                print(f"警告: 未知的关节名称 '{joint_name}'")
        print("所有关节角度已更新")

    def get_current_joint_angles(self):
        return self._robot.data.joint_pos[0].cpu().numpy()

    def _set_initial_joint_angles(self):
        print("设置机械臂初始关节角度...")
        try:
            if not hasattr(self._robot, "joint_names"):
                print("  警告: 机器人对象尚未完全初始化，跳过关节角度设置")
                return

            all_joint_names = self._robot.joint_names
            print(f"  可用关节: {all_joint_names}")

            joint_name_mapping = {
                "joint1": "panda_joint1",
                "joint2": "panda_joint2",
                "joint3": "panda_joint3",
                "joint4": "panda_joint4",
                "joint5": "panda_joint5",
                "joint6": "panda_joint6",
                "joint7": "panda_joint7",
                "finger_left": "panda_finger_joint1",
                "finger_right": "panda_finger_joint2",
            }

            for joint_name, angle in self.joint_angles.items():
                actual_joint_name = joint_name_mapping.get(joint_name, joint_name)
                if actual_joint_name in all_joint_names:
                    joint_idx = all_joint_names.index(actual_joint_name)
                    self._robot.data.joint_pos_target[:, joint_idx] = angle
                    print(
                        f"  设置 {joint_name} ({actual_joint_name}) 角度为 "
                        f"{angle:.4f} 弧度 ({angle * 180 / 3.14159:.1f} 度)"
                    )
                else:
                    print(f"  警告: 关节 {joint_name} ({actual_joint_name}) 不存在于机器人配置中")

            self._robot.write_joint_state_to_sim(
                self._robot.data.joint_pos_target,
                self._robot.data.joint_vel,
            )
            print("机械臂初始关节角度设置完成")
        except Exception as e:
            print(f"  错误: 设置关节角度时发生异常: {e}")
            print("  将使用默认关节角度继续运行")

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        wrist_rgb = self.wrist_camera.data.output["rgb"]
        # 使用相机深度（已在 GelSight 实现中做归一化到 0~255，单通道，形状 N×H×W×1）
        tactile_left_depth = self.gsmini_left.data.output.get("camera_depth")
        tactile_right_depth = self.gsmini_right.data.output.get("camera_depth")
        tactile_left_down_depth = (
            self.gsmini_left_down.data.output.get("camera_depth") if getattr(self, "gsmini_left_down", None) else None
        )
        tactile_right_down_depth = (
            self.gsmini_right_down.data.output.get("camera_depth") if getattr(self, "gsmini_right_down", None) else None
        )

        return {
            "proprio_obs": proprio_obs,
            "wrist_rgb": wrist_rgb,
            "tactile_left_depth": tactile_left_depth,
            "tactile_right_depth": tactile_right_depth,
            "tactile_left_down_depth": tactile_left_down_depth,
            "tactile_right_down_depth": tactile_right_down_depth,
        }

    def _get_rewards(self) -> torch.Tensor:
        cylinder_pos = self._cylinder.data.root_pos_w
        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        distance = torch.norm(gripper_pos - cylinder_pos, dim=-1)
        reward = torch.exp(-distance / 0.1)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return dones, dones

    def _reset_idx(self, env_ids: torch.Tensor):
        self._robot.reset(env_ids)

        cylinder_pos = torch.zeros(len(env_ids), 3, device=self.device)
        cylinder_pos[:, 0] = 0.5
        cylinder_pos[:, 1] = 0.0
        cylinder_pos[:, 2] = 0.05

        cylinder_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        cylinder_pose = torch.cat([cylinder_pos, cylinder_rot], dim=-1)
        self._cylinder.write_root_pose_to_sim(cylinder_pose, env_ids)
        self._cylinder.write_root_velocity_to_sim(
            torch.zeros_like(cylinder_pose[:, :6]), env_ids
        )


class CylinderGraspingFSMTestEnv(CylinderGraspingDemo):
    """State-machine test environment for cylinder grasping."""

    STAGE_APPROACH = 0
    STAGE_PROBE = 1
    STAGE_GRASP = 2
    STAGE_LIFT = 3

    STAGE_NAMES = {
        STAGE_APPROACH: "approach",
        STAGE_PROBE: "probe",
        STAGE_GRASP: "grasp",
        STAGE_LIFT: "lift",
    }


def run_simulator(env: CylinderGraspingDemo):
    """Runs the simulation loop."""

    tactile_render_mode = args_cli.tactile_render
    tactile_baselines: dict[str, torch.Tensor | None] = {
        "left": None,
        "right": None,
        "left_down": None,
        "right_down": None,
    }
    baseline_diff_threshold = 0.05
    baseline_area_threshold = 0.01

    def _set_tactile_render_mode(mode: str):
        show_depth = mode == "depth"
        show_rgb = mode == "rgb"
        for sensor_name in ("gsmini_left", "gsmini_right", "gsmini_left_down", "gsmini_right_down"):
            sensor = getattr(env, sensor_name, None)
            cfg = getattr(env.cfg, sensor_name, None)
            if sensor is None or cfg is None or not getattr(cfg, "debug_vis", False):
                continue
            try:
                prim = sensor._prim_view.prims[0]  # show env 0 only to avoid window spam
                depth_attr = prim.GetAttribute("debug_camera_depth")
                if depth_attr and depth_attr.IsValid():
                    depth_attr.Set(show_depth)
                rgb_attr = prim.GetAttribute("debug_tactile_rgb")
                if rgb_attr and rgb_attr.IsValid():
                    rgb_attr.Set(show_rgb)
            except Exception as e:
                print(f"[WARN] Cannot set tactile render mode for {sensor_name}: {e}")

    if getattr(env.cfg.gsmini_left, "debug_vis", False):
        _set_tactile_render_mode(tactile_render_mode)
        print(f"[INFO] Tactile render mode: {tactile_render_mode} (press V to toggle)")

    print(f"Starting cylinder grasping demo with {env.num_envs} env(s)")
    print("目标：预设姿态 + 方向键/WASD 控制 Y/Z，Q/E 控制 X 方向平移。")

    save_wrist_images = True
    wrist_images_to_save = 4
    wrist_image_env_index = 0
    wrist_image_output_dir = Path("outputs/wrist_camera_demo")
    wrist_images_saved = 0
    if save_wrist_images:
        wrist_image_output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] 腕部相机帧将保存到 {wrist_image_output_dir.resolve()} "
            f"(最多 {wrist_images_to_save} 张，env={wrist_image_env_index})"
        )

    wrist_tex_provider = None
    wrist_window = None
    if omni.ui is not None:
        try:
            wrist_tex_provider = omni.ui.TextureProvider()
            wrist_window = omni.ui.Window("Wrist Camera (env 0)", width=280, height=320)
            with wrist_window.frame:
                with omni.ui.VStack():
                    omni.ui.Label("Wrist camera feed", height=24)
                    omni.ui.ImageWithProvider(wrist_tex_provider)
            print("[INFO] Wrist camera窗口已创建，实时显示 env 0 的画面。")
        except Exception as e:
            print(f"[WARN] 无法创建腕部相机窗口: {e}")
            wrist_tex_provider = None

    env.reset()
    env._set_initial_joint_angles()
    for _ in range(60):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env.sim.render()

    ee_pos_initial, _ = env._compute_frame_pose()
    print(
        f"初始末端位置: [{ee_pos_initial[0,0]:.3f}, {ee_pos_initial[0,1]:.3f}, {ee_pos_initial[0,2]:.3f}] m"
    )

    hand_pos = env._robot.data.body_link_pos_w[:, env._body_idx]
    hand_quat = env._robot.data.body_link_quat_w[:, env._body_idx]
    wrist_offset_pos = torch.tensor(env.cfg.wrist_camera.offset.pos, device=env.device, dtype=torch.float32)
    wrist_offset_rot = torch.tensor(env.cfg.wrist_camera.offset.rot, device=env.device, dtype=torch.float32)
    wrist_offset_pos = wrist_offset_pos.unsqueeze(0).repeat(env.num_envs, 1)
    wrist_offset_rot = wrist_offset_rot.unsqueeze(0).repeat(env.num_envs, 1)
    wrist_pos, _ = math_utils.combine_frame_transforms(hand_pos, hand_quat, wrist_offset_pos, wrist_offset_rot)
    print(
        f"腕部相机位置: [{wrist_pos[0,0]:.3f}, {wrist_pos[0,1]:.3f}, {wrist_pos[0,2]:.3f}] m"
    )

    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    joint_pos_initial = env._robot.data.joint_pos[0, :7].detach().cpu().numpy()
    print("初始关节角度（弧度）：")
    for name, angle in zip(joint_names, joint_pos_initial):
        print(f"  {name}: {angle:.4f} rad ({angle * 180.0 / np.pi:.2f}°)")

    kb_step_lin_m = 0.1  # 4mm
    action_scale = getattr(env, "action_scale", 0.1)
    kb_step_units = kb_step_lin_m / max(action_scale, 1e-6)
    print(
        "保持仿真运行：↑↓/W,S 控制 Z，上下；←→/A,D 控制 Y，左右；Q/E 控制 X 前后；V 切换触觉显示(RGB/depth)。按 Ctrl+C 退出。"
    )

    # 键盘事件订阅
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_iface = carb_input.acquire_input_interface()

    # 多加 forward/backward 两个状态
    pressed = {
        "up": False,
        "down": False,
        "left": False,
        "right": False,
        "forward": False,
        "backward": False,
        "open": False,
        "close": False,
    }
    # gripper action value in "action units" (see _apply_joint_control: target = action * 0.5)
    # to fully open to ~0.04 m, need action ≈ 0.08; fully close = 0.0
    gripper_action_units = 0.04  # start from a small opening (0.02 target after *0.5)

    def _on_kb_event(event, *args, **kwargs):
        nonlocal tactile_render_mode
        if event.type in (
            carb_input.KeyboardEventType.KEY_PRESS,
            carb_input.KeyboardEventType.KEY_REPEAT,
            carb_input.KeyboardEventType.KEY_RELEASE,
        ):
            is_down = event.type in (
                carb_input.KeyboardEventType.KEY_PRESS,
                carb_input.KeyboardEventType.KEY_REPEAT,
            )
            if event.input in (carb_input.KeyboardInput.UP, carb_input.KeyboardInput.W):
                pressed["up"] = is_down
            elif event.input in (carb_input.KeyboardInput.DOWN, carb_input.KeyboardInput.S):
                pressed["down"] = is_down
            elif event.input in (carb_input.KeyboardInput.LEFT, carb_input.KeyboardInput.A):
                pressed["left"] = is_down
            elif event.input in (carb_input.KeyboardInput.RIGHT, carb_input.KeyboardInput.D):
                pressed["right"] = is_down
            # E 向前 (+X)
            elif event.input == carb_input.KeyboardInput.E:
                pressed["forward"] = is_down
            # Q 向后 (-X)
            elif event.input == carb_input.KeyboardInput.Q:
                pressed["backward"] = is_down
            # J: 打开夹爪（设为最大开度）
            elif event.input == carb_input.KeyboardInput.J and is_down:
                pressed["open"] = True
            # K: 关闭夹爪（设为最小开度）
            elif event.input == carb_input.KeyboardInput.K and is_down:
                pressed["close"] = True
            # V: toggle tactile render mode
            elif event.input == carb_input.KeyboardInput.V and event.type == carb_input.KeyboardEventType.KEY_PRESS:
                tactile_render_mode = "rgb" if tactile_render_mode == "depth" else "depth"
                _set_tactile_render_mode(tactile_render_mode)
                print(f"[INFO] Tactile render mode switched to: {tactile_render_mode}")
        return True

    kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)
    print("[INFO] Keyboard event subscription created. 点击 3D 视口，然后按 ↑↓←→ / WASD / Q/E / V。")

    step_counter = 0
    # print tactile depth stats every 0.5s (based on sim dt)
    steps_per_half_sec = max(int(0.5 / max(env.physics_dt, 1e-6)), 1)
    # print arm joint angles every ~2s
    steps_per_joint_log = max(int(2.0 / max(env.physics_dt, 1e-6)), 1)

    def _update_wrist_window():
        if wrist_tex_provider is None:
            return
        try:
            wrist_rgb = env.wrist_camera.data.output.get("rgb")
            if wrist_rgb is None or wrist_rgb.numel() == 0:
                return
            img = wrist_rgb[0]
            # convert to uint8 RGBA for UI
            if img.dtype != torch.uint8:
                img = torch.clamp(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).to(torch.uint8)
            if img.shape[-1] == 1:
                img = img.repeat(1, 1, 3)
            if img.shape[-1] == 3:
                alpha = torch.full((*img.shape[:2], 1), 255, dtype=torch.uint8, device=img.device)
                img = torch.cat([img, alpha], dim=-1)
            img_np = img.detach().cpu().numpy().copy(order="C")
            wrist_tex_provider.set_bytes_data(img_np, [img_np.shape[1], img_np.shape[0]])
        except Exception as exc:
            print(f"[WARN] 更新腕部相机窗口失败: {exc}")

    def _maybe_save_wrist_image(step_idx: int):
        nonlocal wrist_images_saved
        if not save_wrist_images or wrist_images_saved >= wrist_images_to_save:
            return
        try:
            wrist_rgb = env.wrist_camera.data.output.get("rgb")
            if wrist_rgb is None or wrist_rgb.numel() == 0:
                return
            env_id = min(wrist_image_env_index, wrist_rgb.shape[0] - 1)
            img = wrist_rgb[env_id]
            if img.dtype != torch.uint8:
                scale = 255.0 if img.max() <= 1.0 else 1.0
                img = torch.clamp(img * scale, 0, 255).to(torch.uint8)
            if img.shape[-1] == 1:
                img = img.repeat(1, 1, 3)
            filename = wrist_image_output_dir / f"wrist_step_{step_idx:05d}_env{env_id}.png"
            try:
                import imageio.v2 as imageio

                imageio.imwrite(filename, img.detach().cpu().numpy())
            except Exception:
                try:
                    from torchvision.utils import save_image

                    save_image(img.float() / 255.0, filename)
                except Exception as exc2:
                    print(f"[WARN] 保存腕部相机图片失败: {exc2}")
                    wrist_images_saved = wrist_images_to_save
                    return
            wrist_images_saved += 1
            print(f"[INFO] 已保存腕部相机图片: {filename}")
        except Exception as exc:
            print(f"[WARN] 保存腕部相机图片失败: {exc}")
            wrist_images_saved = wrist_images_to_save
    try:
        while simulation_app.is_running():
            up = pressed["up"]
            down = pressed["down"]
            left = pressed["left"]
            right = pressed["right"]
            forward = pressed["forward"]
            backward = pressed["backward"]

            # handle gripper commands (latching)
            if pressed["open"]:
                gripper_action_units = 0.08  # -> finger target ≈ 0.04 m (0.08 * 0.5)
                pressed["open"] = False
            if pressed["close"]:
                gripper_action_units = 0.0   # -> finger target ≈ 0.0 m
                pressed["close"] = False

            # X: 前后 (Q/E)
            dx_dir = (1.0 if forward else 0.0) + (-1.0 if backward else 0.0)
            # Y: 左右 (←→, A/D)
            dy_dir = (-1.0 if left else 0.0) + (1.0 if right else 0.0)
            # Z: 上下 (↑↓, W/S)
            dz_dir = (1.0 if up else 0.0) + (-1.0 if down else 0.0)

            dx_units = dx_dir * kb_step_units
            dy_units = dy_dir * kb_step_units
            dz_units = dz_dir * kb_step_units

            if hasattr(env, "manual_move_units"):
                try:
                    dx_units += float(env.manual_move_units[0, 0].item())
                    dy_units += float(env.manual_move_units[0, 1].item())
                    env.manual_move_units.zero_()
                except Exception:
                    pass

            actions = torch.zeros((env.num_envs, 5), device=env.device)
            actions[:, 0] = dx_units   # X 前后(Q/E)
            actions[:, 1] = dy_units   # Y 左右(←→/A,D)
            actions[:, 2] = dz_units   # Z 上下(↑↓/W,S)
            # yaw 保持 0；第 5 维为夹爪开合（绝对目标，见 _apply_joint_control）
            actions[:, 4] = gripper_action_units

            env._pre_physics_step(actions)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            env.sim.render()
            _update_wrist_window()
            _maybe_save_wrist_image(step_counter)

            # 每 0.5s 打印一次触觉深度统计（归一化到 [0,1]）和 4 路触觉 gate 值
            if step_counter % steps_per_half_sec == 0:
                try:
                    def _depth_mean_norm(d):
                        if d is None:
                            return torch.zeros((env.num_envs,), device=env.device)
                        dn = d.to(device=env.device, dtype=torch.float32) / 255.0
                        return dn.mean(dim=(1, 2, 3), keepdim=False)

                    def _depth_01(d):
                        if d is None:
                            return None
                        return d.to(device=env.device, dtype=torch.float32) / 255.0

                    def _sensor_gate(name: str, d):
                        current = _depth_01(d)
                        if current is None:
                            return (
                                torch.zeros((env.num_envs,), device=env.device),
                                torch.zeros((env.num_envs,), device=env.device),
                            )
                        if tactile_baselines[name] is None or tactile_baselines[name].shape != current.shape:
                            tactile_baselines[name] = current.clone()
                        diff = torch.abs(current - tactile_baselines[name])
                        ratio = (diff >= baseline_diff_threshold).to(torch.float32).mean(dim=(1, 2, 3), keepdim=False)
                        gate = (ratio > baseline_area_threshold).to(torch.float32)
                        return ratio, gate

                    sensor_depths = {
                        "left": env.gsmini_left.data.output.get("camera_depth") if hasattr(env, "gsmini_left") else None,
                        "right": env.gsmini_right.data.output.get("camera_depth") if hasattr(env, "gsmini_right") else None,
                        "left_down": (
                            env.gsmini_left_down.data.output.get("camera_depth")
                            if hasattr(env, "gsmini_left_down")
                            else None
                        ),
                        "right_down": (
                            env.gsmini_right_down.data.output.get("camera_depth")
                            if hasattr(env, "gsmini_right_down")
                            else None
                        ),
                    }

                    dmeans = {name: _depth_mean_norm(depth).mean().item() for name, depth in sensor_depths.items()}
                    ratios_and_gates = {name: _sensor_gate(name, depth) for name, depth in sensor_depths.items()}
                    ratios = {name: rg[0].mean().item() for name, rg in ratios_and_gates.items()}
                    gates = {name: rg[1].mean().item() for name, rg in ratios_and_gates.items()}
                    any_contact = max(gates.values()) if gates else 0.0

                    print(
                        f"[触觉深度] step {step_counter}: "
                        f"diff_thr={baseline_diff_threshold:.3f}, area_thr={baseline_area_threshold:.3f}, any_contact={any_contact:.0f}, "
                        f"left(dmean={dmeans['left']:.4f}, ratio={ratios['left']:.3f}, gate={gates['left']:.0f}), "
                        f"right(dmean={dmeans['right']:.4f}, ratio={ratios['right']:.3f}, gate={gates['right']:.0f}), "
                        f"left_down(dmean={dmeans['left_down']:.4f}, ratio={ratios['left_down']:.3f}, gate={gates['left_down']:.0f}), "
                        f"right_down(dmean={dmeans['right_down']:.4f}, ratio={ratios['right_down']:.3f}, gate={gates['right_down']:.0f})"
                    )
                except Exception:
                    pass

            if step_counter % 60 == 0:
                wrist_rgb = env.wrist_camera.data.output.get("rgb")
                if wrist_rgb is not None:
                    mean_val = wrist_rgb[0].float().mean().item()
                    print(f"[step {step_counter}] wrist_camera mean pixel: {mean_val:.3f}")
                print(
                    f"[step {step_counter}] kb pressed: "
                    f"up={up} down={down} left={left} right={right} "
                    f"forward={forward} backward={backward} "
                    f"-> dx={dx_units:.4f} dy={dy_units:.4f} dz={dz_units:.4f} gripper={gripper_action_units:.3f}"
                )
                gripper_joint_pos = env._robot.data.joint_pos[:, env._finger_joint_ids]
                gripper_open_single = gripper_joint_pos.mean().item()
                gripper_open_total = gripper_open_single * 2.0
                print(
                    f"[step {step_counter}] gripper opening: "
                    f"single={gripper_open_single:.4f} m, total={gripper_open_total:.4f} m"
                )
                hand_pos_loop = env._robot.data.body_link_pos_w[:, env._body_idx]
                hand_quat_loop = env._robot.data.body_link_quat_w[:, env._body_idx]
                wrist_pos_loop, _ = math_utils.combine_frame_transforms(
                    hand_pos_loop, hand_quat_loop, wrist_offset_pos, wrist_offset_rot
                )
                ee_pos_loop, _ = env._compute_frame_pose()
                print(
                    f"[step {step_counter}] 末端夹爪位置: "
                    f"[{ee_pos_loop[0,0]:.3f}, {ee_pos_loop[0,1]:.3f}, {ee_pos_loop[0,2]:.3f}] m"
                )
                print(
                    f"[step {step_counter}] 腕部相机位置: "
                    f"[{wrist_pos_loop[0,0]:.3f}, {wrist_pos_loop[0,1]:.3f}, {wrist_pos_loop[0,2]:.3f}] m"
                )
            if step_counter % steps_per_joint_log == 0:
                try:
                    joints_rad = env._robot.data.joint_pos[0, :7].detach().cpu().numpy()
                    joints_deg = np.degrees(joints_rad)
                    names = [
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
                    ]
                    angles_str = ", ".join(f"{n}: {r:.3f} rad ({d:.1f}°)" for n, r, d in zip(names, joints_rad, joints_deg))
                    print(f"[step {step_counter}] 关节角: {angles_str}")
                except Exception as e:
                    print(f"[WARN] 无法读取关节角: {e}")
            step_counter += 1
    finally:
        try:
            input_iface.unsubscribe_from_keyboard_events(keyboard, kb_sub)
        except Exception:
            pass
        env.close()
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError_Uninitialized:
            pass


def run_fsm_test_env(env: CylinderGraspingFSMTestEnv):
    """Run a 4-stage FSM control loop for grasp testing."""

    tactile_baselines: dict[str, torch.Tensor | None] = {
        "left": None,
        "right": None,
        "left_down": None,
        "right_down": None,
    }
    baseline_diff_threshold = 0.05
    baseline_area_threshold = 0.01

    near_probe_distance_m = 0.085
    lifted_height_delta_m = 0.02

    action_scale = max(float(getattr(env, "action_scale", 0.1)), 1e-6)
    approach_offset = torch.tensor([0.0, 0.0, 0.11], device=env.device, dtype=torch.float32).unsqueeze(0)
    probe_offset = torch.tensor([0.0, 0.0, 0.075], device=env.device, dtype=torch.float32).unsqueeze(0)
    grasp_offset = torch.tensor([0.0, 0.0, 0.08], device=env.device, dtype=torch.float32).unsqueeze(0)
    lift_offset = torch.tensor([0.0, 0.0, 0.18], device=env.device, dtype=torch.float32).unsqueeze(0)

    def _depth_01(depth_img: torch.Tensor | None) -> torch.Tensor | None:
        if depth_img is None:
            return None
        return depth_img.to(device=env.device, dtype=torch.float32) / 255.0

    def _sensor_gate(name: str, depth_img: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        current = _depth_01(depth_img)
        if current is None:
            ratio = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
            gate = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
            return ratio, gate

        if tactile_baselines[name] is None or tactile_baselines[name].shape != current.shape:
            tactile_baselines[name] = current.clone()

        diff = torch.abs(current - tactile_baselines[name])
        ratio = (diff >= baseline_diff_threshold).to(torch.float32).mean(dim=(1, 2, 3), keepdim=False)
        gate = ratio > baseline_area_threshold
        return ratio, gate

    def _delta_to_action_units(delta_xyz: torch.Tensor, max_step_m: float) -> torch.Tensor:
        delta_clamped = torch.clamp(delta_xyz, min=-max_step_m, max=max_step_m)
        return delta_clamped / action_scale

    def _apply_stage_action(
        actions: torch.Tensor,
        mask: torch.Tensor,
        ee_tip_pos_w: torch.Tensor,
        cylinder_pos_w: torch.Tensor,
        offset: torch.Tensor,
        max_step_m: float,
        gripper_value: float,
    ):
        if not bool(mask.any()):
            return
        target_pos = cylinder_pos_w + offset
        delta = target_pos[mask] - ee_tip_pos_w[mask]
        actions[mask, :3] = _delta_to_action_units(delta, max_step_m=max_step_m)
        actions[mask, 4] = gripper_value

    env.reset()
    env._set_initial_joint_angles()
    for _ in range(60):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env.sim.render()

    object_init_height = env._cylinder.data.root_pos_w[:, 2].clone()
    prev_stage = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    steps_per_log = max(int(0.5 / max(env.physics_dt, 1e-6)), 1)
    kb_step_lin_m = 0.02
    kb_step_units = kb_step_lin_m / action_scale

    print(f"Starting FSM test environment with {env.num_envs} env(s)")
    print("FSM stages: 0=approach, 1=probe, 2=grasp, 3=lift")
    print("FSM+键盘：↑↓/W,S 控制Z；←→/A,D 控制Y；Q/E 控制X；J/K 强制夹爪开/合。")

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_iface = carb_input.acquire_input_interface()
    pressed = {
        "up": False,
        "down": False,
        "left": False,
        "right": False,
        "forward": False,
        "backward": False,
        "open": False,
        "close": False,
    }

    def _on_kb_event(event, *args, **kwargs):
        if event.type in (
            carb_input.KeyboardEventType.KEY_PRESS,
            carb_input.KeyboardEventType.KEY_REPEAT,
            carb_input.KeyboardEventType.KEY_RELEASE,
        ):
            is_down = event.type in (
                carb_input.KeyboardEventType.KEY_PRESS,
                carb_input.KeyboardEventType.KEY_REPEAT,
            )
            if event.input in (carb_input.KeyboardInput.UP, carb_input.KeyboardInput.W):
                pressed["up"] = is_down
            elif event.input in (carb_input.KeyboardInput.DOWN, carb_input.KeyboardInput.S):
                pressed["down"] = is_down
            elif event.input in (carb_input.KeyboardInput.LEFT, carb_input.KeyboardInput.A):
                pressed["left"] = is_down
            elif event.input in (carb_input.KeyboardInput.RIGHT, carb_input.KeyboardInput.D):
                pressed["right"] = is_down
            elif event.input == carb_input.KeyboardInput.E:
                pressed["forward"] = is_down
            elif event.input == carb_input.KeyboardInput.Q:
                pressed["backward"] = is_down
            elif event.input == carb_input.KeyboardInput.J:
                pressed["open"] = is_down
            elif event.input == carb_input.KeyboardInput.K:
                pressed["close"] = is_down
        return True

    kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)
    print("[INFO] FSM 测试环境键盘订阅已创建。点击 3D 视口后再按键。")

    try:
        step_counter = 0
        while simulation_app.is_running():
            up = pressed["up"]
            down = pressed["down"]
            left = pressed["left"]
            right = pressed["right"]
            forward = pressed["forward"]
            backward = pressed["backward"]

            dx_dir = (1.0 if forward else 0.0) + (-1.0 if backward else 0.0)
            dy_dir = (-1.0 if left else 0.0) + (1.0 if right else 0.0)
            dz_dir = (1.0 if up else 0.0) + (-1.0 if down else 0.0)

            manual_dx_units = dx_dir * kb_step_units
            manual_dy_units = dy_dir * kb_step_units
            manual_dz_units = dz_dir * kb_step_units

            if hasattr(env, "manual_move_units"):
                try:
                    manual_dx_units += float(env.manual_move_units[0, 0].item())
                    manual_dy_units += float(env.manual_move_units[0, 1].item())
                    env.manual_move_units.zero_()
                except Exception:
                    pass

            sensor_depths = {
                "left": env.gsmini_left.data.output.get("camera_depth") if hasattr(env, "gsmini_left") else None,
                "right": env.gsmini_right.data.output.get("camera_depth") if hasattr(env, "gsmini_right") else None,
                "left_down": (
                    env.gsmini_left_down.data.output.get("camera_depth")
                    if hasattr(env, "gsmini_left_down")
                    else None
                ),
                "right_down": (
                    env.gsmini_right_down.data.output.get("camera_depth")
                    if hasattr(env, "gsmini_right_down")
                    else None
                ),
            }

            sensor_ratios: dict[str, torch.Tensor] = {}
            sensor_gates: dict[str, torch.Tensor] = {}
            for sensor_name, depth_img in sensor_depths.items():
                ratio, gate = _sensor_gate(sensor_name, depth_img)
                sensor_ratios[sensor_name] = ratio
                sensor_gates[sensor_name] = gate

            hand_pos_w = env._robot.data.body_link_pos_w[:, env._body_idx]
            hand_quat_w = env._robot.data.body_link_quat_w[:, env._body_idx]
            ee_tip_pos_w, _ = math_utils.combine_frame_transforms(
                hand_pos_w, hand_quat_w, env._offset_pos, env._offset_rot
            )
            cylinder_pos_w = env._cylinder.data.root_pos_w

            inner_contact = sensor_gates["left"] | sensor_gates["right"]
            bottom_contact = sensor_gates["left_down"] | sensor_gates["right_down"]
            near_probe_zone = torch.norm(ee_tip_pos_w - cylinder_pos_w, dim=-1) < near_probe_distance_m
            object_lifted = cylinder_pos_w[:, 2] > (object_init_height + lifted_height_delta_m)

            # Priority:
            # if object_lifted -> lift
            # elif inner_contact -> grasp
            # elif near_probe_zone or bottom_contact -> probe
            # else -> approach
            stage = torch.full(
                (env.num_envs,),
                fill_value=CylinderGraspingFSMTestEnv.STAGE_APPROACH,
                dtype=torch.long,
                device=env.device,
            )
            stage = torch.where(
                near_probe_zone | bottom_contact,
                torch.full_like(stage, CylinderGraspingFSMTestEnv.STAGE_PROBE),
                stage,
            )
            stage = torch.where(
                inner_contact,
                torch.full_like(stage, CylinderGraspingFSMTestEnv.STAGE_GRASP),
                stage,
            )
            stage = torch.where(
                object_lifted,
                torch.full_like(stage, CylinderGraspingFSMTestEnv.STAGE_LIFT),
                stage,
            )

            actions = torch.zeros((env.num_envs, 5), device=env.device)
            actions[:, 3] = 0.0

            _apply_stage_action(
                actions,
                mask=stage == CylinderGraspingFSMTestEnv.STAGE_APPROACH,
                ee_tip_pos_w=ee_tip_pos_w,
                cylinder_pos_w=cylinder_pos_w,
                offset=approach_offset,
                max_step_m=0.006,
                gripper_value=0.08,
            )
            _apply_stage_action(
                actions,
                mask=stage == CylinderGraspingFSMTestEnv.STAGE_PROBE,
                ee_tip_pos_w=ee_tip_pos_w,
                cylinder_pos_w=cylinder_pos_w,
                offset=probe_offset,
                max_step_m=0.003,
                gripper_value=0.08,
            )
            _apply_stage_action(
                actions,
                mask=stage == CylinderGraspingFSMTestEnv.STAGE_GRASP,
                ee_tip_pos_w=ee_tip_pos_w,
                cylinder_pos_w=cylinder_pos_w,
                offset=grasp_offset,
                max_step_m=0.002,
                gripper_value=0.0,
            )
            _apply_stage_action(
                actions,
                mask=stage == CylinderGraspingFSMTestEnv.STAGE_LIFT,
                ee_tip_pos_w=ee_tip_pos_w,
                cylinder_pos_w=cylinder_pos_w,
                offset=lift_offset,
                max_step_m=0.008,
                gripper_value=0.0,
            )

            # Manual keyboard increments are additive on top of FSM motion.
            actions[:, 0] += manual_dx_units
            actions[:, 1] += manual_dy_units
            actions[:, 2] += manual_dz_units
            if pressed["open"]:
                actions[:, 4] = 0.08
            elif pressed["close"]:
                actions[:, 4] = 0.0

            env._pre_physics_step(actions)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            env.sim.render()

            changed = stage != prev_stage
            if bool(changed.any()):
                changed_ids = torch.nonzero(changed, as_tuple=False).squeeze(-1).tolist()
                for env_id in changed_ids:
                    stage_id = int(stage[env_id].item())
                    print(f"[FSM] env={env_id} step={step_counter} -> stage {stage_id} ({env.STAGE_NAMES[stage_id]})")
                prev_stage = stage.clone()

            if step_counter % steps_per_log == 0:
                env0 = 0
                stage0 = int(stage[env0].item())
                dist0 = torch.norm(ee_tip_pos_w[env0] - cylinder_pos_w[env0]).item()
                print(
                    f"[FSM] step={step_counter} stage={stage0}({env.STAGE_NAMES[stage0]}), "
                    f"lifted={int(object_lifted[env0].item())}, "
                    f"inner_contact={int(inner_contact[env0].item())}, "
                    f"bottom_contact={int(bottom_contact[env0].item())}, "
                    f"near_probe_zone={int(near_probe_zone[env0].item())}, "
                    f"dist={dist0:.4f}, "
                    f"ratio_l={sensor_ratios['left'][env0].item():.3f}, "
                    f"ratio_r={sensor_ratios['right'][env0].item():.3f}, "
                    f"ratio_ld={sensor_ratios['left_down'][env0].item():.3f}, "
                    f"ratio_rd={sensor_ratios['right_down'][env0].item():.3f}, "
                    f"manual(dx={manual_dx_units:.3f}, dy={manual_dy_units:.3f}, dz={manual_dz_units:.3f})"
                )

            step_counter += 1
    finally:
        try:
            input_iface.unsubscribe_from_keyboard_events(keyboard, kb_sub)
        except Exception:
            pass
        env.close()
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError_Uninitialized:
            pass


def main():
    env_cfg = CylinderGraspingDemoCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis
    env_cfg.gsmini_right.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, "gsmini_left_down"):
        env_cfg.gsmini_left_down.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, "gsmini_right_down"):
        env_cfg.gsmini_right_down.debug_vis = args_cli.debug_vis

    if args_cli.fsm_test:
        experiment = CylinderGraspingFSMTestEnv(env_cfg)
        run_fsm_test_env(env=experiment)
    else:
        experiment = CylinderGraspingDemo(env_cfg)
        run_simulator(env=experiment)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
