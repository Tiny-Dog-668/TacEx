from __future__ import annotations

import argparse
from datetime import datetime
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
    "--terminal_input",
    action="store_true",
    default=False,
    help="Enable terminal keyboard fallback (not used now, kept for compatibility).",
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
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tacex import GelSightSensor
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg


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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.02]),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.06,
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
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )

    # robot configuration
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
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
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            activate_contact_sensors=True,
            rigid_props=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.rigid_props.replace(
                disable_gravity=True,
            ),
        ),
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

    left_finger_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/cylinder"],
    )
    right_finger_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/cylinder"],
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
        data_types=["tactile_rgb"],
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

    gsmini_left_outer = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_outer",
    )
    gsmini_right_outer = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_outer",
    )

    gsmini_left_out = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_out",
    )
    gsmini_right_out = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_out",
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
            "joint1": -0.3136,
            "joint2": -0.4448,
            "joint3": 0.3858,
            "joint4": -3.0236,
            "joint5": 0.2319,
            "joint6": 2.5776,
            "joint7": 0.6249,
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

        self.left_finger_contact = ContactSensor(self.cfg.left_finger_contact)
        self.scene.sensors["left_finger_contact"] = self.left_finger_contact
        self.right_finger_contact = ContactSensor(self.cfg.right_finger_contact)
        self.scene.sensors["right_finger_contact"] = self.right_finger_contact

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

        # outer sensors
        try:
            self.gsmini_left_outer = GelSightSensor(self.cfg.gsmini_left_outer)
            self.scene.sensors["gsmini_left_outer"] = self.gsmini_left_outer
        except Exception as e:
            print(f"[WARN] Skipping gsmini_left_outer: {e}")
            self.gsmini_left_outer = None
        try:
            self.gsmini_right_outer = GelSightSensor(self.cfg.gsmini_right_outer)
            self.scene.sensors["gsmini_right_outer"] = self.gsmini_right_outer
        except Exception as e:
            print(f"[WARN] Skipping gsmini_right_outer: {e}")
            self.gsmini_right_outer = None

        # "out" sensors
        try:
            self.gsmini_left_out = GelSightSensor(self.cfg.gsmini_left_out)
            self.scene.sensors["gsmini_left_out"] = self.gsmini_left_out
        except Exception as e:
            print(f"[WARN] Skipping gsmini_left_out: {e}")
            self.gsmini_left_out = None
        try:
            self.gsmini_right_out = GelSightSensor(self.cfg.gsmini_right_out)
            self.scene.sensors["gsmini_right_out"] = self.gsmini_right_out
        except Exception as e:
            print(f"[WARN] Skipping gsmini_right_out: {e}")
            self.gsmini_right_out = None

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
        tactile_left = self.gsmini_left.data.output["tactile_rgb"]
        tactile_right = self.gsmini_right.data.output["tactile_rgb"]

        lf = self.left_finger_contact.data.net_forces_w
        rf = self.right_finger_contact.data.net_forces_w
        if lf.ndim == 3 and lf.shape[1] == 1:
            lf = lf.squeeze(1)
        if rf.ndim == 3 and rf.shape[1] == 1:
            rf = rf.squeeze(1)

        return {
            "proprio_obs": proprio_obs,
            "wrist_rgb": wrist_rgb,
            "tactile_left": tactile_left,
            "tactile_right": tactile_right,
            "contact_force_left": lf,
            "contact_force_right": rf,
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
        cylinder_pos[:, 2] = 0.03

        cylinder_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        cylinder_pose = torch.cat([cylinder_pos, cylinder_rot], dim=-1)
        self._cylinder.write_root_pose_to_sim(cylinder_pose, env_ids)
        self._cylinder.write_root_velocity_to_sim(
            torch.zeros_like(cylinder_pose[:, :6]), env_ids
        )


def run_simulator(env: CylinderGraspingDemo):
    """Runs the simulation loop."""

    # tactile debug
    if env.cfg.gsmini_left.debug_vis:
        for data_type in env.cfg.gsmini_left.data_types:
            env.gsmini_left._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
    if env.cfg.gsmini_right.debug_vis:
        for data_type in env.cfg.gsmini_right.data_types:
            env.gsmini_right._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)

    # outer sensors
    if (
        hasattr(env, "gsmini_left_outer")
        and env.gsmini_left_outer is not None
        and hasattr(env.cfg, "gsmini_left_outer")
        and env.cfg.gsmini_left_outer
        and env.cfg.gsmini_left_outer.debug_vis
    ):
        for data_type in env.cfg.gsmini_left_outer.data_types:
            try:
                env.gsmini_left_outer._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            except Exception as e:
                print(f"[WARN] Cannot enable debug for gsmini_left_outer {data_type}: {e}")
    if (
        hasattr(env, "gsmini_right_outer")
        and env.gsmini_right_outer is not None
        and hasattr(env.cfg, "gsmini_right_outer")
        and env.cfg.gsmini_right_outer
        and env.cfg.gsmini_right_outer.debug_vis
    ):
        for data_type in env.cfg.gsmini_right_outer.data_types:
            try:
                env.gsmini_right_outer._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            except Exception as e:
                print(f"[WARN] Cannot enable debug for gsmini_right_outer {data_type}: {e}")

    if (
        hasattr(env, "gsmini_left_out")
        and env.gsmini_left_out is not None
        and hasattr(env.cfg, "gsmini_left_out")
        and env.cfg.gsmini_left_out
        and env.cfg.gsmini_left_out.debug_vis
    ):
        for data_type in env.cfg.gsmini_left_out.data_types:
            try:
                env.gsmini_left_out._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            except Exception as e:
                print(f"[WARN] Cannot enable debug for gsmini_left_out {data_type}: {e}")
    if (
        hasattr(env, "gsmini_right_out")
        and env.gsmini_right_out is not None
        and hasattr(env.cfg, "gsmini_right_out")
        and env.cfg.gsmini_right_out
        and env.cfg.gsmini_right_out.debug_vis
    ):
        for data_type in env.cfg.gsmini_right_out.data_types:
            try:
                env.gsmini_right_out._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            except Exception as e:
                print(f"[WARN] Cannot enable debug for gsmini_right_out {data_type}: {e}")

    print(f"Starting cylinder grasping demo with {env.num_envs} env(s)")
    print("目标：预设姿态 + 方向键/WASD 控制 Y/Z，Q/E 控制 X 方向平移。")

    # Wrist camera frame dump to a unique subfolder per run (for easy comparison)
    save_wrist_images = True
    wrist_images_to_save = 6
    wrist_image_env_index = 0
    # 放在仓库根目录下的 outputs，并按序号新建子目录
    base_dir = Path(__file__).resolve().parents[2] / "outputs"
    prefix = "wrist_camera_demo"
    idx = 0
    while (base_dir / f"{prefix}{idx}").exists():
        idx += 1
    wrist_image_output_dir = base_dir / f"{prefix}{idx}"
    wrist_images_saved = 0
    if save_wrist_images:
        wrist_image_output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] 腕部相机帧将保存到 {wrist_image_output_dir.resolve()} "
            f"(最多 {wrist_images_to_save} 张，env={wrist_image_env_index})"
        )

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
    print("保持仿真运行：↑↓/W,S 控制 Z，上下；←→/A,D 控制 Y，左右；Q/E 控制 X 前后。按 Ctrl+C 退出。")

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
        return True

    kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)
    print("[INFO] Keyboard event subscription created. 点击 3D 视口，然后按 ↑↓←→ / WASD / Q/E。")

    step_counter = 0
    print_every = max(1, int(round(0.1 / env.physics_dt)))
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
            _maybe_save_wrist_image(step_counter)

            if step_counter % print_every == 0:
                lf = env.left_finger_contact.data.net_forces_w
                rf = env.right_finger_contact.data.net_forces_w
                if lf.ndim == 3 and lf.shape[1] == 1:
                    lf = lf.squeeze(1)
                if rf.ndim == 3 and rf.shape[1] == 1:
                    rf = rf.squeeze(1)
                print(f"[step {step_counter}] contact L:", lf[0].tolist(), "R:", rf[0].tolist())
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
            step_counter += 1
    finally:
        try:
            input_iface.unsubscribe_from_keyboard_events(keyboard, kb_sub)
        except Exception:
            pass
        env.close()
        pynvml.nvmlShutdown()


def main():
    env_cfg = CylinderGraspingDemoCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis
    env_cfg.gsmini_right.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, "gsmini_left_outer"):
        env_cfg.gsmini_left_outer.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, "gsmini_right_outer"):
        env_cfg.gsmini_right_outer.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, "gsmini_left_out"):
        env_cfg.gsmini_left_out.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, "gsmini_right_out"):
        env_cfg.gsmini_right_out.debug_vis = args_cli.debug_vis

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
