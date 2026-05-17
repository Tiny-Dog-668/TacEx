from __future__ import annotations

import argparse

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
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
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
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


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

    
    # plate - 修复为正确的盘子配置
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
            scale=(10, 10, 0.01),  # 宽而薄的盘子形状
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,  # 增加位置迭代
                solver_velocity_iteration_count=4,   # 增加速度迭代
                max_angular_velocity=100.0,          # 降低最大角速度
                max_linear_velocity=10.0,            # 降低最大线速度
                max_depenetration_velocity=1.0,      # 降低最大穿透速度
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,  # 减小接触偏移
                rest_offset=0.0005,    # 增加静止偏移
            ),
        ),
    )

    cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.02]),
        spawn=sim_utils.UsdFileCfg(
            func=spawn_ycb_can,
            usd_path=YCB_CAN_USD_PATH,
            scale=YCB_CAN_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,  # 增加位置迭代
                solver_velocity_iteration_count=4,   # 增加速度迭代
                max_angular_velocity=100.0,          # 降低最大角速度
                max_linear_velocity=10.0,            # 降低最大线速度
                max_depenetration_velocity=1.0,      # 降低最大穿透速度
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,  # 减小接触偏移
                rest_offset=0.0005,    # 增加静止偏移，避免过度接触
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
                "panda_joint1": -0.4510,
                "panda_joint2": 0.1890,
                "panda_joint3": 0.4750,
                "panda_joint4": -2.3660,
                "panda_joint5": -0.2250,
                "panda_joint6": 2.5450,
                "panda_joint7": 0.9910,
                "panda_finger_joint.*": 0.02,
            },
        ),
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            rigid_props=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.rigid_props.replace(
                disable_gravity=True,
            ),
        ),
    )

    # wrist camera
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        update_period=0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=40.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0, -0.12),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )

    # third-person camera (fixed to env root)
    third_person_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=30.0,
            focus_distance=400.0,
            horizontal_aperture=40.0,
            clipping_range=(0.05, 30.0),
        ),
        # Place the camera up and back relative to env origin (ROS convention)
        offset=CameraCfg.OffsetCfg(
            pos=(0.7, 0.0, 1.1),
            # Look along negative X (rotate 180 deg around Z)
            rot=(0.69888, 0.10757, 0.10757, 0.69888),#（W,X,Y,Z）
            convention="opengl",
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
        debug_vis=True,  # for rendering sensor output in the gui
        # update Taxim cfg
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"],  # marker_motion
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

    # add two outer tactile sensors (mirror inner settings)
    gsmini_left_outer = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_outer",
    )
    gsmini_right_outer = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_outer",
    )



    # IK controller - 使用相对模式，仿照cylinder_grasping_privileged.py
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,  # 使用相对模式
        ik_method="dls",  # 可选: 'pinv', 'svd', 'trans', 'dls'
        # 添加更多参数以提高精度
    )
    episode_length_s = 0
    action_space = 5  # 5维动作空间：xyz + rz + gripper
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

        # Get end-effector body index
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # Index of fingers -> first id is left, second id is right finger
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.

        self._jacobi_body_idx = self._body_idx - 1

        # ee offset w.r.t panda hand -> based on the asset
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # ---

        # create buffer to store actions (= ik_commands)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        
        # 动作空间配置 - 仿照cylinder_grasping_privileged.py
        self.action_scale = 0.1  # 动作缩放因子
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        
        # 当前动作状态
        self.current_actions = torch.zeros((self.num_envs, 5), device=self.device)  # [dx, dy, dz, droll, gripper]

        self.step_count = 0
        
        # 关节角度控制参数 - 可在这里修改关节初始角度
        self.joint_angles = {

            'joint1': -0.4510,
            'joint2': 0.1890,
            'joint3': 0.4750,
            'joint4': -2.3660,
            'joint5': -0.2250,
            'joint6': 2.5450,
            'joint7': 0.9910,
            
            'finger_left': 0.02,
            'finger_right': 0.02,
        }

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)


    def _setup_scene(self):

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self._cylinder

        # plate
        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        self.wrist_camera = Camera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera 
        # third-person camera
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

        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left

        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

        # outer sensors (only if prims exist); prevent crash if USD lacks these prims
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

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )


        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step."""
        # 存储当前动作
        self.current_actions = actions.clone()
        
        # 仿照cylinder_grasping_privileged.py的动作处理
        # 将5维动作扩展为6维给IK控制器
        # 获取当前末端执行器位姿
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        
        # 构造6维动作：dx, dy, dz, droll, dpitch, dyaw
        # 我们的5维动作：[dx, dy, dz, drz, gripper] (第4维是绕Z轴旋转)
        # 扩展为6维：[dx, dy, dz, droll=0, dpitch=0, dyaw=drz]
        self.processed_actions[:, :3] = self.current_actions[:, :3] * self.action_scale  # dx, dy, dz
        self.processed_actions[:, 3] = 0.0  # droll = 0 (绕X轴旋转)
        self.processed_actions[:, 4] = 0.0  # dpitch = 0 (绕Y轴旋转)
        self.processed_actions[:, 5] = self.current_actions[:, 3] * self.action_scale *10   # dyaw (绕Z轴旋转)
        
        # 设置IK命令（相对模式）
        self._ik_controller.set_command(self.processed_actions, ee_pos_curr_b, ee_quat_curr_b)
        
        # 应用关节控制
        self._apply_joint_control()

    def _apply_joint_control(self):
        """Apply joint control to the robot using IK controller."""
        # 仿照cylinder_grasping_privileged.py设计：使用IK控制器计算关节位置
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        
        # compute the delta in joint-space for arm joints only
        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            # IK controller returns all joint positions, but we only need the first 7 (arm joints)
            arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
            # Extract only the arm joint positions (first 7)
            arm_joint_pos_des = arm_joint_pos_des[:, :7]
        else:
            arm_joint_pos_des = joint_pos[:, :7].clone()
        
        # 设置夹爪动作（第5维是夹爪动作）
        gripper_action = self.current_actions[:, -1]  # 第5维是夹爪动作
        
        # 获取当前夹爪关节位置
        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]
        
        # 转换夹爪动作为位置（假设动作在[-1, 1]范围内）
        # -1 = 完全打开, 1 = 完全闭合
        # Franka夹爪有2个手指关节，都应该一起移动
        gripper_pos_des = gripper_action.unsqueeze(-1) * 0.5  # 缩放到夹爪范围
        gripper_pos_des = gripper_pos_des.expand(-1, len(self._finger_joint_ids))  # 扩展到两个手指关节
        
        # 合并机械臂和夹爪关节目标
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_pos_des], dim=1)
        
        # 应用关节目标位置
        self._robot.set_joint_position_target(joint_pos_des)

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame."""
        # obtain quantities from simulation
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        
        # account for the offset
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame."""
        # 获取基座雅可比矩阵
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]
        
        # 转换到基座坐标系
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        
        # 考虑末端执行器偏移量
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        
        return jacobian

    def set_joint_angle(self, joint_name: str, angle: float):
        """设置指定关节的角度
        
        Args:
            joint_name: 关节名称 ('joint1' 到 'joint7', 'finger_left', 'finger_right')
            angle: 角度值 (弧度)
        """
        if joint_name in self.joint_angles:
            self.joint_angles[joint_name] = angle
        else:
            print(f"错误: 未知的关节名称 '{joint_name}'")
            print(f"可用的关节: {list(self.joint_angles.keys())}")

    def set_all_joint_angles(self, angles: dict):
        """批量设置所有关节角度
        
        Args:
            angles: 关节角度字典，例如 {'joint1': 0.5, 'joint2': -1.0, ...}
        """
        for joint_name, angle in angles.items():
            if joint_name in self.joint_angles:
                self.joint_angles[joint_name] = angle
            else:
                print(f"警告: 未知的关节名称 '{joint_name}'")

    def get_current_joint_angles(self):
        """获取当前关节角度"""    # print("平移后关节角度（弧度）：")
    # for name, angle in zip(joint_names, moved_joint_angles.numpy()):
    #     print(f"  {name}: {angle:.4f} rad ({angle * 180.0 / np.pi:.2f}°)")
        return self._robot.data.joint_pos[0].cpu().numpy()
    
    def _set_initial_joint_angles(self):
        """设置机械臂初始关节角度"""
        try:
            # 检查机器人是否已正确初始化
            if not hasattr(self._robot, 'joint_names'):
                print("  警告: 机器人对象尚未完全初始化，跳过关节角度设置")
                return
                
            # 获取所有关节名称
            all_joint_names = self._robot.joint_names
            
            # 关节名称映射：从我们的命名到实际关节名称
            joint_name_mapping = {
                'joint1': 'panda_joint1',
                'joint2': 'panda_joint2', 
                'joint3': 'panda_joint3',
                'joint4': 'panda_joint4',
                'joint5': 'panda_joint5',
                'joint6': 'panda_joint6',
                'joint7': 'panda_joint7',
                'finger_left': 'panda_finger_joint1',
                'finger_right': 'panda_finger_joint2'
            }
            
            # 设置每个关节的角度
            for joint_name, angle in self.joint_angles.items():
                actual_joint_name = joint_name_mapping.get(joint_name, joint_name)
                
                if actual_joint_name in all_joint_names:
                    # 直接设置关节目标位置
                    joint_idx = all_joint_names.index(actual_joint_name)
                    self._robot.data.joint_pos_target[:, joint_idx] = angle
                else:
                    print(f"  警告: 关节 {joint_name} ({actual_joint_name}) 不存在于机器人配置中")
            
            # 应用关节目标位置到仿真
            self._robot.write_joint_state_to_sim(
                self._robot.data.joint_pos_target,
                self._robot.data.joint_vel
            )
        except Exception as e:
            print(f"  错误: 设置关节角度时发生异常: {e}")
            print("  将使用默认关节角度继续运行")
    
    def move_end_effector(self, delta_xyz: tuple[float, float, float], settle_steps: int = 120) -> torch.Tensor:
        """Translate the end-effector by ``delta_xyz`` (meters) using the IK controller and return joint angles."""
        delta = torch.zeros((self.num_envs, 6), device=self.device)
        delta[:, 0] = delta_xyz[0]
        delta[:, 1] = delta_xyz[1]
        delta[:, 2] = delta_xyz[2]

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        self._ik_controller.set_command(delta, ee_pos_curr_b, ee_quat_curr_b)

        joint_pos = self._robot.data.joint_pos[:, :]
        jacobian = self._compute_frame_jacobian()
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        arm_joint_pos_des = arm_joint_pos_des[:, :7]

        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_joint_pos], dim=1)
        self._robot.set_joint_position_target(joint_pos_des)

        for _ in range(settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self.sim.render()
        return self._robot.data.joint_pos[0, :7].detach().cpu()

    def rotate_end_effector(self, delta_rpy: tuple[float, float, float], settle_steps: int = 120) -> torch.Tensor:
        """Rotate the end-effector by ``delta_rpy`` = (droll, dpitch, dyaw) in radians using relative IK.

        Returns the updated 7 arm joint angles (CPU tensor).
        """
        delta = torch.zeros((self.num_envs, 6), device=self.device)
        # fill rotational components only (relative mode)
        delta[:, 3] = delta_rpy[0]
        delta[:, 4] = delta_rpy[1]
        delta[:, 5] = delta_rpy[2]

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        self._ik_controller.set_command(delta, ee_pos_curr_b, ee_quat_curr_b)

        joint_pos = self._robot.data.joint_pos[:, :]
        jacobian = self._compute_frame_jacobian()
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        arm_joint_pos_des = arm_joint_pos_des[:, :7]

        # keep current gripper joints
        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_joint_pos], dim=1)
        self._robot.set_joint_position_target(joint_pos_des)

        for _ in range(settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self.sim.render()
        return self._robot.data.joint_pos[0, :7].detach().cpu()

    def move_and_rotate_end_effector(
        self,
        delta_xyz: tuple[float, float, float],
        delta_rpy: tuple[float, float, float],
        settle_steps: int = 120,
    ) -> torch.Tensor:
        """Apply a combined (dx, dy, dz, droll, dpitch, dyaw) relative command via IK.

        Returns the updated 7 arm joint angles (CPU tensor).
        """
        delta = torch.zeros((self.num_envs, 6), device=self.device)
        delta[:, 0] = delta_xyz[0]
        delta[:, 1] = delta_xyz[1]
        delta[:, 2] = delta_xyz[2]
        delta[:, 3] = delta_rpy[0]
        delta[:, 4] = delta_rpy[1]
        delta[:, 5] = delta_rpy[2]

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        self._ik_controller.set_command(delta, ee_pos_curr_b, ee_quat_curr_b)

        joint_pos = self._robot.data.joint_pos[:, :]
        jacobian = self._compute_frame_jacobian()
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        arm_joint_pos_des = arm_joint_pos_des[:, :7]

        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_joint_pos], dim=1)
        self._robot.set_joint_position_target(joint_pos_des)

        for _ in range(settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self.sim.render()
        return self._robot.data.joint_pos[0, :7].detach().cpu()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get observations from the environment."""
        # Proprioceptive observations
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)
        
        # Vision observations
        wrist_rgb = self.wrist_camera.data.output["rgb"]
        tactile_left = self.gsmini_left.data.output["tactile_rgb"]
        tactile_right = self.gsmini_right.data.output["tactile_rgb"]
        
        return {
            "proprio_obs": proprio_obs,
            "wrist_rgb": wrist_rgb,
            "tactile_left": tactile_left,
            "tactile_right": tactile_right,
        }

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards for the current state."""
        # Simple reward based on distance to cylinder
        cylinder_pos = self._cylinder.data.root_pos_w
        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        
        distance = torch.norm(gripper_pos - cylinder_pos, dim=-1)
        reward = torch.exp(-distance / 0.1)
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate done flags."""
        # No episode termination for demo
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return dones, dones

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at the given indices."""
        # Reset robot
        self._robot.reset(env_ids)
        
        # Reset cylinder position
        cylinder_pos = torch.zeros(len(env_ids), 3, device=self.device)
        cylinder_pos[:, 0] = 0.5 + torch.randn(len(env_ids), device=self.device) * 0.1
        cylinder_pos[:, 1] = torch.randn(len(env_ids), device=self.device) * 0.1
        cylinder_pos[:, 2] = 0.05
        
        cylinder_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        
        # Combine position and rotation into pose tensor
        cylinder_pose = torch.cat([cylinder_pos, cylinder_rot], dim=-1)  # Shape: (len(env_ids), 7)
        self._cylinder.write_root_pose_to_sim(cylinder_pose, env_ids)
        self._cylinder.write_root_velocity_to_sim(
            torch.zeros_like(cylinder_pose[:,:6]), env_ids
        )


def run_simulator(env: CylinderGraspingDemo):
    """Runs the simulation loop."""

    # 用于显示tactile图像
    if env.cfg.gsmini_left.debug_vis:
        for data_type in env.cfg.gsmini_left.data_types:
            env.gsmini_left._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            
    if env.cfg.gsmini_right.debug_vis:
        for data_type in env.cfg.gsmini_right.data_types:
            env.gsmini_right._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
    # outer sensors (if available): require both cfg enabled and sensor instance present
    if (
        hasattr(env, 'gsmini_left_outer')
        and env.gsmini_left_outer is not None
        and hasattr(env.cfg, 'gsmini_left_outer')
        and env.cfg.gsmini_left_outer
        and env.cfg.gsmini_left_outer.debug_vis
    ):
        for data_type in env.cfg.gsmini_left_outer.data_types:
            try:
                env.gsmini_left_outer._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            except Exception as e:
                print(f"[WARN] Cannot enable debug for gsmini_left_outer {data_type}: {e}")
    if (
        hasattr(env, 'gsmini_right_outer')
        and env.gsmini_right_outer is not None
        and hasattr(env.cfg, 'gsmini_right_outer')
        and env.cfg.gsmini_right_outer
        and env.cfg.gsmini_right_outer.debug_vis
    ):
        for data_type in env.cfg.gsmini_right_outer.data_types:
            try:
                env.gsmini_right_outer._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            except Exception as e:
                print(f"[WARN] Cannot enable debug for gsmini_right_outer {data_type}: {e}")
    

    print(f"Starting cylinder grasping demo with {env.num_envs} env(s)")

    env.reset()
    
    # 设置机械臂初始关节角度
    env._set_initial_joint_angles()
    # 让仿真前进几步，确保初始角度应用
    for _ in range(60):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env.sim.render()


    zero_actions = torch.zeros((env.num_envs, 5), device=env.device)
    while simulation_app.is_running():
        env._pre_physics_step(zero_actions)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        env.sim.render()

    env.close()
    pynvml.nvmlShutdown()


def main():
    """Main function."""
    # Define simulation env
    env_cfg = CylinderGraspingDemoCfg()
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis
    env_cfg.gsmini_right.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, 'gsmini_left_outer'):
        env_cfg.gsmini_left_outer.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, 'gsmini_right_outer'):
        env_cfg.gsmini_right_outer.debug_vis = args_cli.debug_vis

    experiment = CylinderGraspingDemo(env_cfg)

    
    # Run the simulator
    run_simulator(env=experiment)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim apply
        simulation_app.close()
