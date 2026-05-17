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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tacex import GelSightSensor

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

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

    # plate
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
        ),
    )

    # cylinder object
    cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.05]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
            scale=(0.5, 0.5, 1.0),  # Make it more cylindrical
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
    )

    # robot configuration
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # wrist camera
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
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

    # IK controller
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    episode_length_s = 0
    action_space = 0
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
        # self.ik_commands[:, 3:] = torch.tensor([0,1,0,0],device=self.device)

        self.step_count = 0
        
        # 关节角度控制参数 - 可以在这里修改各个关节的角度
        self.joint_angles = {
            'joint1': 0.0,      # 关节1角度 (弧度)
            'joint2': -0.5,     # 关节2角度 (弧度)
            'joint3': 0.0,      # 关节3角度 (弧度)
            'joint4': -2.0,     # 关节4角度 (弧度)
            'joint5': 0.0,      # 关节5角度 (弧度)
            'joint6': 2.0,      # 关节6角度 (弧度)
            'joint7': 0.5,      # 关节7角度 (弧度)
            'finger_left': 0.02,  # 左手指角度 (弧度)
            'finger_right': 0.02, # 右手指角度 (弧度)
        }
        
        # 关节角度变化控制参数
        self.joint_angle_ranges = {
            'joint1': (-3.14, 3.14),    # 关节角度范围 (弧度)
            'joint2': (-1.57, 1.57),
            'joint3': (-3.14, 3.14),
            'joint4': (-3.14, 3.14),
            'joint5': (-3.14, 3.14),
            'joint6': (-3.14, 3.14),
            'joint7': (-3.14, 3.14),
            'finger_left': (0.0, 0.04),
            'finger_right': (0.0, 0.04),
        }
        
        # 关节角度变化速度 (弧度/步)
        self.joint_velocities = {
            'joint1': 0.01,
            'joint2': 0.01,
            'joint3': 0.01,
            'joint4': 0.01,
            'joint5': 0.01,
            'joint6': 0.01,
            'joint7': 0.01,
            'finger_left': 0.001,
            'finger_right': 0.001,
        }
        
        # 关节角度变化模式
        self.movement_mode = "random"  # "static", "sinusoidal", "random", "sequence"
        self.time_step = 0

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)


    def _setup_scene(self):

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self._cylinder

        self.wrist_camera = Camera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera 

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

        RigidObject(self.cfg.plate)

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
        # 设置IK控制器命令
        self._ik_controller.set_command(self.ik_commands)
        
        # 应用关节控制
        self._apply_joint_control()

    def _apply_joint_control(self):
        """Apply joint control to the robot."""
        # 根据运动模式更新关节角度
        self._update_joint_angles()
        
        # 从self.joint_angles字典中获取目标关节角度
        target_joint_angles = torch.tensor([
            [self.joint_angles['joint1'], self.joint_angles['joint2'], 
             self.joint_angles['joint3'], self.joint_angles['joint4'],
             self.joint_angles['joint5'], self.joint_angles['joint6'], 
             self.joint_angles['joint7'],
             self.joint_angles['finger_left'], self.joint_angles['finger_right']]
        ], device=self.device)
        
        # 直接设置关节目标位置
        self._robot.set_joint_position_target(target_joint_angles)
        
        # 更新步数
        self.time_step += 1
        
        # 方法2: 使用IK控制 (基于末端执行器位置) - 更精确的方法
        # 取消注释下面的代码来使用IK控制
        """
        # 设置目标末端执行器位置和姿态
        target_pos = torch.tensor([[0.5, 0.0, 0.3]], device=self.device)  # 目标位置
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)  # 目标姿态
        
        # 更新IK命令
        self.ik_commands[:, :3] = target_pos
        self.ik_commands[:, 3:] = target_quat
        
        # 计算IK解
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        jacobian = self._compute_frame_jacobian()
        joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        
        # 设置手指位置
        joint_pos_des[0, self._finger_joint_ids[0]] = 0.02  # 左手指
        joint_pos_des[0, self._finger_joint_ids[1]] = 0.02  # 右手指
        
        # 应用关节目标位置
        self._robot.set_joint_position_target(joint_pos_des)
        """

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

    def _update_joint_angles(self):
        """根据运动模式更新关节角度"""
        if self.movement_mode == "static":
            # 静态模式：保持当前角度不变
            pass
        elif self.movement_mode == "sinusoidal":
            # 正弦波模式：关节角度按正弦波变化
            for joint_name in self.joint_angles:
                if joint_name.startswith('joint'):
                    # 计算正弦波角度
                    frequency = 0.1  # 频率
                    amplitude = 1.0  # 振幅
                    phase = self.time_step * frequency
                    angle_offset = amplitude * np.sin(phase)
                    
                    # 限制在关节范围内
                    min_angle, max_angle = self.joint_angle_ranges[joint_name]
                    base_angle = (min_angle + max_angle) / 2
                    new_angle = base_angle + angle_offset
                    self.joint_angles[joint_name] = np.clip(new_angle, min_angle, max_angle)
        elif self.movement_mode == "random":
            # 随机模式：随机改变关节角度
            for joint_name in self.joint_angles:
                if self.time_step % 100 == 0:  # 每100步改变一次
                    min_angle, max_angle = self.joint_angle_ranges[joint_name]
                    self.joint_angles[joint_name] = np.random.uniform(min_angle, max_angle)
        elif self.movement_mode == "sequence":
            # 序列模式：按预定义序列运动
            sequence_steps = [
                {'joint1': 0.0, 'joint2': -0.5, 'joint3': 0.0, 'joint4': -2.0, 'joint5': 0.0, 'joint6': 2.0, 'joint7': 0.5},
                {'joint1': 0.5, 'joint2': -1.0, 'joint3': 0.3, 'joint4': -1.5, 'joint5': 0.2, 'joint6': 1.5, 'joint7': 0.8},
                {'joint1': -0.5, 'joint2': -0.3, 'joint3': -0.3, 'joint4': -2.5, 'joint5': -0.2, 'joint6': 2.5, 'joint7': 0.2},
                {'joint1': 0.0, 'joint2': -0.5, 'joint3': 0.0, 'joint4': -2.0, 'joint5': 0.0, 'joint6': 2.0, 'joint7': 0.5},
            ]
            
            if self.time_step % 200 == 0:  # 每200步切换到下一个序列
                sequence_idx = (self.time_step // 200) % len(sequence_steps)
                for joint_name, angle in sequence_steps[sequence_idx].items():
                    if joint_name in self.joint_angles:
                        self.joint_angles[joint_name] = angle

    def set_joint_angle(self, joint_name: str, angle: float):
        """设置指定关节的角度
        
        Args:
            joint_name: 关节名称 ('joint1' 到 'joint7', 'finger_left', 'finger_right')
            angle: 角度值 (弧度)
        """
        if joint_name in self.joint_angles:
            self.joint_angles[joint_name] = angle
            print(f"设置 {joint_name} 角度为 {angle} 弧度 ({angle * 180 / 3.14159:.1f} 度)")
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
        print("所有关节角度已更新")

    def get_current_joint_angles(self):
        """获取当前关节角度"""
        return self._robot.data.joint_pos[0].cpu().numpy()

    def set_movement_mode(self, mode: str):
        """设置运动模式
        
        Args:
            mode: 运动模式 ("static", "sinusoidal", "random", "sequence")
        """
        if mode in ["static", "sinusoidal", "random", "sequence"]:
            self.movement_mode = mode
            print(f"运动模式已设置为: {mode}")
        else:
            print(f"错误: 未知的运动模式 '{mode}'")
            print("可用的模式: static, sinusoidal, random, sequence")

    def set_joint_velocity(self, joint_name: str, velocity: float):
        """设置关节角度变化速度
        
        Args:
            joint_name: 关节名称
            velocity: 速度 (弧度/步)
        """
        if joint_name in self.joint_velocities:
            self.joint_velocities[joint_name] = velocity
            print(f"设置 {joint_name} 速度为 {velocity} 弧度/步")
        else:
            print(f"错误: 未知的关节名称 '{joint_name}'")

    def reset_to_home_position(self):
        """重置到初始位置"""
        self.joint_angles = {
            'joint1': 0.0,
            'joint2': -0.5,
            'joint3': 0.0,
            'joint4': -2.0,
            'joint5': 0.0,
            'joint6': 2.0,
            'joint7': 0.5,
            'finger_left': 0.02,
            'finger_right': 0.02,
        }
        print("机器人已重置到初始位置")

    def print_joint_status(self):
        """打印当前关节状态"""
        print("=== 当前关节状态 ===")
        for joint_name, angle in self.joint_angles.items():
            degrees = angle * 180 / np.pi
            print(f"{joint_name}: {angle:.3f} 弧度 ({degrees:.1f} 度)")
        print(f"运动模式: {self.movement_mode}")
        print(f"时间步: {self.time_step}")
        print("==================")

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
    

    print(f"Starting Franka robot movement demo with {env.num_envs} envs")
    
    env.reset()
    
    # 演示不同的运动模式
    print("=== Franka 机器人运动演示 ===")
    print("可用的运动模式:")
    print("1. static - 静态模式 (保持当前角度)")
    print("2. sinusoidal - 正弦波模式 (关节按正弦波运动)")
    print("3. random - 随机模式 (随机改变关节角度)")
    print("4. sequence - 序列模式 (按预定义序列运动)")
    
    # 设置初始运动模式
    env.set_movement_mode("sinusoidal")  # 开始使用正弦波模式
    
    # 打印初始关节状态
    env.print_joint_status()
    
    # 演示：每1000步切换运动模式
    mode_switch_interval = 1000
    current_mode_index = 0
    modes = ["sinusoidal", "random", "sequence", "static"]

    # Simulation loop
    step_count = 0
    while simulation_app.is_running():
        # toggle flags
        env._window.reset = False
        env._window.new_action = False
        # env.reset()

        # 每1000步切换运动模式
        if step_count % mode_switch_interval == 0 and step_count > 0:
            current_mode_index = (current_mode_index + 1) % len(modes)
            new_mode = modes[current_mode_index]
            env.set_movement_mode(new_mode)
            print(f"\n=== 切换到运动模式: {new_mode} ===")
            env.print_joint_status()

        # 每500步打印一次关节状态
        if step_count % 500 == 0:
            print(f"\n--- 步数: {step_count} ---")
            env.print_joint_status()

        # perform physics step
        env._pre_physics_step(torch.zeros(1, device=env.device))
        env.scene.write_data_to_sim()
        env.sim.step(render=False)

        # update isaac buffers() -> also updates sensors
        env.scene.update(dt=env.physics_dt)
        # render scene for cameras (used by sensor)
        env.sim.render()
        
        step_count += 1

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
