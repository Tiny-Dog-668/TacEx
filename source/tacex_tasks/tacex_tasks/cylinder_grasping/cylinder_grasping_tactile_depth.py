# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cylinder grasping environment with privileged state observations and 4 GelSight depth sensors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# for Domain Randomization
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_error_magnitude,
    sample_uniform,
    wrap_to_pi,
)
from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelCfg,
    UniformNoiseCfg,
    gaussian_noise,
)
from tacex import GelSightSensor
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
from torch.cuda.amp import autocast

# from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)

from tacex_tasks.utils import DirectLiveVisualizer


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
class CylinderGraspingFourTactileDepthCfg(DirectRLEnvCfg):
    """Configuration for the cylinder grasping environment with 4 tactile depth sensors (no vision cameras)."""

    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    debug_vis = False
    ui_window_class_type = CustomEnvWindow

    decimation = 1
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            solver_type=1,
            max_position_iteration_count=32,
            max_velocity_iteration_count=4,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,  # 使用4个环境进行测试
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.03]),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,  # 圆柱体半径 3cm（原 2cm 的 1.5 倍）
            height=0.06,  # 圆柱体高度 6cm（原 4cm 的 1.5 倍）
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
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),  # 红色圆柱体
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )

    # robot configuration - 学习ball_rolling_tactile_rgb.py的固定初始位置
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        # Use the 4-pad GelSight Mini asset (adds *_down pads expected by this env).
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.down.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                'panda_joint1': -0.3136,
                'panda_joint2': -0.4448,
                'panda_joint3': 0.3858,
                'panda_joint4': -3.0236,
                'panda_joint5': 0.2319,
                'panda_joint6': 2.5776,
                'panda_joint7': 0.6249,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    tactile_img_res_hw = (96, 128)
    
    gsmini_left: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
    )
    gsmini_left.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left.data_types = ["camera_depth"]
    # only use depth; drop marker motion simulation
    gsmini_left.marker_motion_sim_cfg = None
    # set tactile image resolution explicitly (no try/except needed as default optical cfg exists)
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_right: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )
    gsmini_right.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right.data_types = ["camera_depth"]
    gsmini_right.marker_motion_sim_cfg = None
    gsmini_right.optical_sim_cfg = gsmini_right.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    # additional down-facing tactile sensors (suffix _down)
    gsmini_left_down: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_down",
    )
    gsmini_left_down.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left_down.data_types = ["camera_depth"]
    gsmini_left_down.marker_motion_sim_cfg = None
    gsmini_left_down.optical_sim_cfg = gsmini_left_down.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_right_down: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_down",
    )
    gsmini_right_down.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right_down.data_types = ["camera_depth"]
    gsmini_right_down.marker_motion_sim_cfg = None
    gsmini_right_down.optical_sim_cfg = gsmini_right_down.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    # IK controller - 优化配置
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose", #是求解关节角度还是位姿
        use_relative_mode=True,  # 使用相对模式，期望6维动作
        ik_method="dls",  # 使用阻尼最小二乘法，更稳定
    )

    # noise models
    gaussian_noise_cfg = GaussianNoiseCfg(mean=0.0, std=0.002, operation="add")

    # environment settings
    episode_length_s = 5.0  # 这个会被转换为步数
    max_episode_length = 300  # 直接设置最大步数
    action_space = 5  # 5维动作空间
    # 动作含义：
    # [0] dx: X方向位移增量 (m)
    # [1] dy: Y方向位移增量 (m) 
    # [2] dz: Z方向位移增量 (m)
    # [3] dyaw: 绕Z轴旋转增量 (rad) - 只控制yaw，roll和pitch固定为0
    # [4] gripper: 夹爪开合动作 (m)
    # 注意：前4维给IK控制器（xyz + yaw），第5维直接控制夹爪
    observation_space = {
        "proprio_obs": 18,  # joint positions and velocities (9 joints * 2)
        "tactile_left_depth_resnet": 128,
        "tactile_right_depth_resnet": 128,
        "tactile_left_down_depth_resnet": 128,
        "tactile_right_down_depth_resnet": 128,
        "critic_cylinder_pos": 3,
        "critic_cylinder_quat": 4,
        "critic_cylinder_lin_vel": 3,
        "critic_cylinder_ang_vel": 3,
        "critic_gripper_pos": 3,
        "critic_gripper_quat": 4,
        "critic_gripper_lin_vel": 3,
        "critic_gripper_ang_vel": 3,
        "critic_target_pos": 3,
        "critic_target_distance": 1,
    }
    state_space = 0
    action_scale = 0.05  # [m] for position, [rad] for orientation - 减小单步动作
    
    # reward configuration
    reach_sigma = 0.1
    reach_weight = 5.0
    minimal_lift_height = 0.04
    lift_weight = 15.0
    penalty_schedule_steps = 150_000
    action_rate_weight_initial = -1e-4
    action_rate_weight_final = -1e-1
    joint_vel_weight_initial = -1e-4
    joint_vel_weight_final = -1e-1
    success_height = 0.10
    success_reward_weight = 100.0
    # tactile contact reward (inner sensors only: left/right)
    tactile_contact_weight_inner = 15.0
    tactile_contact_threshold = 0.8  # depth_norm threshold in [0,1] to count a pixel as contact
    tactile_ratio_scale = 0.01  # contact ratio scale before tanh
    
    # 安全最小离板间隙（用于动作门控与奖励扣分）
    safety_plate_clearance = 0.002
    # 指尖几何相对 link 原点的近似 z 偏移（m）——保守估计
    finger_tip_offset = 0.02
    # 板面相对板中心的 z 偏移（m）（block.usd 厚度 0.01 -> 顶面偏移 ~0.005）
    plate_top_offset = 0.005

    # task specific parameters 
    cylinder_height = 0.06  # 匹配CylinderCfg中的高度（放大 1.5x）
    cylinder_radius = 0.03  # 匹配CylinderCfg中的半径（放大 1.5x）
    # 用于相对抬升目标的参考（与 done 判定解耦，done 使用 success_height 绝对阈值）
    lift_height = 0.06
    ground_height = 0.0  # 地面高度，用于判断机械臂关节是否碰撞地面

    # randomization
    cylinder_pos_range = 0.00  # plate half-width for random placement
    cylinder_rot_range = 0.00
    
    # 机械臂随机化配置（打破并行环境的同步起步）
    robot_joint_pos_noise = 0.02  # 关节位置随机化范围 (弧度)
    robot_joint_vel_noise = 0.00  # 关节速度随机化范围
    action_noise_scale = 0.01  # 动作噪声缩放
    reset_jitter_max_steps = 15  # 重置后给 episode 计数随机抖动，错峰超时
    reward_print_interval = 200


class CylinderGraspingFourTactileDepthEnv(DirectRLEnv):
    """Cylinder grasping environment with privileged state and 4 tactile depth sensors."""

    cfg: CylinderGraspingFourTactileDepthCfg

    def __init__(self, cfg: CylinderGraspingFourTactileDepthCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        self.step_count = 0
        self.reward_print_interval = self.cfg.reward_print_interval

        # --- For IK actions ---
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_idx - 1

        # ee offset w.r.t panda hand -> based on the asset
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # create auxiliary variables for computing applied action, observations and rewards
        self.processed_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.action_history = torch.zeros_like(self.processed_actions)

        # Lightweight 1-channel tactile encoder shared for four tactile streams -> 128-D
        # Conv blocks with adaptive pooling to be resolution-agnostic.
        self._tactile_feature_dim = 128
        self._tactile_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self._tactile_feature_dim),
            nn.ReLU(inplace=True),
        ).to(self.device)
        self._tactile_encoder.eval()
        for p in self._tactile_encoder.parameters():
            p.requires_grad_(False)

        # Index of fingers -> first id is left, second id is right finger
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])
        self._left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self._right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        # add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Initialize cylinder positions randomly on the plate
        self._initialize_cylinder_positions()
        # 每个环境的真实 spawn 高度（考虑 z 抖动与 env 原点偏移）
        self._cylinder_spawn_height_per_env = self._cylinder.data.root_pos_w[:, 2].clone()
        self._lift_target_height_per_env = self._cylinder_spawn_height_per_env + self.cfg.lift_height

        # （门禁逻辑已移除）

    def _setup_scene(self):
        """Setup the scene."""
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # cylinder
        self._cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self._cylinder

        # plate
        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        # clone environments first
        self.scene.clone_environments(copy_from_source=False)

        # tactile sensors
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right
        # additional down-facing sensors
        self.gsmini_left_down = GelSightSensor(self.cfg.gsmini_left_down)
        self.scene.sensors["gsmini_left_down"] = self.gsmini_left_down
        self.gsmini_right_down = GelSightSensor(self.cfg.gsmini_right_down)
        self.scene.sensors["gsmini_right_down"] = self.gsmini_right_down

        # add frame transformer for end-effector
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

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _initialize_cylinder_positions(self):
        """Initialize cylinder positions randomly on the plate for all environments - 只在xy平面随机，保持z轴高度不变."""
        # 获取默认根状态并修改位置
        cylinder_state = self._cylinder.data.default_root_state.clone()
        
        # 只在xy平面添加随机偏移，保持z轴高度不变
        xy_offset = sample_uniform(-self.cfg.cylinder_pos_range, self.cfg.cylinder_pos_range, (self.num_envs, 2), self.device)
        cylinder_state[:, :2] += xy_offset  # 只修改x和y坐标
        
        # 轻微 z 抖动与随机 yaw，打破完全一致的初态
        cylinder_state[:, 2] += sample_uniform(-0.002, 0.002, (self.num_envs,), self.device)
        rand_yaw = sample_uniform(-self.cfg.cylinder_rot_range, self.cfg.cylinder_rot_range, (self.num_envs,), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        cylinder_state[:, 3:7] = rand_quat
        
        # 添加环境原点偏移（包含z轴）
        cylinder_state[:, :3] += self.scene.env_origins
        
        # 写入仿真
        self._cylinder.write_root_state_to_sim(cylinder_state, torch.arange(self.num_envs, device=self.device))

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step."""
        # store actions for reward computation
        self.actions = actions.clone()

        # scale actions
        self.processed_actions = self.actions * self.cfg.action_scale
        # sanitize processed actions (avoid NaN/Inf propagating into IK)
        finite_mask = torch.isfinite(self.processed_actions)
        self.processed_actions = torch.where(finite_mask, self.processed_actions, torch.zeros_like(self.processed_actions))
        # conservative clamp
        self.processed_actions = torch.clamp(self.processed_actions, -self.cfg.action_scale, self.cfg.action_scale)

        # 添加动作噪声以增加多样性（打破并行环境同步动作）
        action_noise = sample_uniform(
            -self.cfg.action_noise_scale,
            self.cfg.action_noise_scale,
            (self.num_envs, self.cfg.action_space),
            self.device,
        )
        # 不对姿态的 roll/pitch 施加噪声：
        # - 目前动作维度为 [dx, dy, dz, dyaw, gripper]，无 roll/pitch 维度
        # - 为避免通过 yaw 噪声间接影响 roll/pitch（IK 耦合/数值效应），此处也移除对 yaw 的噪声
        noise_mask = torch.ones((self.num_envs, self.cfg.action_space), device=self.device)
        noise_mask[:, 3] = 0.0  # 关闭 dyaw 噪声
        self.processed_actions += action_noise * noise_mask

        # set commands for IK controller (only arm joint actions, not gripper)
        # 将5维动作转换为6维：前4维给IK控制器，第5维给夹爪
        # 构造6维动作：xyz + rpy，其中 roll 和 pitch 设为 0，yaw 来自第 4 维动作
        arm_actions = torch.zeros((self.num_envs, 6), device=self.device)
        arm_actions[:, :3] = self.processed_actions[:, :3]  # xyz
        arm_actions[:, 3] = 0.0  # roll = 0
        arm_actions[:, 4] = 0.0  # pitch = 0  
        arm_actions[:, 5] = self.processed_actions[:, 3]  # yaw (第4维动作)
        
        # 获取当前末端执行器位姿（相对模式需要）
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()

        # yaw 不做自动纠偏，仅由策略控制

        # 设置IK命令（相对模式需要传递当前位姿）
        # 动作门控（更宽松）：
        # - 当指尖接近板面但不在物体附近时，阻止进一步下探；
        # - 当接近物体时，允许小幅下探以促成接触与夹持；
        finger_l = self._robot.data.body_link_pos_w[:, self._left_finger_body_idx]
        finger_r = self._robot.data.body_link_pos_w[:, self._right_finger_body_idx]
        min_finger_tip_z = torch.minimum(finger_l[:, 2], finger_r[:, 2]) - self.cfg.finger_tip_offset
        plate_top_z = self._plate.data.root_pos_w[:, 2] + self.cfg.plate_top_offset
        # 末端与圆柱在 XY 平面上的距离（用于判断是否已对准）
        ee_xy = self._robot.data.body_link_pos_w[:, self._body_idx, :2]
        obj_xy = self._cylinder.data.root_pos_w[:, :2]
        near_obj_xy = torch.norm(obj_xy - ee_xy, dim=-1) < (2.0 * self.cfg.cylinder_radius)
        near_plate = min_finger_tip_z < (plate_top_z + self.cfg.safety_plate_clearance)
        if (near_plate & (~near_obj_xy)).any():
            dz = arm_actions[:, 2]
            dz = torch.where(near_plate & (~near_obj_xy), torch.clamp(dz, min=0.0), dz)
            arm_actions[:, 2] = dz
        # 若已在物体上方且接近板面，允许小幅负向 dz（避免“悬停学不会接触”）
        if (near_plate & near_obj_xy).any():
            dz = arm_actions[:, 2]
            dz = torch.where(near_plate & near_obj_xy, torch.clamp(dz, min=-0.005), dz)
            arm_actions[:, 2] = dz
        # 近板面限制 yaw，避免刮板引发姿态突变
        if near_plate.any():
            dyaw = arm_actions[:, 5]
            dyaw = torch.where(near_plate, dyaw * 0.2, dyaw)
            arm_actions[:, 5] = dyaw

        self._ik_controller.set_command(arm_actions, ee_pos_curr_b, ee_quat_curr_b)
        # 暂存上一时刻动作作为历史特征
        self.action_history = self.prev_actions.clone()

    def _apply_action(self):
        """Apply actions to the robot using IK controller."""
        # obtain quantities from simulation
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        
        # compute the delta in joint-space for arm joints（始终计算，避免零范数分支造成行为差异）
        jacobian = self._compute_frame_jacobian()
        # IK controller expects all joint positions, but only updates the first 7 (arm joints)
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        # Extract only the arm joint positions (first 7)
        arm_joint_pos_des = arm_joint_pos_des[:, :7]
        # sanitize IK output
        arm_joint_pos_des = torch.where(torch.isfinite(arm_joint_pos_des), arm_joint_pos_des, joint_pos[:, :7])
        # clamp to soft joint limits
        joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, :7, 0]
        joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, :7, 1]
        arm_joint_pos_des = torch.max(arm_joint_pos_des, joint_lower_limits.unsqueeze(0))
        arm_joint_pos_des = torch.min(arm_joint_pos_des, joint_upper_limits.unsqueeze(0))
        
        # handle gripper action (last action)
        # 将夹爪动作解释为“增量”而非“绝对位置”，使策略更易从任意开度收拢/放开
        # action ∈ [-1, 1]，负数趋向闭合（减小开度），正数趋向张开（增大开度）
        gripper_action = self.processed_actions[:, -1]

        # 当前夹爪关节位置（两指应相同）
        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]

        # 增量步长（m）：每步最多 5 mm 变化，避免数值不稳定与反复冲击
        gripper_delta = gripper_action.unsqueeze(-1) * 0.2
        gripper_pos_des = gripper_joint_pos + gripper_delta
        gripper_pos_des = torch.where(torch.isfinite(gripper_pos_des), gripper_pos_des, gripper_joint_pos)
        gripper_pos_des = torch.clamp(gripper_pos_des, 0.0, 0.04)
        
        # combine arm and gripper joint targets
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_pos_des], dim=1)
        # final safety: replace non-finite with current joint positions
        joint_pos_des = torch.where(torch.isfinite(joint_pos_des), joint_pos_des, joint_pos)

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

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Get observations from the environment."""
        # Proprioceptive observations
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        # 特权信息观察 - 圆柱体信息
        cylinder_pos = self._cylinder.data.root_pos_w
        cylinder_quat = self._cylinder.data.root_quat_w
        cylinder_lin_vel = self._cylinder.data.root_lin_vel_w
        cylinder_ang_vel = self._cylinder.data.root_ang_vel_w

        # 特权信息观察 - 夹爪信息
        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        gripper_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        gripper_lin_vel = self._robot.data.body_link_lin_vel_w[:, self._body_idx]
        gripper_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._body_idx]

        # Target information
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        # normalize quaternions used in computation path
        def _norm_quat(q: torch.Tensor) -> torch.Tensor:
            n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
            return q / n
        hand_quat = _norm_quat(hand_quat)
        offset_rot = _norm_quat(self._offset_rot)
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, offset_rot
        )
        target_pos_relative = cylinder_pos - ee_pos  # 相对位置
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        # Tactile depth from GelSight sensors (left/right + left_down/right_down); use camera_depth in [0,255]
        tact_l_raw = self.gsmini_left.data.output.get("camera_depth")
        tact_r_raw = self.gsmini_right.data.output.get("camera_depth")
        tact_ld_raw = self.gsmini_left_down.data.output.get("camera_depth") if hasattr(self, "gsmini_left_down") else None
        tact_rd_raw = self.gsmini_right_down.data.output.get("camera_depth") if hasattr(self, "gsmini_right_down") else None
        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (240, 320))
        def _prep_depth_1ch(depth_tensor):
            # expected NHWC with C=1; convert to float [0,1], resize if needed; return NCHW with C=1
            if depth_tensor is None:
                return torch.zeros((self.num_envs, 1, target_h, target_w), dtype=torch.float32, device=self.device)
            d = depth_tensor.to(device=self.device, dtype=torch.float32) / 255.0  # NHWC1 in [0,1]
            dn = d.permute(0, 3, 1, 2).contiguous()  # NCHW (C=1)
            if dn.shape[2] != target_h or dn.shape[3] != target_w:
                dn = F.interpolate(dn, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return dn

        tact_l = _prep_depth_1ch(tact_l_raw)
        tact_r = _prep_depth_1ch(tact_r_raw)
        tact_ld = _prep_depth_1ch(tact_ld_raw)
        tact_rd = _prep_depth_1ch(tact_rd_raw)

        # （门禁逻辑已移除）

        # Tactile features (using FP16 autocast on CUDA if available)
        # 判断是否为 CUDA 设备（self.device 可能是 torch.device 或 str）
        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        use_amp = (dev_type == "cuda")

        with torch.no_grad(), autocast(enabled=use_amp, dtype=torch.float16):
            fl = self._tactile_encoder(tact_l).view(self.num_envs, self._tactile_feature_dim)
            fr = self._tactile_encoder(tact_r).view(self.num_envs, self._tactile_feature_dim)
            fld = self._tactile_encoder(tact_ld).view(self.num_envs, self._tactile_feature_dim)
            frd = self._tactile_encoder(tact_rd).view(self.num_envs, self._tactile_feature_dim)

        obs = {
            "proprio_obs": proprio_obs,
            "tactile_left_depth_resnet": fl,
            "tactile_right_depth_resnet": fr,
            "tactile_left_down_depth_resnet": fld,
            "tactile_right_down_depth_resnet": frd,
            "critic_cylinder_pos": cylinder_pos,
            "critic_cylinder_quat": cylinder_quat,
            "critic_cylinder_lin_vel": cylinder_lin_vel,
            "critic_cylinder_ang_vel": cylinder_ang_vel,
            "critic_gripper_pos": gripper_pos,
            "critic_gripper_quat": gripper_quat,
            "critic_gripper_lin_vel": gripper_lin_vel,
            "critic_gripper_ang_vel": gripper_ang_vel,
            "critic_target_pos": target_pos_relative,
            "critic_target_distance": target_distance,
        }
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards based on reaching, lifting, action rate, and joint velocity penalties."""
        cylinder_pos = self._cylinder.data.root_pos_w
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )

        # reaching_object: r_reach = 1 - tanh(||p_obj - p_ee|| / sigma)
        sigma = max(self.cfg.reach_sigma, 1e-6)
        reach_distance = torch.norm(cylinder_pos - ee_pos, dim=-1)
        reach_reward = 1.0 - torch.tanh(reach_distance / sigma)

        # lifting_object: base reward above h_min + linear reward for height above h_min
        h_min = self.cfg.minimal_lift_height
        height = cylinder_pos[:, 2]
        base_reward = (height > h_min).float()
        denom = max(self.cfg.success_height - h_min, 1e-6)
        linear_reward = torch.clamp((height - h_min) / denom, 0.0, 1.0)
        lift_reward = base_reward * 0.5 + linear_reward

        # success bonus when object height exceeds success_height
        success_reward = (cylinder_pos[:, 2] > self.cfg.success_height).float()

        # action_rate penalty: use sanitized, scaled actions to avoid explosion
        action_curr = torch.where(torch.isfinite(self.processed_actions), self.processed_actions, torch.zeros_like(self.processed_actions))
        action_prev = torch.where(torch.isfinite(self.prev_actions), self.prev_actions, torch.zeros_like(self.prev_actions))
        action_diff = action_curr - action_prev
        action_rate_penalty = torch.sum(action_diff * action_diff, dim=-1)

        # joint_vel penalty: sum_j omega_j^2
        joint_vel = self._robot.data.joint_vel
        joint_vel = torch.where(torch.isfinite(joint_vel), joint_vel, torch.zeros_like(joint_vel))
        joint_vel_penalty = torch.sum(joint_vel * joint_vel, dim=-1)

        # schedule for penalty weights
        schedule_reached = self.step_count >= self.cfg.penalty_schedule_steps
        action_weight = (
            self.cfg.action_rate_weight_final if schedule_reached else self.cfg.action_rate_weight_initial
        )
        joint_weight = (
            self.cfg.joint_vel_weight_final if schedule_reached else self.cfg.joint_vel_weight_initial
        )

        # tactile contact reward (inner tactile sensors: left/right)
        tact_l = self.gsmini_left.data.output.get("camera_depth") if hasattr(self, "gsmini_left") else None
        tact_r = self.gsmini_right.data.output.get("camera_depth") if hasattr(self, "gsmini_right") else None

        def _to_01(depth_tensor: torch.Tensor) -> torch.Tensor:
            d = depth_tensor.to(device=self.device, dtype=torch.float32)
            if d.dtype == torch.uint8 or (d.numel() > 0 and d.max() > 2.0):
                d = d / 255.0
            return d

        def _contact_ratio_from_depth(depth_tensor, thr: float) -> torch.Tensor:
            if depth_tensor is None:
                return torch.zeros((self.num_envs,), device=self.device)
            d = _to_01(depth_tensor)
            contact_px = (d < thr).float()
            return contact_px.mean(dim=(1, 2, 3), keepdim=False)

        thr = float(getattr(self.cfg, "tactile_contact_threshold", 0.8))
        ratio_l = _contact_ratio_from_depth(tact_l, thr)
        ratio_r = _contact_ratio_from_depth(tact_r, thr)
        ratio_scale = float(getattr(self.cfg, "tactile_ratio_scale", 0.01))
        s_l = torch.tanh(ratio_l / (ratio_scale + 1e-6))
        s_r = torch.tanh(ratio_r / (ratio_scale + 1e-6))
        tactile_contact_inner = 0.5 * (s_l + s_r)
        tactile_weight = float(getattr(self.cfg, "tactile_contact_weight_inner", 0.0))

        rewards = (
            self.cfg.reach_weight * reach_reward
            + self.cfg.lift_weight * lift_reward
            + self.cfg.success_reward_weight * success_reward
            + tactile_weight * tactile_contact_inner
            + action_weight * action_rate_penalty
            + joint_weight * joint_vel_penalty
        )

        if (
            self.reward_print_interval > 0
            and (self.step_count + 1) % self.reward_print_interval == 0
        ):
            print(
                f"[奖励] step {self.step_count + 1}: "
                f"reach={reach_reward.mean().item():.3f} "
                f"(w={self.cfg.reach_weight}), lift={lift_reward.mean().item():.3f} "
                f"(w={self.cfg.lift_weight}), success={success_reward.mean().item():.3f} "
                f"(w={self.cfg.success_reward_weight}), tactile_inner={tactile_contact_inner.mean().item():.3f} "
                f"(w={tactile_weight}), "
                f"action_penalty={action_rate_penalty.mean().item():.6f} "
                f"(w={action_weight}), joint_penalty={joint_vel_penalty.mean().item():.6f} "
                f"(w={joint_weight}), total={rewards.mean().item():.3f}"
            )

        self.step_count += 1
        self.prev_actions = action_curr.detach().clone()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate done flags."""
        # Episode timeout - 基于步数检查
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = time_out.bool()

        # Success condition - cylinder lifted above absolute threshold (与奖励中的 success_height 一致)
        current_height = self._cylinder.data.root_pos_w[:, 2]
        success = current_height > self.cfg.success_height
        success = success.bool()

        # 失败条件 - 机械臂关节碰撞地面
        # 获取机械臂所有关节的世界坐标位置
        joint_positions = self._robot.data.body_link_pos_w  # [num_envs, num_links, 3]
        
        # 检查是否有任何关节的Z坐标低于地面高度
        joint_z_positions = joint_positions[:, :, 2]  # [num_envs, num_links]
        collision_with_ground = torch.any(joint_z_positions < self.cfg.ground_height, dim=1)  # [num_envs]
        collision_with_ground = collision_with_ground.bool()

        # Combine done conditions
        dones = time_out | success | collision_with_ground
        
        # 简化的Episode结束信息
        # if dones.any():
        #     success_envs = success.nonzero(as_tuple=False).squeeze(-1).cpu().numpy() if success.any() else []
        #     collision_envs = collision_with_ground.nonzero(as_tuple=False).squeeze(-1).cpu().numpy() if collision_with_ground.any() else []
        #     timeout_envs = time_out.nonzero(as_tuple=False).squeeze(-1).cpu().numpy() if time_out.any() else []
            
        #     print(f"\n🏁 Episode结束 (步骤 {self.step_count}): 成功{len(success_envs)} | 碰撞{len(collision_envs)} | 超时{len(timeout_envs)}")

        return dones, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at the given indices - 学习ball_rolling_tactile_rgb.py的重置机制."""
        # 调用父类重置方法
        super()._reset_idx(env_ids)
        
        # 重置机械臂到随机化初始位置
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        
        # 添加关节位置随机化 - 只对arm关节（前7个）
        joint_noise = sample_uniform(
            -self.cfg.robot_joint_pos_noise, 
            self.cfg.robot_joint_pos_noise, 
            (len(env_ids), 7), 
            self.device
        )
        joint_pos[:, :7] += joint_noise
        
        # 限制在关节范围内
        joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, :7, 0]
        joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, :7, 1]
        joint_pos[:, :7] = torch.clamp(
            joint_pos[:, :7],
            joint_lower_limits.unsqueeze(0),
            joint_upper_limits.unsqueeze(0)
        )
        
        # 添加关节速度随机化
        joint_vel = sample_uniform(
            -self.cfg.robot_joint_vel_noise, 
            self.cfg.robot_joint_vel_noise, 
            (len(env_ids), joint_pos.shape[1]), 
            self.device
        )
        
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 重置圆柱体位置 - 随机放置在盘子上
        cylinder_state = self._cylinder.data.default_root_state[env_ids].clone()
        
        # 在 xy 平面添加随机偏移
        xy_offset = sample_uniform(-self.cfg.cylinder_pos_range, self.cfg.cylinder_pos_range, (len(env_ids), 2), self.device)
        cylinder_state[:, :2] += xy_offset
        
        # 轻微 z 抖动与随机 yaw，打破完全一致的初态
        cylinder_state[:, 2] += sample_uniform(-0.002, 0.002, (len(env_ids),), self.device)
        rand_yaw = sample_uniform(-self.cfg.cylinder_rot_range, self.cfg.cylinder_rot_range, (len(env_ids),), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        cylinder_state[:, 3:7] = rand_quat
        
        # 添加环境原点偏移（包含z轴）
        cylinder_state[:, :3] += self.scene.env_origins[env_ids]
        
        # 写入仿真
        self._cylinder.write_root_state_to_sim(cylinder_state, env_ids=env_ids)

        # 记录每个 env 的真实 spawn 高度与抬升阈值
        self._cylinder_spawn_height_per_env[env_ids] = cylinder_state[:, 2].clone()
        self._lift_target_height_per_env[env_ids] = self._cylinder_spawn_height_per_env[env_ids] + self.cfg.lift_height

        # 重置动作缓冲区，并为超时步数加入抖动，错峰重置
        self.actions[env_ids] = 0
        self.processed_actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.action_history[env_ids] = 0
        max_jitter_steps = int(getattr(self.cfg, "reset_jitter_max_steps", 0))
        if max_jitter_steps > 0:
            jitter = torch.randint(0, max_jitter_steps, (len(env_ids),), device=self.device)
        else:
            jitter = torch.zeros((len(env_ids),), device=self.device, dtype=self.episode_length_buf.dtype)
        self.episode_length_buf[env_ids] = jitter

    def _compute_collision_penalty(
        self, left_finger_pos: torch.Tensor, right_finger_pos: torch.Tensor, cylinder_pos: torch.Tensor
    ) -> torch.Tensor:
        """计算与桌面/地面的穿透性碰撞（工具函数，当前奖励未使用）。"""
        plate_top_z = self._plate.data.root_pos_w[:, 2] + self.cfg.plate_top_offset
        ground_height = self.cfg.ground_height

        left_below_plate = left_finger_pos[:, 2] < (plate_top_z - 0.001)
        right_below_plate = right_finger_pos[:, 2] < (plate_top_z - 0.001)
        object_below_ground = cylinder_pos[:, 2] < ground_height

        collision_detected = left_below_plate | right_below_plate | object_below_ground
        return collision_detected.float()
