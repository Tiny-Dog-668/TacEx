# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can Grasping Environment with Privileged State Observations plus Wrist Camera."""

from __future__ import annotations

import os
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
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
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
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom
try:
    import torchvision
    from torchvision.models import ResNet18_Weights
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from tacex_tasks.utils import DirectLiveVisualizer
from .can_grasping_vision_only_resnet18 import (
    CanGraspingVisionOnlyCfg as CanGraspingVisionOnlyResNet18Cfg,
    CanGraspingVisionOnlyEnv as CanGraspingVisionOnlyResNet18Env,
)

YCB_CAN_USD_PATH = "/home/tinydog/Projects/ycb-tools/models/ycb/002_master_chef_can/can.usda"
YCB_CAN_SCALE = (0.5, 0.5, 0.5)


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
class CanGraspingVisionOnlyCfg(DirectRLEnvCfg):
    """Configuration for the can grasping environment with wrist-camera vision."""

    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    debug_vis = False
    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,  # 更小的时间步长，提高稳定性
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            solver_type=1,  # 使用TGS求解器，更稳定
            max_position_iteration_count=32,  # 增加位置迭代次数
            max_velocity_iteration_count=4,   # 增加速度迭代次数
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,  # 降低摩擦系数，避免过度粘性
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,  # 使用4个环境进行测试
        env_spacing=1.5,
        replicate_physics=False,  # 允许 per-env 视觉材质修改
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
            # visible=False,
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

    can = RigidObjectCfg(
        prim_path="/World/envs/env_.*/can",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.005]),
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
                # 固定物体：启用 kinematic，使其不受末端碰撞/外力影响（位置不会被推走）
                # kinematic_enabled=True,
                # disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,  # 减小接触偏移
                rest_offset=0.0005,    # 增加静止偏移，避免过度接触
            ),
        ),
    )

    # robot configuration - 学习ball_rolling_tactile_rgb.py的固定初始位置
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # 'panda_joint1': -0.4846,
                # 'panda_joint2': -0.3678,
                # 'panda_joint3': 0.2810,
                # 'panda_joint4': -2.9674,
                # 'panda_joint5': 0.1246,
                # 'panda_joint6': 2.6000,
                # 'panda_joint7': 0.5239,
                
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

    # wrist camera (tiled rendering for batched obs)
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        update_period=0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0, -0.12),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )

    # wrist camera preprocessing (optional)
    wrist_preprocess_enabled = True
    wrist_blur_kernel_size = 45
    wrist_blur_sigma = 1.2
    wrist_gaussian_noise_std = 0.2
    wrist_save_images_enabled = True
    wrist_save_images_dir = "outputs/wrist_debug"
    wrist_save_images_max = 6
    wrist_save_images_interval = 1
    wrist_save_images_env_id = 0

    # IK controller - 优化配置
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,  # 使用相对模式，期望6维动作
        ik_method="dls",  # 使用阻尼最小二乘法，更稳定
    )

    # noise models
    gaussian_noise_cfg = GaussianNoiseCfg(mean=0.0, std=0.002, operation="add")

    # environment settings
    episode_length_s =  2.5 # 这个会被转换为步数
    max_episode_length = 150  # 直接设置最大步数
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
        "gripper_state": 1,  # gripper opening state
        "action_history": 5,  # previous actions
        "wrist_resnet": 512,  # resnet18 feature vector (if available)
        "critic_can_pos": 3,
        "critic_can_quat": 4,
        "critic_can_lin_vel": 3,
        "critic_can_ang_vel": 3,
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
    success_height = 0.07
    success_reward_weight = 100.0
    
    # 安全最小离板间隙（用于动作门控与奖励扣分）
    safety_plate_clearance = 0.002
    # 指尖几何相对 link 原点的近似 z 偏移（m）——保守估计
    finger_tip_offset = 0.02
    # 板面相对板中心的 z 偏移（m）（block.usd 厚度 0.01 -> 顶面偏移 ~0.005）
    plate_top_offset = 0.005

    # task specific parameters 
    can_height = 0.06  # approximate can height (m)
    can_radius = 0.03  # approximate can radius (m)
    lift_height = 0.05
    ground_height = 0.0  # 地面高度，用于判断机械臂关节是否碰撞地面

    # randomization
    can_pos_range = 0.05  # plate half-width for random placement
    can_rot_range = 0.0
    plate_color_min = (0.1, 0.1, 0.1)
    plate_color_max = (0.9, 0.9, 0.9)
    
    # 机械臂随机化配置（打破并行环境的同步起步）
    robot_joint_pos_noise = 0.03  # 关节位置随机化范围 (弧度)
    robot_joint_vel_noise = 0.10  # 关节速度随机化范围
    action_noise_scale = 0.01  # 动作噪声缩放
    reset_jitter_max_steps = 15  # 重置后给 episode 计数随机抖动，错峰超时
    reward_print_interval = 200


@configclass
class CanGraspingVisionSixTactileCfg(CanGraspingVisionOnlyResNet18Cfg):
    """Configuration for can grasping with wrist-camera vision and 6 tactile RGB sensors."""

    tactile_img_res_hw = (96, 128)

    # Use the 6-pad GelSight Mini asset (adds *_down and *_out pads expected by this env).
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.six.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.3136,
                "panda_joint2": -0.4448,
                "panda_joint3": 0.3858,
                "panda_joint4": -3.0236,
                "panda_joint5": 0.2319,
                "panda_joint6": 2.5776,
                "panda_joint7": 0.6249,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Six tactile RGB sensors (left/right + left_down/right_down + left_out/right_out).
    gsmini_left: GelSightMiniCfg = GelSightMiniCfg(prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left")
    gsmini_left.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_left.marker_motion_sim_cfg = None
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_right: GelSightMiniCfg = GelSightMiniCfg(prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right")
    gsmini_right.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_right.marker_motion_sim_cfg = None
    gsmini_right.optical_sim_cfg = gsmini_right.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_left_down: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_down"
    )
    gsmini_left_down.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left_down.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_left_down.marker_motion_sim_cfg = None
    gsmini_left_down.optical_sim_cfg = gsmini_left_down.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_right_down: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_down"
    )
    gsmini_right_down.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right_down.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_right_down.marker_motion_sim_cfg = None
    gsmini_right_down.optical_sim_cfg = gsmini_right_down.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_left_out: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_out"
    )
    gsmini_left_out.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left_out.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_left_out.marker_motion_sim_cfg = None
    gsmini_left_out.optical_sim_cfg = gsmini_left_out.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    gsmini_right_out: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_out"
    )
    gsmini_right_out.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=2,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right_out.data_types = ["camera_depth", "tactile_rgb"]
    gsmini_right_out.marker_motion_sim_cfg = None
    gsmini_right_out.optical_sim_cfg = gsmini_right_out.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    observation_space = {
        "proprio_obs": 18,
        "gripper_state": 1,
        "action_history": 5,
        "wrist_resnet": 512,
        "tactile_left_resnet": 128,
        "tactile_right_resnet": 128,
        "tactile_left_down_resnet": 128,
        "tactile_right_down_resnet": 128,
        "tactile_left_out_resnet": 128,
        "tactile_right_out_resnet": 128,
        "critic_can_pos": 3,
        "critic_can_quat": 4,
        "critic_can_lin_vel": 3,
        "critic_can_ang_vel": 3,
        "critic_gripper_pos": 3,
        "critic_gripper_quat": 4,
        "critic_gripper_lin_vel": 3,
        "critic_gripper_ang_vel": 3,
        "critic_target_pos": 3,
        "critic_target_distance": 1,
    }


class CanGraspingVisionOnlyEnv(DirectRLEnv):
    """Can grasping environment with privileged state and additional camera stream."""

    cfg: CanGraspingVisionOnlyCfg

    def __init__(self, cfg: CanGraspingVisionOnlyCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        self.step_count = 0
        self.reward_print_interval = self.cfg.reward_print_interval
        self._plate_material_bound = [False] * self.num_envs

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
        # IK controller handles 6D actions (arm joints), but our action space is 7D (6 arm joints + gripper)
        self.processed_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.action_history = torch.zeros_like(self.processed_actions)

        # wrist camera offset (relative to panda_hand)
        wrist_offset_pos = torch.tensor(self.cfg.wrist_camera.offset.pos, dtype=torch.float32, device=self.device)
        wrist_offset_rot = torch.tensor(self.cfg.wrist_camera.offset.rot, dtype=torch.float32, device=self.device)
        self._wrist_offset_pos = wrist_offset_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self._wrist_offset_rot = wrist_offset_rot.unsqueeze(0).repeat(self.num_envs, 1)

        # Optional ResNet18 backbone for wrist RGB -> 512-d feature
        self._use_resnet18 = _HAS_TORCHVISION
        if self._use_resnet18:
            backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # remove final fc; keep feature extractor
            self._resnet18 = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
            for p in self._resnet18.parameters():
                p.requires_grad_(False)
            # ImageNet normalization constants
            self._imgnet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self._imgnet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # wrist camera preprocessing
        self._wrist_preprocess_enabled = bool(self.cfg.wrist_preprocess_enabled)
        self._wrist_blur_kernel = None
        self._wrist_blur_padding = 0
        if self._wrist_preprocess_enabled:
            kernel_size = int(self.cfg.wrist_blur_kernel_size)
            sigma = float(self.cfg.wrist_blur_sigma)
            if kernel_size > 1 and sigma > 0.0:
                if kernel_size % 2 == 0:
                    kernel_size += 1
                self._wrist_blur_padding = kernel_size // 2
                self._wrist_blur_kernel = self._build_box_kernel(kernel_size)
        self._wrist_save_images_enabled = bool(self.cfg.wrist_save_images_enabled)
        if self._wrist_save_images_enabled and not _HAS_TORCHVISION:
            print("[WARN] wrist_save_images_enabled requires torchvision; disabling image dump.")
            self._wrist_save_images_enabled = False
        self._wrist_save_dir = self.cfg.wrist_save_images_dir
        self._wrist_save_max = int(self.cfg.wrist_save_images_max)
        self._wrist_save_interval = max(int(self.cfg.wrist_save_images_interval), 1)
        self._wrist_save_env_id = int(self.cfg.wrist_save_images_env_id)
        self._wrist_save_count = 0

        # Index of fingers -> first id is left, second id is right finger
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])
        self._left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self._right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        # add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Initialize can positions randomly on the plate
        self._initialize_can_positions()
        # 每个环境的真实 spawn 高度（考虑 z 抖动与 env 原点偏移）
        self._can_spawn_height_per_env = self._can.data.root_com_pos_w[:, 2].clone()
        self._lift_target_height_per_env = self._can_spawn_height_per_env + self.cfg.lift_height

    def _setup_scene(self):
        """Setup the scene."""
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # can
        self._can = RigidObject(self.cfg.can)
        self.scene.rigid_objects["can"] = self._can

        # plate
        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        # clone environments first
        self.scene.clone_environments(copy_from_source=False)

        # remove auto-added dome light so wrist camera matches intended lighting
        if prim_utils.is_prim_path_valid("/World/defaultDomeLight"):
            prim_utils.delete_prim("/World/defaultDomeLight")

        # wrist camera sensor
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera

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

    def _initialize_can_positions(self):
        """Initialize can positions randomly on the plate for all environments - 只在xy平面随机，保持z轴高度不变."""
        # 获取默认根状态并修改位置
        can_state = self._can.data.default_root_state.clone()
        
        # 只在xy平面添加随机偏移，保持z轴高度不变
        xy_offset = sample_uniform(-self.cfg.can_pos_range, self.cfg.can_pos_range, (self.num_envs, 2), self.device)
        can_state[:, :2] += xy_offset  # 只修改x和y坐标
        
        # 轻微 z 抖动与随机 yaw，打破完全一致的初态
        can_state[:, 2] += sample_uniform(-0.002, 0.002, (self.num_envs,), self.device)
        rand_yaw = sample_uniform(-self.cfg.can_rot_range, self.cfg.can_rot_range, (self.num_envs,), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        can_state[:, 3:7] = rand_quat
        
        # 添加环境原点偏移（包含z轴）
        can_state[:, :3] += self.scene.env_origins
        
        # 写入仿真
        self._can.write_root_state_to_sim(can_state, torch.arange(self.num_envs, device=self.device))

    def _apply_plate_material(
        self,
        material_path: str,
        target_prim_path: str,
        color: tuple[float, float, float],
        bind: bool = True,
    ):
        stage = sim_utils.stage_utils.get_current_stage()
        try:
            prim = stage.GetPrimAtPath(material_path)
            if not prim.IsValid():
                sim_utils.spawn_preview_surface(material_path, sim_utils.PreviewSurfaceCfg())
                bind = True
            if bind:
                sim_utils.bind_visual_material(target_prim_path, material_path)
            shader_prim = stage.GetPrimAtPath(f"{material_path}/Shader")
            if shader_prim.IsValid():
                sim_utils.safe_set_attribute_on_usd_prim(
                    shader_prim, "inputs:diffuse_color", color, camel_case=True
                )
        except Exception as exc:
            print(f"[WARN] plate material update failed for {material_path}: {exc}")

    def _randomize_plate_colors(self, env_ids: torch.Tensor):
        if self.cfg.scene.replicate_physics:
            print("[WARN] replicate_physics=True: per-env plate color randomization is disabled.")
            return
        env_id_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        if not env_id_list:
            return
        color_min = torch.tensor(self.cfg.plate_color_min, device=self.device)
        color_max = torch.tensor(self.cfg.plate_color_max, device=self.device)
        colors = torch.rand((len(env_id_list), 3), device=self.device)
        colors = colors * (color_max - color_min) + color_min
        colors = colors.detach().cpu().tolist()
        plate_paths = self._plate.root_physx_view.prim_paths
        for idx, env_id in enumerate(env_id_list):
            plate_root = plate_paths[env_id]
            color = tuple(float(c) for c in colors[idx])
            material_path = f"{plate_root}/material"
            bind = not self._plate_material_bound[env_id]
            self._apply_plate_material(material_path, plate_root, color, bind=bind)
            if bind:
                self._plate_material_bound[env_id] = True

    def _build_box_kernel(self, kernel_size: int) -> torch.Tensor:
        kernel = torch.ones((kernel_size, kernel_size), device=self.device, dtype=torch.float32)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel.repeat(3, 1, 1, 1)

    def _process_wrist_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if self._wrist_blur_kernel is not None:
            x = F.conv2d(x, self._wrist_blur_kernel, padding=self._wrist_blur_padding, groups=3)
        noise_std = float(self.cfg.wrist_gaussian_noise_std)
        if noise_std > 0.0:
            x = x + torch.randn_like(x) * noise_std
        return torch.clamp(x, 0.0, 1.0)

    def _maybe_save_wrist_images(self, x_raw: torch.Tensor, x_proc: torch.Tensor):
        if not self._wrist_save_images_enabled:
            return
        if self._wrist_save_count >= self._wrist_save_max:
            return
        if self._wrist_save_env_id < 0 or self._wrist_save_env_id >= self.num_envs:
            print(f"[WARN] wrist_save_images_env_id out of range: {self._wrist_save_env_id}")
            self._wrist_save_images_enabled = False
            return
        step_id = int(getattr(self, "common_step_counter", 0))
        if step_id % self._wrist_save_interval != 0:
            return
        os.makedirs(self._wrist_save_dir, exist_ok=True)
        env_id = self._wrist_save_env_id
        raw = x_raw[env_id].detach().cpu()
        proc = x_proc[env_id].detach().cpu()
        idx = self._wrist_save_count
        raw_path = os.path.join(self._wrist_save_dir, f"wrist_raw_env{env_id}_step{step_id:06d}_{idx:03d}.png")
        proc_path = os.path.join(self._wrist_save_dir, f"wrist_proc_env{env_id}_step{step_id:06d}_{idx:03d}.png")
        torchvision.utils.save_image(raw, raw_path)
        torchvision.utils.save_image(proc, proc_path)
        self._wrist_save_count += 1

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
        self.processed_actions += action_noise

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
        obj_xy = self._can.data.root_com_pos_w[:, :2]
        near_obj_xy = torch.norm(obj_xy - ee_xy, dim=-1) < (2.0 * self.cfg.can_radius)
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
        can_pos = self._can.data.root_com_pos_w
        can_quat = self._can.data.root_quat_w
        can_lin_vel = self._can.data.root_com_lin_vel_w
        can_ang_vel = self._can.data.root_com_ang_vel_w

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
        target_pos_relative = can_pos - ee_pos  # 相对位置
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        # Wrist camera world position
        wrist_offset_rot = _norm_quat(self._wrist_offset_rot)
        wrist_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._wrist_offset_pos, wrist_offset_rot
        )

        if (
            self.reward_print_interval > 0
            and self.step_count % self.reward_print_interval == 0
        ):
            ee_np = ee_pos[0].detach().cpu().numpy()
            wrist_np = wrist_pos[0].detach().cpu().numpy()

        # Gripper state
        gripper_state = self._robot.data.joint_pos[:, self._finger_joint_ids[0:1]]

        # Wrist camera observation (fallback to zeros if unavailable)
        wrist_rgb = self.wrist_camera.data.output.get("rgb")
        wrist_rgb_valid = wrist_rgb is not None
        if wrist_rgb is None:
            wrist_rgb = torch.zeros(
                (self.num_envs, self.cfg.wrist_camera.height, self.cfg.wrist_camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            # normalize to [0, 1] float32 for stable CNN training
            wrist_rgb = wrist_rgb.to(torch.float32) / 255.0

        # compute ResNet18 features if available
        if hasattr(self, "_use_resnet18") and self._use_resnet18:
            x = wrist_rgb.permute(0, 3, 1, 2).contiguous()
            if self._wrist_preprocess_enabled:
                x_proc = self._process_wrist_rgb(x)
            else:
                x_proc = x
            if wrist_rgb_valid:
                self._maybe_save_wrist_images(x, x_proc)
            x = x_proc
            # resize to 224x224 to match ResNet18 training resolution
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            # ImageNet normalization
            x = (x - self._imgnet_mean) / self._imgnet_std
            with torch.no_grad():
                feat = self._resnet18(x)  # [N, 512, 1, 1]
                feat = feat.view(self.num_envs, 512)
            wrist_feat = feat
        else:
            wrist_feat = torch.zeros((self.num_envs, 512), device=self.device, dtype=torch.float32)

        obs = {
            "proprio_obs": proprio_obs,
            "gripper_state": gripper_state,
            "action_history": self.action_history,
            "wrist_resnet": wrist_feat,
            "critic_can_pos": can_pos,
            "critic_can_quat": can_quat,
            "critic_can_lin_vel": can_lin_vel,
            "critic_can_ang_vel": can_ang_vel,
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
        can_pos = self._can.data.root_com_pos_w
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )

        # reaching_object: r_reach = 1 - tanh(||p_obj - p_ee|| / sigma)
        sigma = max(self.cfg.reach_sigma, 1e-6)
        reach_distance = torch.norm(can_pos - ee_pos, dim=-1)
        reach_reward = 1.0 - torch.tanh(reach_distance / sigma)

        # lifting_object: base reward above h_min + linear reward for height above h_min
        h_min = self.cfg.minimal_lift_height
        height = can_pos[:, 2]
        # print("can height:", height.mean().item())
        base_reward = (height > h_min).float() 
        denom = max(self.cfg.success_height - h_min, 1e-6)
        linear_reward = torch.clamp((height - h_min) / denom, 0.0, 1.0)
        lift_reward = base_reward * 0.5 + linear_reward


        # success bonus when object height exceeds success_height
        success_reward = (can_pos[:, 2] > self.cfg.success_height).float()

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

        rewards = (
            self.cfg.reach_weight * reach_reward
            + self.cfg.lift_weight * lift_reward
            + self.cfg.success_reward_weight * success_reward
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
                f"(w={self.cfg.success_reward_weight}), "
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

        # Success condition - can lifted above threshold (基于实时高度，避免依赖上一周期缓存)
        current_height = self._can.data.root_com_pos_w[:, 2]
        success = current_height > self._lift_target_height_per_env
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
        can_state = self._can.data.default_root_state[env_ids].clone()
        
        # 在 xy 平面添加随机偏移
        xy_offset = sample_uniform(-self.cfg.can_pos_range, self.cfg.can_pos_range, (len(env_ids), 2), self.device)
        can_state[:, :2] += xy_offset
        
        # 轻微 z 抖动与随机 yaw，打破完全一致的初态
        # can_state[:, 2] += sample_uniform(-0.002, 0.002, (len(env_ids),), self.device)
        rand_yaw = sample_uniform(-self.cfg.can_rot_range, self.cfg.can_rot_range, (len(env_ids),), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        can_state[:, 3:7] = rand_quat
        
        # 添加环境原点偏移（包含z轴）
        can_state[:, :3] += self.scene.env_origins[env_ids]
        
        # 写入仿真
        self._can.write_root_state_to_sim(can_state, env_ids=env_ids)

        # 记录每个 env 的真实 spawn 高度与抬升阈值
        com_pos_b = self._can.data.body_com_pos_b[env_ids, 0]
        com_quat_b = self._can.data.body_com_quat_b[env_ids, 0]
        com_pos_w, _ = math_utils.combine_frame_transforms(can_state[:, :3], can_state[:, 3:7], com_pos_b, com_quat_b)
        self._can_spawn_height_per_env[env_ids] = com_pos_w[:, 2].clone()
        self._lift_target_height_per_env[env_ids] = self._can_spawn_height_per_env[env_ids] + self.cfg.lift_height

        # 重置动作缓冲区，并为超时步数加入抖动，错峰重置
        self.actions[env_ids] = 0
        self.processed_actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.action_history[env_ids] = 0
        jitter = torch.randint(0, self.cfg.reset_jitter_max_steps, (len(env_ids),), device=self.device)
        self.episode_length_buf[env_ids] = jitter
        self._randomize_plate_colors(env_ids)

    def _compute_collision_penalty(
        self, left_finger_pos: torch.Tensor, right_finger_pos: torch.Tensor, can_pos: torch.Tensor
    ) -> torch.Tensor:
        """计算与桌面/地面的穿透性碰撞（工具函数，当前奖励未使用）。"""
        plate_top_z = self._plate.data.root_pos_w[:, 2] + self.cfg.plate_top_offset
        ground_height = self.cfg.ground_height

        left_below_plate = left_finger_pos[:, 2] < (plate_top_z - 0.001)
        right_below_plate = right_finger_pos[:, 2] < (plate_top_z - 0.001)
        object_below_ground = can_pos[:, 2] < ground_height

        collision_detected = left_below_plate | right_below_plate | object_below_ground
        return collision_detected.float()


class CanGraspingVisionSixTactileEnv(CanGraspingVisionOnlyResNet18Env):
    """Can grasping environment with wrist-camera vision and 6 tactile RGB sensors."""

    cfg: CanGraspingVisionSixTactileCfg

    def __init__(self, cfg: CanGraspingVisionSixTactileCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._tactile_feature_dim = 128
        if _HAS_TORCHVISION:
            resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            resnet.fc = nn.Linear(resnet.fc.in_features, self._tactile_feature_dim)
            self._tactile_encoder = resnet.to(self.device)
            self._tactile_encoder.eval()
            for p in self._tactile_encoder.parameters():
                p.requires_grad_(False)
            self._tactile_imgnet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self._tactile_imgnet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:
            self._tactile_encoder = None
            self._tactile_imgnet_mean = None
            self._tactile_imgnet_std = None

    def _setup_scene(self):
        super()._setup_scene()

        # tactile sensors
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right
        self.gsmini_left_down = GelSightSensor(self.cfg.gsmini_left_down)
        self.scene.sensors["gsmini_left_down"] = self.gsmini_left_down
        self.gsmini_right_down = GelSightSensor(self.cfg.gsmini_right_down)
        self.scene.sensors["gsmini_right_down"] = self.gsmini_right_down
        self.gsmini_left_out = GelSightSensor(self.cfg.gsmini_left_out)
        self.scene.sensors["gsmini_left_out"] = self.gsmini_left_out
        self.gsmini_right_out = GelSightSensor(self.cfg.gsmini_right_out)
        self.scene.sensors["gsmini_right_out"] = self.gsmini_right_out

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs_dict = super()._get_observations()
        obs = obs_dict["policy"]

        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))

        def _prep_rgb_3ch(rgb_tensor: torch.Tensor | None) -> torch.Tensor:
            if rgb_tensor is None or self._tactile_encoder is None:
                return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
            x = rgb_tensor.to(device=self.device, dtype=torch.float32)
            max_val = x.max()
            if torch.isfinite(max_val) and max_val > 1.5:
                x = x / 255.0
            x = x.clamp(0.0, 1.0)
            xn = x.permute(0, 3, 1, 2).contiguous()
            if xn.shape[2] != target_h or xn.shape[3] != target_w:
                xn = F.interpolate(xn, size=(target_h, target_w), mode="bilinear", align_corners=False)
            xn = (xn - self._tactile_imgnet_mean) / self._tactile_imgnet_std
            return xn

        tact_l_raw = getattr(self, "gsmini_left", None)
        tact_l_raw = tact_l_raw.data.output.get("tactile_rgb") if tact_l_raw is not None else None
        tact_r_raw = getattr(self, "gsmini_right", None)
        tact_r_raw = tact_r_raw.data.output.get("tactile_rgb") if tact_r_raw is not None else None
        tact_ld_raw = getattr(self, "gsmini_left_down", None)
        tact_ld_raw = tact_ld_raw.data.output.get("tactile_rgb") if tact_ld_raw is not None else None
        tact_rd_raw = getattr(self, "gsmini_right_down", None)
        tact_rd_raw = tact_rd_raw.data.output.get("tactile_rgb") if tact_rd_raw is not None else None
        tact_lo_raw = getattr(self, "gsmini_left_out", None)
        tact_lo_raw = tact_lo_raw.data.output.get("tactile_rgb") if tact_lo_raw is not None else None
        tact_ro_raw = getattr(self, "gsmini_right_out", None)
        tact_ro_raw = tact_ro_raw.data.output.get("tactile_rgb") if tact_ro_raw is not None else None

        tact_l = _prep_rgb_3ch(tact_l_raw)
        tact_r = _prep_rgb_3ch(tact_r_raw)
        tact_ld = _prep_rgb_3ch(tact_ld_raw)
        tact_rd = _prep_rgb_3ch(tact_rd_raw)
        tact_lo = _prep_rgb_3ch(tact_lo_raw)
        tact_ro = _prep_rgb_3ch(tact_ro_raw)

        if self._tactile_encoder is None:
            zeros = torch.zeros((self.num_envs, self._tactile_feature_dim), device=self.device, dtype=torch.float32)
            fl = fr = fld = frd = flo = fro = zeros
        else:
            dev = self.device
            dev_type = getattr(dev, "type", None)
            if dev_type is None:
                dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
            use_amp = dev_type == "cuda"
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                fl = self._tactile_encoder(tact_l).view(self.num_envs, self._tactile_feature_dim).float()
                fr = self._tactile_encoder(tact_r).view(self.num_envs, self._tactile_feature_dim).float()
                fld = self._tactile_encoder(tact_ld).view(self.num_envs, self._tactile_feature_dim).float()
                frd = self._tactile_encoder(tact_rd).view(self.num_envs, self._tactile_feature_dim).float()
                flo = self._tactile_encoder(tact_lo).view(self.num_envs, self._tactile_feature_dim).float()
                fro = self._tactile_encoder(tact_ro).view(self.num_envs, self._tactile_feature_dim).float()

        obs.update(
            {
                "tactile_left_resnet": fl,
                "tactile_right_resnet": fr,
                "tactile_left_down_resnet": fld,
                "tactile_right_down_resnet": frd,
                "tactile_left_out_resnet": flo,
                "tactile_right_out_resnet": fro,
            }
        )

        return {"policy": obs}
