# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cube Grasping Environment with Privileged State Observations plus Wrist Camera."""

from __future__ import annotations

import os
import random
from typing import List, Tuple
import torch
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
from isaaclab.utils import configclass
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
import omni.usd
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
try:
    import torchvision
    from torchvision.models import ResNet18_Weights
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

# from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)

from tacex_tasks.utils import DirectLiveVisualizer


def add_random_protrusions_to_cube(
    cube_prim_path: str,
    cube_size: Tuple[float, float, float],
    fixed_radius: float = 0.0125,
    faces: Tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z"),
    gap: float = 0.001,
    seed: int | None = None,
    enable_visual: bool = True,
    bump_friction: Tuple[float, float] = (0.5, 0.5),
    bump_restitution: float = 0.0,
    max_tries_per_face: int = 200,
) -> None:
    """Add spherical protrusions (collision spheres) on cube surface."""
    if seed is not None:
        random.seed(seed)

    stage = omni.usd.get_context().get_stage()
    container_path = f"{cube_prim_path}/Protrusions"
    material_suffix = cube_prim_path.replace("/", "_").strip("_")
    material_path = f"/World/Materials/bump_mat_{material_suffix}"

    if stage.GetPrimAtPath(material_path).IsValid():
        bump_material_path = stage.GetPrimAtPath(material_path).GetPath()
    else:
        bump_mat = sim_utils.spawn_rigid_body_material(
            prim_path=material_path,
            cfg=sim_utils.RigidBodyMaterialCfg(
                static_friction=bump_friction[0],
                dynamic_friction=bump_friction[1],
                restitution=bump_restitution,
            ),
        )
        bump_material_path = bump_mat.GetPath()

    if stage.GetPrimAtPath(container_path).IsValid():
        stage.RemovePrim(container_path)

    UsdGeom.Xform.Define(stage, container_path)

    hx, hy, hz = cube_size[0] * 0.5, cube_size[1] * 0.5, cube_size[2] * 0.5
    placed: List[Tuple[Gf.Vec3d, float]] = []

    def sample_on_face(face: str, r: float) -> Gf.Vec3d:
        mx = hx - r - gap
        my = hy - r - gap
        mz = hz - r - gap

        if face == "+X":
            x = hx
            y = random.uniform(-my, my)
            z = random.uniform(-mz, mz)
        elif face == "-X":
            x = -hx
            y = random.uniform(-my, my)
            z = random.uniform(-mz, mz)
        elif face == "+Y":
            x = random.uniform(-mx, mx)
            y = hy
            z = random.uniform(-mz, mz)
        elif face == "-Y":
            x = random.uniform(-mx, mx)
            y = -hy
            z = random.uniform(-mz, mz)
        elif face == "+Z":
            x = random.uniform(-mx, mx)
            y = random.uniform(-my, my)
            z = hz
        else:
            raise ValueError(face)

        return Gf.Vec3d(x, y, z)

    def is_non_overlapping(pos: Gf.Vec3d, r: float) -> bool:
        for p, pr in placed:
            if (pos - p).GetLength() < (r + pr + gap):
                return False
        return True

    created = 0

    for i, face in enumerate(faces):
        r = float(fixed_radius)
        pos = None
        for _ in range(max_tries_per_face):
            cand = sample_on_face(face, r)
            if is_non_overlapping(cand, r):
                pos = cand
                break

        if pos is None:
            if face == "+X":
                pos = Gf.Vec3d(hx, 0.0, 0.0)
            elif face == "-X":
                pos = Gf.Vec3d(-hx, 0.0, 0.0)
            elif face == "+Y":
                pos = Gf.Vec3d(0.0, hy, 0.0)
            elif face == "-Y":
                pos = Gf.Vec3d(0.0, -hy, 0.0)
            elif face == "+Z":
                pos = Gf.Vec3d(0.0, 0.0, hz)
            else:
                continue

        placed.append((pos, r))

        bump_path = f"{container_path}/bump_{i:02d}"
        sphere = UsdGeom.Sphere.Define(stage, bump_path)
        sphere.GetRadiusAttr().Set(r)

        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(pos)

        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
        sim_utils.bind_physics_material(bump_path, bump_material_path)

        if enable_visual:
            sphere.CreateDisplayColorAttr().Set([(0.0, 0.0, 0.0)])
            sphere.CreateDisplayOpacityAttr().Set([1.0])

        created += 1

    print(f"[INFO] Added {created}/{len(faces)} protrusions under {cube_prim_path}.")


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
class CubeGraspingVisionOnlyCfg(DirectRLEnvCfg):
    """Configuration for the cube grasping environment with wrist-camera vision."""

    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.6, 1.2, 0.7)
    viewer.lookat = (0.5, 0.0, 0.05)

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
        num_envs=128,
        env_spacing=1.5,
        replicate_physics=False,  # 允许 per-env 视觉材质修改
        lazy_sensor_update=True,
    )

    board_size = (0.6, 0.6, 0.01)
    cube_size = (0.06, 0.06, 0.06)

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

    board = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wood_board",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, board_size[2] * 0.5)),
        spawn=sim_utils.CuboidCfg(
            size=board_size,
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
                diffuse_color=(0.65, 0.45, 0.30),
                roughness=0.6,
                metallic=0.0,
            ),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, board_size[2] + cube_size[2] * 0.5 + 0.002)
        ),
        spawn=sim_utils.CuboidCfg(
            size=cube_size,
            rigid_props=RigidBodyPropertiesCfg(
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
                diffuse_color=(0.2, 0.2, 0.8),
                roughness=0.5,
                metallic=0.0,
            ),
        ),
    )

    # robot configuration
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # "panda_joint1": -0.4510,
                # "panda_joint2": 0.1890,
                # "panda_joint3": 0.4750,
                # "panda_joint4": -2.3660,
                # "panda_joint5": -0.2250,
                # "panda_joint6": 2.5450,
                # "panda_joint7": 0.9910,

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
            pos=(0.12, 0, -0.12),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )

    # freeze ResNet18 feature extractor parameters
    resnet18_frozen = True

    # wrist camera preprocessing (optional)
    wrist_preprocess_enabled = True
    wrist_blur_kernel_size = 0
    wrist_blur_sigma = 1.2
    wrist_gaussian_noise_std = 0.0
    wrist_save_images_enabled = True
    wrist_save_images_dir = "auto"
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
        "critic_cube_pos": 3,
        "critic_cube_quat": 4,
        "critic_cube_lin_vel": 3,
        "critic_cube_ang_vel": 3,
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
    minimal_lift_height = board_size[2] + cube_size[2] * 0.5 + 0.01
    lift_weight = 15.0
    success_height = board_size[2] + cube_size[2] * 0.5 + 0.04
    # require N consecutive steps above success height before done
    success_hold_steps = 5
    success_reward_weight = 100.0
    drop_penalty_enabled = True
    
    # task specific parameters 
    ground_height = 0.0  # 地面高度，用于判断机械臂关节是否碰撞地面

    # randomization
    cube_pos_range = 0.05  # board half-width for random placement
    cube_rot_range = 0.0
    board_color_min = (0.1, 0.1, 0.1)
    board_color_max = (0.9, 0.9, 0.9)
    
    # 机械臂随机化配置（打破并行环境的同步起步）
    robot_joint_pos_noise = 0.03  # 关节位置随机化范围 (弧度)
    robot_joint_vel_noise = 0.10  # 关节速度随机化范围
    action_noise_scale = 0.01  # 动作噪声缩放
    reset_jitter_max_steps = 15  # 重置后给 episode 计数随机抖动，错峰超时
    # Every N steps, print windowed episode-average reward (across completed envs).
    reward_print_interval = 500
    # Print per-env episode average reward when an env finishes an episode.
    print_per_env_episode_avg = False


class CubeGraspingVisionOnlyEnv(DirectRLEnv):
    """Cube grasping environment with privileged state and additional camera stream."""

    cfg: CubeGraspingVisionOnlyCfg

    def __init__(self, cfg: CubeGraspingVisionOnlyCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        self.step_count = 0
        self.reward_print_interval = self.cfg.reward_print_interval
        self._board_material_bound = [False] * self.num_envs
        self._ep_return = torch.zeros((self.num_envs,), device=self.device)
        self._ep_len = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._ep_avg_accum_sum = torch.tensor(0.0, device=self.device)
        self._ep_avg_accum_count = torch.tensor(0, dtype=torch.long, device=self.device)
        self._ever_lifted_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._dropped_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._episode_start_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._success_steps_buf = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self._success_hold_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._success_achieved_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_success_steps_mean = 0.0
        self._last_success_steps_min = 0
        self._last_success_steps_max = 0
        # windowed success-steps stats for periodic prints
        self._success_steps_window_sum = 0.0
        self._success_steps_window_count = 0
        self._success_steps_window_min = None
        self._success_steps_window_max = None
        # windowed success counts (per print interval)
        self._success_window_done = 0
        self._success_window_success = 0
        self._success_window_timeout = 0
        self._success_window_collision = 0

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
            trainable = not bool(self.cfg.resnet18_frozen)
            self._resnet18.train(trainable)
            for p in self._resnet18.parameters():
                p.requires_grad_(trainable)
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
        self._wrist_save_dir = self._resolve_wrist_save_dir(self.cfg.wrist_save_images_dir)
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

        # Initialize cube positions randomly on the board
        self._initialize_cube_positions()
        # 每个环境的真实 spawn 高度（考虑 z 抖动与 env 原点偏移）
        self._cube_spawn_height_per_env = self._cube.data.root_com_pos_w[:, 2].clone()

    def _resolve_wrist_save_dir(self, save_dir: str | None) -> str:
        if save_dir is None:
            normalized = ""
        else:
            normalized = str(save_dir).strip()
        if not normalized or normalized.lower() == "auto":
            return os.path.join(self._find_latest_skrl_run_dir(), "wrist_debug")
        if "{run_dir}" in normalized:
            return normalized.format(run_dir=self._find_latest_skrl_run_dir())
        return normalized

    def _find_latest_skrl_run_dir(self) -> str:
        env_dir = (
            os.environ.get("TACEX_RUN_DIR")
            or os.environ.get("SKRL_RUN_DIR")
            or os.environ.get("SKRL_LOG_DIR")
        )
        if env_dir:
            return env_dir
        log_root = os.path.abspath(os.path.join("logs", "skrl", "cube_grasping"))
        if os.path.isdir(log_root):
            candidates = [
                os.path.join(log_root, name)
                for name in os.listdir(log_root)
                if os.path.isdir(os.path.join(log_root, name))
            ]
            if candidates:
                return max(candidates, key=os.path.getmtime)
        return log_root

    def _setup_scene(self):
        """Setup the scene."""
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # cube
        self._cube = RigidObject(self.cfg.cube)
        self.scene.rigid_objects["cube"] = self._cube

        # board
        self._board = RigidObject(self.cfg.board)
        self.scene.rigid_objects["board"] = self._board

        # clone environments first
        self.scene.clone_environments(copy_from_source=False)

        # remove auto-added dome light so wrist camera matches intended lighting
        if prim_utils.is_prim_path_valid("/World/defaultDomeLight"):
            prim_utils.delete_prim("/World/defaultDomeLight")

        # add protrusions to cube (per env)
        for env_id in range(self.num_envs):
            try:
                cube_root = self._cube.root_physx_view.prim_paths[env_id]
            except Exception:
                cube_root = f"/World/envs/env_{env_id}/cube"
            add_random_protrusions_to_cube(
                cube_prim_path=cube_root,
                cube_size=self.cfg.cube_size,
                fixed_radius=0.0125,
                faces=("+X", "-X", "+Y", "-Y", "+Z"),
                gap=0.001,
                seed=None,
                enable_visual=True,
            )

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

    def _initialize_cube_positions(self):
        """Initialize cube positions randomly on the board for all environments - 只在xy平面随机，保持z轴高度不变."""
        # 获取默认根状态并修改位置
        cube_state = self._cube.data.default_root_state.clone()
        
        # 只在xy平面添加随机偏移，保持z轴高度不变
        xy_offset = sample_uniform(-self.cfg.cube_pos_range, self.cfg.cube_pos_range, (self.num_envs, 2), self.device)
        cube_state[:, :2] += xy_offset  # 只修改x和y坐标
        
        # 轻微 z 抖动与随机 yaw，打破完全一致的初态
        cube_state[:, 2] += sample_uniform(-0.002, 0.002, (self.num_envs,), self.device)
        rand_yaw = sample_uniform(-self.cfg.cube_rot_range, self.cfg.cube_rot_range, (self.num_envs,), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        cube_state[:, 3:7] = rand_quat
        
        # 添加环境原点偏移（包含z轴）
        cube_state[:, :3] += self.scene.env_origins
        
        # 写入仿真
        self._cube.write_root_state_to_sim(cube_state, torch.arange(self.num_envs, device=self.device))

    def _apply_board_material(
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
            print(f"[WARN] board material update failed for {material_path}: {exc}")

    def _randomize_board_colors(self, env_ids: torch.Tensor):
        if self.cfg.scene.replicate_physics:
            print("[WARN] replicate_physics=True: per-env board color randomization is disabled.")
            return
        env_id_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        if not env_id_list:
            return
        color_min = torch.tensor(self.cfg.board_color_min, device=self.device)
        color_max = torch.tensor(self.cfg.board_color_max, device=self.device)
        colors = torch.rand((len(env_id_list), 3), device=self.device)
        colors = colors * (color_max - color_min) + color_min
        colors = colors.detach().cpu().tolist()
        board_paths = self._board.root_physx_view.prim_paths
        for idx, env_id in enumerate(env_id_list):
            board_root = board_paths[env_id]
            color = tuple(float(c) for c in colors[idx])
            material_path = f"{board_root}/material"
            bind = not self._board_material_bound[env_id]
            self._apply_board_material(material_path, board_root, color, bind=bind)
            if bind:
                self._board_material_bound[env_id] = True

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

        gripper_delta = gripper_action.unsqueeze(-1) * 1.0
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

        # 特权信息观察 - 立方体信息
        cube_pos = self._cube.data.root_com_pos_w
        cube_quat = self._cube.data.root_quat_w
        cube_lin_vel = self._cube.data.root_com_lin_vel_w
        cube_ang_vel = self._cube.data.root_com_ang_vel_w

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
        target_pos_relative = cube_pos - ee_pos  # 相对位置
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
            if self.cfg.resnet18_frozen:
                with torch.no_grad():
                    feat = self._resnet18(x)  # [N, 512, 1, 1]
                    feat = feat.view(self.num_envs, 512)
            else:
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
            "critic_cube_pos": cube_pos,
            "critic_cube_quat": cube_quat,
            "critic_cube_lin_vel": cube_lin_vel,
            "critic_cube_ang_vel": cube_ang_vel,
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
        cube_pos = self._cube.data.root_com_pos_w
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )

        # reaching_object: r_reach = 1 - tanh(||p_obj - p_ee|| / sigma)
        sigma = max(self.cfg.reach_sigma, 1e-6)
        reach_distance = torch.norm(cube_pos - ee_pos, dim=-1)
        reach_reward = 1.0 - torch.tanh(reach_distance / sigma)

        # lifting_object: base reward above h_min + linear reward for height above h_min
        h_min = self.cfg.minimal_lift_height
        height = cube_pos[:, 2]
        # print("cube height:", height.mean().item())
        base_reward = (height > h_min).float() 
        denom = max(self.cfg.success_height - h_min, 1e-6)
        linear_reward = torch.clamp((height - h_min) / denom, 0.0, 1.0)
        lift_reward = base_reward * 0.5 + linear_reward


        # success bonus: give once when height stays above threshold for N consecutive steps
        hold_steps = max(1, int(getattr(self.cfg, "success_hold_steps", 1)))
        success_now = height > self.cfg.success_height
        if hold_steps > 1:
            self._success_hold_buf = torch.where(
                success_now,
                torch.clamp(self._success_hold_buf + 1, max=hold_steps),
                torch.zeros_like(self._success_hold_buf),
            )
            success_hold = self._success_hold_buf >= hold_steps
        else:
            success_hold = success_now
        success_first = success_hold & ~self._success_achieved_buf
        self._success_achieved_buf |= success_first
        success_reward = success_first.float()

        rewards = (
            self.cfg.reach_weight * reach_reward
            + self.cfg.lift_weight * lift_reward
            + self.cfg.success_reward_weight * success_reward
        )

        # drop penalty: if lifted before and now falls below lift threshold
        lifted = height > self.cfg.minimal_lift_height
        self._ever_lifted_buf |= lifted
        dropped_now = self._ever_lifted_buf & ~self._dropped_buf & (height <= self.cfg.minimal_lift_height)
        drop_penalty = (-self.cfg.success_reward_weight) * dropped_now.float()
        if not self.cfg.drop_penalty_enabled:
            drop_penalty = torch.zeros_like(drop_penalty)
        if dropped_now.any() and self.cfg.drop_penalty_enabled:
            rewards = rewards + drop_penalty
            self._dropped_buf |= dropped_now

        self._ep_return = self._ep_return + rewards
        self._ep_len = self._ep_len + 1
        if success_first.any():
            success_ids = success_first.nonzero(as_tuple=False).squeeze(-1)
            success_steps = torch.clamp(self._ep_len[success_ids], min=1)
            self._success_steps_buf[success_ids] = success_steps

        if (
            self.reward_print_interval > 0
            and (self.step_count + 1) % self.reward_print_interval == 0
        ):
            if self._success_steps_window_count > 0:
                window_avg = self._success_steps_window_sum / self._success_steps_window_count
                window_min = self._success_steps_window_min
                window_max = self._success_steps_window_max
            else:
                window_avg = 0.0
                window_min = 0
                window_max = 0
            if self._success_window_done > 0:
                window_success_rate = self._success_window_success / self._success_window_done
            else:
                window_success_rate = 0.0
            self.extras.setdefault("log", {})
            self.extras["log"]["success_rate"] = torch.tensor(window_success_rate, device=self.device)
            self.extras["log"]["info/success_steps_mean_window"] = torch.tensor(window_avg, device=self.device)
            self.extras["log"]["info/success_steps_min_window"] = torch.tensor(float(window_min), device=self.device)
            self.extras["log"]["info/success_steps_max_window"] = torch.tensor(float(window_max), device=self.device)
            cube_height_mean = float(cube_pos[:, 2].mean().item())
            success_step = float(success_reward.mean().item())
            success_term = float(success_step * self.cfg.success_reward_weight)
            print(
                f"[奖励] step {self.step_count + 1}: "
                f"reach={reach_reward.mean().item():.3f} "
                f"(w={self.cfg.reach_weight}), lift={lift_reward.mean().item():.3f} "
                f"(w={self.cfg.lift_weight}), success={window_success_rate:.3f} "
                f"(w={self.cfg.success_reward_weight}), "
                f"success_step={success_step:.3f} "
                f"(term={success_term:.3f}), "
                f"cube_h={cube_height_mean:.4f}, "
                f"drop_penalty={drop_penalty.mean().item():.3f} "
                f"(count={int(dropped_now.sum().item())}), "
                f"success_steps(avg={window_avg:.1f},"
                f"min={window_min},max={window_max}), "
                f"total={rewards.mean().item():.3f}"
            )
            # reset windowed success-steps stats after printing
            self._success_steps_window_sum = 0.0
            self._success_steps_window_count = 0
            self._success_steps_window_min = None
            self._success_steps_window_max = None
            # reset windowed success counts after printing
            self._success_window_done = 0
            self._success_window_success = 0
            self._success_window_timeout = 0
            self._success_window_collision = 0

        self.step_count += 1
        self.prev_actions = self.processed_actions.detach().clone()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate done flags."""
        # Episode timeout - 基于步数检查
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = time_out.bool()

        # Success tracking (no longer ends episode)
        success = self._success_achieved_buf

        # 失败条件 - 机械臂关节碰撞地面
        # 获取机械臂所有关节的世界坐标位置
        joint_positions = self._robot.data.body_link_pos_w  # [num_envs, num_links, 3]
        
        # 检查是否有任何关节的Z坐标低于地面高度
        joint_z_positions = joint_positions[:, :, 2]  # [num_envs, num_links]
        collision_with_ground = torch.any(joint_z_positions < self.cfg.ground_height, dim=1)  # [num_envs]
        collision_with_ground = collision_with_ground.bool()

        # Combine done conditions
        dones = time_out | collision_with_ground

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.extras.setdefault("log", {})
            num_done = int(done_ids.numel())
            num_success = int(success[done_ids].sum().item())
            num_timeout = int(time_out[done_ids].sum().item())
            num_collision = int(collision_with_ground[done_ids].sum().item())
            self._success_window_done += num_done
            self._success_window_success += num_success
            self._success_window_timeout += num_timeout
            self._success_window_collision += num_collision
            ep_len = torch.clamp(self._ep_len[done_ids].to(torch.float32), min=1.0)
            ep_avg = self._ep_return[done_ids] / ep_len
            self._ep_avg_accum_sum = self._ep_avg_accum_sum + ep_avg.sum()
            self._ep_avg_accum_count = self._ep_avg_accum_count + ep_avg.numel()
            success_done_mask = success[done_ids]
            success_done_ids = done_ids[success_done_mask]
            if success_done_ids.numel() > 0:
                success_steps = torch.clamp(self._success_steps_buf[success_done_ids], min=1)
                self._last_success_steps_mean = float(success_steps.float().mean().item())
                self._last_success_steps_min = int(success_steps.min().item())
                self._last_success_steps_max = int(success_steps.max().item())
                steps_sum = float(success_steps.sum().item())
                steps_count = int(success_steps.numel())
                steps_min = int(success_steps.min().item())
                steps_max = int(success_steps.max().item())
                self._success_steps_window_sum += steps_sum
                self._success_steps_window_count += steps_count
                if self._success_steps_window_min is None:
                    self._success_steps_window_min = steps_min
                    self._success_steps_window_max = steps_max
                else:
                    self._success_steps_window_min = min(self._success_steps_window_min, steps_min)
                    self._success_steps_window_max = max(self._success_steps_window_max, steps_max)
            if getattr(self.cfg, "print_per_env_episode_avg", False):
                ids = done_ids.detach().cpu().numpy().tolist()
                vals = ep_avg.detach().cpu().numpy().tolist()
                details = " ".join([f"env{eid}={v:.3f}" for eid, v in zip(ids, vals)])
                print(f"[Episode Avg Reward per-env] {details}")
            self._ep_return[done_ids] = 0.0
            self._ep_len[done_ids] = 0
        else:
            pass

        if self.reward_print_interval > 0 and (self.step_count % self.reward_print_interval == 0):
            cnt = int(self._ep_avg_accum_count.item())
            if cnt > 0:
                avg = float(self._ep_avg_accum_sum.item()) / cnt
            else:
                avg = 0.0
            self.extras.setdefault("log", {})
            self.extras["log"]["info/episode_avg_reward_window"] = torch.tensor(avg, device=self.device)
            if cnt > 0:
                print(f"[Episode Avg Reward] avg={avg:.3f}")
                self._ep_avg_accum_sum = torch.tensor(0.0, device=self.device)
                self._ep_avg_accum_count = torch.tensor(0, dtype=torch.long, device=self.device)
        
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
        self._ep_return[env_ids] = 0.0
        self._ep_len[env_ids] = 0
        
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

        # 重置立方体位置 - 随机放置在木板上
        cube_state = self._cube.data.default_root_state[env_ids].clone()
        
        # 在 xy 平面添加随机偏移
        xy_offset = sample_uniform(-self.cfg.cube_pos_range, self.cfg.cube_pos_range, (len(env_ids), 2), self.device)
        cube_state[:, :2] += xy_offset
        
        # 轻微 z 抖动与随机 yaw，打破完全一致的初态
        # cube_state[:, 2] += sample_uniform(-0.002, 0.002, (len(env_ids),), self.device)
        rand_yaw = sample_uniform(-self.cfg.cube_rot_range, self.cfg.cube_rot_range, (len(env_ids),), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        cube_state[:, 3:7] = rand_quat
        
        # 添加环境原点偏移（包含z轴）
        cube_state[:, :3] += self.scene.env_origins[env_ids]
        
        # 写入仿真
        self._cube.write_root_state_to_sim(cube_state, env_ids=env_ids)

        # 记录每个 env 的真实 spawn 高度与抬升阈值
        com_pos_b = self._cube.data.body_com_pos_b[env_ids, 0]
        com_quat_b = self._cube.data.body_com_quat_b[env_ids, 0]
        com_pos_w, _ = math_utils.combine_frame_transforms(cube_state[:, :3], cube_state[:, 3:7], com_pos_b, com_quat_b)
        self._cube_spawn_height_per_env[env_ids] = com_pos_w[:, 2].clone()

        # 重置动作缓冲区，并为超时步数加入抖动，错峰重置
        self.actions[env_ids] = 0
        self.processed_actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.action_history[env_ids] = 0
        jitter = torch.randint(0, self.cfg.reset_jitter_max_steps, (len(env_ids),), device=self.device)
        self.episode_length_buf[env_ids] = jitter
        self._episode_start_length_buf[env_ids] = jitter
        self._randomize_board_colors(env_ids)
        self._ever_lifted_buf[env_ids] = False
        self._dropped_buf[env_ids] = False
        self._success_steps_buf[env_ids] = -1
        self._success_hold_buf[env_ids] = 0
        self._success_achieved_buf[env_ids] = False
