"""Minimal sim-to-real grasping scene using Franka and a fixed third-person camera."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
import torch.nn.functional as F
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom

from ..cylinder_grasping.cylinder_grasping_vision_only_resnet18 import (
    CylinderGraspingVisionOnlyCfg,
    CylinderGraspingVisionOnlyEnv,
)


@configclass
class Sim2RealGraspEnvCfg(CylinderGraspingVisionOnlyCfg):
    """Cylinder grasping scene simplified for a sim-to-real baseline."""

    bottle_height = 0.10
    bottle_radius = 0.025
    cap_height = 0.024
    cap_radius = 0.026
    cap_center_offset_z = 0.039

    observation_space = {
        "proprio_obs": 15,  # 7 arm joint positions + 7 arm joint velocities + 1 gripper opening width
        "action_history": 4,
        "wrist_resnet": 512,
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

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (0.98, 0.00, 1.2)
    viewer.lookat = (0.50, 0.00, 0.04)

    # Allow per-environment visual material overrides for plate color randomization.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=1.5,
        replicate_physics=False,
        lazy_sensor_update=True,
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.0, 0.0, 0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # Reuse the base env's "plate" slot as a simple black floor panel under the object.
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/floor_panel",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.005)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
            scale=(16, 24, 0.01),
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
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0),
                metallic=0.0,
                roughness=0.98,
            ),
        ),
    )

    cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        # Simple grasping rigid body for the bottle: keep one clean cylinder for physics.
        # The floor panel top is at z=0.01 m, so the center should start around z=0.06 m.
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0.0, 0.01 + 0.5 * bottle_height]),
        spawn=sim_utils.CylinderCfg(
            radius=bottle_radius,
            height=bottle_height,
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
                diffuse_color=(0.95, 0.95, 0.94),
                metallic=0.0,
                roughness=0.88,
            ),
        ),
    )

    cap = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cap",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.01 + bottle_height - 0.5 * cap_height]),
        spawn=sim_utils.CylinderCfg(
            radius=cap_radius,
            height=cap_height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=1.0,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.76, 0.76, 0.78),
                metallic=0.9,
                roughness=0.3,
            ),
        ),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.30803592681804604,
                "panda_joint2": -0.1153251832265955,
                "panda_joint3": 0.3018773037895521,
                "panda_joint4": -2.273053513963647,
                "panda_joint5": 0.04098702578428312,
                "panda_joint6": 2.1623620899157245,
                "panda_joint7": 0.7541767871490952,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Keep the inherited observation interface intact by feeding the third-person camera
    # through the base env's existing visual encoder path.
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        # Start from the RealSense D435 intrinsics, but tighten the effective field-of-view
        # to better match the cropped real deployment frame used for sim-to-real alignment.
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                820.0,
                0.0,
                322.553558349609,
                0.0,
                820.0,
                260.0,
                0.0,
                0.0,
                1.0,
            ],
            width=640,
            height=480,
            focal_length=1.0,
            focus_distance=400.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # A more top-down, zoomed-in deployment view that better matches the real robot.
            pos=(1.190000, 0.000000, 1.260000),
            rot=(0.683818, 0.179979, 0.179979, 0.683818),
            convention="opengl",
        ),
    )

    # Third-person image randomization applied before the ResNet18 encoder.
    wrist_visual_randomization_enabled = True
    wrist_brightness_randomization_enabled = True
    wrist_brightness_range = (0.80, 1.20)
    wrist_gamma_randomization_enabled = True
    wrist_gamma_range = (0.85, 1.20)
    wrist_contrast_randomization_enabled = True
    wrist_contrast_range = (0.80, 1.20)
    wrist_blur_randomization_enabled = True
    wrist_blur_probability = 0.35
    wrist_blur_kernel_sizes = (3, 5, 7)
    wrist_gaussian_noise_randomization_enabled = True
    wrist_gaussian_noise_std_range = (0.0, 0.025)

    # Scene appearance randomization.
    light_randomization_enabled = True
    light_intensity_range = (1200.0, 3200.0)
    light_color_temperature_range = (3200.0, 7600.0)
    ground_color_randomization_enabled = True
    ground_color_min = (0.02, 0.02, 0.02)
    ground_color_max = (0.22, 0.22, 0.22)
    plate_color_randomization_enabled = True
    plate_color_min = (0.02, 0.02, 0.02)
    plate_color_max = (0.80, 0.80, 0.80)

    # Decouple x/y spawn ranges for the bottle in the sim2real setup.
    cylinder_x_pos_range = 0.10
    cylinder_y_pos_range = 0.05

    # Require the bottle to stay reasonably upright before awarding lift/success rewards.
    lift_upright_tilt_threshold_deg = 20.0

    # Disable inherited training-time noise sources to keep the scene visually stable.
    cylinder_height = bottle_height
    cylinder_radius = bottle_radius
    lift_height = 0.0625
    lift_reward_start_lowest_height = 0.007
    success_lowest_height = 0.05
    success_hold_steps = 5
    max_gripper_opening_width = 0.08  # [m] total opening width across both fingers
    robot_joint_pos_noise = 0.0
    robot_joint_vel_noise = 0.0
    action_noise_scale = 0.0
    # Keep reset jitter effectively disabled while satisfying the inherited randint(0, max_steps) call.
    reset_jitter_max_steps = 1
    # Standard Franka can report base-link z very close to zero; keep a small tolerance.
    ground_height = -0.02


class Sim2RealGraspEnv(CylinderGraspingVisionOnlyEnv):
    """Minimal third-person Franka cylinder grasping environment."""

    cfg: Sim2RealGraspEnvCfg

    def __init__(self, cfg: Sim2RealGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # The inherited base env uses the tactile-gripper hand offset (0.11841 m).
        # Standard Franka Panda hand in Isaac Lab uses ~0.107 m for IK body offset.
        self._offset_pos = torch.tensor([0.0, 0.0, 0.107], device=self.device).repeat(self.num_envs, 1)
        self._max_gripper_joint_opening = 0.5 * self.cfg.max_gripper_opening_width
        self._cap_offset_pos = torch.tensor(
            [0.0, 0.0, self.cfg.cap_center_offset_z], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._cap_offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self._success_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._light_prim_path = "/World/Light"
        self._ground_material_bound = False
        self._plate_material_bound = [False] * self.num_envs
        self._blur_kernel_cache: dict[int, torch.Tensor] = {}
        self._cylinder_local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self._lift_upright_cos_threshold = math.cos(math.radians(float(self.cfg.lift_upright_tilt_threshold_deg)))
        self._sync_cap_visual()
        self._randomize_scene_visuals(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _setup_scene(self):
        """Setup the scene with a simple bottle body plus a visual-only silver cap."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self._cylinder

        self._cap = RigidObject(self.cfg.cap)
        self.scene.rigid_objects["cap"] = self._cap

        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        self.scene.clone_environments(copy_from_source=False)

        if prim_utils.is_prim_path_valid("/World/defaultDomeLight"):
            prim_utils.delete_prim("/World/defaultDomeLight")

        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera

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
                        pos=(0.0, 0.0, 0.107),
                    ),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _sample_cylinder_xy_offsets(self, count: int) -> torch.Tensor:
        offsets = torch.empty((count, 2), device=self.device, dtype=torch.float32)
        offsets[:, 0].uniform_(-float(self.cfg.cylinder_x_pos_range), float(self.cfg.cylinder_x_pos_range))
        offsets[:, 1].uniform_(-float(self.cfg.cylinder_y_pos_range), float(self.cfg.cylinder_y_pos_range))
        return offsets

    def _initialize_cylinder_positions(self):
        """Initialize bottle positions with wider x randomization and no z jitter."""
        cylinder_state = self._cylinder.data.default_root_state.clone()
        cylinder_state[:, :2] += self._sample_cylinder_xy_offsets(self.num_envs)

        rand_yaw = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        rand_yaw.uniform_(-float(self.cfg.cylinder_rot_range), float(self.cfg.cylinder_rot_range))
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        cylinder_state[:, 3:7] = rand_quat

        cylinder_state[:, :3] += self.scene.env_origins
        self._cylinder.write_root_state_to_sim(cylinder_state, torch.arange(self.num_envs, device=self.device))

    def _compute_cylinder_upright_cos(self, cylinder_quat: torch.Tensor) -> torch.Tensor:
        local_z_axis = self._cylinder_local_z_axis[: cylinder_quat.shape[0]]
        cylinder_up_axis = math_utils.quat_apply(cylinder_quat, local_z_axis)
        return torch.clamp(cylinder_up_axis[:, 2], min=-1.0, max=1.0)

    def _compute_cylinder_lowest_height(self, cylinder_pos: torch.Tensor, cylinder_quat: torch.Tensor) -> torch.Tensor:
        axis_z = torch.abs(self._compute_cylinder_upright_cos(cylinder_quat))
        radial_component = torch.sqrt(torch.clamp(1.0 - axis_z * axis_z, min=0.0))
        return cylinder_pos[:, 2] - 0.5 * self.cfg.bottle_height * axis_z - self.cfg.bottle_radius * radial_component

    def _sample_uniform_scalar(self, low: float, high: float, min_value: float | None = None) -> float:
        low = float(low)
        high = float(high)
        if min_value is not None:
            low = max(low, min_value)
            high = max(high, min_value)
        if high < low:
            high = low
        return float(torch.empty((), device=self.device).uniform_(low, high).item())

    def _sample_color(
        self, color_min: tuple[float, float, float], color_max: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        color_min_t = torch.tensor(color_min, dtype=torch.float32, device=self.device)
        color_max_t = torch.tensor(color_max, dtype=torch.float32, device=self.device)
        color = color_min_t + torch.rand((3,), device=self.device) * (color_max_t - color_min_t)
        return tuple(float(v) for v in color.tolist())

    def _resolve_visual_target_prim(self, root_prim_path: str) -> str:
        mesh_prims = sim_utils.get_all_matching_child_prims(
            root_prim_path,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
        )
        if mesh_prims:
            return mesh_prims[0].GetPath().pathString
        return root_prim_path

    def _apply_visual_material(
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
            print(f"[WARN] visual material update failed for {material_path}: {exc}")

    def _kelvin_to_rgb(self, temperature_kelvin: float) -> tuple[float, float, float]:
        temperature = max(1000.0, min(float(temperature_kelvin), 40000.0)) / 100.0
        if temperature <= 66.0:
            red = 255.0
            green = 99.4708025861 * math.log(max(temperature, 1e-6)) - 161.1195681661
            if temperature <= 19.0:
                blue = 0.0
            else:
                blue = 138.5177312231 * math.log(max(temperature - 10.0, 1e-6)) - 305.0447927307
        else:
            red = 329.698727446 * math.pow(max(temperature - 60.0, 1e-6), -0.1332047592)
            green = 288.1221695283 * math.pow(max(temperature - 60.0, 1e-6), -0.0755148492)
            blue = 255.0

        def _clamp_channel(value: float) -> float:
            return max(0.0, min(255.0, value)) / 255.0

        return (_clamp_channel(red), _clamp_channel(green), _clamp_channel(blue))

    def _randomize_dome_light(self):
        if not bool(getattr(self.cfg, "light_randomization_enabled", False)):
            return
        stage = sim_utils.stage_utils.get_current_stage()
        light_prim = stage.GetPrimAtPath(self._light_prim_path)
        if not light_prim.IsValid():
            return

        intensity = self._sample_uniform_scalar(*self.cfg.light_intensity_range, min_value=0.0)
        color_temperature = self._sample_uniform_scalar(
            *self.cfg.light_color_temperature_range,
            min_value=1000.0,
        )
        light_color = self._kelvin_to_rgb(color_temperature)

        try:
            sim_utils.safe_set_attribute_on_usd_prim(light_prim, "inputs:intensity", intensity, camel_case=True)
            sim_utils.safe_set_attribute_on_usd_prim(light_prim, "inputs:color", light_color, camel_case=True)
            sim_utils.safe_set_attribute_on_usd_prim(
                light_prim, "inputs:enable_color_temperature", True, camel_case=True
            )
            sim_utils.safe_set_attribute_on_usd_prim(
                light_prim, "inputs:color_temperature", color_temperature, camel_case=True
            )
        except Exception as exc:
            print(f"[WARN] dome light randomization failed: {exc}")

    def _randomize_ground_color(self):
        if not bool(getattr(self.cfg, "ground_color_randomization_enabled", False)):
            return
        ground_root = self.cfg.ground.prim_path
        ground_target = self._resolve_visual_target_prim(ground_root)
        ground_color = self._sample_color(self.cfg.ground_color_min, self.cfg.ground_color_max)
        bind = not self._ground_material_bound
        self._apply_visual_material(f"{ground_root}/material", ground_target, ground_color, bind=bind)
        if bind:
            self._ground_material_bound = True

    def _randomize_plate_colors(self, env_ids: torch.Tensor):
        if not bool(getattr(self.cfg, "plate_color_randomization_enabled", False)):
            return
        env_id_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        if not env_id_list:
            return
        plate_paths = self._plate.root_physx_view.prim_paths
        for env_id in env_id_list:
            plate_root = plate_paths[env_id]
            plate_target = self._resolve_visual_target_prim(plate_root)
            plate_color = self._sample_color(self.cfg.plate_color_min, self.cfg.plate_color_max)
            bind = not self._plate_material_bound[env_id]
            self._apply_visual_material(f"{plate_root}/material", plate_target, plate_color, bind=bind)
            if bind:
                self._plate_material_bound[env_id] = True

    def _randomize_scene_visuals(self, env_ids: torch.Tensor):
        self._randomize_dome_light()
        self._randomize_ground_color()
        self._randomize_plate_colors(env_ids)

    def _build_box_kernel(self, kernel_size: int) -> torch.Tensor:
        kernel = torch.ones((kernel_size, kernel_size), device=self.device, dtype=torch.float32)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel.repeat(3, 1, 1, 1)

    def _get_blur_kernel(self, kernel_size: int) -> torch.Tensor:
        if kernel_size not in self._blur_kernel_cache:
            self._blur_kernel_cache[kernel_size] = self._build_box_kernel(kernel_size)
        return self._blur_kernel_cache[kernel_size]

    def _sample_blur_kernel_size(self) -> int:
        kernel_sizes = tuple(int(k) for k in getattr(self.cfg, "wrist_blur_kernel_sizes", ()))
        kernel_sizes = tuple((k + 1) if (k > 1 and k % 2 == 0) else k for k in kernel_sizes if k > 1)
        if not kernel_sizes:
            return 0
        index = int(torch.randint(0, len(kernel_sizes), (1,), device=self.device).item())
        return kernel_sizes[index]

    def _apply_wrist_visual_randomization(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self.cfg, "wrist_visual_randomization_enabled", False)):
            return x

        x = torch.clamp(x, 0.0, 1.0)

        if bool(getattr(self.cfg, "wrist_blur_randomization_enabled", False)):
            blur_probability = float(getattr(self.cfg, "wrist_blur_probability", 0.0))
            if blur_probability > 0.0 and float(torch.rand((), device=self.device).item()) < blur_probability:
                kernel_size = self._sample_blur_kernel_size()
                if kernel_size > 1:
                    kernel = self._get_blur_kernel(kernel_size)
                    x = F.conv2d(x, kernel, padding=kernel_size // 2, groups=3)

        if bool(getattr(self.cfg, "wrist_brightness_randomization_enabled", False)):
            low, high = self.cfg.wrist_brightness_range
            low = max(0.0, float(low))
            high = max(low, float(high))
            brightness = torch.empty((x.shape[0], 1, 1, 1), device=self.device).uniform_(low, high)
            x = x * brightness

        if bool(getattr(self.cfg, "wrist_contrast_randomization_enabled", False)):
            low, high = self.cfg.wrist_contrast_range
            low = max(0.0, float(low))
            high = max(low, float(high))
            contrast = torch.empty((x.shape[0], 1, 1, 1), device=self.device).uniform_(low, high)
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * contrast + mean

        x = torch.clamp(x, 0.0, 1.0)

        if bool(getattr(self.cfg, "wrist_gamma_randomization_enabled", False)):
            low, high = self.cfg.wrist_gamma_range
            low = max(1e-3, float(low))
            high = max(low, float(high))
            gamma = torch.empty((x.shape[0], 1, 1, 1), device=self.device).uniform_(low, high)
            x = torch.pow(torch.clamp(x, min=1e-6, max=1.0), gamma)

        if bool(getattr(self.cfg, "wrist_gaussian_noise_randomization_enabled", False)):
            low, high = self.cfg.wrist_gaussian_noise_std_range
            low = max(0.0, float(low))
            high = max(low, float(high))
            noise_std = torch.empty((x.shape[0], 1, 1, 1), device=self.device).uniform_(low, high)
            x = x + torch.randn_like(x) * noise_std

        return torch.clamp(x, 0.0, 1.0)

    def _sync_cap_visual(self, env_ids: torch.Tensor | None = None):
        """Keep the silver cap aligned with the main bottle rigid body."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return

        cylinder_pos = self._cylinder.data.root_pos_w[env_ids]
        cylinder_quat = self._cylinder.data.root_quat_w[env_ids]
        cap_pos, cap_quat = math_utils.combine_frame_transforms(
            cylinder_pos,
            cylinder_quat,
            self._cap_offset_pos[env_ids],
            self._cap_offset_rot[env_ids],
        )
        cap_pose = torch.cat([cap_pos, cap_quat], dim=-1)
        cap_velocity = torch.cat(
            [self._cylinder.data.root_lin_vel_w[env_ids], self._cylinder.data.root_ang_vel_w[env_ids]], dim=-1
        )
        self._cap.write_root_pose_to_sim(cap_pose, env_ids=env_ids)
        self._cap.write_root_velocity_to_sim(cap_velocity, env_ids=env_ids)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Use a 15-D proprio vector: 7 arm positions + 7 arm velocities + 1 gripper opening width."""
        self._sync_cap_visual()
        use_resnet18 = bool(getattr(self, "_use_resnet18", False))
        if use_resnet18:
            self._use_resnet18 = False
        try:
            observations = super()._get_observations()
        finally:
            if use_resnet18:
                self._use_resnet18 = True
        obs = observations["policy"]

        wrist_rgb = self.wrist_camera.data.output.get("rgb")
        if wrist_rgb is None:
            wrist_feat = torch.zeros((self.num_envs, 512), device=self.device, dtype=torch.float32)
        elif use_resnet18:
            # The shared ImageNet encoder is frozen and must never update
            # BatchNorm running statistics during environment rollouts.
            self._resnet18.eval()
            x = wrist_rgb.to(torch.float32).permute(0, 3, 1, 2).contiguous() / 255.0
            x = self._apply_wrist_visual_randomization(x)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - self._imgnet_mean) / self._imgnet_std
            # Keep a regular detached tensor: the downstream Actor still needs
            # to save this input while computing gradients for its own weights.
            with torch.no_grad():
                wrist_feat = self._resnet18(x)
                wrist_feat = wrist_feat.view(self.num_envs, 512)
        else:
            wrist_feat = torch.zeros((self.num_envs, 512), device=self.device, dtype=torch.float32)

        arm_joint_pos = self._robot.data.joint_pos[:, :7]
        arm_joint_vel = self._robot.data.joint_vel[:, :7]
        gripper_opening = self._robot.data.joint_pos[:, self._finger_joint_ids].sum(dim=-1, keepdim=True)
        gripper_opening = torch.clamp(gripper_opening, 0.0, self.cfg.max_gripper_opening_width)
        obs["proprio_obs"] = torch.cat([arm_joint_pos, arm_joint_vel, gripper_opening], dim=-1)
        obs["wrist_resnet"] = wrist_feat
        obs.pop("gripper_state", None)

        return observations

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset selected environments and realign the visual bottle cap."""
        super(CylinderGraspingVisionOnlyEnv, self)._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros((len(env_ids), joint_pos.shape[1]), device=self.device, dtype=joint_pos.dtype)

        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        cylinder_state = self._cylinder.data.default_root_state[env_ids].clone()
        cylinder_state[:, :2] += self._sample_cylinder_xy_offsets(len(env_ids))

        rand_yaw = torch.empty((len(env_ids),), device=self.device, dtype=torch.float32)
        rand_yaw.uniform_(-float(self.cfg.cylinder_rot_range), float(self.cfg.cylinder_rot_range))
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        cylinder_state[:, 3:7] = rand_quat

        cylinder_state[:, :3] += self.scene.env_origins[env_ids]
        self._cylinder.write_root_state_to_sim(cylinder_state, env_ids=env_ids)

        self._cylinder_spawn_height_per_env[env_ids] = cylinder_state[:, 2].clone()
        self._lift_target_height_per_env[env_ids] = self._cylinder_spawn_height_per_env[env_ids] + self.cfg.lift_height

        self.actions[env_ids] = 0
        self.processed_actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.action_history[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self._success_hold_counter[env_ids] = 0
        self._randomize_scene_visuals(env_ids)
        self._sync_cap_visual(env_ids)

    def _apply_action(self):
        """Apply actions and clamp the gripper to the configured 0.08 m total opening width."""
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        jacobian = self._compute_frame_jacobian()
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        arm_joint_pos_des = arm_joint_pos_des[:, :7]
        arm_joint_pos_des = torch.where(torch.isfinite(arm_joint_pos_des), arm_joint_pos_des, joint_pos[:, :7])
        joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, :7, 0]
        joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, :7, 1]
        arm_joint_pos_des = torch.max(arm_joint_pos_des, joint_lower_limits.unsqueeze(0))
        arm_joint_pos_des = torch.min(arm_joint_pos_des, joint_upper_limits.unsqueeze(0))

        gripper_action = self.processed_actions[:, -1]
        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]
        gripper_delta = gripper_action.unsqueeze(-1) * 0.2
        gripper_pos_des = gripper_joint_pos + gripper_delta
        gripper_pos_des = torch.where(torch.isfinite(gripper_pos_des), gripper_pos_des, gripper_joint_pos)
        gripper_pos_des = torch.clamp(gripper_pos_des, 0.0, self._max_gripper_joint_opening)

        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_pos_des], dim=1)
        joint_pos_des = torch.where(torch.isfinite(joint_pos_des), joint_pos_des, joint_pos)
        self._robot.set_joint_position_target(joint_pos_des)

    def _get_rewards(self) -> torch.Tensor:
        """Use reach + thresholded lift shaping + success reward with fixed absolute height thresholds."""
        cylinder_pos = self._cylinder.data.root_pos_w
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )

        sigma = max(self.cfg.reach_sigma, 1e-6)
        reach_distance = torch.norm(cylinder_pos - ee_pos, dim=-1)
        reach_reward = 1.0 - torch.tanh(reach_distance / sigma)

        upright_cos = self._compute_cylinder_upright_cos(self._cylinder.data.root_quat_w)
        upright_tilt_deg = torch.rad2deg(torch.acos(upright_cos))
        upright_mask = (upright_cos >= self._lift_upright_cos_threshold).float()

        current_height = cylinder_pos[:, 2]
        current_lowest_height = self._compute_cylinder_lowest_height(cylinder_pos, self._cylinder.data.root_quat_w)
        lift_start_height = float(self.cfg.lift_reward_start_lowest_height)
        success_height = float(self.cfg.success_lowest_height)
        lift_span = max(success_height - lift_start_height, 1e-6)
        base_reward = (current_lowest_height > lift_start_height).float()
        linear_reward = torch.clamp((current_lowest_height - lift_start_height) / lift_span, min=0.0, max=1.0)
        lift_reward = (base_reward + linear_reward) * upright_mask

        success_reward = (current_lowest_height > success_height).float() * upright_mask

        rewards = (
            self.cfg.reach_weight * reach_reward
            + self.cfg.lift_weight * lift_reward
            + self.cfg.success_reward_weight * success_reward
        )

        log = self.extras.setdefault("log", {})
        log["reward/reach"] = reach_reward.mean().detach()
        log["reward/lift"] = lift_reward.mean().detach()
        log["reward/success"] = success_reward.mean().detach()
        log["reward/total"] = rewards.mean().detach()
        log["info/reach_distance"] = reach_distance.mean().detach()
        log["info/cylinder_avg_height"] = current_height.mean().detach()
        log["info/cylinder_lowest_height"] = current_lowest_height.mean().detach()
        log["info/cylinder_upright_cos"] = upright_cos.mean().detach()
        log["info/cylinder_tilt_deg"] = upright_tilt_deg.mean().detach()
        log["info/success_hold_steps"] = self._success_hold_counter.to(torch.float32).mean().detach()

        if (
            self.reward_print_interval > 0
            and (self.step_count + 1) % self.reward_print_interval == 0
        ):
            print(
                f"[奖励] step {self.step_count + 1}: "
                f"reach={reach_reward.mean().item():.3f} (w={self.cfg.reach_weight}), "
                f"lift={lift_reward.mean().item():.3f} (w={self.cfg.lift_weight}), "
                f"success={success_reward.mean().item():.3f} (w={self.cfg.success_reward_weight}), "
                f"tilt={upright_tilt_deg.mean().item():.2f} deg, "
                f"hold={self._success_hold_counter.float().mean().item():.2f}/{self.cfg.success_hold_steps}, "
                f"center_z={current_height.mean().item():.4f} m, "
                f"lowest_z={current_lowest_height.mean().item():.4f} m, "
                f"total={rewards.mean().item():.3f}"
            )

        action_curr = torch.where(
            torch.isfinite(self.processed_actions),
            self.processed_actions,
            torch.zeros_like(self.processed_actions),
        )
        self.prev_actions = action_curr.detach().clone()
        self.step_count += 1

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate on timeout, ground collision, or sustained upright lift success."""
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).bool()

        current_lowest_height = self._compute_cylinder_lowest_height(
            self._cylinder.data.root_pos_w,
            self._cylinder.data.root_quat_w,
        )
        upright_cos = self._compute_cylinder_upright_cos(self._cylinder.data.root_quat_w)
        above_success_height = (current_lowest_height > float(self.cfg.success_lowest_height)) & (
            upright_cos >= self._lift_upright_cos_threshold
        )
        self._success_hold_counter = torch.where(
            above_success_height,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        success = self._success_hold_counter >= self.cfg.success_hold_steps

        joint_positions = self._robot.data.body_link_pos_w
        joint_z_positions = joint_positions[:, :, 2]
        collision_with_ground = torch.any(joint_z_positions < self.cfg.ground_height, dim=1).bool()

        dones = time_out | collision_with_ground | success
        return dones, time_out
