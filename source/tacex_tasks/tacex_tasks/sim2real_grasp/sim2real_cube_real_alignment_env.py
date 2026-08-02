"""Real-reference-aligned Franka cube grasping environment.

The measured values in this module come from
``20260711_214450_real_alignment_reference/alignment_reference.json``.
The current camera intrinsics and pose were supplied separately; their
calibration provenance remains pending confirmation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    TiledCamera,
    TiledCameraCfg,
)
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from .sim2real_cube_grasp_env import Sim2RealCubeGraspEnv, Sim2RealCubeGraspEnvCfg


_TABLE_COLLISION_ROBOT_BODY_NAMES = (
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
)

_CLEAN_PLATE_COLOR = (0.02, 0.02, 0.02)
_CLEAN_BACKDROP_COLOR = (0.01, 0.01, 0.01)
_CLEAN_DOME_LIGHT_COLOR = (0.75, 0.75, 0.75)


@configclass
class Sim2RealCubeRealAlignmentEnvCfg(Sim2RealCubeGraspEnvCfg):
    """Cube task aligned to the 2026-07-11 real Franka/D435 reference capture."""

    alignment_reference_id = "20260711_214450_real_alignment_reference"
    camera_extrinsics_source = "user_provided_base_T_camera_color_optical_20260726"
    camera_intrinsics_source = "20260726_194811_d435_crop_samples"
    deployment_camera_serial = "215322076207"
    camera_raw_resolution = (640, 480)
    camera_raw_intrinsic_matrix = (
        604.897400,
        0.0,
        320.980103,
        0.0,
        605.085815,
        247.913223,
        0.0,
        0.0,
        1.0,
    )
    camera_crop_roi_xywh = (100, 34, 400, 398)
    camera_model_intrinsic_matrix = (
        338.742544,
        0.0,
        123.748857,
        0.0,
        340.550811,
        120.393372,
        0.0,
        0.0,
        1.0,
    )
    # Omniverse 4.5 forces square pixels and a centered principal point. Render
    # a slightly wider centered view, then use one fixed batched GPU warp so
    # the RGB seen by ResNet has camera_model_intrinsic_matrix exactly. A
    # 300-pixel native focal length covers every target ray without padding.
    camera_native_render_intrinsic_matrix = (
        300.0,
        0.0,
        112.0,
        0.0,
        300.0,
        112.0,
        0.0,
        0.0,
        1.0,
    )
    camera_nominal_intrinsic_compensation_enabled = True
    camera_nominal_intrinsic_compensation_mode = (
        "gpu_affine_grid_centered_square_pixel_render_to_calibrated_k"
    )
    camera_resize_interpolation = "bilinear"
    gripper_control_mode = "total_width_delta_cached_target"
    action_scale = 0.05  # [m] Cartesian increment per 30 Hz policy step at |action|=1
    gripper_width_delta_scale = 0.01  # [m] total-width increment per policy step at |action|=1
    arm_ik_tcp_source = "panda_hand_fixed_offset"
    arm_ik_tcp_offset_m = (0.0, 0.0, 0.1034)
    reach_center_source = "mean_of_left_and_right_fingertip_centers"
    fingertip_local_offset_m = (0.0, 0.0, 0.045)

    # Measured real target: 5 cm cube. Keep this asset local to the aligned task
    # so TacEx-Sim2Real-Cube-Grasp-v0 retains its existing 6 cm target.
    cube_size = (0.05, 0.05, 0.05)
    cube_half_xy_extent = 0.5 * max(cube_size[0], cube_size[1])
    cylinder_radius = cube_half_xy_extent
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.001 + 0.5 * cube_size[2])),
        spawn=sim_utils.CuboidCfg(
            size=cube_size,
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

    # Shared Cube reward uses settled center-of-mass lift: 0--35 mm maps
    # linearly to [0, 1]. The 35 mm target is 70% of the measured cube edge.
    lift_reference_mode = "center_of_mass"
    lift_reward_start_delta = 0.0
    success_lift_delta = 0.035
    success_requires_upright = False
    lift_tilt_curriculum_enabled = False
    episode_success_rate_window_steps = 200

    # Keep 60 Hz physics for contact stability while camera, rendering, policy,
    # action application, observations, rewards, and history advance at 30 Hz.
    decimation = 2
    episode_length_s = 5.0  # 5 s * 30 policy steps/s = 150 policy steps

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.3077768694457032,
                "panda_joint2": -0.11490349419892411,
                "panda_joint3": 0.30181493015057165,
                "panda_joint4": -2.2731998141397707,
                "panda_joint5": 0.040613268755509774,
                "panda_joint6": 2.162421075317457,
                "panda_joint7": 0.7543247225501507,
                # Real gripper width was 0.0400015078 m; Panda finger joints each
                # contribute half of the total opening width.
                "panda_finger_joint.*": 0.020000753924250603,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # The real D435 path captures 640x480, crops x=[100, 500), y=[34, 432),
    # then bilinearly resizes the 400x398 crop to 224x224. Keep the simulator
    # camera buffer at 224x224 and compensate Omniverse's centered/square-pixel
    # limitation in the GPU observation path.
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=1.0 / 30.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=list(camera_native_render_intrinsic_matrix),
            width=224,
            height=224,
            focal_length=1.0,
            focus_distance=0.8,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # User-provided base_T_camera_color_optical pose. Quaternion is
            # converted from ROS xyzw to Isaac wxyz without changing axes.
            pos=(1.172904219177, 0.031653013416, 0.512212537004),
            rot=(-0.375287920087, 0.607013774710, 0.582420741286, -0.389203461533),
            convention="ros",
        ),
    )

    # A finite 1 mm thickness retains robust rigid-body contacts while making
    # the visible board top effectively coincide with z=0.
    plate_thickness_m = 0.001
    plate_top_height_m = 0.001
    plate_top_offset = 0.5 * plate_thickness_m
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/floor_panel",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.50, 0.0, 0.5 * plate_thickness_m)),
        spawn=sim_utils.CuboidCfg(
            size=(3.0, 3.0, plate_thickness_m),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
                diffuse_color=_CLEAN_PLATE_COLOR,
                metallic=0.0,
                roughness=0.98,
            ),
        ),
    )

    table_collision_robot_body_names = _TABLE_COLLISION_ROBOT_BODY_NAMES
    table_collision_force_threshold_n = 1.0
    table_collision_penalty = -10.0
    table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        # One sensor body per environment is required for filtered contacts.
        prim_path="/World/envs/env_.*/floor_panel",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        filter_prim_paths_expr=[
            f"/World/envs/env_.*/Robot/{body_name}"
            for body_name in _TABLE_COLLISION_ROBOT_BODY_NAMES
        ],
    )

    # The clean profile uses the near-black curtain visible in the new capture.
    # This asset is visual-only and cannot affect contacts or rewards.
    backdrop = RigidObjectCfg(
        prim_path="/World/envs/env_.*/real_alignment_backdrop",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.32, 0.0, 1.25)),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 2.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=_CLEAN_BACKDROP_COLOR,
                metallic=0.0,
                roughness=0.95,
            ),
        ),
    )

    # Clean alignment profile: only the cube XY reset position is randomized.
    wrist_visual_randomization_enabled = False
    wrist_brightness_randomization_enabled = False
    wrist_gamma_randomization_enabled = False
    wrist_contrast_randomization_enabled = False
    wrist_blur_randomization_enabled = False
    wrist_gaussian_noise_randomization_enabled = False
    light_randomization_enabled = False
    ground_color_randomization_enabled = False
    plate_color_randomization_enabled = False

    # Absolute robot-root position is default cube position (0.50, 0.00) plus
    # these symmetric offsets. Training starts near the center and expands.
    cube_image_aligned_xy_offset = (0.0, 0.0)
    cube_x_pos_range = 0.05
    cube_y_pos_range = 0.05
    cube_position_curriculum_enabled = True
    cube_position_curriculum_initial_x_range = 0.02
    cube_position_curriculum_initial_y_range = 0.02
    cube_position_curriculum_start_step = 20_000
    cube_position_curriculum_end_step = 100_000
    cube_position_curriculum_step_offset = 0
    cube_position_curriculum_force_full_range = False

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation


@configclass
class Sim2RealCubeRealAlignmentDREnvCfg(Sim2RealCubeRealAlignmentEnvCfg):
    """Real-alignment Cube profile with curriculum-scaled visual randomization."""

    # DirectRLEnv.common_step_counter advances once per 30 Hz policy step,
    # independent of num_envs. Keep the first 100k steps on the exact Clean
    # visual profile so grasping and the cube-position curriculum can converge
    # before every DR range grows continuously from zero to full scale.
    dr_curriculum_enabled = True
    dr_curriculum_start_step = 100_000
    dr_curriculum_end_step = 220_000
    dr_curriculum_initial_scale = 0.0
    # Explicit resume hook: set this to the restored outer training timestep.
    dr_curriculum_step_offset = 0

    # Per-env camera perturbation in the calibrated camera frame:
    # T_sample = T_calib * delta_T.
    camera_pose_randomization_enabled = True
    camera_position_delta_max_m = (0.003, 0.003, 0.003)
    camera_rotation_delta_max_deg = (1.0, 1.0, 1.0)

    # Omniverse does not render a non-centred principal point reliably. Apply
    # focal/principal perturbations as a batched GPU image warp.
    camera_intrinsic_warp_enabled = True
    camera_focal_scale_range = (0.985, 1.015)
    camera_principal_point_shift_max_px = (2.0, 2.0)

    # Per-env parameters are sampled once per episode. Pixel noise itself is
    # sampled each frame using the episode-fixed noise standard deviation.
    wrist_visual_randomization_enabled = True
    wrist_brightness_randomization_enabled = True
    wrist_brightness_range = (0.85, 1.15)
    wrist_gamma_randomization_enabled = True
    wrist_gamma_range = (0.85, 1.15)
    wrist_contrast_randomization_enabled = True
    wrist_contrast_range = (0.85, 1.15)
    wrist_saturation_randomization_enabled = True
    wrist_saturation_range = (0.85, 1.15)
    wrist_hue_randomization_enabled = True
    wrist_hue_max_deg = 5.0
    wrist_white_balance_randomization_enabled = True
    wrist_white_balance_shift_max = 0.08
    wrist_blur_randomization_enabled = True
    wrist_blur_probability = 0.15
    wrist_blur_kernel_sizes = (3,)
    wrist_gaussian_noise_randomization_enabled = True
    wrist_gaussian_noise_std_range = (0.0, 0.01)

    # DomeLight is stage-global, so update it only on all-env resets. Per-env
    # exposure and white balance are handled by GPU post-processing.
    light_randomization_enabled = True
    light_nominal_intensity = 2000.0
    light_nominal_color = _CLEAN_DOME_LIGHT_COLOR
    light_intensity_range = (1000.0, 3000.0)
    light_nominal_color_temperature = 5500.0
    light_color_temperature_range = (3800.0, 7200.0)

    # Keep the global ground fixed. Both material distributions are centered on
    # the exact Clean colors so scale=0 reproduces the Clean rendered scene.
    ground_color_randomization_enabled = False
    plate_color_randomization_enabled = True
    plate_color_center = _CLEAN_PLATE_COLOR
    plate_color_min = (0.01, 0.01, 0.01)
    plate_color_max = (0.05, 0.05, 0.05)
    backdrop_color_randomization_enabled = True
    backdrop_color_center = _CLEAN_BACKDROP_COLOR
    backdrop_color_min = (0.005, 0.005, 0.005)
    backdrop_color_max = (0.03, 0.03, 0.03)


class Sim2RealCubeRealAlignmentEnv(Sim2RealCubeGraspEnv):
    """Cube grasping environment with real-reference robot and camera parameters."""

    cfg: Sim2RealCubeRealAlignmentEnvCfg

    def __init__(self, cfg: Sim2RealCubeRealAlignmentEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._offset_pos = torch.tensor(
            self.cfg.arm_ik_tcp_offset_m,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._fingertip_local_offset = torch.tensor(
            self.cfg.fingertip_local_offset_m,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._reset_episode_success_statistics()

    def _reset_episode_success_statistics(self) -> None:
        """Reset cumulative and rolling policy-step episode outcome statistics."""
        window_steps = max(
            1,
            int(getattr(self.cfg, "episode_success_rate_window_steps", 200)),
        )
        self._episode_success_window_steps = window_steps
        self._episode_completed_per_step = torch.zeros(
            (window_steps,), device=self.device, dtype=torch.float32
        )
        self._episode_success_per_step = torch.zeros(
            (window_steps,), device=self.device, dtype=torch.float32
        )
        self._episode_success_window_write_idx = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self._episode_success_window_step_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self._episode_completed_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self._episode_success_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )

    def _ensure_episode_success_statistics(self) -> None:
        """Lazily initialize statistics for legacy pickled environment configs."""
        expected_steps = max(
            1,
            int(getattr(self.cfg, "episode_success_rate_window_steps", 200)),
        )
        if (
            not hasattr(self, "_episode_completed_per_step")
            or self._episode_completed_per_step.shape != (expected_steps,)
        ):
            self._reset_episode_success_statistics()

    def _record_episode_outcomes_for_step(
        self,
        completed_count: torch.Tensor,
        success_count: torch.Tensor,
    ) -> None:
        """Append aggregate completed/successful episode counts for one policy step."""
        self._ensure_episode_success_statistics()
        completed_count = completed_count.to(device=self.device, dtype=torch.long).reshape(())
        success_count = success_count.to(device=self.device, dtype=torch.long).reshape(())
        write_idx = self._episode_success_window_write_idx
        self._episode_completed_per_step[write_idx] = completed_count.to(torch.float32)
        self._episode_success_per_step[write_idx] = success_count.to(torch.float32)
        self._episode_success_window_write_idx.copy_(
            (write_idx + 1) % self._episode_success_window_steps
        )
        self._episode_success_window_step_count.add_(1).clamp_(
            max=self._episode_success_window_steps
        )
        self._episode_completed_count.add_(completed_count)
        self._episode_success_count.add_(success_count)

    def _episode_success_statistics(self) -> dict[str, torch.Tensor]:
        """Return scalar tensors for cumulative and recent episode success rates."""
        self._ensure_episode_success_statistics()
        completed = self._episode_completed_count.to(dtype=torch.float32)
        successes = self._episode_success_count.to(dtype=torch.float32)
        window_completed = self._episode_completed_per_step.sum()
        window_successes = self._episode_success_per_step.sum()
        window_step_count = self._episode_success_window_step_count.to(dtype=torch.float32)
        cumulative_rate = successes / torch.clamp(completed, min=1.0)
        window_rate = window_successes / torch.clamp(window_completed, min=1.0)
        return {
            "cumulative_rate": cumulative_rate,
            "window_rate": window_rate,
            "completed_count": completed,
            "success_count": successes,
            "window_completed_count": window_completed,
            "window_success_count": window_successes,
            "window_step_count": window_step_count,
        }

    def _publish_episode_success_statistics(self) -> None:
        """Expose stable metric keys to skrl training and play CSV logging."""
        metrics = self._episode_success_statistics()
        log = self.extras.setdefault("log", {})
        log["success_rate"] = metrics["cumulative_rate"].detach()
        log["recent_success_rate"] = metrics["window_rate"].detach()
        log["episode_success_rate_window"] = metrics["window_rate"].detach()
        log["info/episode_success_rate_window"] = metrics["window_rate"].detach()
        log["info/episode_success_rate_cumulative"] = metrics["cumulative_rate"].detach()
        log["info/episode_completed_count"] = metrics["completed_count"].detach()
        log["info/episode_success_count"] = metrics["success_count"].detach()
        log["info/episode_completed_count_window"] = metrics[
            "window_completed_count"
        ].detach()
        log["info/episode_success_count_window"] = metrics[
            "window_success_count"
        ].detach()
        log["info/episode_success_window_step_count"] = metrics[
            "window_step_count"
        ].detach()

    def _apply_nominal_camera_intrinsic_compensation(self, x: torch.Tensor) -> torch.Tensor:
        """Map the centered Omniverse render to the calibrated 224x224 camera K."""
        if not bool(self.cfg.camera_nominal_intrinsic_compensation_enabled):
            return x

        batch_size, _, height, width = x.shape
        if (width, height) != (
            int(self.cfg.wrist_camera.width),
            int(self.cfg.wrist_camera.height),
        ):
            raise ValueError(
                "Nominal camera intrinsic compensation expects the native camera "
                f"resolution {(self.cfg.wrist_camera.width, self.cfg.wrist_camera.height)}, "
                f"got {(width, height)}."
            )

        target = self.cfg.camera_model_intrinsic_matrix
        native = self.cfg.camera_native_render_intrinsic_matrix
        scale_x = float(native[0]) / float(target[0])
        scale_y = float(native[4]) / float(target[4])
        # align_corners=False maps normalized coordinates through pixel centers.
        # Include the half-pixel term so target (cx, cy) samples the native
        # centered principal point exactly even when the focal scales differ.
        translate_x = (
            2.0
            * (float(native[2]) + 0.5 - scale_x * (float(target[2]) + 0.5))
            / float(width)
            - 1.0
            + scale_x
        )
        translate_y = (
            2.0
            * (float(native[5]) + 0.5 - scale_y * (float(target[5]) + 0.5))
            / float(height)
            - 1.0
            + scale_y
        )

        cache_key = (height, width, x.device, x.dtype)
        if getattr(self, "_nominal_intrinsic_grid_key", None) != cache_key:
            theta = torch.tensor(
                [
                    [scale_x, 0.0, translate_x],
                    [0.0, scale_y, translate_y],
                ],
                device=x.device,
                dtype=x.dtype,
            ).unsqueeze(0)
            self._nominal_intrinsic_grid = F.affine_grid(
                theta,
                size=(1, x.shape[1], height, width),
                align_corners=False,
            )
            self._nominal_intrinsic_grid_key = cache_key

        grid = self._nominal_intrinsic_grid.expand(batch_size, -1, -1, -1)
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def _apply_wrist_visual_randomization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fixed calibrated-K warp before the clean profile hooks."""
        x = self._apply_nominal_camera_intrinsic_compensation(x)
        return super()._apply_wrist_visual_randomization(x)

    def _compute_fingertip_positions_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform the nominal inner fingertip centers into world coordinates."""
        left_pos = self._robot.data.body_link_pos_w[:, self._left_finger_body_idx]
        left_quat = self._robot.data.body_link_quat_w[:, self._left_finger_body_idx]
        right_pos = self._robot.data.body_link_pos_w[:, self._right_finger_body_idx]
        right_quat = self._robot.data.body_link_quat_w[:, self._right_finger_body_idx]
        left_tip, _ = math_utils.combine_frame_transforms(
            left_pos,
            left_quat,
            self._fingertip_local_offset,
            self._offset_rot,
        )
        right_tip, _ = math_utils.combine_frame_transforms(
            right_pos,
            right_quat,
            self._fingertip_local_offset,
            self._offset_rot,
        )
        return left_tip, right_tip

    def _compute_reach_center_world(self) -> torch.Tensor:
        """Return the midpoint of the two nominal fingertip contact centers."""
        left_tip, right_tip = self._compute_fingertip_positions_world()
        return 0.5 * (left_tip + right_tip)

    def _compute_action_gate_finger_tip_z(self) -> torch.Tensor:
        """Use the same physical fingertip midpoint as the reach center."""
        return self._compute_reach_center_world()[:, 2]

    def _compute_action_gate_center_xy(self) -> torch.Tensor:
        """Use the physical fingertip midpoint for object-alignment gating."""
        return self._compute_reach_center_world()[:, :2]

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Make privileged gripper state use the same fingertip midpoint as reach."""
        observations = super()._get_observations()
        obs = observations["policy"]
        left_tip, right_tip = self._compute_fingertip_positions_world()
        center = 0.5 * (left_tip + right_tip)

        left_quat = self._robot.data.body_link_quat_w[:, self._left_finger_body_idx]
        right_quat = self._robot.data.body_link_quat_w[:, self._right_finger_body_idx]
        left_offset_w = math_utils.quat_apply(left_quat, self._fingertip_local_offset)
        right_offset_w = math_utils.quat_apply(right_quat, self._fingertip_local_offset)
        left_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._left_finger_body_idx]
        right_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._right_finger_body_idx]
        left_tip_vel = (
            self._robot.data.body_link_lin_vel_w[:, self._left_finger_body_idx]
            + torch.cross(left_ang_vel, left_offset_w, dim=-1)
        )
        right_tip_vel = (
            self._robot.data.body_link_lin_vel_w[:, self._right_finger_body_idx]
            + torch.cross(right_ang_vel, right_offset_w, dim=-1)
        )

        target_pos = self._cube.data.root_pos_w - center
        obs["critic_gripper_pos"] = center
        obs["critic_gripper_quat"] = self._robot.data.body_link_quat_w[:, self._body_idx]
        obs["critic_gripper_lin_vel"] = 0.5 * (left_tip_vel + right_tip_vel)
        obs["critic_gripper_ang_vel"] = 0.5 * (left_ang_vel + right_ang_vel)
        obs["critic_target_pos"] = target_pos
        obs["critic_target_distance"] = torch.norm(target_pos, dim=-1, keepdim=True)
        return observations

    def _current_cube_xy_half_ranges(self) -> tuple[float, float]:
        """Return curriculum-scaled XY half-ranges in the robot-root frame."""
        full_x = float(self.cfg.cube_x_pos_range)
        full_y = float(self.cfg.cube_y_pos_range)
        if bool(self.cfg.cube_position_curriculum_force_full_range):
            return full_x, full_y
        if not bool(self.cfg.cube_position_curriculum_enabled):
            return full_x, full_y

        start = int(self.cfg.cube_position_curriculum_start_step)
        end = max(int(self.cfg.cube_position_curriculum_end_step), start + 1)
        step = int(getattr(self, "common_step_counter", 0)) + int(
            self.cfg.cube_position_curriculum_step_offset
        )
        progress = min(max((step - start) / float(end - start), 0.0), 1.0)
        initial_x = float(self.cfg.cube_position_curriculum_initial_x_range)
        initial_y = float(self.cfg.cube_position_curriculum_initial_y_range)
        return (
            initial_x + progress * (full_x - initial_x),
            initial_y + progress * (full_y - initial_y),
        )

    def _sample_cube_xy_offsets(self, count: int) -> torch.Tensor:
        """Sample around absolute nominal position (0.50, 0.00) using the curriculum."""
        range_x, range_y = self._current_cube_xy_half_ranges()
        offsets = torch.empty((count, 2), device=self.device, dtype=torch.float32)
        offsets[:, 0].uniform_(-range_x, range_x)
        offsets[:, 1].uniform_(-range_y, range_y)
        return offsets

    def _register_table_contact_sensor(self) -> None:
        """Register one filtered table-contact sensor per parallel environment."""
        self.table_contact_sensor = ContactSensor(self.cfg.table_contact_sensor)
        self.scene.sensors["table_contact_sensor"] = self.table_contact_sensor

    def _compute_table_robot_contact_force(self) -> torch.Tensor:
        """Return maximum normal table/robot force over both physics substeps."""
        force_history = self.table_contact_sensor.data.force_matrix_w_history
        if force_history is None:
            raise RuntimeError(
                "table_contact_sensor has no filtered force matrix; "
                "verify plate contact reporting and robot filter prim paths."
            )
        # force_history: [N, physics_history=2, table_body=1, robot_filters=10, xyz=3].
        force_norm = torch.linalg.vector_norm(force_history, dim=-1)
        return force_norm.amax(dim=(1, 2, 3))

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Record episode outcomes in a rolling 200-policy-step window."""
        dones, time_out = super()._get_dones()
        success = self._success_hold_counter >= int(self.cfg.success_hold_steps)
        self._record_episode_outcomes_for_step(
            completed_count=dones.to(dtype=torch.long).sum(),
            success_count=(dones & success).to(dtype=torch.long).sum(),
        )
        self._publish_episode_success_statistics()
        return dones, time_out

    def _compute_additional_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply one fixed penalty when any active Franka link contacts the table."""
        max_force = self._compute_table_robot_contact_force()
        collision = max_force > float(self.cfg.table_collision_force_threshold_n)
        penalty = collision.to(dtype=max_force.dtype) * float(self.cfg.table_collision_penalty)
        self._last_table_collision_force = max_force.detach().clone()
        self._last_table_collision = collision.detach().clone()
        self._last_table_collision_penalty = penalty.detach().clone()
        return penalty, {
            "reward/table_collision": penalty.mean().detach(),
            "info/table_collision_fraction": collision.to(torch.float32).mean().detach(),
            "info/table_collision_max_force_n": max_force.max().detach(),
            "info/table_collision_mean_force_n": max_force.mean().detach(),
        }

    def _additional_reward_print_fields(self) -> str:
        fields = ""
        if hasattr(self, "_last_table_collision"):
            fields = (
                f"table_collision={self._last_table_collision.float().mean().item():.3f}, "
                f"table_force_max={self._last_table_collision_force.max().item():.2f} N, "
                f"table_penalty={self._last_table_collision_penalty.mean().item():.3f}, "
            )
        metrics = self._episode_success_statistics()
        return (
            fields
            + f"episode_success_window={metrics['window_rate'].item():.3f} "
            + f"({int(metrics['window_success_count'].item())}/"
            + f"{int(metrics['window_completed_count'].item())}, "
            + f"last {int(metrics['window_step_count'].item())} policy steps), "
            + f"episode_success_total={metrics['cumulative_rate'].item():.3f} "
            + f"({int(metrics['success_count'].item())}/"
            + f"{int(metrics['completed_count'].item())}), "
        )

    def _ensure_desired_gripper_width_buffer(self) -> None:
        """Create the per-environment cached total-width target when first needed."""
        if hasattr(self, "_desired_gripper_width") and self._desired_gripper_width.shape == (self.num_envs,):
            return

        default_finger_pos = self._robot.data.default_joint_pos[:, self._finger_joint_ids]
        default_width = default_finger_pos.sum(dim=-1)
        self._desired_gripper_width = torch.clamp(
            default_width,
            min=0.0,
            max=float(self.cfg.max_gripper_opening_width),
        ).detach().clone()

    def _pre_physics_step(self, actions: torch.Tensor):
        """Update arm commands and integrate one total-width gripper increment."""
        super()._pre_physics_step(actions)
        self._ensure_desired_gripper_width_buffer()

        raw_gripper_action = self.actions[:, -1]
        bounded_gripper_action = torch.where(
            torch.isfinite(raw_gripper_action),
            torch.clamp(raw_gripper_action, min=-1.0, max=1.0),
            torch.zeros_like(raw_gripper_action),
        )
        requested_width_delta = bounded_gripper_action * float(self.cfg.gripper_width_delta_scale)
        self._desired_gripper_width = torch.clamp(
            self._desired_gripper_width + requested_width_delta,
            min=0.0,
            max=float(self.cfg.max_gripper_opening_width),
        )

        # The fourth processed/history component is the requested total-width
        # increment. It intentionally remains the requested value at width
        # saturation, matching the pre-limit history semantics used for xyz.
        self.processed_actions[:, -1] = requested_width_delta
        self.action_history[:, -1] = requested_width_delta

    def _apply_action(self):
        """Apply arm targets and resend one cached symmetric gripper-width target."""
        super()._apply_action()
        self._ensure_desired_gripper_width_buffer()

        per_finger_target = (0.5 * self._desired_gripper_width).unsqueeze(-1)
        per_finger_target = per_finger_target.expand(-1, len(self._finger_joint_ids))
        self._robot.set_joint_position_target(
            per_finger_target,
            joint_ids=self._finger_joint_ids,
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset the inherited environment and restore selected width targets."""
        super()._reset_idx(env_ids)
        self.table_contact_sensor.reset(env_ids)
        self._ensure_desired_gripper_width_buffer()

        reset_finger_pos = self._robot.data.default_joint_pos[env_ids][:, self._finger_joint_ids]
        reset_width = torch.clamp(
            reset_finger_pos.sum(dim=-1),
            min=0.0,
            max=float(self.cfg.max_gripper_opening_width),
        )
        self._desired_gripper_width[env_ids] = reset_width

    def _setup_scene(self):
        """Set up the inherited cube task plus a per-environment visual backdrop."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cube = RigidObject(self.cfg.cube)
        self._cylinder = self._cube
        self.scene.rigid_objects["cube"] = self._cube

        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        self._backdrop = RigidObject(self.cfg.backdrop)
        self.scene.rigid_objects["real_alignment_backdrop"] = self._backdrop

        self.scene.clone_environments(copy_from_source=False)
        self._register_table_contact_sensor()

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
                    offset=OffsetCfg(pos=self.cfg.arm_ik_tcp_offset_m),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
                    name="left_fingertip",
                    offset=OffsetCfg(pos=self.cfg.fingertip_local_offset_m),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
                    name="right_fingertip",
                    offset=OffsetCfg(pos=self.cfg.fingertip_local_offset_m),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path,
            ground.spawn,
            translation=ground.init_state.pos,
            orientation=ground.init_state.rot,
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=_CLEAN_DOME_LIGHT_COLOR)
        light_cfg.func("/World/Light", light_cfg)


class Sim2RealCubeRealAlignmentDREnv(Sim2RealCubeRealAlignmentEnv):
    """Real-alignment Cube environment with curriculum-scaled per-env visual DR."""

    cfg: Sim2RealCubeRealAlignmentDREnvCfg

    def _normalize_dr_env_ids(self, env_ids: torch.Tensor | list[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _dr_curriculum_scale(self) -> float:
        """Return the global DR range multiplier for the current outer policy step."""
        if not bool(getattr(self.cfg, "dr_curriculum_enabled", False)):
            return 1.0
        start = int(self.cfg.dr_curriculum_start_step)
        end = max(int(self.cfg.dr_curriculum_end_step), start + 1)
        initial = min(max(float(self.cfg.dr_curriculum_initial_scale), 0.0), 1.0)
        step = int(getattr(self, "common_step_counter", 0)) + int(self.cfg.dr_curriculum_step_offset)
        progress = min(max((step - start) / float(end - start), 0.0), 1.0)
        return initial + (1.0 - initial) * progress

    def _compute_additional_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Keep the inherited table penalty and expose DR curriculum progress."""
        reward, log = super()._compute_additional_reward()
        log["info/dr_curriculum_scale"] = torch.tensor(
            self._dr_curriculum_scale(),
            device=self.device,
            dtype=torch.float32,
        )
        return reward, log

    @staticmethod
    def _curriculum_bounds(low: float, high: float, center: float, scale: float) -> tuple[float, float]:
        """Shrink a full range around its nominal center by ``scale``."""
        low = float(low)
        high = max(float(high), low)
        center = float(center)
        scale = min(max(float(scale), 0.0), 1.0)
        return center + scale * (low - center), center + scale * (high - center)

    def _ensure_dr_randomization_state(self) -> None:
        """Lazily allocate per-env GPU state before the first reset callback."""
        if getattr(self, "_dr_randomization_initialized", False):
            return

        scalar_shape = (self.num_envs, 1, 1, 1)
        self._dr_brightness = torch.ones(scalar_shape, device=self.device)
        self._dr_contrast = torch.ones(scalar_shape, device=self.device)
        self._dr_saturation = torch.ones(scalar_shape, device=self.device)
        self._dr_gamma = torch.ones(scalar_shape, device=self.device)
        self._dr_hue_rad = torch.zeros(scalar_shape, device=self.device)
        self._dr_white_balance = torch.ones((self.num_envs, 3, 1, 1), device=self.device)
        self._dr_noise_std = torch.zeros(scalar_shape, device=self.device)
        self._dr_blur_mask = torch.zeros(scalar_shape, device=self.device, dtype=torch.bool)
        self._dr_focal_scale = torch.ones((self.num_envs,), device=self.device)
        self._dr_principal_shift_px = torch.zeros((self.num_envs, 2), device=self.device)
        self._dr_camera_delta_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._dr_camera_delta_rpy_rad = torch.zeros((self.num_envs, 3), device=self.device)

        nominal_pos = torch.tensor(self.cfg.wrist_camera.offset.pos, dtype=torch.float32, device=self.device)
        nominal_quat = torch.tensor(self.cfg.wrist_camera.offset.rot, dtype=torch.float32, device=self.device)
        nominal_quat = nominal_quat / torch.linalg.norm(nominal_quat).clamp(min=1e-9)
        self._dr_nominal_camera_pos = nominal_pos
        self._dr_nominal_camera_quat = nominal_quat
        self._dr_camera_pos_w = nominal_pos.unsqueeze(0) + self.scene.env_origins
        self._dr_camera_quat_w = nominal_quat.unsqueeze(0).repeat(self.num_envs, 1)

        self._dr_plate_colors = torch.tensor(
            self.cfg.plate_color_center, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._dr_backdrop_colors = torch.tensor(
            self.cfg.backdrop_color_center, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._backdrop_material_bound = [False] * self.num_envs
        self._dr_current_curriculum_scale = self._dr_curriculum_scale()
        self._dr_global_light_intensity = float(self.cfg.light_nominal_intensity)
        self._dr_global_light_color_temperature = float(self.cfg.light_nominal_color_temperature)
        self._dr_global_light_color = tuple(float(value) for value in self.cfg.light_nominal_color)

        gaussian_1d = torch.tensor([1.0, 2.0, 1.0], device=self.device)
        gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        self._dr_gaussian_blur_kernel = gaussian_2d.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self._dr_randomization_initialized = True

    def _sample_dr_episode_parameters(self, env_ids: torch.Tensor, scale: float) -> None:
        """Sample cached per-env parameters for one episode."""
        count = int(env_ids.numel())
        if count == 0:
            return

        def sample_range(low: float, high: float, center: float = 1.0) -> torch.Tensor:
            effective_low, effective_high = self._curriculum_bounds(low, high, center, scale)
            return torch.empty((count, 1, 1, 1), device=self.device).uniform_(
                effective_low, effective_high
            )

        self._dr_brightness[env_ids] = sample_range(*self.cfg.wrist_brightness_range)
        self._dr_contrast[env_ids] = sample_range(*self.cfg.wrist_contrast_range)
        self._dr_saturation[env_ids] = sample_range(*self.cfg.wrist_saturation_range)
        self._dr_gamma[env_ids] = sample_range(*self.cfg.wrist_gamma_range)

        hue_limit = math.radians(float(self.cfg.wrist_hue_max_deg)) * scale
        self._dr_hue_rad[env_ids] = torch.empty((count, 1, 1, 1), device=self.device).uniform_(
            -hue_limit, hue_limit
        )
        wb_limit = float(self.cfg.wrist_white_balance_shift_max) * scale
        wb_shift = torch.empty((count, 1, 1, 1), device=self.device).uniform_(-wb_limit, wb_limit)
        self._dr_white_balance[env_ids] = torch.cat(
            [1.0 + wb_shift, torch.ones_like(wb_shift), 1.0 - wb_shift],
            dim=1,
        )

        noise_low, noise_high = self._curriculum_bounds(
            *self.cfg.wrist_gaussian_noise_std_range,
            center=0.0,
            scale=scale,
        )
        self._dr_noise_std[env_ids] = torch.empty((count, 1, 1, 1), device=self.device).uniform_(
            noise_low, noise_high
        )
        blur_probability = min(max(float(self.cfg.wrist_blur_probability) * scale, 0.0), 1.0)
        self._dr_blur_mask[env_ids] = (
            torch.rand((count, 1, 1, 1), device=self.device) < blur_probability
        )

        focal_low, focal_high = self._curriculum_bounds(
            *self.cfg.camera_focal_scale_range,
            center=1.0,
            scale=scale,
        )
        self._dr_focal_scale[env_ids] = torch.empty((count,), device=self.device).uniform_(
            focal_low, focal_high
        )
        principal_limit = (
            torch.tensor(
                self.cfg.camera_principal_point_shift_max_px,
                dtype=torch.float32,
                device=self.device,
            )
            * scale
        )
        self._dr_principal_shift_px[env_ids] = (
            2.0 * torch.rand((count, 2), device=self.device) - 1.0
        ) * principal_limit

        position_limit = (
            torch.tensor(self.cfg.camera_position_delta_max_m, dtype=torch.float32, device=self.device)
            * scale
        )
        self._dr_camera_delta_pos[env_ids] = (
            2.0 * torch.rand((count, 3), device=self.device) - 1.0
        ) * position_limit
        rotation_limit = torch.deg2rad(
            torch.tensor(
                self.cfg.camera_rotation_delta_max_deg,
                dtype=torch.float32,
                device=self.device,
            )
        ) * scale
        self._dr_camera_delta_rpy_rad[env_ids] = (
            2.0 * torch.rand((count, 3), device=self.device) - 1.0
        ) * rotation_limit

        def sample_colors(
            color_min: tuple[float, float, float],
            color_max: tuple[float, float, float],
            color_center: tuple[float, float, float],
        ) -> torch.Tensor:
            center = torch.tensor(color_center, dtype=torch.float32, device=self.device)
            low = center + scale * (
                torch.tensor(color_min, dtype=torch.float32, device=self.device) - center
            )
            high = center + scale * (
                torch.tensor(color_max, dtype=torch.float32, device=self.device) - center
            )
            return low + torch.rand((count, 3), device=self.device) * (high - low)

        self._dr_plate_colors[env_ids] = sample_colors(
            self.cfg.plate_color_min,
            self.cfg.plate_color_max,
            self.cfg.plate_color_center,
        )
        self._dr_backdrop_colors[env_ids] = sample_colors(
            self.cfg.backdrop_color_min,
            self.cfg.backdrop_color_max,
            self.cfg.backdrop_color_center,
        )

    def _apply_dr_camera_poses(self, env_ids: torch.Tensor) -> None:
        """Apply T_calib * delta_T independently to selected TiledCamera views."""
        if not bool(self.cfg.camera_pose_randomization_enabled) or env_ids.numel() == 0:
            return
        count = int(env_ids.numel())
        nominal_pos_w = self._dr_nominal_camera_pos.unsqueeze(0) + self.scene.env_origins[env_ids]
        nominal_quat = self._dr_nominal_camera_quat.unsqueeze(0).repeat(count, 1)
        delta_rpy = self._dr_camera_delta_rpy_rad[env_ids]
        delta_quat = math_utils.quat_from_euler_xyz(
            delta_rpy[:, 0],
            delta_rpy[:, 1],
            delta_rpy[:, 2],
        )
        camera_pos_w, camera_quat_w = math_utils.combine_frame_transforms(
            nominal_pos_w,
            nominal_quat,
            self._dr_camera_delta_pos[env_ids],
            delta_quat,
        )
        self.wrist_camera.set_world_poses(
            positions=camera_pos_w,
            orientations=camera_quat_w,
            env_ids=env_ids,
            convention=str(self.cfg.wrist_camera.offset.convention),
        )
        self._dr_camera_pos_w[env_ids] = camera_pos_w
        self._dr_camera_quat_w[env_ids] = camera_quat_w

    def _apply_dr_material_colors(self, env_ids: torch.Tensor) -> None:
        """Update unique per-env plate/backdrop materials on reset."""
        env_id_list = env_ids.detach().cpu().tolist()
        if not env_id_list:
            return

        if bool(self.cfg.plate_color_randomization_enabled):
            plate_paths = self._plate.root_physx_view.prim_paths
            plate_colors = self._dr_plate_colors[env_ids].detach().cpu().tolist()
            for env_id, color in zip(env_id_list, plate_colors):
                plate_root = plate_paths[env_id]
                plate_target = self._resolve_visual_target_prim(plate_root)
                bind = not self._plate_material_bound[env_id]
                self._apply_visual_material(
                    f"{plate_root}/material",
                    plate_target,
                    tuple(float(channel) for channel in color),
                    bind=bind,
                )
                if bind:
                    self._plate_material_bound[env_id] = True

        if bool(self.cfg.backdrop_color_randomization_enabled):
            backdrop_paths = self._backdrop.root_physx_view.prim_paths
            backdrop_colors = self._dr_backdrop_colors[env_ids].detach().cpu().tolist()
            for env_id, color in zip(env_id_list, backdrop_colors):
                backdrop_root = backdrop_paths[env_id]
                backdrop_target = self._resolve_visual_target_prim(backdrop_root)
                bind = not self._backdrop_material_bound[env_id]
                self._apply_visual_material(
                    f"{backdrop_root}/material",
                    backdrop_target,
                    tuple(float(channel) for channel in color),
                    bind=bind,
                )
                if bind:
                    self._backdrop_material_bound[env_id] = True

    def _randomize_dr_global_light(self, scale: float) -> None:
        """Randomize the one stage-global DomeLight for the current env batch."""
        if not bool(self.cfg.light_randomization_enabled):
            return

        stage = sim_utils.stage_utils.get_current_stage()
        light_prim = stage.GetPrimAtPath(getattr(self, "_light_prim_path", "/World/Light"))
        if not light_prim.IsValid():
            return

        intensity_low, intensity_high = self._curriculum_bounds(
            *self.cfg.light_intensity_range,
            center=float(self.cfg.light_nominal_intensity),
            scale=scale,
        )
        temperature_low, temperature_high = self._curriculum_bounds(
            *self.cfg.light_color_temperature_range,
            center=float(self.cfg.light_nominal_color_temperature),
            scale=scale,
        )
        intensity = self._sample_uniform_scalar(intensity_low, intensity_high, min_value=0.0)
        color_temperature = self._sample_uniform_scalar(
            temperature_low,
            temperature_high,
            min_value=1000.0,
        )

        # Express color temperature as an RGB tint relative to the nominal
        # temperature. This keeps scale=0 exactly equal to the Clean DomeLight
        # color instead of switching to another renderer temperature model.
        nominal_temperature_rgb = self._kelvin_to_rgb(
            float(self.cfg.light_nominal_color_temperature)
        )
        sampled_temperature_rgb = self._kelvin_to_rgb(color_temperature)
        light_color = tuple(
            min(
                max(
                    float(base_channel)
                    * float(sampled_channel)
                    / max(float(nominal_channel), 1e-6),
                    0.0,
                ),
                1.0,
            )
            for base_channel, sampled_channel, nominal_channel in zip(
                self.cfg.light_nominal_color,
                sampled_temperature_rgb,
                nominal_temperature_rgb,
            )
        )

        try:
            sim_utils.safe_set_attribute_on_usd_prim(
                light_prim,
                "inputs:intensity",
                intensity,
                camel_case=True,
            )
            sim_utils.safe_set_attribute_on_usd_prim(
                light_prim,
                "inputs:color",
                light_color,
                camel_case=True,
            )
            sim_utils.safe_set_attribute_on_usd_prim(
                light_prim,
                "inputs:enable_color_temperature",
                False,
                camel_case=True,
            )
            sim_utils.safe_set_attribute_on_usd_prim(
                light_prim,
                "inputs:color_temperature",
                color_temperature,
                camel_case=True,
            )
            self._dr_global_light_intensity = intensity
            self._dr_global_light_color_temperature = color_temperature
            self._dr_global_light_color = light_color
        except Exception as exc:
            print(f"[WARN] DR dome light randomization failed: {exc}")

    def _randomize_scene_visuals(self, env_ids: torch.Tensor) -> None:
        """Resample episode-fixed DR state for selected environments."""
        self._ensure_dr_randomization_state()
        env_ids = self._normalize_dr_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        scale = self._dr_curriculum_scale()
        self._dr_current_curriculum_scale = scale
        self._sample_dr_episode_parameters(env_ids, scale)
        self._apply_dr_camera_poses(env_ids)
        self._apply_dr_material_colors(env_ids)

        # A DomeLight belongs to the shared USD stage rather than an individual
        # cloned environment. Updating it on a partial reset would silently
        # change every running environment, so only update it for a full batch.
        if int(env_ids.numel()) == self.num_envs:
            self._randomize_dr_global_light(scale)

    def _apply_dr_intrinsic_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-env focal/principal-point perturbations as a GPU warp."""
        if not bool(self.cfg.camera_intrinsic_warp_enabled):
            return x

        batch_size, _, height, width = x.shape
        focal_scale = self._dr_focal_scale[:batch_size].clamp(min=1e-4)
        principal_shift = self._dr_principal_shift_px[:batch_size]
        theta = torch.zeros((batch_size, 2, 3), device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = 1.0 / focal_scale
        theta[:, 1, 1] = 1.0 / focal_scale
        # affine_grid maps output coordinates to input coordinates. A positive
        # virtual principal-point shift therefore requires a negative sampling
        # translation. The division by focal_scale keeps the affine form exact.
        theta[:, 0, 2] = -2.0 * principal_shift[:, 0] / (float(width) * focal_scale)
        theta[:, 1, 2] = -2.0 * principal_shift[:, 1] / (float(height) * focal_scale)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def _apply_dr_hue_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate RGB hue per environment without leaving the GPU."""
        batch_size = x.shape[0]
        angle = self._dr_hue_rad[:batch_size, 0, 0, 0].to(dtype=x.dtype)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        matrix = torch.empty((batch_size, 3, 3), device=x.device, dtype=x.dtype)

        matrix[:, 0, 0] = 0.213 + 0.787 * cos_angle - 0.213 * sin_angle
        matrix[:, 0, 1] = 0.715 - 0.715 * cos_angle - 0.715 * sin_angle
        matrix[:, 0, 2] = 0.072 - 0.072 * cos_angle + 0.928 * sin_angle
        matrix[:, 1, 0] = 0.213 - 0.213 * cos_angle + 0.143 * sin_angle
        matrix[:, 1, 1] = 0.715 + 0.285 * cos_angle + 0.140 * sin_angle
        matrix[:, 1, 2] = 0.072 - 0.072 * cos_angle - 0.283 * sin_angle
        matrix[:, 2, 0] = 0.213 - 0.213 * cos_angle - 0.787 * sin_angle
        matrix[:, 2, 1] = 0.715 - 0.715 * cos_angle + 0.715 * sin_angle
        matrix[:, 2, 2] = 0.072 + 0.928 * cos_angle + 0.072 * sin_angle
        return torch.einsum("nij,njhw->nihw", matrix, x)

    def _apply_wrist_visual_randomization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cached episode DR parameters and frame-wise sensor noise."""
        x = self._apply_nominal_camera_intrinsic_compensation(x)
        if not bool(self.cfg.wrist_visual_randomization_enabled):
            return x

        self._ensure_dr_randomization_state()
        if self._dr_current_curriculum_scale <= 0.0:
            # Avoid even an identity grid_sample/color pipeline so the first
            # curriculum stage is pixel-identical to the Clean observation path.
            return x
        batch_size = x.shape[0]
        if batch_size > self.num_envs:
            raise ValueError(
                f"DR image batch ({batch_size}) exceeds configured environments ({self.num_envs})"
            )

        # ResNet consumes RGB only. Some RTX annotators may append alpha.
        x = torch.clamp(x[:, :3], 0.0, 1.0)
        x = self._apply_dr_intrinsic_warp(x)

        x = x * self._dr_white_balance[:batch_size]
        x = x * self._dr_brightness[:batch_size]

        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * self._dr_contrast[:batch_size] + mean

        luminance = (
            0.2126 * x[:, 0:1]
            + 0.7152 * x[:, 1:2]
            + 0.0722 * x[:, 2:3]
        )
        x = (
            luminance
            + self._dr_saturation[:batch_size] * (x - luminance)
        )
        x = self._apply_dr_hue_rotation(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = torch.pow(
            torch.clamp(x, min=1e-6, max=1.0),
            self._dr_gamma[:batch_size],
        )

        if bool(self.cfg.wrist_blur_randomization_enabled):
            blurred = F.conv2d(
                x,
                self._dr_gaussian_blur_kernel.to(dtype=x.dtype),
                padding=1,
                groups=3,
            )
            x = torch.where(self._dr_blur_mask[:batch_size], blurred, x)

        if bool(self.cfg.wrist_gaussian_noise_randomization_enabled):
            x = x + torch.randn_like(x) * self._dr_noise_std[:batch_size]

        return torch.clamp(x, 0.0, 1.0)
