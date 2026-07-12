"""Real-reference-aligned Franka cube grasping environment.

The measured values in this module come from
``20260711_214450_real_alignment_reference/alignment_reference.json``.
Camera extrinsics were not measured in that capture, so the existing manually
aligned third-person pose is retained and explicitly treated as approximate.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from .sim2real_cube_grasp_env import Sim2RealCubeGraspEnv, Sim2RealCubeGraspEnvCfg


@configclass
class Sim2RealCubeRealAlignmentEnvCfg(Sim2RealCubeGraspEnvCfg):
    """Cube task aligned to the 2026-07-11 real Franka/D435 reference capture."""

    alignment_reference_id = "20260711_214450_real_alignment_reference"
    camera_extrinsics_source = "approximate_existing_sim_pose"

    # Measured real target: 5 cm cube. Keep this asset local to the aligned task
    # so TacEx-Sim2Real-Cube-Grasp-v0 retains its existing 6 cm target.
    cube_size = (0.05, 0.05, 0.05)
    cube_half_xy_extent = 0.5 * max(cube_size[0], cube_size[1])
    cylinder_radius = cube_half_xy_extent
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.01 + 0.5 * cube_size[2])),
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

    # Preserve a 5 mm dead band above contact jitter. A 35 mm lift is 70% of
    # the measured cube edge and replaces the 40 mm threshold used for 6 cm.
    lift_reward_start_delta = 0.005
    success_lift_delta = 0.035

    # The real RGB stream is 30 Hz. Physics remains 60 Hz and the policy advances
    # every two simulation steps, so camera and policy observations update at 30 Hz.
    decimation = 2

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

    # The real pipeline crops x=[80, 560) from the native 640x480 image and then
    # resizes the resulting 480x480 RGB crop to 224x224. Rendering directly at
    # 224x224 with the transformed intrinsics is projection-equivalent.
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=1.0 / 30.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                282.46025390625,
                0.0,
                113.19166056315105,
                0.0,
                282.2679077148438,
                116.30194091796875,
                0.0,
                0.0,
                1.0,
            ],
            width=224,
            height=224,
            focal_length=1.0,
            focus_distance=0.8,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # Pending hand-eye calibration: this image-derived pose uses a
            # 15 degree downward optical-axis pitch. It is not a measured
            # camera-to-robot transform.
            pos=(1.90, 0.0, 0.468),
            rot=(0.5609855268, 0.4304593346, 0.4304593346, 0.5609855268),
            convention="opengl",
        ),
    )

    # The real image's table rear edge is near row 110 in the 224x224 policy
    # image. With the approximate camera pose, x=0.0 m projects to that row.
    # The cube nominal centre remains x=0.60 m and is therefore supported close
    # to the rear edge, as in the captured scene.
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/floor_panel",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.50, 0.0, 0.005)),
        spawn=sim_utils.CuboidCfg(
            size=(3.0, 3.0, 0.01),
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
                diffuse_color=(0.24, 0.29, 0.25),
                metallic=0.0,
                roughness=0.98,
            ),
        ),
    )

    # Approximate the green curtain visible behind the real Franka. It is visual
    # only and therefore cannot affect contacts or rewards.
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
                diffuse_color=(0.08, 0.34, 0.07),
                metallic=0.0,
                roughness=0.95,
            ),
        ),
    )

    # Reference-centred appearance randomization: narrower than the original
    # broad sim2real task but still covers auto-exposure/white-balance variation.
    wrist_visual_randomization_enabled = True
    wrist_brightness_randomization_enabled = True
    wrist_brightness_range = (0.90, 1.10)
    wrist_gamma_randomization_enabled = True
    wrist_gamma_range = (0.95, 1.05)
    wrist_contrast_randomization_enabled = True
    wrist_contrast_range = (0.90, 1.10)
    wrist_blur_randomization_enabled = True
    wrist_blur_probability = 0.15
    wrist_blur_kernel_sizes = (3,)
    wrist_gaussian_noise_randomization_enabled = True
    wrist_gaussian_noise_std_range = (0.0, 0.01)

    # The D435 capture used auto white balance. Keep the simulator light neutral;
    # applying the measured 4600 K setting again in rendering would double-tint
    # already white-balanced RGB frames.
    light_randomization_enabled = False
    ground_color_randomization_enabled = True
    # The inherited ground plane is visible behind the table in this camera pose;
    # tint it together with the backdrop to match the green real background.
    ground_color_min = (0.04, 0.22, 0.035)
    ground_color_max = (0.08, 0.32, 0.07)
    plate_color_randomization_enabled = True
    plate_color_min = (0.02, 0.035, 0.025)
    plate_color_max = (0.05, 0.07, 0.05)

    # The reference file does not contain object pose. This nominal offset is
    # estimated from the cube projection in rgb_model_median.png and is therefore
    # deliberately kept separate from measured robot/camera parameters.
    cube_image_aligned_xy_offset = (0.04, -0.10)
    cube_x_pos_range = 0.05
    cube_y_pos_range = 0.04

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation


class Sim2RealCubeRealAlignmentEnv(Sim2RealCubeGraspEnv):
    """Cube grasping environment with real-reference robot and camera parameters."""

    cfg: Sim2RealCubeRealAlignmentEnvCfg

    def _sample_cube_xy_offsets(self, count: int) -> torch.Tensor:
        """Sample around the image-derived nominal cube location."""
        offsets = super()._sample_cube_xy_offsets(count)
        nominal = torch.tensor(
            self.cfg.cube_image_aligned_xy_offset,
            device=self.device,
            dtype=offsets.dtype,
        )
        return offsets + nominal.unsqueeze(0)

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
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.107)),
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

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
