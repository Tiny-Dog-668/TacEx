"""Occluded grasping baseline with third-person vision only."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)

from .occluded_grasping_cabinet_scene import make_fixed_panel_cfg
from .vt_box import (
    CAN_COLLISION_REST_OFFSET,
    CAN_HEIGHT,
    CAN_RANDOM_CENTER_X,
    CAN_RANDOM_CENTER_Y,
    CAN_RADIUS,
    CAN_RESET_ROOT_Z as BOX_CAN_RESET_ROOT_Z,
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
)

# Keep the RL drawer scene consistent with the interactive demo scene.
DRAWER_SCALE = 1.8
DRAWER_DEPTH_SCALE = 1.0
DRAWER_WIDTH_SCALE = 1.0
DRAWER_HEIGHT_SCALE = 0.8
DRAWER_OPEN_RATIO = 0.6
DRAWER_FRONT_FACE_X = 0.435
DRAWER_CENTER_Y = 0.00

BASE_DRAWER_INNER_X = 0.22
BASE_DRAWER_INNER_Y = 0.26
BASE_DRAWER_WALL_THICKNESS = 0.015
BASE_DRAWER_FLOOR_THICKNESS = 0.02
BASE_DRAWER_WALL_HEIGHT = 0.10
BASE_DRAWER_FRONT_PANEL_HEIGHT = 0.14
BASE_OBJECT_Z_OFFSET = 0.015

BASE_DRAWER_CABINET_WALL_THICKNESS = 0.015
BASE_DRAWER_CABINET_SIDE_CLEARANCE_Y = 0.004
BASE_DRAWER_CABINET_TOP_CLEARANCE_Z = 0.006
BASE_DRAWER_CABINET_REAR_CLEARANCE_X = 0.020
BASE_DRAWER_FRONT_PANEL_SIDE_GAP = 0.004

BASE_DRAWER_HANDLE_DEPTH = 0.018
BASE_DRAWER_HANDLE_WIDTH_Y = 0.07
BASE_DRAWER_HANDLE_HEIGHT_Z = 0.014

DRAWER_X_SCALE = DRAWER_SCALE * DRAWER_DEPTH_SCALE
DRAWER_Y_SCALE = DRAWER_SCALE * DRAWER_WIDTH_SCALE
DRAWER_Z_SCALE = DRAWER_SCALE * DRAWER_HEIGHT_SCALE

BOX_INNER_X = BASE_DRAWER_INNER_X * DRAWER_X_SCALE
BOX_INNER_Y = BASE_DRAWER_INNER_Y * DRAWER_Y_SCALE
BOX_WALL_THICKNESS = BASE_DRAWER_WALL_THICKNESS * DRAWER_SCALE
BOX_FLOOR_THICKNESS = BASE_DRAWER_FLOOR_THICKNESS * DRAWER_Z_SCALE
BOX_WALL_HEIGHT = BASE_DRAWER_WALL_HEIGHT * DRAWER_Z_SCALE
BOX_FRONT_PANEL_HEIGHT = BASE_DRAWER_FRONT_PANEL_HEIGHT * DRAWER_Z_SCALE

DRAWER_OUTER_DEPTH_X = BOX_INNER_X + 2 * BOX_WALL_THICKNESS
DRAWER_OUTER_WIDTH_Y = BOX_INNER_Y + 2 * BOX_WALL_THICKNESS
DRAWER_BODY_HEIGHT_Z = BOX_FLOOR_THICKNESS + BOX_WALL_HEIGHT

BOX_CENTER_X = DRAWER_FRONT_FACE_X + DRAWER_OUTER_DEPTH_X * 0.5
BOX_CENTER_Y = DRAWER_CENTER_Y
BOX_WALL_CENTER_Z = BOX_FLOOR_THICKNESS + BOX_WALL_HEIGHT * 0.5
BOX_FRONT_PANEL_CENTER_Z = BOX_FRONT_PANEL_HEIGHT * 0.5
BOX_OBJECT_Z = BOX_FLOOR_THICKNESS + BASE_OBJECT_Z_OFFSET * DRAWER_Z_SCALE
CAN_RESET_ROOT_Z = BOX_CAN_RESET_ROOT_Z

DRAWER_CABINET_WALL_THICKNESS = BASE_DRAWER_CABINET_WALL_THICKNESS * DRAWER_SCALE
DRAWER_CABINET_SIDE_CLEARANCE_Y = BASE_DRAWER_CABINET_SIDE_CLEARANCE_Y * DRAWER_Y_SCALE
DRAWER_CABINET_TOP_CLEARANCE_Z = BASE_DRAWER_CABINET_TOP_CLEARANCE_Z * DRAWER_Z_SCALE
DRAWER_CABINET_REAR_CLEARANCE_X = BASE_DRAWER_CABINET_REAR_CLEARANCE_X * DRAWER_X_SCALE
DRAWER_FRONT_PANEL_SIDE_GAP = BASE_DRAWER_FRONT_PANEL_SIDE_GAP * DRAWER_Y_SCALE
DRAWER_FRONT_PROTRUSION_X = DRAWER_OUTER_DEPTH_X * DRAWER_OPEN_RATIO

DRAWER_CABINET_CENTER_Y = BOX_CENTER_Y
DRAWER_CABINET_FRONT_X = BOX_CENTER_X - DRAWER_OUTER_DEPTH_X * 0.5 + DRAWER_FRONT_PROTRUSION_X
DRAWER_CABINET_OUTER_DEPTH_X = DRAWER_OUTER_DEPTH_X - DRAWER_FRONT_PROTRUSION_X + DRAWER_CABINET_REAR_CLEARANCE_X
DRAWER_CABINET_CENTER_X = DRAWER_CABINET_FRONT_X + DRAWER_CABINET_OUTER_DEPTH_X * 0.5
DRAWER_CABINET_OUTER_WIDTH_Y = (
    DRAWER_OUTER_WIDTH_Y + 2 * (DRAWER_CABINET_SIDE_CLEARANCE_Y + DRAWER_CABINET_WALL_THICKNESS)
)
DRAWER_CABINET_HEIGHT = DRAWER_BODY_HEIGHT_Z + DRAWER_CABINET_TOP_CLEARANCE_Z + DRAWER_CABINET_WALL_THICKNESS
DRAWER_FRONT_PANEL_WIDTH_Y = DRAWER_CABINET_OUTER_WIDTH_Y - 2 * DRAWER_FRONT_PANEL_SIDE_GAP

DRAWER_HANDLE_DEPTH = BASE_DRAWER_HANDLE_DEPTH * DRAWER_SCALE
DRAWER_HANDLE_WIDTH_Y = BASE_DRAWER_HANDLE_WIDTH_Y * DRAWER_Y_SCALE
DRAWER_HANDLE_HEIGHT_Z = BASE_DRAWER_HANDLE_HEIGHT_Z * DRAWER_Z_SCALE

@configclass
class OccludedGraspingVisionOnlyBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for the occluded grasping vision-only baseline."""

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.45, 0.85, 0.52)
    viewer.lookat = (BOX_CENTER_X, BOX_CENTER_Y, 0.08)

    box_floor = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_floor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BOX_CENTER_X, BOX_CENTER_Y, BOX_FLOOR_THICKNESS * 0.5)
        ),
        spawn=make_fixed_panel_cfg(
            size=(
                BOX_INNER_X + 2 * BOX_WALL_THICKNESS,
                BOX_INNER_Y + 2 * BOX_WALL_THICKNESS,
                BOX_FLOOR_THICKNESS,
            ),
            color=(0.40, 0.27, 0.18),
        ),
    )

    box_wall_front = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_wall_front",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X - BOX_INNER_X * 0.5 - BOX_WALL_THICKNESS * 0.5,
                BOX_CENTER_Y,
                BOX_FRONT_PANEL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(BOX_WALL_THICKNESS, DRAWER_FRONT_PANEL_WIDTH_Y, BOX_FRONT_PANEL_HEIGHT),
            color=(0.56, 0.39, 0.26),
        ),
    )

    box_wall_back = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_wall_back",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X + BOX_INNER_X * 0.5 + BOX_WALL_THICKNESS * 0.5,
                BOX_CENTER_Y,
                BOX_WALL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(BOX_WALL_THICKNESS, BOX_INNER_Y, BOX_WALL_HEIGHT),
            color=(0.50, 0.35, 0.24),
        ),
    )

    box_wall_left = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_wall_left",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X,
                BOX_CENTER_Y - BOX_INNER_Y * 0.5 - BOX_WALL_THICKNESS * 0.5,
                BOX_WALL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(BOX_INNER_X, BOX_WALL_THICKNESS, BOX_WALL_HEIGHT),
            color=(0.50, 0.35, 0.24),
        ),
    )

    box_wall_right = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_wall_right",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X,
                BOX_CENTER_Y + BOX_INNER_Y * 0.5 + BOX_WALL_THICKNESS * 0.5,
                BOX_WALL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(BOX_INNER_X, BOX_WALL_THICKNESS, BOX_WALL_HEIGHT),
            color=(0.50, 0.35, 0.24),
        ),
    )

    drawer_cabinet_left_side = RigidObjectCfg(
        prim_path="/World/envs/env_.*/drawer_cabinet_left_side",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                DRAWER_CABINET_CENTER_X,
                DRAWER_CABINET_CENTER_Y - DRAWER_CABINET_OUTER_WIDTH_Y * 0.5 + DRAWER_CABINET_WALL_THICKNESS * 0.5,
                DRAWER_CABINET_HEIGHT * 0.5,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(
                DRAWER_CABINET_OUTER_DEPTH_X,
                DRAWER_CABINET_WALL_THICKNESS,
                DRAWER_CABINET_HEIGHT,
            ),
            color=(0.44, 0.31, 0.22),
        ),
    )

    drawer_cabinet_right_side = drawer_cabinet_left_side.replace(
        prim_path="/World/envs/env_.*/drawer_cabinet_right_side",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                DRAWER_CABINET_CENTER_X,
                DRAWER_CABINET_CENTER_Y + DRAWER_CABINET_OUTER_WIDTH_Y * 0.5 - DRAWER_CABINET_WALL_THICKNESS * 0.5,
                DRAWER_CABINET_HEIGHT * 0.5,
            )
        ),
    )

    drawer_cabinet_back = RigidObjectCfg(
        prim_path="/World/envs/env_.*/drawer_cabinet_back",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                DRAWER_CABINET_CENTER_X + DRAWER_CABINET_OUTER_DEPTH_X * 0.5 - DRAWER_CABINET_WALL_THICKNESS * 0.5,
                DRAWER_CABINET_CENTER_Y,
                DRAWER_CABINET_HEIGHT * 0.5,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(
                DRAWER_CABINET_WALL_THICKNESS,
                DRAWER_CABINET_OUTER_WIDTH_Y - 2 * DRAWER_CABINET_WALL_THICKNESS,
                DRAWER_CABINET_HEIGHT,
            ),
            color=(0.44, 0.31, 0.22),
        ),
    )

    drawer_cabinet_top = RigidObjectCfg(
        prim_path="/World/envs/env_.*/drawer_cabinet_top",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                DRAWER_CABINET_CENTER_X,
                DRAWER_CABINET_CENTER_Y,
                DRAWER_CABINET_HEIGHT - DRAWER_CABINET_WALL_THICKNESS * 0.5,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(
                DRAWER_CABINET_OUTER_DEPTH_X,
                DRAWER_CABINET_OUTER_WIDTH_Y,
                DRAWER_CABINET_WALL_THICKNESS,
            ),
            color=(0.46, 0.32, 0.22),
        ),
    )

    drawer_handle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/drawer_handle",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X - BOX_INNER_X * 0.5 - BOX_WALL_THICKNESS - DRAWER_HANDLE_DEPTH * 0.5,
                BOX_CENTER_Y,
                BOX_FRONT_PANEL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(DRAWER_HANDLE_DEPTH, DRAWER_HANDLE_WIDTH_Y, DRAWER_HANDLE_HEIGHT_Z),
            color=(0.72, 0.72, 0.74),
        ),
    )

    cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[CAN_RANDOM_CENTER_X, CAN_RANDOM_CENTER_Y, CAN_RESET_ROOT_Z]),
        spawn=sim_utils.CylinderCfg(
            radius=CAN_RADIUS,
            height=CAN_HEIGHT,
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
                rest_offset=CAN_COLLISION_REST_OFFSET,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.84, 0.84, 0.86),
                roughness=0.45,
                metallic=0.0,
            ),
        ),
    )
    third_person_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=30.0,
            focus_distance=200.0,
            horizontal_aperture=40.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.2, -0.35, 1.2),
            rot=(0.68301, 0.18301, -0.18301, -0.68301),
            convention="opengl",
        ),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.4716,
                "panda_joint2": 0.0114,
                "panda_joint3": 0.5090,
                "panda_joint4": -2.3263,
                "panda_joint5": -0.0645,
                "panda_joint6": 2.3565,
                "panda_joint7": 0.8718,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    observation_space = {
        "proprio_obs": 18,
        "third_resnet": 256,
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

    box_floor_top_offset = BOX_FLOOR_THICKNESS * 0.5


class OccludedGraspingVisionOnlyBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Occluded grasping environment using only third-person vision and proprioception."""

    cfg: OccludedGraspingVisionOnlyBoxCfg

    def _setup_scene(self):
        """Setup the drawer scene without tactile sensors."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self._cylinder
        # Compatibility alias: inherited reset logic still references self._can.
        self._can = self._cylinder

        self._spawn_drawer_scene_geometry()

        self.scene.clone_environments(copy_from_source=False)

        self.third_person_camera = TiledCamera(self.cfg.third_person_camera)
        self.scene.sensors["third_person_camera"] = self.third_person_camera

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
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.11841)),
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

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Get vision-only observations from the environment."""
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        cylinder_pos = self._cylinder.data.root_pos_w
        cylinder_quat = self._cylinder.data.root_quat_w
        cylinder_lin_vel = self._cylinder.data.root_lin_vel_w
        cylinder_ang_vel = self._cylinder.data.root_ang_vel_w

        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        gripper_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        gripper_lin_vel = self._robot.data.body_link_lin_vel_w[:, self._body_idx]
        gripper_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._body_idx]

        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]

        def _norm_quat(q: torch.Tensor) -> torch.Tensor:
            n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
            return q / n

        hand_quat = _norm_quat(hand_quat)
        offset_rot = _norm_quat(self._offset_rot)
        ee_pos, _ = math_utils.combine_frame_transforms(hand_pos, hand_quat, self._offset_pos, offset_rot)
        target_pos_relative = cylinder_pos - ee_pos
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        third_rgb = self.third_person_camera.data.output.get("rgb")
        if third_rgb is None:
            third_rgb = torch.zeros(
                (self.num_envs, self.cfg.third_person_camera.height, self.cfg.third_person_camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            third_rgb = third_rgb.to(device=self.device, dtype=torch.float32) / 255.0

        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        use_amp = dev_type == "cuda"

        if hasattr(self, "_use_resnet18") and self._use_resnet18:
            xt = third_rgb.permute(0, 3, 1, 2).contiguous().to(self.device)
            if hasattr(self, "_imagenet_mean"):
                xt = (xt - self._imagenet_mean) / self._imagenet_std
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                third_feat = self._resnet18(xt).view(self.num_envs, 256)
        else:
            third_feat = torch.zeros((self.num_envs, 256), device=self.device)

        obs = {
            "proprio_obs": proprio_obs,
            "third_resnet": third_feat,
            "critic_can_pos": cylinder_pos,
            "critic_can_quat": cylinder_quat,
            "critic_can_lin_vel": cylinder_lin_vel,
            "critic_can_ang_vel": cylinder_ang_vel,
            "critic_gripper_pos": gripper_pos,
            "critic_gripper_quat": gripper_quat,
            "critic_gripper_lin_vel": gripper_lin_vel,
            "critic_gripper_ang_vel": gripper_ang_vel,
            "critic_target_pos": target_pos_relative,
            "critic_target_distance": target_distance,
        }
        return {"policy": obs}

@configclass
class OccludedGraspingVisionOnlyWristBoxCfg(OccludedGraspingVisionOnlyBoxCfg):
    """Configuration for the occluded grasping vision-only baseline using a wrist camera."""

    third_person_camera: TiledCameraCfg = TiledCameraCfg(
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
            pos=(0.12, 0.0, -0.12),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )


class OccludedGraspingVisionOnlyWristBoxEnv(OccludedGraspingVisionOnlyBoxEnv):
    """Occluded grasping environment using wrist-camera vision and proprioception."""

    cfg: OccludedGraspingVisionOnlyWristBoxCfg
