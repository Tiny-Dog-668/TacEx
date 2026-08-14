"""State-only diagnostic variant of the real-aligned Franka cube task.

This task deliberately exposes simulator ground-truth positions to the Actor.
It is an upper-bound experiment for separating reward/control problems from
third-person visual localization problems and is not deployable on the real
robot.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .sim2real_cube_real_alignment_env import (
    Sim2RealCubeRealAlignmentEnv,
    Sim2RealCubeRealAlignmentEnvCfg,
)


@configclass
class Sim2RealCubeRealAlignmentPrivilegedEnvCfg(Sim2RealCubeRealAlignmentEnvCfg):
    """Real-alignment physics with a privileged position-only Actor."""

    # Keep this diagnostic task on its existing control/reset distribution.
    # The v9 5 mm gripper and +/-5 cm XY changes are scoped to Clean and DR.
    gripper_width_delta_scale = 0.002
    cube_x_pos_range = 0.10
    cube_y_pos_range = 0.10

    camera_sensor_enabled = False
    vision_encoder_enabled = False
    privileged_actor_observation = (
        "proprio_obs+action_history+cube_pos_root+gripper_pos_root+target_pos_root"
    )

    observation_space = {
        "proprio_obs": 15,
        "action_history": 4,
        # Actor-only diagnostic keys. Positions use the robot-root-aligned,
        # per-environment frame, so parallel environment origins do not leak in.
        "privileged_cube_pos": 3,
        "privileged_gripper_pos": 3,
        "privileged_target_pos": 3,
        # Keep the asymmetric critic contract used by the vision baseline.
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


class Sim2RealCubeRealAlignmentPrivilegedEnv(Sim2RealCubeRealAlignmentEnv):
    """No-camera upper-bound task using simulator object/gripper positions."""

    cfg: Sim2RealCubeRealAlignmentPrivilegedEnvCfg

    def _setup_scene(self) -> None:
        """Create the aligned physical scene without camera or visual backdrop."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._setup_cube_target()

        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        self.scene.clone_environments(copy_from_source=False)
        self._register_table_contact_sensor()

        if prim_utils.is_prim_path_valid("/World/defaultDomeLight"):
            prim_utils.delete_prim("/World/defaultDomeLight")

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

        # Lighting is not required for state observations. Keep one low-cost
        # light so optional GUI inspection remains readable.
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return state observations without reading or encoding an RGB buffer.

        Shapes:
            proprio_obs: [N, 15]
            action_history: [N, 4]
            privileged_*_pos: [N, 3], robot-root-aligned per-env frame
        """
        arm_joint_pos = self._robot.data.joint_pos[:, :7]
        arm_joint_vel = self._robot.data.joint_vel[:, :7]
        gripper_opening = self._robot.data.joint_pos[:, self._finger_joint_ids].sum(dim=-1, keepdim=True)
        gripper_opening = torch.clamp(gripper_opening, 0.0, self.cfg.max_gripper_opening_width)
        proprio_obs = torch.cat([arm_joint_pos, arm_joint_vel, gripper_opening], dim=-1)

        cube_pos_w = self._cube.data.root_pos_w
        cube_quat_w = self._cube.data.root_quat_w
        cube_lin_vel_w = self._cube.data.root_lin_vel_w
        cube_ang_vel_w = self._cube.data.root_ang_vel_w

        left_tip_w, right_tip_w = self._compute_fingertip_positions_world()
        gripper_center_w = 0.5 * (left_tip_w + right_tip_w)

        # The robot base is fixed at each environment origin with identity
        # orientation in this task. Removing env_origins therefore expresses
        # positions in the same robot-root-aligned frame for every environment.
        # Input/output shapes: [N, 3] world positions -> [N, 3] root positions.
        # CPU PhysX can retain scene origins on CPU while asset state stays on
        # the configured CUDA policy device, so align both device and dtype.
        env_origins = self.scene.env_origins.to(
            device=cube_pos_w.device,
            dtype=cube_pos_w.dtype,
        )
        cube_pos_root = cube_pos_w - env_origins
        gripper_pos_root = gripper_center_w - env_origins
        target_pos_root = cube_pos_root - gripper_pos_root

        left_quat_w = self._robot.data.body_link_quat_w[:, self._left_finger_body_idx]
        right_quat_w = self._robot.data.body_link_quat_w[:, self._right_finger_body_idx]
        left_offset_w = math_utils.quat_apply(left_quat_w, self._fingertip_local_offset)
        right_offset_w = math_utils.quat_apply(right_quat_w, self._fingertip_local_offset)
        left_ang_vel_w = self._robot.data.body_link_ang_vel_w[:, self._left_finger_body_idx]
        right_ang_vel_w = self._robot.data.body_link_ang_vel_w[:, self._right_finger_body_idx]
        left_tip_vel_w = (
            self._robot.data.body_link_lin_vel_w[:, self._left_finger_body_idx]
            + torch.cross(left_ang_vel_w, left_offset_w, dim=-1)
        )
        right_tip_vel_w = (
            self._robot.data.body_link_lin_vel_w[:, self._right_finger_body_idx]
            + torch.cross(right_ang_vel_w, right_offset_w, dim=-1)
        )

        obs = {
            "proprio_obs": proprio_obs,
            "action_history": self.action_history,
            "privileged_cube_pos": cube_pos_root,
            "privileged_gripper_pos": gripper_pos_root,
            "privileged_target_pos": target_pos_root,
            "critic_cube_pos": cube_pos_root,
            "critic_cube_quat": cube_quat_w,
            "critic_cube_lin_vel": cube_lin_vel_w,
            "critic_cube_ang_vel": cube_ang_vel_w,
            "critic_gripper_pos": gripper_pos_root,
            "critic_gripper_quat": self._robot.data.body_link_quat_w[:, self._body_idx],
            "critic_gripper_lin_vel": 0.5 * (left_tip_vel_w + right_tip_vel_w),
            "critic_gripper_ang_vel": 0.5 * (left_ang_vel_w + right_ang_vel_w),
            "critic_target_pos": target_pos_root,
            "critic_target_distance": torch.norm(target_pos_root, dim=-1, keepdim=True),
        }
        return {"policy": obs}
