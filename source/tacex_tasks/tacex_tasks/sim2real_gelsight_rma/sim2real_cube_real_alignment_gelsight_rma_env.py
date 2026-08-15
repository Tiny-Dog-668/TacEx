"""GelSight-equipped RMA variants of the Real-Alignment cube task.

The GelSight Mini geometry defines its own hand-frame center and lowest-point
contract. Teacher does not render tactile RGB. Student uses third-person RGB for
cube position and left/right GelSight tactile RGB for contact prediction.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from tacex import GelSightSensor
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
    create_gelsight_standard_franka_arm_visual_usd,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from tacex_tasks.sim2real_grasp.sim2real_cube_real_alignment_env import (
    Sim2RealCubeRealAlignmentEnvCfg,
)
from tacex_tasks.sim2real_grasp.sim2real_cube_real_alignment_rma_env import (
    Sim2RealCubeRealAlignmentRMAStudentDREnv,
    Sim2RealCubeRealAlignmentRMAStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMAStudentEnv,
    Sim2RealCubeRealAlignmentRMAStudentEnvCfg,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnv,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnvCfg,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapEnv,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapEnvCfg,
    Sim2RealCubeRealAlignmentRMATeacherEnv,
    Sim2RealCubeRealAlignmentRMATeacherEnvCfg,
)

from .gelsight_geometry import (
    GELSIGHT_HAND_TO_FINGERTIP_BOTTOM_M,
    GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M,
    GELSIGHT_TABLE_CLEARANCE_MIN_M,
    GELSIGHT_TABLE_CLEARANCE_PENALTY,
    table_clearance_penalty,
)


_GELSIGHT_PROFILE = "franka_gsmini_gripper_rigid_left_right"
_GELSIGHT_SENSOR_NAMES = ("gsmini_left", "gsmini_right")
_GELSIGHT_TACTILE_RGB_SHAPE = (96, 128, 3)
_GELSIGHT_SENSOR_PRIMS = (
    "/World/envs/env_.*/Robot/gelsight_mini_case_left",
    "/World/envs/env_.*/Robot/gelsight_mini_case_right",
)
_GELSIGHT_CONTACT_FILTER_PRIMS = (
    "/World/envs/env_.*/Robot/gelpad_left",
    "/World/envs/env_.*/Robot/gelpad_right",
)

def _make_gelsight_robot_cfg(
    *, align_standard_franka_visuals: bool = False
) -> ArticulationCfg:
    """Return the Real-Alignment initial state on the two-GelSight Franka asset."""
    reference_robot = Sim2RealCubeRealAlignmentEnvCfg().robot
    robot_cfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=reference_robot.init_state,
    )
    if align_standard_franka_visuals:
        robot_cfg.spawn.usd_path = create_gelsight_standard_franka_arm_visual_usd()
    hand_actuator = robot_cfg.actuators["panda_hand"]
    hand_actuator.effort_limit_sim = 40.0
    hand_actuator.stiffness = 400.0
    hand_actuator.damping = 40.0
    return robot_cfg


def _make_gelsight_sensor_cfg(prim_path: str) -> GelSightMiniCfg:
    sensor_cfg = GelSightMiniCfg(prim_path=prim_path)
    sensor_cfg.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=0.0,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
        update_latest_camera_pose=False,
    )
    sensor_cfg.data_types = ["tactile_rgb"]
    sensor_cfg.marker_motion_sim_cfg = None
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        tactile_img_res=(128, 96)
    )
    return sensor_cfg


def _make_gelsight_cube_contact_sensor_cfg() -> ContactSensorCfg:
    """Return cube ContactSensor filters that match the GelSight contact bodies."""
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/cube",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        filter_prim_paths_expr=list(_GELSIGHT_CONTACT_FILTER_PRIMS),
    )


class _RMAGelSightSensorMixin:
    """Optionally register left/right GelSight Mini sensors."""

    def _setup_scene(self) -> None:
        super()._setup_scene()
        if not bool(getattr(self.cfg, "rma_gelsight_tactile_sensor_enabled", False)):
            self.gsmini_left = None
            self.gsmini_right = None
            return
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

    def get_gelsight_tactile_rgb(self) -> dict[str, object | None]:
        """Return current left/right tactile RGB tensors for debug tooling."""
        if self.gsmini_left is None or self.gsmini_right is None:
            return {"gsmini_left": None, "gsmini_right": None}
        return {
            "gsmini_left": self.gsmini_left.data.output.get("tactile_rgb"),
            "gsmini_right": self.gsmini_right.data.output.get("tactile_rgb"),
        }


class _RMAGelSightGeometryMixin:
    """Use the rigid dual-GelSight geometry for reward, critic, and safety."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        self._gelsight_center_offset_hand = torch.tensor(
            self.cfg.gelsight_center_offset_hand_m,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._gelsight_bottom_offset_hand = torch.tensor(
            self.cfg.gelsight_fingertip_bottom_offset_hand_m,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)

    def _hand_offset_world(self, offset_hand_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform batched ``panda_hand`` local offsets into world coordinates."""
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        return math_utils.combine_frame_transforms(
            hand_pos,
            hand_quat,
            offset_hand_m,
            self._offset_rot,
        )

    def _compute_reach_center_world(self) -> torch.Tensor:
        """Return the midpoint of the two GelSight contact faces in world coordinates."""
        center, _ = self._hand_offset_world(self._gelsight_center_offset_hand)
        return center

    def _compute_gelsight_fingertip_bottom_world(self) -> torch.Tensor:
        """Return panda_fingertip_centered's hand-frame reference in world coordinates."""
        bottom, _ = self._hand_offset_world(self._gelsight_bottom_offset_hand)
        return bottom

    def _compute_additional_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Add a geometry-only penalty before the lowest gripper point reaches the table."""
        reward, log = super()._compute_additional_reward()
        clearance = (
            self._compute_gelsight_fingertip_bottom_world()[:, 2]
            - float(self.cfg.plate_top_height_m)
        )
        below_clearance, penalty = table_clearance_penalty(
            clearance,
            min_clearance_m=float(self.cfg.gelsight_table_clearance_min_m),
            penalty_value=float(self.cfg.gelsight_table_clearance_penalty),
        )
        self._last_gelsight_table_clearance_m = clearance.detach().clone()
        self._last_gelsight_table_clearance_penalty = penalty.detach().clone()
        log.update(
            {
                "reward/gelsight_table_clearance": penalty.mean().detach(),
                "info/gelsight_fingertip_bottom_clearance_m": clearance.mean().detach(),
                "info/gelsight_table_clearance_fraction": below_clearance.float().mean().detach(),
            }
        )
        return reward + penalty, log

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Keep privileged gripper state aligned with the GelSight reward center."""
        observations = super()._get_observations()
        obs = observations["policy"]
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        hand_lin_vel = self._robot.data.body_link_lin_vel_w[:, self._body_idx]
        hand_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._body_idx]
        center = self._compute_reach_center_world()
        center_offset_w = math_utils.quat_apply(hand_quat, self._gelsight_center_offset_hand)
        center_lin_vel = hand_lin_vel + torch.cross(hand_ang_vel, center_offset_w, dim=-1)
        target_pos = self._cube.data.root_pos_w - center
        obs["critic_gripper_pos"] = center
        obs["critic_gripper_quat"] = hand_quat
        obs["critic_gripper_lin_vel"] = center_lin_vel
        obs["critic_gripper_ang_vel"] = hand_ang_vel
        obs["critic_target_pos"] = target_pos
        obs["critic_target_distance"] = torch.norm(target_pos, dim=-1, keepdim=True)
        return observations

    def _additional_reward_print_fields(self) -> str:
        fields = super()._additional_reward_print_fields()
        if not hasattr(self, "_last_gelsight_table_clearance_m"):
            return fields
        return (
            fields
            + "gelsight_tip_clearance="
            + f"{self._last_gelsight_table_clearance_m.mean().item():.4f} m, "
            + "gelsight_tip_penalty="
            + f"{self._last_gelsight_table_clearance_penalty.mean().item():.3f}, "
        )


class _RMAGelSightStudentObservationMixin:
    """Expose GelSight tactile RGB to Student adaptation without touching Actor inputs."""

    def _tactile_rgb_or_zeros(self, sensor_name: str) -> torch.Tensor:
        sensor = self.scene.sensors.get(sensor_name)
        tactile = None if sensor is None else sensor.data.output.get("tactile_rgb")
        if tactile is None:
            return torch.zeros(
                (self.num_envs, *_GELSIGHT_TACTILE_RGB_SHAPE),
                device=self.device,
                dtype=torch.uint8,
            )
        image = tactile[..., :3]
        if image.dtype.is_floating_point:
            # The configured GPU Taxim backend returns float RGB in [0,1].
            # Use the explicit task contract instead of a per-step GPU->CPU
            # max().item() range probe for each sensor.
            image = image * float(self.cfg.rma_gelsight_tactile_rgb_float_scale)
            image = image.round().clamp(0, 255).to(torch.uint8)
        else:
            image = image.clamp(0, 255).to(torch.uint8)
        return image

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]
        left = self._tactile_rgb_or_zeros("gsmini_left")
        right = self._tactile_rgb_or_zeros("gsmini_right")
        if bool(getattr(self.cfg, "rma_gelsight_reference_enabled", False)):
            self._ensure_gelsight_reference_buffers()
            pending = self._gelsight_reference_pending
            if pending.any():
                # SensorBase.data above forces the first valid post-reset update.
                # Copy only pending rows so asynchronous resets cannot replace
                # another environment's episode reference.
                self._gelsight_left_reference_rgb[pending] = left[pending]
                self._gelsight_right_reference_rgb[pending] = right[pending]
                self._gelsight_reference_pending[pending] = False
            obs["gsmini_left_reference_rgb"] = self._gelsight_left_reference_rgb.clone()
            obs["gsmini_right_reference_rgb"] = self._gelsight_right_reference_rgb.clone()
        obs["gsmini_left_rgb"] = left
        obs["gsmini_right_rgb"] = right
        return observations

    def _ensure_gelsight_reference_buffers(self) -> None:
        shape = (self.num_envs, *_GELSIGHT_TACTILE_RGB_SHAPE)
        if hasattr(self, "_gelsight_left_reference_rgb"):
            return
        self._gelsight_left_reference_rgb = torch.zeros(
            shape, device=self.device, dtype=torch.uint8
        )
        self._gelsight_right_reference_rgb = torch.zeros(
            shape, device=self.device, dtype=torch.uint8
        )
        self._gelsight_reference_pending = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        if bool(getattr(self.cfg, "rma_gelsight_reference_enabled", False)):
            self._ensure_gelsight_reference_buffers()
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            self._gelsight_left_reference_rgb[env_ids] = 0
            self._gelsight_right_reference_rgb[env_ids] = 0
            self._gelsight_reference_pending[env_ids] = True


_GELSIGHT_TACTILE_OBSERVATION_SPACE = {
    "gsmini_left_rgb": gym.spaces.Box(
        low=0,
        high=255,
        shape=_GELSIGHT_TACTILE_RGB_SHAPE,
        dtype=np.uint8,
    ),
    "gsmini_right_rgb": gym.spaces.Box(
        low=0,
        high=255,
        shape=_GELSIGHT_TACTILE_RGB_SHAPE,
        dtype=np.uint8,
    ),
}

_GELSIGHT_REFERENCE_OBSERVATION_SPACE = {
    "gsmini_left_reference_rgb": gym.spaces.Box(
        low=0, high=255, shape=_GELSIGHT_TACTILE_RGB_SHAPE, dtype=np.uint8
    ),
    "gsmini_right_reference_rgb": gym.spaces.Box(
        low=0, high=255, shape=_GELSIGHT_TACTILE_RGB_SHAPE, dtype=np.uint8
    ),
}


def _make_student_observation_space(base_cfg_cls: type) -> dict:
    return {
        **base_cfg_cls().observation_space,
        **_GELSIGHT_TACTILE_OBSERVATION_SPACE,
    }


@configclass
class _RMAGelSightCfgMixin:
    """Shared static GelSight profile fields for RMA contracts."""

    robot = _make_gelsight_robot_cfg()
    rma_robot_profile = _GELSIGHT_PROFILE
    rma_gelsight_enabled = True
    arm_ik_tcp_source = "panda_hand_fixed_offset_gelpad_midpoint"
    arm_ik_tcp_offset_m = (0.0, 0.0, GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M)
    reach_center_source = "panda_hand_plus_z_gelpad_left_right_midpoint"
    gelsight_center_offset_hand_m = (0.0, 0.0, GELSIGHT_HAND_TO_GELPAD_MIDPOINT_M)
    gelsight_fingertip_bottom_offset_hand_m = (0.0, 0.0, GELSIGHT_HAND_TO_FINGERTIP_BOTTOM_M)
    gelsight_table_clearance_min_m = GELSIGHT_TABLE_CLEARANCE_MIN_M
    gelsight_table_clearance_penalty = GELSIGHT_TABLE_CLEARANCE_PENALTY
    rma_gelsight_sensor_names = _GELSIGHT_SENSOR_NAMES
    rma_gelsight_sensor_prims = _GELSIGHT_SENSOR_PRIMS
    rma_gelsight_actor_observation = "none"
    rma_student_contact_observation_source = "physics_contact_label"
    rma_gelsight_contact_filter_prims = _GELSIGHT_CONTACT_FILTER_PRIMS
    rma_cube_contact_sensor = _make_gelsight_cube_contact_sensor_cfg()
    rma_gelsight_tactile_sensor_enabled = False
    rma_gelsight_reference_enabled = False
    rma_gelsight_tactile_rgb_float_scale = 255.0
    gsmini_left = _make_gelsight_sensor_cfg(_GELSIGHT_SENSOR_PRIMS[0])
    gsmini_right = _make_gelsight_sensor_cfg(_GELSIGHT_SENSOR_PRIMS[1])

    def __post_init__(self) -> None:
        # configclass injects a non-forwarding __post_init__ into field-only
        # mixins. Forward explicitly so the Real-Alignment base can align RTX
        # rendering with the 30 Hz policy cadence.
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg(
    _RMAGelSightCfgMixin,
    Sim2RealCubeRealAlignmentRMATeacherEnvCfg,
):
    """Privileged RMA Teacher running on the two-GelSight Franka asset."""

    rma_task_id = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Teacher-v0"
    rma_role = "teacher"


class Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv(
    _RMAGelSightGeometryMixin,
    _RMAGelSightSensorMixin,
    Sim2RealCubeRealAlignmentRMATeacherEnv,
):
    """GelSight-equipped RMA Teacher; GelSight is not an Actor input."""

    cfg: Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightStudentEnvCfg(
    _RMAGelSightCfgMixin,
    Sim2RealCubeRealAlignmentRMAStudentEnvCfg,
):
    """Clean visual Student on the same two-GelSight robot profile."""

    rma_task_id = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-v0"
    rma_role = "student"
    rma_student_contact_observation_source = "gelsight_tactile_rgb"
    rma_gelsight_tactile_sensor_enabled = True
    observation_space = _make_student_observation_space(
        Sim2RealCubeRealAlignmentRMAStudentEnvCfg
    )


class Sim2RealCubeRealAlignmentRMAGelSightStudentEnv(
    _RMAGelSightStudentObservationMixin,
    _RMAGelSightSensorMixin,
    _RMAGelSightGeometryMixin,
    Sim2RealCubeRealAlignmentRMAStudentEnv,
):
    """GelSight-equipped RMA visual Student environment."""

    cfg: Sim2RealCubeRealAlignmentRMAGelSightStudentEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapEnvCfg(
    _RMAGelSightCfgMixin,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapEnvCfg,
):
    """Heatmap-supervised Clean Student on the GelSight robot profile."""

    rma_task_id = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-v0"
    rma_role = "student"
    rma_student_contact_observation_source = "gelsight_tactile_rgb"
    rma_gelsight_tactile_sensor_enabled = True
    observation_space = _make_student_observation_space(
        Sim2RealCubeRealAlignmentRMAStudentHeatmapEnvCfg
    )


class Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapEnv(
    _RMAGelSightStudentObservationMixin,
    _RMAGelSightSensorMixin,
    _RMAGelSightGeometryMixin,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapEnv,
):
    """GelSight-equipped heatmap Student environment."""

    cfg: Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg(
    _RMAGelSightCfgMixin,
    Sim2RealCubeRealAlignmentRMAStudentDREnvCfg,
):
    """Full-strength visual DR Student on the GelSight robot profile."""

    rma_task_id = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-DR-v0"
    rma_role = "student"
    rma_student_contact_observation_source = "gelsight_tactile_rgb"
    rma_gelsight_tactile_sensor_enabled = True
    observation_space = _make_student_observation_space(
        Sim2RealCubeRealAlignmentRMAStudentDREnvCfg
    )


class Sim2RealCubeRealAlignmentRMAGelSightStudentDREnv(
    _RMAGelSightStudentObservationMixin,
    _RMAGelSightSensorMixin,
    _RMAGelSightGeometryMixin,
    Sim2RealCubeRealAlignmentRMAStudentDREnv,
):
    """GelSight-equipped DR Student environment."""

    cfg: Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapDREnvCfg(
    _RMAGelSightCfgMixin,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnvCfg,
):
    """Heatmap-supervised full-strength DR Student on the GelSight robot profile."""

    rma_task_id = (
        "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-DR-v0"
    )
    rma_role = "student"
    rma_student_contact_observation_source = "gelsight_tactile_rgb"
    rma_gelsight_tactile_sensor_enabled = True
    observation_space = _make_student_observation_space(
        Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnvCfg
    )


class Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapDREnv(
    _RMAGelSightStudentObservationMixin,
    _RMAGelSightSensorMixin,
    _RMAGelSightGeometryMixin,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnv,
):
    """GelSight-equipped heatmap DR Student environment."""

    cfg: Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapDREnvCfg
