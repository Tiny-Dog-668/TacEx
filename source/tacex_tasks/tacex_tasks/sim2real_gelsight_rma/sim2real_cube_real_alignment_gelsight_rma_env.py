"""GelSight-equipped RMA variants of the Real-Alignment cube task.

These tasks intentionally keep the existing RMA Actor contract unchanged. The
GelSight Mini geometry is part of the simulated robot profile. Teacher does not
render tactile RGB. Student uses third-person RGB for cube position and left/right
GelSight tactile RGB for contact prediction; the frozen Actor contract remains
unchanged.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
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


def _make_gelsight_robot_cfg() -> ArticulationCfg:
    """Return the Real-Alignment initial state on the two-GelSight Franka asset."""
    reference_robot = Sim2RealCubeRealAlignmentEnvCfg().robot
    robot_cfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=reference_robot.init_state,
    )
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
            if image.numel() > 0 and image.max().item() <= 1.0 + 1.0e-6:
                image = image * 255.0
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
    rma_gelsight_sensor_names = _GELSIGHT_SENSOR_NAMES
    rma_gelsight_sensor_prims = _GELSIGHT_SENSOR_PRIMS
    rma_gelsight_actor_observation = "none"
    rma_student_contact_observation_source = "physics_contact_label"
    rma_gelsight_contact_filter_prims = _GELSIGHT_CONTACT_FILTER_PRIMS
    rma_cube_contact_sensor = _make_gelsight_cube_contact_sensor_cfg()
    rma_gelsight_tactile_sensor_enabled = False
    rma_gelsight_reference_enabled = False
    gsmini_left = _make_gelsight_sensor_cfg(_GELSIGHT_SENSOR_PRIMS[0])
    gsmini_right = _make_gelsight_sensor_cfg(_GELSIGHT_SENSOR_PRIMS[1])


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg(
    _RMAGelSightCfgMixin,
    Sim2RealCubeRealAlignmentRMATeacherEnvCfg,
):
    """Privileged RMA Teacher running on the two-GelSight Franka asset."""

    rma_task_id = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Teacher-v0"
    rma_role = "teacher"


class Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv(
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
    Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnv,
):
    """GelSight-equipped heatmap DR Student environment."""

    cfg: Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapDREnvCfg
