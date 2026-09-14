"""Four-tactile Pulled-Drawer Progress Teacher and binary Student environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from tacex import GelSightSensor
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_PROFILE,
    GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_USD,
)

from .sim2real_cube_real_alignment_gelsight_pulled_drawer_env import (
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnv,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg,
)
from .sim2real_cube_real_alignment_gelsight_x040_progress_env import (
    _GelSightX040ProgressCfgMixin,
    _GelSightX040ProgressRewardMixin,
)
from .sim2real_cube_real_alignment_gelsight_x040_three_frame_env import (
    linear_collision_threshold,
)
from .sim2real_cube_real_alignment_gelsight_rma_env import (
    _GELSIGHT_TACTILE_RGB_SHAPE,
    _make_gelsight_sensor_cfg,
)


GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-Progress-Four-Tactile-Teacher-v0"
)
GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-Progress-Four-Tactile-"
    "Three-Frame-Binary-Direct-Action-Student-DR-v0"
)
FOUR_TACTILE_CONTACT_ORDER = ("left_inner", "right_inner", "left_down", "right_down")
FOUR_TACTILE_CONTACT_WEIGHTS = (1.5, 1.5, 1.0, 1.0)
# Conservative hand-frame corners of the two downward GelPad contact faces.
# Values come from the V8 composed USD bounds relative to ``panda_hand``.
# Hand local +Z points toward the support surface in the task's fixed grasp pose.
FOUR_TACTILE_DOWN_FACE_POINTS_HAND_M = (
    (-0.011597, 0.010610, 0.183310),
    (-0.011597, 0.035863, 0.183310),
    (0.009238, 0.010610, 0.183310),
    (0.009238, 0.035863, 0.183310),
    (-0.014165, -0.036317, 0.183098),
    (-0.014165, -0.011064, 0.183098),
    (0.006669, -0.036317, 0.183098),
    (0.006669, -0.011064, 0.183098),
)
FOUR_TACTILE_LOWEST_POINT_MAX_HAND_Z_M = 0.183310
FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_START_N = 50.0
FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_END_N = 10.0
FOUR_TACTILE_DOWN_COLLISION_CURRICULUM_START_STEP = 0
FOUR_TACTILE_DOWN_COLLISION_CURRICULUM_END_STEP = 100_000
FOUR_TACTILE_DOWN_COLLISION_PENALTY = -10.0
_CONTACT_PRIMS = (
    "/World/envs/env_.*/Robot/gelpad_left",
    "/World/envs/env_.*/Robot/gelpad_right",
    "/World/envs/env_.*/Robot/gelpad_left_01",
    "/World/envs/env_.*/Robot/gelpad_right_01",
)
_DOWN_SENSOR_PRIMS = (
    "/World/envs/env_.*/Robot/gelsight_mini_case_left_down",
    "/World/envs/env_.*/Robot/gelsight_mini_case_right_down",
)


def _four_contact_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/cube", update_period=0.0, history_length=2,
        debug_vis=False, filter_prim_paths_expr=list(_CONTACT_PRIMS)
    )


def _four_robot_cfg():
    robot = Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg().robot.copy()
    robot.spawn = robot.spawn.copy()
    robot.spawn.usd_path = GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_USD
    return robot


def _four_cube_illegal_cfg():
    cfg = Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg().cube_illegal_contact_sensor.copy()
    cfg.filter_prim_paths_expr = list(cfg.filter_prim_paths_expr) + [
        "/World/envs/env_.*/Robot/gelsight_mini_case_left_down",
        "/World/envs/env_.*/Robot/gelsight_mini_case_right_down",
    ]
    return cfg


def _four_surface_sensor_cfgs():
    result = {}
    base = Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg().pulled_drawer_surface_contact_sensors
    additions = [
        "/World/envs/env_.*/Robot/gelsight_mini_case_left_down",
        "/World/envs/env_.*/Robot/gelsight_mini_case_right_down",
        "/World/envs/env_.*/Robot/gelpad_left_01",
        "/World/envs/env_.*/Robot/gelpad_right_01",
    ]
    for name, sensor in base.items():
        copied = sensor.copy()
        copied.filter_prim_paths_expr = list(copied.filter_prim_paths_expr) + additions
        result[name] = copied
    return result


def _four_observation_space(base: dict, *, include_down_tactile: bool) -> dict:
    result = dict(base)
    result["rma_contact_state"] = gym.spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
    if include_down_tactile:
        for side in ("left_down", "right_down"):
            result[f"gsmini_{side}_rgb"] = gym.spaces.Box(
                0, 255, shape=_GELSIGHT_TACTILE_RGB_SHAPE, dtype=np.uint8
            )
            result[f"gsmini_{side}_reference_rgb"] = gym.spaces.Box(
                0, 255, shape=_GELSIGHT_TACTILE_RGB_SHAPE, dtype=np.uint8
            )
    return result


def lowest_world_point_from_hand_frame(
    hand_position_w: torch.Tensor,
    hand_quaternion_w: torch.Tensor,
    points_hand: torch.Tensor,
) -> torch.Tensor:
    """Transform clearance samples and return the minimum-world-Z point, ``[N,S,3] -> [N,3]``."""
    if hand_position_w.ndim != 2 or hand_position_w.shape[-1] != 3:
        raise ValueError("hand_position_w must have shape [N,3]")
    if hand_quaternion_w.shape != (hand_position_w.shape[0], 4):
        raise ValueError("hand_quaternion_w must have shape [N,4]")
    if points_hand.ndim != 2 or points_hand.shape[-1] != 3:
        raise ValueError("points_hand must have shape [S,3]")
    batch_size = hand_position_w.shape[0]
    point_count = points_hand.shape[0]
    rotations = hand_quaternion_w.unsqueeze(1).expand(-1, point_count, -1)
    local_points = points_hand.unsqueeze(0).expand(batch_size, -1, -1)
    world_points = math_utils.quat_apply(
        rotations.reshape(-1, 4), local_points.reshape(-1, 3)
    ).reshape(batch_size, point_count, 3)
    world_points = world_points + hand_position_w.unsqueeze(1)
    lowest_indices = world_points[..., 2].argmin(dim=1)
    return world_points[
        torch.arange(batch_size, device=world_points.device), lowest_indices
    ]


def down_tactile_collision_penalty(
    four_tactile_forces_n: torch.Tensor,
    *,
    threshold_n: float,
    penalty_value: float = FOUR_TACTILE_DOWN_COLLISION_PENALTY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize excessive Cube force on either downward pad without terminating, ``[N,4]``."""
    if four_tactile_forces_n.ndim != 2 or four_tactile_forces_n.shape[-1] != 4:
        raise ValueError("four_tactile_forces_n must have shape [N,4]")
    if threshold_n <= 0.0 or penalty_value > 0.0:
        raise ValueError("threshold must be positive and penalty must be non-positive")
    down_force_max = four_tactile_forces_n[:, 2:].amax(dim=-1)
    collision = down_force_max > float(threshold_n)
    return collision.to(four_tactile_forces_n.dtype) * float(penalty_value), collision


def four_tactile_down_collision_threshold(
    policy_step: int,
    *,
    start_n: float = FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_START_N,
    end_n: float = FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_END_N,
    start_step: int = FOUR_TACTILE_DOWN_COLLISION_CURRICULUM_START_STEP,
    end_step: int = FOUR_TACTILE_DOWN_COLLISION_CURRICULUM_END_STEP,
) -> float:
    """Return the clamped linear downward-contact threshold for one policy step."""
    return linear_collision_threshold(
        policy_step,
        start_n=start_n,
        end_n=end_n,
        start_step=start_step,
        end_step=end_step,
    )


class _FourTactileMixin:
    """Expand contact/reward and optional RGB sensing from two pads to four."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        # Input [S=8,3] in panda_hand coordinates. Output clearance is the
        # minimum world-Z across both downward GelPad face corners per env.
        self._four_tactile_clearance_points_hand = torch.tensor(
            self.cfg.four_tactile_clearance_points_hand_m,
            device=self.device,
            dtype=torch.float32,
        )

    def _compute_gelsight_fingertip_bottom_world(self) -> torch.Tensor:
        hand_position_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quaternion_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        return lowest_world_point_from_hand_frame(
            hand_position_w,
            hand_quaternion_w,
            self._four_tactile_clearance_points_hand,
        )

    def _rma_contact_force_history(self) -> torch.Tensor:
        history = self.rma_cube_contact_sensor.data.force_matrix_w_history
        if history is None or history.shape[-2] != 4:
            shape = None if history is None else tuple(history.shape)
            raise RuntimeError(f"Expected four RMA contact filters, got {shape}")
        return history

    def _compute_rma_contact_state_from_forces(self, forces: torch.Tensor) -> torch.Tensor:
        return (forces >= float(self.cfg.rma_contact_force_threshold_n)).to(torch.float32)

    def _current_four_tactile_down_collision_threshold_n(self) -> float:
        """Return the linearly scheduled downward-contact threshold for this policy step."""
        policy_step = int(getattr(self, "common_step_counter", 0)) + int(
            self.cfg.illegal_collision_curriculum_step_offset
        )
        return four_tactile_down_collision_threshold(
            policy_step,
            start_n=float(self.cfg.four_tactile_down_collision_threshold_start_n),
            end_n=float(self.cfg.four_tactile_down_collision_threshold_end_n),
            start_step=int(self.cfg.four_tactile_down_collision_curriculum_start_step),
            end_step=int(self.cfg.four_tactile_down_collision_curriculum_end_step),
        )

    def _compute_additional_reward(self):
        reward, log = super()._compute_additional_reward()
        inherited = self._last_rma_contact_reward
        weights = torch.as_tensor(
            FOUR_TACTILE_CONTACT_WEIGHTS, device=self.device, dtype=torch.float32
        )
        weighted = (self._last_rma_contact_state * weights).sum(dim=-1)
        reward = reward - inherited + weighted
        self._last_rma_contact_reward = weighted.detach().clone()
        log["reward/rma_contact"] = weighted.mean().detach()
        for index, name in enumerate(FOUR_TACTILE_CONTACT_ORDER):
            log[f"info/rma_{name}_contact_fraction"] = (
                self._last_rma_contact_state[:, index].mean().detach()
            )
            log[f"info/rma_{name}_contact_force_n"] = (
                self._last_rma_contact_forces[:, index].mean().detach()
            )
        down_collision_threshold_n = (
            self._current_four_tactile_down_collision_threshold_n()
        )
        down_collision_penalty, down_collision = down_tactile_collision_penalty(
            self._last_rma_contact_forces,
            threshold_n=down_collision_threshold_n,
            penalty_value=float(self.cfg.four_tactile_down_collision_penalty),
        )
        self._last_four_tactile_down_collision_penalty = (
            down_collision_penalty.detach().clone()
        )
        log.update(
            {
                "reward/down_tactile_illegal_collision": (
                    down_collision_penalty.mean().detach()
                ),
                "info/down_tactile_illegal_collision_fraction": (
                    down_collision.float().mean().detach()
                ),
                "info/down_tactile_collision_threshold_n": torch.tensor(
                    down_collision_threshold_n,
                    device=self.device,
                    dtype=self._last_rma_contact_forces.dtype,
                ),
                "info/down_tactile_contact_force_max_n": (
                    self._last_rma_contact_forces[:, 2:].amax(dim=-1).max().detach()
                ),
            }
        )
        return reward + down_collision_penalty, log

    def _rma_collision_reward_print_fields(
        self, log: dict[str, torch.Tensor]
    ) -> str:
        fields = super()._rma_collision_reward_print_fields(log)
        return (
            fields
            + "down_tactile_threshold="
            + f"{log['info/down_tactile_collision_threshold_n'].item():.2f} N, "
        )

    def _setup_scene(self) -> None:
        super()._setup_scene()
        if not bool(getattr(self.cfg, "rma_gelsight_tactile_sensor_enabled", False)):
            self.gsmini_left_down = self.gsmini_right_down = None
            return
        self.gsmini_left_down = GelSightSensor(self.cfg.gsmini_left_down)
        self.gsmini_right_down = GelSightSensor(self.cfg.gsmini_right_down)
        self.scene.sensors["gsmini_left_down"] = self.gsmini_left_down
        self.scene.sensors["gsmini_right_down"] = self.gsmini_right_down

    def _ensure_down_reference_buffers(self) -> None:
        if hasattr(self, "_gelsight_left_down_reference_rgb"):
            return
        shape = (self.num_envs, *_GELSIGHT_TACTILE_RGB_SHAPE)
        self._gelsight_left_down_reference_rgb = torch.zeros(shape, device=self.device, dtype=torch.uint8)
        self._gelsight_right_down_reference_rgb = torch.zeros(shape, device=self.device, dtype=torch.uint8)
        self._gelsight_down_reference_pending = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

    def _get_observations(self):
        observations = super()._get_observations()
        if not bool(getattr(self.cfg, "rma_gelsight_tactile_sensor_enabled", False)):
            return observations
        obs = observations["policy"]
        left = self._tactile_rgb_or_zeros("gsmini_left_down")
        right = self._tactile_rgb_or_zeros("gsmini_right_down")
        self._ensure_down_reference_buffers()
        pending = self._gelsight_down_reference_pending
        if pending.any():
            self._gelsight_left_down_reference_rgb[pending] = left[pending]
            self._gelsight_right_down_reference_rgb[pending] = right[pending]
            pending[pending] = False
        obs["gsmini_left_down_rgb"] = left
        obs["gsmini_right_down_rgb"] = right
        obs["gsmini_left_down_reference_rgb"] = self._gelsight_left_down_reference_rgb.clone()
        obs["gsmini_right_down_reference_rgb"] = self._gelsight_right_down_reference_rgb.clone()
        return observations

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        if hasattr(self, "_gelsight_down_reference_pending"):
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            self._gelsight_left_down_reference_rgb[env_ids] = 0
            self._gelsight_right_down_reference_rgb[env_ids] = 0
            self._gelsight_down_reference_pending[env_ids] = True


@configclass
class _FourTactileCfgMixin:
    robot = _four_robot_cfg()
    rma_robot_profile = GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_PROFILE
    pulled_drawer_robot_asset_profile = GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_PROFILE
    rma_gelsight_sensor_names = ("gsmini_left", "gsmini_right", "gsmini_left_down", "gsmini_right_down")
    rma_gelsight_sensor_prims = (
        "/World/envs/env_.*/Robot/gelsight_mini_case_left",
        "/World/envs/env_.*/Robot/gelsight_mini_case_right",
        *_DOWN_SENSOR_PRIMS,
    )
    rma_gelsight_contact_filter_prims = _CONTACT_PRIMS
    rma_contact_order = FOUR_TACTILE_CONTACT_ORDER
    rma_contact_reward_weights = FOUR_TACTILE_CONTACT_WEIGHTS
    rma_contact_threshold_comparison = "greater_than_or_equal"
    rma_excess_contact_force_penalty_mode = "quadratic_normalized_max_four_tactile_excess"
    four_tactile_clearance_source = "minimum_world_z_of_down_gelpad_face_corners"
    four_tactile_clearance_points_hand_m = FOUR_TACTILE_DOWN_FACE_POINTS_HAND_M
    four_tactile_down_collision_curriculum_schedule = (
        "linear_clamped_global_policy_step"
    )
    four_tactile_down_collision_threshold_start_n = (
        FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_START_N
    )
    four_tactile_down_collision_threshold_end_n = (
        FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_END_N
    )
    four_tactile_down_collision_curriculum_start_step = (
        FOUR_TACTILE_DOWN_COLLISION_CURRICULUM_START_STEP
    )
    four_tactile_down_collision_curriculum_end_step = (
        FOUR_TACTILE_DOWN_COLLISION_CURRICULUM_END_STEP
    )
    four_tactile_down_collision_curriculum_step_offset_source = (
        "illegal_collision_curriculum_step_offset"
    )
    four_tactile_down_collision_penalty = FOUR_TACTILE_DOWN_COLLISION_PENALTY
    four_tactile_down_collision_terminates_episode = False
    gelsight_fingertip_bottom_offset_hand_m = (
        0.0,
        0.0,
        FOUR_TACTILE_LOWEST_POINT_MAX_HAND_Z_M,
    )
    rma_cube_contact_sensor = _four_contact_sensor_cfg()
    cube_illegal_contact_sensor = _four_cube_illegal_cfg()
    pulled_drawer_surface_contact_sensors = _four_surface_sensor_cfgs()
    gsmini_left_down = _make_gelsight_sensor_cfg(_DOWN_SENSOR_PRIMS[0])
    gsmini_right_down = _make_gelsight_sensor_cfg(_DOWN_SENSOR_PRIMS[1])
    illegal_collision_terminates_episode = True
    pulled_drawer_collision_scope = (
        "cube_non_four_gelpads_robot_or_cabinet_and_nonbase_robot_floor_cabinet_tray"
    )


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg(
    _FourTactileCfgMixin,
    _GelSightX040ProgressCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg,
):
    rma_task_id = GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    observation_space = _four_observation_space(
        Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg().observation_space,
        include_down_tactile=False,
    )


class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnv(
    _GelSightX040ProgressRewardMixin,
    _FourTactileMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg(
    _FourTactileCfgMixin,
    _GelSightX040ProgressCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg,
):
    rma_task_id = GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK
    observation_space = _four_observation_space(
        Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg().observation_space,
        include_down_tactile=True,
    )


class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnv(
    _GelSightX040ProgressRewardMixin,
    _FourTactileMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg


__all__ = tuple(name for name in globals() if name.startswith(("GELSIGHT_", "Sim2Real")))
