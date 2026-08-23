"""GelSight RMA tasks with the X040-Wide reset/DR and three-frame contract."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_PROFILE,
    GELSIGHT_STANDARD_FRANKA_BASE_LED_COLOR_RGB,
    GELSIGHT_STANDARD_FRANKA_BASE_LED_SUBSET_PATH,
)

from .sim2real_cube_real_alignment_gelsight_rma_env import (
    _GELSIGHT_REFERENCE_OBSERVATION_SPACE,
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg,
    _make_gelsight_robot_cfg,
)
from .sim2real_cube_real_alignment_gelsight_size_buckets_env import (
    GELSIGHT_SIZE_BUCKETS_M,
    _GelSightFixedSizeCubeMixin,
    _cube_illegal_sensor_cfg,
    _full_cell_backdrop_cfg,
    _full_cell_plate_cfg,
    _isolated_scene_cfg,
    _table_illegal_sensor_cfg,
    _visible_ground_cfg,
)


GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-X040-DR-Size-Buckets-"
    "Teacher-v0"
)
GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-X040-DR-Size-Buckets-"
    "Three-Frame-Direct-Action-Student-DR-v0"
)


def linear_collision_threshold(
    step: int,
    *,
    start_n: float,
    end_n: float,
    start_step: int = 0,
    end_step: int = 100_000,
) -> float:
    """Linearly interpolate a collision-force threshold and clamp its endpoints."""
    if end_step <= start_step:
        raise ValueError("collision curriculum end_step must exceed start_step")
    progress = min(max((int(step) - start_step) / float(end_step - start_step), 0.0), 1.0)
    return float(start_n) + progress * (float(end_n) - float(start_n))


class _GelSightX040SafetyMixin:
    """Attach fixed-Cube GelSight safety sensors and the shared curriculum."""

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.cube_illegal_contact_sensor = ContactSensor(
            self.cfg.cube_illegal_contact_sensor
        )
        self.scene.sensors["cube_illegal_contact_sensor"] = self.cube_illegal_contact_sensor

    @staticmethod
    def _max_filtered_contact_force(sensor: ContactSensor) -> torch.Tensor:
        history = sensor.data.force_matrix_w_history
        if history is None:
            raise RuntimeError("Filtered contact sensor has no force history")
        # [N,H,source_bodies,filters,xyz] -> [N].
        return torch.linalg.vector_norm(history, dim=-1).flatten(start_dim=1).amax(dim=1)

    def _compute_illegal_collision_force(self) -> torch.Tensor:
        cube_force = self._max_filtered_contact_force(self.cube_illegal_contact_sensor)
        table_force = self._compute_table_robot_contact_force()
        return torch.maximum(cube_force, table_force)

    def _collision_curriculum_step(self) -> int:
        return int(getattr(self, "common_step_counter", 0)) + int(
            self.cfg.illegal_collision_curriculum_step_offset
        )

    def _collision_threshold(self, prefix: str) -> float:
        return linear_collision_threshold(
            self._collision_curriculum_step(),
            start_n=float(getattr(self.cfg, f"{prefix}_start_n")),
            end_n=float(getattr(self.cfg, f"{prefix}_end_n")),
            start_step=int(self.cfg.illegal_collision_curriculum_start_step),
            end_step=int(self.cfg.illegal_collision_curriculum_end_step),
        )

    def _compute_additional_reward(self):
        reward, log = super()._compute_additional_reward()
        illegal_force = self._compute_illegal_collision_force()
        penalty_threshold = self._collision_threshold("illegal_collision_penalty_threshold")
        penalty = (illegal_force > penalty_threshold).to(illegal_force.dtype) * float(
            self.cfg.illegal_collision_penalty
        )
        # Replace the inherited table-only collision penalty with the unified rule.
        reward = reward - self._last_table_collision_penalty + penalty
        self._last_illegal_collision_force = illegal_force.detach().clone()
        self._last_illegal_collision_penalty = penalty.detach().clone()
        log.pop("reward/table_collision", None)
        log.update(
            {
                "reward/illegal_collision": penalty.mean().detach(),
                "info/illegal_collision_max_force_n": illegal_force.max().detach(),
                "info/illegal_collision_penalty_threshold_n": torch.tensor(
                    penalty_threshold, device=self.device, dtype=illegal_force.dtype
                ),
            }
        )
        if bool(self.cfg.illegal_collision_terminates_episode):
            log["info/illegal_collision_termination_threshold_n"] = torch.tensor(
                self._collision_threshold("illegal_collision_termination_threshold"),
                device=self.device,
                dtype=illegal_force.dtype,
            )
        return reward, log

    def _rma_collision_reward_print_fields(
        self, log: dict[str, torch.Tensor]
    ) -> str:
        """Format the X040 unified collision metrics for the RMA summary."""
        termination_field = ""
        if "info/illegal_collision_termination_threshold_n" in log:
            termination_field = (
                "illegal_termination_threshold="
                f"{log['info/illegal_collision_termination_threshold_n'].item():.2f} N, "
            )
        return (
            f"illegal_collision={log['reward/illegal_collision'].item():.3f}, "
            f"illegal_threshold="
            f"{log['info/illegal_collision_penalty_threshold_n'].item():.2f} N, "
            f"{termination_field}"
            f"illegal_force_max={log['info/illegal_collision_max_force_n'].item():.2f} N, "
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).bool()
        current_height = self._cube.data.root_pos_w[:, 2]
        upright_cos = self._compute_cube_upright_cos(self._cube.data.root_quat_w)
        _, _, above_success_height = self._compute_cube_lift_terms(current_height, upright_cos)
        self._success_hold_counter = torch.where(
            above_success_height,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        success = self._success_hold_counter >= int(self.cfg.success_hold_steps)
        self._rma_episode_success_ever |= success
        ground_collision = torch.any(
            self._robot.data.body_link_pos_w[:, :, 2] < float(self.cfg.ground_height), dim=1
        )
        if bool(self.cfg.illegal_collision_terminates_episode):
            illegal_force = self._compute_illegal_collision_force()
            termination_threshold = self._collision_threshold(
                "illegal_collision_termination_threshold"
            )
            illegal_termination = illegal_force > termination_threshold
        else:
            # X040 keeps the force penalty but disables force-based termination.
            illegal_termination = torch.zeros_like(ground_collision)
        terminated = ground_collision | illegal_termination
        completed = terminated | time_out
        self._record_episode_outcomes_for_step(
            completed_count=completed.long().sum(),
            success_count=(completed & self._rma_episode_success_ever).long().sum(),
        )
        self._publish_episode_success_statistics()
        self._last_rma_success_nonterminal = success.detach().clone()
        self._last_illegal_collision_termination = illegal_termination.detach().clone()
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self.cube_illegal_contact_sensor.reset(env_ids)
        self.table_contact_sensor.reset(env_ids)

        # Input/output joint state: [K, 9]. Only the seven arm joints receive
        # clipped Gaussian reset noise; finger positions retain calibration.
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        std_rad = float(self.cfg.arm_joint_reset_noise_std_rad)
        clip_rad = float(self.cfg.arm_joint_reset_noise_clip_rad)
        if std_rad < 0.0 or clip_rad < 0.0:
            raise ValueError("Arm-joint reset noise std/clip must be non-negative")
        noise = torch.randn(
            (env_ids.numel(), 7),
            device=self.device,
            dtype=self._robot.data.default_joint_pos.dtype,
        ) * std_rad
        noise.clamp_(min=-clip_rad, max=clip_rad)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        arm_lower = self._robot.data.soft_joint_pos_limits[env_ids, :7, 0]
        arm_upper = self._robot.data.soft_joint_pos_limits[env_ids, :7, 1]
        noisy_arm_pos = torch.clamp(
            joint_pos[:, :7] + noise,
            min=arm_lower,
            max=arm_upper,
        )
        joint_pos[:, :7] = noisy_arm_pos
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if not hasattr(self, "_last_arm_joint_reset_noise_rad"):
            self._last_arm_joint_reset_noise_rad = torch.zeros(
                (self.num_envs, 7),
                device=self.device,
                dtype=joint_pos.dtype,
            )
        self._last_arm_joint_reset_noise_rad[env_ids] = (
            noisy_arm_pos - self._robot.data.default_joint_pos[env_ids, :7]
        )


@configclass
class _GelSightX040CfgMixin:
    """Physical, reset, and safety fields shared by the paired task family."""

    cube_size_buckets_m = GELSIGHT_SIZE_BUCKETS_M
    cube_size_sampling = "fixed_round_robin_by_environment"
    cube_size_assignment = "env_id_mod_8"
    cube_size_object_count_per_environment = 1
    cube_size_requires_num_envs_multiple_of_bucket_count = True
    enable_gpu_dynamics = True
    arm_joint_reset_noise_distribution = "normal_clipped_per_environment_per_reset"
    arm_joint_reset_noise_std_rad = 0.01
    arm_joint_reset_noise_clip_rad = 0.03
    cube_illegal_contact_sensor = _cube_illegal_sensor_cfg()
    table_contact_sensor = _table_illegal_sensor_cfg()
    illegal_collision_scope = "cube_non_gelpad_robot_or_nonbase_robot_table"
    illegal_collision_curriculum_start_step = 0
    illegal_collision_curriculum_end_step = 100_000
    illegal_collision_curriculum_step_offset = 0
    illegal_collision_penalty_threshold_start_n = 100.0
    illegal_collision_penalty_threshold_end_n = 10.0
    illegal_collision_terminates_episode = False
    illegal_collision_penalty = -10.0
    cube_x_pos_range = 0.08
    cube_y_pos_range = 0.10
    cube_position_curriculum_enabled = False
    cube_position_curriculum_force_full_range = True
    robot = _make_gelsight_robot_cfg()
    rma_franka_visual_profile = GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_PROFILE
    rma_base_status_led_subset_path = GELSIGHT_STANDARD_FRANKA_BASE_LED_SUBSET_PATH
    rma_base_status_led_color_rgb = GELSIGHT_STANDARD_FRANKA_BASE_LED_COLOR_RGB

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg(
    _GelSightX040CfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg,
):
    """GelSight Teacher on the X040 reset distribution and collision curriculum."""

    rma_task_id = GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK
    scene = _isolated_scene_cfg(Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg)
    ground = _visible_ground_cfg(Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg)
    plate = _full_cell_plate_cfg(Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg)
    backdrop = _full_cell_backdrop_cfg(Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg)
    cube = Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg().cube.copy()
    cube.init_state = cube.init_state.copy()
    cube.init_state.pos = (0.40, 0.00, 0.026)


class Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnv(
    _GelSightX040SafetyMixin,
    _GelSightFixedSizeCubeMixin,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg(
    _GelSightX040CfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg,
):
    """GelSight Student with X040 DR and three wrist RGB frames."""

    rma_task_id = GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK
    scene = _isolated_scene_cfg(Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg)
    ground = _visible_ground_cfg(Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg)
    plate = _full_cell_plate_cfg(Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg)
    backdrop = _full_cell_backdrop_cfg(Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg)
    cube = Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg().cube.copy()
    cube.init_state = cube.init_state.copy()
    cube.init_state.pos = (0.40, 0.00, 0.026)
    camera_position_delta_max_m = (0.01, 0.01, 0.01)
    camera_rotation_delta_max_deg = (2.0, 2.0, 2.0)
    rma_gelsight_reference_enabled = True
    rma_student_contact_observation_source = "gelsight_signed_current_minus_reference"
    wrist_rgb_history_length = 3
    wrist_rgb_history_stride_policy_steps = 1
    wrist_rgb_history_order = "oldest_to_newest"
    wrist_rgb_history_reset_fill = "repeat_first_post_reset_frame"
    observation_space = {
        **{
            name: value
            for name, value in Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg()
            .observation_space.items()
            if name != "wrist_rgb"
        },
        "wrist_rgb_history": gym.spaces.Box(
            low=0, high=255, shape=(3, 224, 224, 3), dtype=np.uint8
        ),
        **_GELSIGHT_REFERENCE_OBSERVATION_SPACE,
    }


class Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnv(
    _GelSightX040SafetyMixin,
    _GelSightFixedSizeCubeMixin,
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnv,
):
    """GelSight Student with reset-safe wrist history and one fixed Cube per env."""

    cfg: Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg

    def _ensure_wrist_rgb_history(self) -> None:
        expected_shape = (self.num_envs, 3, 224, 224, 3)
        if not hasattr(self, "_wrist_rgb_history") or self._wrist_rgb_history.shape != expected_shape:
            self._wrist_rgb_history = torch.zeros(
                expected_shape, device=self.device, dtype=torch.uint8
            )
            self._wrist_rgb_history_reset_pending = torch.ones(
                (self.num_envs,), device=self.device, dtype=torch.bool
            )

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self._ensure_wrist_rgb_history()
        self._wrist_rgb_history_reset_pending[
            env_ids.to(device=self.device, dtype=torch.long)
        ] = True

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        policy = observations["policy"]
        current_rgb = policy.pop("wrist_rgb")
        self._ensure_wrist_rgb_history()
        self._wrist_rgb_history[:, 0].copy_(self._wrist_rgb_history[:, 1])
        self._wrist_rgb_history[:, 1].copy_(self._wrist_rgb_history[:, 2])
        self._wrist_rgb_history[:, 2].copy_(current_rgb)
        reset_ids = torch.nonzero(
            self._wrist_rgb_history_reset_pending, as_tuple=False
        ).squeeze(-1)
        if reset_ids.numel() > 0:
            self._wrist_rgb_history[reset_ids] = current_rgb[reset_ids].unsqueeze(1).expand(
                -1, 3, -1, -1, -1
            )
            self._wrist_rgb_history_reset_pending[reset_ids] = False
        policy["wrist_rgb_history"] = self._wrist_rgb_history
        return observations


__all__ = (
    "GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK",
    "GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK",
    "linear_collision_threshold",
    "Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnv",
    "Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnv",
)
