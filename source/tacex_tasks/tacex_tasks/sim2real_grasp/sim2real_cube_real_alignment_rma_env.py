"""RMA-style teacher and visual-student variants of Real-Alignment Clean."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from .sim2real_cube_real_alignment_env import (
    Sim2RealCubeRealAlignmentDREnv,
    Sim2RealCubeRealAlignmentDREnvCfg,
    Sim2RealCubeRealAlignmentEnv,
    Sim2RealCubeRealAlignmentEnvCfg,
)
from .sim2real_cube_real_alignment_privileged_env import (
    Sim2RealCubeRealAlignmentPrivilegedEnv,
)


_CRITIC_OBSERVATION_SPACE = {
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


def _make_rma_cube_contact_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/cube",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/panda_leftfinger",
            "/World/envs/env_.*/Robot/panda_rightfinger",
        ],
    )


def _make_rma_cube_cfg():
    cube_cfg = Sim2RealCubeRealAlignmentEnvCfg().cube.copy()
    cube_cfg.spawn.activate_contact_sensors = True
    return cube_cfg


def _make_rma_robot_cfg():
    robot_cfg = Sim2RealCubeRealAlignmentEnvCfg().robot.copy()
    hand_actuator = robot_cfg.actuators["panda_hand"]
    hand_actuator.effort_limit_sim = 40.0
    hand_actuator.stiffness = 400.0
    hand_actuator.damping = 40.0
    return robot_cfg


class _RMATerminalMixin:
    """Add RMA contact privilege/reward while keeping success non-terminal."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from .rma_models import RMAObservationNormalizer

        normalizer = RMAObservationNormalizer().to(self.device)
        actual_lower = self._robot.data.soft_joint_pos_limits[0, :7, 0]
        actual_upper = self._robot.data.soft_joint_pos_limits[0, :7, 1]
        if not torch.allclose(actual_lower, normalizer.joint_lower, atol=1e-4, rtol=0.0):
            raise RuntimeError("Live Panda lower joint limits differ from the RMA normalization contract")
        if not torch.allclose(actual_upper, normalizer.joint_upper, atol=1e-4, rtol=0.0):
            raise RuntimeError("Live Panda upper joint limits differ from the RMA normalization contract")
        self._rma_episode_success_ever = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._rma_previous_action_history = torch.zeros(
            (self.num_envs, int(self.cfg.action_space)),
            device=self.device,
            dtype=torch.float32,
        )
        self._rma_action_rate_scale = torch.tensor(
            [
                float(self.cfg.action_scale),
                float(self.cfg.action_scale),
                float(self.cfg.action_scale),
                float(self.cfg.gripper_width_delta_scale),
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.rma_cube_contact_sensor = ContactSensor(self.cfg.rma_cube_contact_sensor)
        self.scene.sensors["rma_cube_contact_sensor"] = self.rma_cube_contact_sensor

    def _rma_contact_force_history(self) -> torch.Tensor:
        history = self.rma_cube_contact_sensor.data.force_matrix_w_history
        if history is None:
            raise RuntimeError("RMA cube contact sensor has no filtered force history")
        if history.shape[-2] != 2:
            raise RuntimeError(
                f"Expected two RMA finger contact filters, got shape={tuple(history.shape)}"
            )
        return history

    def _compute_rma_contact_forces(self) -> torch.Tensor:
        """Return max left/right cube-finger force norm over one policy step."""
        # [N, physics_history=2, cube_body=1, finger_filters=2, xyz=3]
        history = self._rma_contact_force_history()
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=(1, 2))

    def _compute_rma_contact_state(self) -> torch.Tensor:
        forces = self._compute_rma_contact_forces()
        return (forces >= float(self.cfg.rma_contact_force_threshold_n)).to(torch.float32)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        observations["policy"]["rma_contact_state"] = self._compute_rma_contact_state()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Keep the RMA console summary focused while preserving all reward logs."""
        print_interval = int(self.reward_print_interval)
        summary_step = self.step_count + 1
        should_print = print_interval > 0 and summary_step % print_interval == 0

        # The inherited implementation owns reward/window accounting and its
        # verbose print. Suppress only that print on RMA summary steps.
        if should_print:
            self.reward_print_interval = 0
        try:
            rewards = super()._get_rewards()
        finally:
            self.reward_print_interval = print_interval

        if not should_print:
            return rewards

        log = self.extras["log"]
        metrics = self._episode_success_statistics()
        average_reward = self._reward_print_window_sum / max(
            int(self._reward_print_window_count), 1
        )
        average_lift_mm = 1000.0 * log["info/cube_lift_delta"].item()
        if abs(average_lift_mm) < 0.005:
            average_lift_mm = 0.0
        print(
            f"[RMA奖励] step {summary_step}: "
            f"reach={float(self.cfg.reach_weight) * log['reward/reach'].item():.3f}, "
            f"lift={float(self.cfg.lift_weight) * log['reward/lift'].item():.3f}, "
            f"success={float(self.cfg.success_reward_weight) * log['reward/success'].item():.3f}, "
            f"contact={log['reward/rma_contact'].item():.3f}, "
            f"table={log['reward/table_collision'].item():.3f}, "
            f"smooth={log['reward/rma_action_rate'].item():.3f}, "
            f"total={log['reward/total'].item():.3f}, "
            f"avg_reward_{print_interval}={average_reward.item():.3f}, "
            f"success_window_{print_interval}={metrics['window_rate'].item():.3f} "
            f"({int(metrics['window_success_count'].item())}/"
            f"{int(metrics['window_completed_count'].item())}), "
            f"avg_lift={average_lift_mm:.2f} mm"
        )
        self._reset_reward_print_window()
        return rewards

    def _compute_rma_action_rate_penalty(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Penalize command jumps in normalized environment-action units."""
        weight = float(self.cfg.rma_action_rate_penalty_weight)
        if weight <= 0.0:
            zeros = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            return zeros, {
                "reward/rma_action_rate": zeros.mean().detach(),
                "info/rma_action_rate_norm_sq_mean": zeros.mean().detach(),
                "info/rma_action_rate_norm_max": zeros.max().detach(),
            }

        delta = (self.action_history - self._rma_previous_action_history) / self._rma_action_rate_scale
        normalized_rate_sq = torch.mean(delta.square(), dim=-1)
        penalty = -weight * normalized_rate_sq
        return penalty, {
            "reward/rma_action_rate": penalty.mean().detach(),
            "info/rma_action_rate_norm_sq_mean": normalized_rate_sq.mean().detach(),
            "info/rma_action_rate_norm_max": torch.linalg.vector_norm(delta, dim=-1).max().detach(),
        }

    def _compute_additional_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward, log = super()._compute_additional_reward()
        forces = self._compute_rma_contact_forces()
        contact = (forces >= float(self.cfg.rma_contact_force_threshold_n)).to(torch.float32)
        left_contact = contact[:, 0]
        right_contact = contact[:, 1]
        bilateral_contact = left_contact * right_contact
        single_contact = 0.5 * (left_contact + right_contact)
        single_fraction = float(self.cfg.rma_single_contact_reward_fraction)
        contact_reward = single_fraction * single_contact + (1.0 - single_fraction) * bilateral_contact
        weighted_contact_reward = float(self.cfg.rma_contact_reward_weight) * contact_reward
        action_rate_penalty, action_rate_log = self._compute_rma_action_rate_penalty()

        self._last_rma_contact_forces = forces.detach().clone()
        self._last_rma_contact_state = contact.detach().clone()
        self._last_rma_contact_reward = weighted_contact_reward.detach().clone()
        self._last_rma_action_rate_penalty = action_rate_penalty.detach().clone()
        log.update(
            {
                "reward/rma_contact": weighted_contact_reward.mean().detach(),
                "info/rma_left_contact_fraction": left_contact.mean().detach(),
                "info/rma_right_contact_fraction": right_contact.mean().detach(),
                "info/rma_bilateral_contact_fraction": bilateral_contact.mean().detach(),
                "info/rma_left_contact_force_n": forces[:, 0].mean().detach(),
                "info/rma_right_contact_force_n": forces[:, 1].mean().detach(),
            }
        )
        log.update(action_rate_log)
        return reward + weighted_contact_reward + action_rate_penalty, log

    def _additional_reward_print_fields(self) -> str:
        fields = super()._additional_reward_print_fields()
        if not hasattr(self, "_last_rma_contact_state"):
            return fields
        bilateral = self._last_rma_contact_state.prod(dim=-1).mean()
        return (
            fields
            + f"rma_contact_lr={self._last_rma_contact_state.mean(dim=0).tolist()}, "
            + f"rma_bilateral={bilateral.item():.3f}, "
            + f"rma_contact_reward={self._last_rma_contact_reward.mean().item():.3f}, "
            + f"rma_action_rate_penalty={self._last_rma_action_rate_penalty.mean().item():.3f}, "
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if hasattr(self, "action_history"):
            self._rma_previous_action_history = self.action_history.detach().clone()
        super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self.rma_cube_contact_sensor.reset(env_ids)
        self._rma_episode_success_ever[env_ids] = False
        self._rma_previous_action_history[env_ids] = 0.0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).bool()

        current_center_height = self._cube.data.root_pos_w[:, 2]
        upright_cos = self._compute_cube_upright_cos(self._cube.data.root_quat_w)
        _, _, above_success_height = self._compute_cube_lift_terms(
            current_center_height, upright_cos
        )
        self._success_hold_counter = torch.where(
            above_success_height,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        success = self._success_hold_counter >= int(self.cfg.success_hold_steps)
        self._rma_episode_success_ever |= success

        joint_positions = self._robot.data.body_link_pos_w
        joint_z_positions = joint_positions[:, :, 2]
        collision_with_ground = torch.any(
            joint_z_positions < self.cfg.ground_height, dim=1
        ).bool()

        dones = time_out | collision_with_ground
        self._record_episode_outcomes_for_step(
            completed_count=dones.to(dtype=torch.long).sum(),
            success_count=(dones & self._rma_episode_success_ever)
            .to(dtype=torch.long)
            .sum(),
        )
        self._publish_episode_success_statistics()
        self._last_rma_success_nonterminal = success.detach().clone()
        return dones, time_out


@configclass
class Sim2RealCubeRealAlignmentRMATeacherEnvCfg(Sim2RealCubeRealAlignmentEnvCfg):
    """Current Clean v9 physics with privileged cube XYZ and finger contact."""

    camera_sensor_enabled = False
    vision_encoder_enabled = False
    robot = _make_rma_robot_cfg()
    cube = _make_rma_cube_cfg()
    rma_cube_contact_sensor = _make_rma_cube_contact_sensor_cfg()
    rma_contact_force_threshold_n = 0.2
    rma_contact_reward_weight = 3.0
    rma_single_contact_reward_fraction = 1.0
    rma_action_rate_penalty_weight = 0.05
    rma_action_rate_penalty_scales = "environment_action_scales"
    rma_role = "teacher"
    rma_position_frame = "robot_root"
    rma_actor_feature_dim = 30
    rma_end_effector_position_source = "embedded_panda_fk_from_proprio_joint_position"
    rma_object_pose_components = "position_xyz_only"
    rma_contact_components = "left_right_cube_finger_binary"
    rma_success_terminates_episode = False
    observation_space = {
        "proprio_obs": 15,
        "action_history": 4,
        "rma_cube_pos": 3,
        "rma_contact_state": 2,
        **_CRITIC_OBSERVATION_SPACE,
    }


class Sim2RealCubeRealAlignmentRMATeacherEnv(
    _RMATerminalMixin, Sim2RealCubeRealAlignmentPrivilegedEnv
):
    """No-camera teacher task with privileged cube position and finger contact."""

    cfg: Sim2RealCubeRealAlignmentRMATeacherEnvCfg

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]
        obs["rma_cube_pos"] = obs.pop("privileged_cube_pos")
        obs.pop("privileged_gripper_pos", None)
        obs.pop("privileged_target_pos", None)
        return observations


@configclass
class Sim2RealCubeRealAlignmentRMAStudentEnvCfg(Sim2RealCubeRealAlignmentEnvCfg):
    """Clean camera task exposing uint8 RGB plus privileged training labels."""

    # The student owns the layer4 ResNet. Avoid constructing the environment's
    # separate global-average-pooled encoder.
    vision_encoder_enabled = False
    robot = _make_rma_robot_cfg()
    cube = _make_rma_cube_cfg()
    rma_cube_contact_sensor = _make_rma_cube_contact_sensor_cfg()
    # Keep the Student rollout distribution identical to the selected Teacher
    # contract. The privileged force/contact values remain loss labels only.
    rma_contact_force_threshold_n = 0.2
    rma_contact_reward_weight = 3.0
    rma_single_contact_reward_fraction = 1.0
    rma_action_rate_penalty_weight = 0.05
    rma_action_rate_penalty_scales = "environment_action_scales"
    rma_heatmap_supervision_enabled = False
    rma_heatmap_loss_weight = 0.0
    rma_heatmap_sigma_px = 1.5
    cube_position_curriculum_force_full_range = True
    rma_role = "student"
    rma_position_frame = "robot_root"
    rma_actor_feature_dim = 30
    rma_end_effector_position_source = "embedded_panda_fk_from_proprio_joint_position"
    rma_object_pose_components = "position_xyz_only"
    rma_contact_components = "left_right_cube_finger_binary"
    rma_success_terminates_episode = False
    observation_space = {
        "proprio_obs": 15,
        "action_history": 4,
        "wrist_rgb": gym.spaces.Box(
            low=0,
            high=255,
            shape=(224, 224, 3),
            dtype=np.uint8,
        ),
        "rma_cube_pos": 3,
        "rma_contact_state": 2,
        **_CRITIC_OBSERVATION_SPACE,
    }


class Sim2RealCubeRealAlignmentRMAStudentEnv(
    _RMATerminalMixin, Sim2RealCubeRealAlignmentEnv
):
    """Visual student task; privileged cube XYZ/contact are loss labels only."""

    cfg: Sim2RealCubeRealAlignmentRMAStudentEnvCfg

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        wrist_rgb = self.wrist_camera.data.output.get("rgb")
        if wrist_rgb is None:
            calibrated_rgb = torch.zeros(
                (self.num_envs, 224, 224, 3),
                device=self.device,
                dtype=torch.uint8,
            )
        else:
            # [N,H,W,C] uint8 -> calibrated Clean model image -> uint8. The
            # fixed nominal intrinsic warp is inherited from Real-Alignment.
            image = wrist_rgb[..., :3].to(torch.float32).permute(0, 3, 1, 2) / 255.0
            image = self._apply_wrist_visual_randomization(image)
            calibrated_rgb = (
                image.permute(0, 2, 3, 1).mul(255.0).round().clamp(0, 255).to(torch.uint8)
            )

        obs.pop("wrist_resnet", None)
        obs["wrist_rgb"] = calibrated_rgb
        obs["rma_cube_pos"] = self._cube.data.root_pos_w - self.scene.env_origins
        return observations


@configclass
class Sim2RealCubeRealAlignmentRMAStudentHeatmapEnvCfg(
    Sim2RealCubeRealAlignmentRMAStudentEnvCfg
):
    """Clean RMA Student with explicit cube-center heatmap supervision enabled."""

    rma_heatmap_supervision_enabled = True
    rma_heatmap_loss_weight = 1.0
    rma_heatmap_sigma_px = 1.5


class Sim2RealCubeRealAlignmentRMAStudentHeatmapEnv(
    Sim2RealCubeRealAlignmentRMAStudentEnv
):
    """Clean visual Student environment for heatmap-supervised distillation."""

    cfg: Sim2RealCubeRealAlignmentRMAStudentHeatmapEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAStudentDREnvCfg(
    Sim2RealCubeRealAlignmentDREnvCfg,
):
    """RMA Student with full-strength visual/scene DR from the first update.

    The RMA Teacher remains the Clean privileged controller. Only the Student's
    camera/rendered appearance is randomized, so the physics, rewards, contact
    labels, action contract, and Teacher manifest stay identical.
    """

    # Unlike the generic PPO DR profile, RMA distillation starts at the full
    # distribution immediately. ``_dr_curriculum_scale`` returns 1.0 whenever
    # this switch is false, independently of common_step_counter.
    dr_curriculum_enabled = False
    dr_curriculum_initial_scale = 1.0

    # Start from the DR configuration rather than inheriting the Clean Student
    # config: configclass copies inherited fields, so the latter would retain
    # Clean's explicit ``wrist_*_randomization_enabled=False`` values.
    vision_encoder_enabled = False
    robot = _make_rma_robot_cfg()
    cube = _make_rma_cube_cfg()
    rma_cube_contact_sensor = _make_rma_cube_contact_sensor_cfg()
    rma_contact_force_threshold_n = 0.2
    rma_contact_reward_weight = 3.0
    rma_single_contact_reward_fraction = 1.0
    rma_action_rate_penalty_weight = 0.05
    rma_action_rate_penalty_scales = "environment_action_scales"
    rma_heatmap_supervision_enabled = False
    rma_heatmap_loss_weight = 0.0
    rma_heatmap_sigma_px = 1.5
    cube_position_curriculum_force_full_range = True
    rma_role = "student"
    rma_position_frame = "robot_root"
    rma_actor_feature_dim = 30
    rma_end_effector_position_source = "embedded_panda_fk_from_proprio_joint_position"
    rma_object_pose_components = "position_xyz_only"
    rma_contact_components = "left_right_cube_finger_binary"
    rma_success_terminates_episode = False
    observation_space = {
        "proprio_obs": 15,
        "action_history": 4,
        "wrist_rgb": gym.spaces.Box(
            low=0,
            high=255,
            shape=(224, 224, 3),
            dtype=np.uint8,
        ),
        "rma_cube_pos": 3,
        "rma_contact_state": 2,
        **_CRITIC_OBSERVATION_SPACE,
    }


class Sim2RealCubeRealAlignmentRMAStudentDREnv(
    Sim2RealCubeRealAlignmentRMAStudentEnv,
    Sim2RealCubeRealAlignmentDREnv,
):
    """RMA visual Student using the shared full-strength DR implementation."""

    cfg: Sim2RealCubeRealAlignmentRMAStudentDREnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnvCfg(
    Sim2RealCubeRealAlignmentRMAStudentDREnvCfg
):
    """DR RMA Student with explicit cube-center heatmap supervision enabled."""

    rma_heatmap_supervision_enabled = True
    rma_heatmap_loss_weight = 1.0
    rma_heatmap_sigma_px = 1.5


class Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnv(
    Sim2RealCubeRealAlignmentRMAStudentDREnv
):
    """DR visual Student environment for heatmap-supervised distillation."""

    cfg: Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnvCfg
