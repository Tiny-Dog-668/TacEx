"""Progress-reward and terminal-success variants of GelSight Size-Buckets."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .sim2real_cube_real_alignment_gelsight_size_buckets_env import (
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnvCfg,
)


GELSIGHT_SIZE_BUCKETS_PROGRESS_TEACHER_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Size-Buckets-Progress-Teacher-v0"
)
GELSIGHT_SIZE_BUCKETS_PROGRESS_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Size-Buckets-Progress-Student-DR-v0"
)


def signed_progress_reward(
    current_progress: torch.Tensor, previous_progress: torch.Tensor
) -> torch.Tensor:
    """Return signed normalized progress ``[N]`` for one transition."""
    if current_progress.shape != previous_progress.shape:
        raise ValueError(
            "current and previous progress must have identical shapes, got "
            f"{tuple(current_progress.shape)} and {tuple(previous_progress.shape)}"
        )
    return current_progress - previous_progress


def quadratic_excess_contact_force_penalty(
    forces_n: torch.Tensor,
    *,
    threshold_n: float,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize the maximum left/right force excess continuously, ``[N,2] -> [N]``."""
    if forces_n.ndim != 2 or forces_n.shape[-1] != 2:
        raise ValueError(f"expected left/right forces [N,2], got {tuple(forces_n.shape)}")
    if threshold_n <= 0.0:
        raise ValueError("excess-contact threshold must be positive")
    if weight < 0.0:
        raise ValueError("excess-contact penalty weight must be non-negative")
    normalized_excess = torch.relu(forces_n - threshold_n) / threshold_n
    maximum_excess = normalized_excess.amax(dim=-1)
    return -weight * maximum_excess.square(), maximum_excess > 0.0


class _GelSightProgressRewardMixin:
    """Replace reward farming with signed lift progress and terminal success."""

    def _ensure_previous_lift_progress(self, reference: torch.Tensor) -> None:
        if not hasattr(self, "_previous_lift_progress"):
            self._previous_lift_progress = torch.zeros_like(reference)

    def _ensure_previous_reach_proximity(self, reference: torch.Tensor) -> None:
        if not hasattr(self, "_previous_reach_proximity"):
            self._previous_reach_proximity = torch.zeros_like(reference)
            self._reach_proximity_initialized = torch.zeros_like(
                reference, dtype=torch.bool
            )

    def _ensure_previous_contact_reward(self, reference: torch.Tensor) -> None:
        if not hasattr(self, "_previous_contact_reward"):
            self._previous_contact_reward = torch.zeros_like(reference)

    def _shape_reach_reward(self, reach_proximity: torch.Tensor) -> torch.Tensor:
        self._ensure_previous_reach_proximity(reach_proximity)
        self._current_reach_proximity = reach_proximity
        progress = signed_progress_reward(
            reach_proximity, self._previous_reach_proximity
        )
        # The reset state has no reliable post-simulation sensor sample yet.
        # Suppress only the first transition instead of granting its absolute
        # proximity as a reset bonus.
        return torch.where(
            self._reach_proximity_initialized,
            progress,
            torch.zeros_like(progress),
        )

    def _shape_lift_reward(
        self, lift_progress: torch.Tensor, upright: torch.Tensor
    ) -> torch.Tensor:
        # Input/output: normalized per-environment lift progress [N]. Upright
        # gating is applied before differencing so losing a required upright
        # pose returns the previously awarded shaping reward.
        absolute_progress = super()._shape_lift_reward(lift_progress, upright)
        self._ensure_previous_lift_progress(absolute_progress)
        self._current_lift_progress = absolute_progress
        return signed_progress_reward(
            absolute_progress, self._previous_lift_progress
        )

    def _shape_success_reward(self, success: torch.Tensor) -> torch.Tensor:
        # Rewards are evaluated before dones. Predict the hold counter that
        # _get_dones will commit for this same transition and emit one bonus on
        # the terminal success step only.
        next_hold = torch.where(
            success,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        success_event = success & (next_hold >= int(self.cfg.success_hold_steps))
        self._last_terminal_success_event = success_event.detach().clone()
        return success_event.to(torch.float32)

    def _compute_additional_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward, log = super()._compute_additional_reward()

        # Replace the inherited per-step contact reward with signed contact
        # acquisition progress. Holding an unchanged contact state earns zero;
        # losing contact returns the previously awarded amount.
        inherited_contact_reward = self._last_rma_contact_reward
        self._ensure_previous_contact_reward(inherited_contact_reward)
        self._current_contact_reward = inherited_contact_reward
        contact_progress_reward = signed_progress_reward(
            inherited_contact_reward, self._previous_contact_reward
        )
        reward = reward - inherited_contact_reward + contact_progress_reward
        self._last_rma_contact_reward = contact_progress_reward.detach().clone()

        # Replace the inherited fixed -5 excessive-force term with a continuous
        # quadratic penalty. At 30 N with a 15 N threshold the penalty is -5.
        inherited_force_penalty = self._last_gelsight_excess_force_penalty
        force_penalty, excessive_force = quadratic_excess_contact_force_penalty(
            self._last_rma_contact_forces,
            threshold_n=float(self.cfg.rma_excess_contact_force_threshold_n),
            weight=float(self.cfg.rma_excess_contact_force_quadratic_weight),
        )
        reward = reward - inherited_force_penalty + force_penalty
        self._last_gelsight_excess_force_penalty = force_penalty.detach().clone()

        # action_history stores physical XYZ/width increments [N,4]. Normalize
        # by the environment action scales before penalizing command magnitude.
        normalized_action = self.action_history / self._rma_action_rate_scale
        action_magnitude_sq = normalized_action.square().mean(dim=-1)
        action_penalty = (
            -float(self.cfg.rma_action_magnitude_penalty_weight)
            * action_magnitude_sq
        )
        reward = reward + action_penalty
        self._last_rma_action_magnitude_penalty = action_penalty.detach().clone()

        log.update(
            {
                "reward/gelsight_excess_contact_force": force_penalty.mean().detach(),
                "info/gelsight_excess_contact_force_fraction": (
                    excessive_force.float().mean().detach()
                ),
                "reward/rma_action_magnitude": action_penalty.mean().detach(),
                "reward/rma_contact": contact_progress_reward.mean().detach(),
                "info/rma_action_magnitude_norm_sq_mean": (
                    action_magnitude_sq.mean().detach()
                ),
            }
        )
        return reward, log

    def _get_rewards(self) -> torch.Tensor:
        rewards = super()._get_rewards()
        self._ensure_previous_lift_progress(self._current_lift_progress)
        self._previous_lift_progress.copy_(self._current_lift_progress.detach())
        self._ensure_previous_reach_proximity(self._current_reach_proximity)
        self._previous_reach_proximity.copy_(
            self._current_reach_proximity.detach()
        )
        self._reach_proximity_initialized.fill_(True)
        self._ensure_previous_contact_reward(self._current_contact_reward)
        self._previous_contact_reward.copy_(self._current_contact_reward.detach())
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if hasattr(self, "_previous_lift_progress"):
            self._previous_lift_progress[env_ids] = 0.0
            self._current_lift_progress[env_ids] = 0.0
        if hasattr(self, "_previous_reach_proximity"):
            self._previous_reach_proximity[env_ids] = 0.0
            self._current_reach_proximity[env_ids] = 0.0
            self._reach_proximity_initialized[env_ids] = False
        if hasattr(self, "_previous_contact_reward"):
            self._previous_contact_reward[env_ids] = 0.0
            self._current_contact_reward[env_ids] = 0.0


@configclass
class _GelSightProgressRewardCfgMixin:
    reach_reward_mode = "signed_normalized_proximity_delta"
    lift_reward_mode = "signed_normalized_progress_delta"
    contact_reward_mode = "signed_contact_acquisition_delta"
    success_reward_mode = "once_on_confirmed_terminal_success"
    rma_success_terminates_episode = True
    rma_action_magnitude_penalty_weight = 0.05
    rma_action_magnitude_penalty_scales = "environment_action_scales"
    rma_excess_contact_force_penalty_mode = "quadratic_normalized_max_side_excess"
    rma_excess_contact_force_quadratic_weight = 5.0

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressTeacherEnvCfg(
    _GelSightProgressRewardCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnvCfg,
):
    rma_task_id = GELSIGHT_SIZE_BUCKETS_PROGRESS_TEACHER_TASK


class Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressTeacherEnv(
    _GelSightProgressRewardMixin,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressTeacherEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressStudentDREnvCfg(
    _GelSightProgressRewardCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnvCfg,
):
    rma_task_id = GELSIGHT_SIZE_BUCKETS_PROGRESS_STUDENT_DR_TASK


class Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressStudentDREnv(
    _GelSightProgressRewardMixin,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressStudentDREnvCfg


__all__ = (
    "GELSIGHT_SIZE_BUCKETS_PROGRESS_TEACHER_TASK",
    "GELSIGHT_SIZE_BUCKETS_PROGRESS_STUDENT_DR_TASK",
    "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressTeacherEnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressTeacherEnv",
    "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsProgressStudentDREnv",
    "quadratic_excess_contact_force_penalty",
    "signed_progress_reward",
)
