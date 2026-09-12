"""X040 three-frame tasks with the GelSight signed-progress reward contract."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .sim2real_cube_real_alignment_gelsight_size_buckets_progress_env import (
    _GelSightProgressRewardCfgMixin,
    _GelSightProgressRewardMixin,
)
from .sim2real_cube_real_alignment_gelsight_x040_three_frame_env import (
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg,
)


GELSIGHT_X040_PROGRESS_TEACHER_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-X040-Progress-Teacher-v0"
)
GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-X040-Progress-"
    "Three-Frame-Direct-Action-Student-DR-v0"
)


@configclass
class _GelSightX040ProgressCfgMixin(_GelSightProgressRewardCfgMixin):
    """Apply the 0911 reward/safety contract to the 0823 X040 distribution."""

    reach_reward_mode = "absolute_normalized_proximity_per_step"
    reach_weight = 2.5
    lift_reward_mode = "absolute_normalized_progress_per_step"
    lift_weight = 2.5
    success_reward_weight = 1000.0
    illegal_collision_penalty_threshold_start_n = 20.0
    illegal_collision_penalty_threshold_end_n = 5.0
    illegal_collision_terminates_episode = False

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()


class _GelSightX040ProgressRewardMixin(_GelSightProgressRewardMixin):
    """Use absolute reach proximity while retaining signed lift/contact progress."""

    def _shape_reach_reward(
        self, reach_proximity: torch.Tensor
    ) -> torch.Tensor:
        # Input/output: current normalized proximity [N]. Lift and contact are
        # still differenced by the inherited progress-reward implementation.
        self._current_reach_proximity = reach_proximity
        return reach_proximity

    def _shape_lift_reward(
        self, lift_progress: torch.Tensor, upright: torch.Tensor
    ) -> torch.Tensor:
        # Input/output: current reset-relative normalized lift progress [N].
        # Unlike contact progress, this absolute lift value is paid every step.
        absolute_progress = (
            lift_progress * upright.to(dtype=lift_progress.dtype)
            if bool(self.cfg.lift_reward_requires_upright)
            else lift_progress
        )
        self._current_lift_progress = absolute_progress
        return absolute_progress


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightX040ProgressTeacherEnvCfg(
    _GelSightX040ProgressCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg,
):
    """Privileged Teacher on X040 geometry with terminal progress rewards."""

    rma_task_id = GELSIGHT_X040_PROGRESS_TEACHER_TASK


class Sim2RealCubeRealAlignmentRMAGelSightX040ProgressTeacherEnv(
    _GelSightX040ProgressRewardMixin,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightX040ProgressTeacherEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightX040ProgressThreeFrameStudentDREnvCfg(
    _GelSightX040ProgressCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg,
):
    """Three-frame Student with full X040 DR and terminal progress rewards."""

    rma_task_id = GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK


class Sim2RealCubeRealAlignmentRMAGelSightX040ProgressThreeFrameStudentDREnv(
    _GelSightX040ProgressRewardMixin,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightX040ProgressThreeFrameStudentDREnvCfg


__all__ = (
    "GELSIGHT_X040_PROGRESS_TEACHER_TASK",
    "GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK",
    "_GelSightX040ProgressRewardMixin",
    "Sim2RealCubeRealAlignmentRMAGelSightX040ProgressTeacherEnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightX040ProgressTeacherEnv",
    "Sim2RealCubeRealAlignmentRMAGelSightX040ProgressThreeFrameStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightX040ProgressThreeFrameStudentDREnv",
)
