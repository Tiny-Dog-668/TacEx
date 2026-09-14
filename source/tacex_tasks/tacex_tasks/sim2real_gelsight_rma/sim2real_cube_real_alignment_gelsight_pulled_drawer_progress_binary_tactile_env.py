"""Pulled-Drawer Progress Teacher and binary-tactile three-frame Student."""

from __future__ import annotations

from isaaclab.utils import configclass

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


GELSIGHT_PULLED_DRAWER_PROGRESS_TEACHER_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-"
    "Progress-Teacher-v0"
)
GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-Progress-"
    "Three-Frame-Binary-Tactile-Direct-Action-Student-DR-v0"
)


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressTeacherEnvCfg(
    _GelSightX040ProgressCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg,
):
    """Progress-reward Teacher retaining Pulled-Drawer collision termination."""

    rma_task_id = GELSIGHT_PULLED_DRAWER_PROGRESS_TEACHER_TASK
    # X040 Progress disables force termination. The drawer profile deliberately
    # keeps its 200 -> 20 N strong-collision termination curriculum.
    illegal_collision_terminates_episode = True


class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressTeacherEnv(
    _GelSightX040ProgressRewardMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressTeacherEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressBinaryTactileThreeFrameStudentDREnvCfg(
    _GelSightX040ProgressCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg,
):
    """Progress Student MDP with drawer DR and binary tactile deployment."""

    rma_task_id = (
        GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK
    )
    illegal_collision_terminates_episode = True


class Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressBinaryTactileThreeFrameStudentDREnv(
    _GelSightX040ProgressRewardMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnv,
):
    cfg: (
        Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressBinaryTactileThreeFrameStudentDREnvCfg
    )


__all__ = (
    "GELSIGHT_PULLED_DRAWER_PROGRESS_TEACHER_TASK",
    "GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK",
    "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressTeacherEnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressTeacherEnv",
    "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressBinaryTactileThreeFrameStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressBinaryTactileThreeFrameStudentDREnv",
)
