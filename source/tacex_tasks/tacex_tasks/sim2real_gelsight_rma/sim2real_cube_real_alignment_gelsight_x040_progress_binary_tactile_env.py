"""X040 Progress Student with single-channel binary GelSight differences."""

from __future__ import annotations

from isaaclab.utils import configclass

from .sim2real_cube_real_alignment_gelsight_x040_progress_env import (
    Sim2RealCubeRealAlignmentRMAGelSightX040ProgressThreeFrameStudentDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightX040ProgressThreeFrameStudentDREnvCfg,
)


GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-X040-Progress-"
    "Three-Frame-Binary-Tactile-Direct-Action-Student-DR-v0"
)


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightX040ProgressBinaryTactileThreeFrameStudentDREnvCfg(
    Sim2RealCubeRealAlignmentRMAGelSightX040ProgressThreeFrameStudentDREnvCfg
):
    """Isolate the binary-tactile Student while preserving the Progress MDP."""

    rma_task_id = GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK


class Sim2RealCubeRealAlignmentRMAGelSightX040ProgressBinaryTactileThreeFrameStudentDREnv(
    Sim2RealCubeRealAlignmentRMAGelSightX040ProgressThreeFrameStudentDREnv
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightX040ProgressBinaryTactileThreeFrameStudentDREnvCfg


__all__ = (
    "GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK",
    "Sim2RealCubeRealAlignmentRMAGelSightX040ProgressBinaryTactileThreeFrameStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightX040ProgressBinaryTactileThreeFrameStudentDREnv",
)
