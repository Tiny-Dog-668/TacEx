"""Large Pulled-Drawer four-tactile RMA pair with fixed camera downsampling."""

from __future__ import annotations

from isaaclab.utils import configclass

from .sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_env import (
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileBinaryStudentEnv,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileBinaryStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnv,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnvCfg,
)


GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Large-Pulled-Drawer-"
    "Progress-Four-Tactile-Downsample-Teacher-v0"
)
GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Large-Pulled-Drawer-"
    "Progress-Four-Tactile-Downsample-Three-Frame-Binary-Direct-Action-Student-DR-v0"
)


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleTeacherEnvCfg(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnvCfg,
):
    """Paired Teacher; its privileged inputs and task semantics are unchanged."""

    rma_task_id = (
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK
    )


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleTeacherEnv(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnv
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleTeacherEnvCfg


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileBinaryStudentEnvCfg,
):
    """Student with fixed 224 -> 32 -> 224 RGB degradation and no blur DR."""

    rma_task_id = (
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK
    )
    wrist_downsample_degradation_enabled = True
    wrist_downsample_size = 32
    wrist_blur_randomization_enabled = False


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnv(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileBinaryStudentEnv
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg


__all__ = tuple(name for name in globals() if name.startswith(("GELSIGHT_", "Sim2Real")))
