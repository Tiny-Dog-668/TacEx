"""Large Pulled-Drawer variants of the high-cylinder four-tactile route."""

from __future__ import annotations

from isaaclab.utils import configclass

from .sim2real_cube_real_alignment_gelsight_pulled_drawer_env import (
    PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M,
    PULLED_DRAWER_TRAY_REFERENCE_CENTER_XY_M,
    pulled_drawer_geometry_contract,
)
from .sim2real_cylinder_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnv,
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnv,
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg,
)


GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Large-Pulled-Drawer-"
    "Progress-Four-Tactile-Teacher-v0"
)
GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Large-Pulled-Drawer-"
    "Progress-Four-Tactile-Three-Frame-Binary-Direct-Action-Student-DR-v0"
)

# Dimensions are depth, width, height in the robot-root-aligned world frame.
LARGE_PULLED_DRAWER_CABINET_NOMINAL_SIZE_M = (0.40, 0.35, 0.18)
LARGE_PULLED_DRAWER_TRAY_NOMINAL_SIZE_M = (0.25, 0.32, 0.15)
# Keep the nominal tray center at the existing task's calibrated workspace
# location. The larger cabinet moves rearward to preserve the contiguous seam.
LARGE_PULLED_DRAWER_TRAY_REFERENCE_CENTER_XY_M = (
    PULLED_DRAWER_TRAY_REFERENCE_CENTER_XY_M
)
LARGE_PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M = (
    PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M[0]
    + 0.5 * LARGE_PULLED_DRAWER_TRAY_NOMINAL_SIZE_M[0]
    + 0.5 * LARGE_PULLED_DRAWER_CABINET_NOMINAL_SIZE_M[0],
    -0.01,
)


def large_pulled_drawer_geometry_contract() -> dict[str, object]:
    """Return the JSON-safe geometry contract for the large drawer."""
    return pulled_drawer_geometry_contract(
        cabinet_nominal_size_m=LARGE_PULLED_DRAWER_CABINET_NOMINAL_SIZE_M,
        cabinet_nominal_center_xy_m=(
            LARGE_PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M
        ),
        tray_nominal_size_m=LARGE_PULLED_DRAWER_TRAY_NOMINAL_SIZE_M,
        tray_reference_center_xy_m=(
            LARGE_PULLED_DRAWER_TRAY_REFERENCE_CENTER_XY_M
        ),
    )


@configclass
class _LargePulledDrawerCfgMixin:
    pulled_drawer_cabinet_nominal_size_m = (
        LARGE_PULLED_DRAWER_CABINET_NOMINAL_SIZE_M
    )
    pulled_drawer_cabinet_nominal_center_xy_m = (
        LARGE_PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M
    )
    pulled_drawer_tray_nominal_size_m = LARGE_PULLED_DRAWER_TRAY_NOMINAL_SIZE_M
    pulled_drawer_tray_reference_center_xy_m = (
        LARGE_PULLED_DRAWER_TRAY_REFERENCE_CENTER_XY_M
    )
    pulled_drawer_geometry_variant = "large_400x350x180_tray_250x320x150"
    cube_x_pos_range = 0.05
    cube_y_pos_range = 0.05
    illegal_collision_penalty = 0.0


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnvCfg(
    _LargePulledDrawerCfgMixin,
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg,
):
    rma_task_id = (
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    )


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnv(
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnv
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnvCfg


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileBinaryStudentEnvCfg(
    _LargePulledDrawerCfgMixin,
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg,
):
    rma_task_id = (
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK
    )


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileBinaryStudentEnv(
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnv
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileBinaryStudentEnvCfg


__all__ = tuple(
    name
    for name in globals()
    if name.startswith(("GELSIGHT_", "LARGE_", "Sim2Real", "large_"))
)
