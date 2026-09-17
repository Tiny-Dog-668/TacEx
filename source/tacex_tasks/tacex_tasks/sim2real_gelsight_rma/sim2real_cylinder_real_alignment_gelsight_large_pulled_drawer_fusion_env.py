"""Large Pulled-Drawer Student task wrappers for the six-way fusion ablation."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_downsample_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnv,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg,
)


GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_VISION_ONLY_DOWNSAMPLE_STUDENT_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Large-Pulled-Drawer-"
    "Progress-Four-Tactile-Vision-Only-Downsample-Three-Frame-Direct-Action-Student-DR-v0"
)
GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_DOWNSAMPLE_STUDENT_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Large-Pulled-Drawer-"
    "Progress-Four-Tactile-Tactile-Cross-Downsample-Three-Frame-Binary-Direct-Action-Student-DR-v0"
)
GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_DOWNSAMPLE_STUDENT_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Large-Pulled-Drawer-"
    "Progress-Four-Tactile-Tactile-Cross-Alpha-Downsample-Three-Frame-Binary-Direct-Action-Student-DR-v0"
)
GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE_STUDENT_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Large-Pulled-Drawer-"
    "Progress-Four-Tactile-Tactile-Cross-Alpha-Aux-Downsample-Three-Frame-Binary-Direct-Action-Student-DR-v0"
)
GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE_STUDENT_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Large-Pulled-Drawer-"
    "Progress-Four-Tactile-Tactile-Cross-Alpha-Aux-GRU-Downsample-Three-Frame-Binary-Direct-Action-Student-DR-v0"
)


class _ScheduledObjectXYResetMixin:
    """One-shot full-vector object XY override used only by paired evaluation."""

    def set_next_reset_object_xy(self, object_xy_root_m) -> None:
        # Input is root-frame XY [N, 2]; the environment reset sampler consumes
        # offsets relative to each replicated object's default root state.
        xy = torch.as_tensor(object_xy_root_m, device=self.device, dtype=torch.float32)
        if xy.shape != (self.num_envs, 2):
            raise ValueError(
                f"Expected evaluation object XY [{self.num_envs},2], got {tuple(xy.shape)}"
            )
        default_xy = self._cube.data.default_root_state[:, :2].to(xy)
        self._next_evaluation_cube_xy_offsets = xy - default_xy

    def _sample_cube_xy_offsets(self, count: int):
        offsets = getattr(self, "_next_evaluation_cube_xy_offsets", None)
        if offsets is None:
            return super()._sample_cube_xy_offsets(count)
        if int(count) != self.num_envs:
            raise RuntimeError(
                "Evaluation XY override is only valid for a full environment reset"
            )
        del self._next_evaluation_cube_xy_offsets
        return offsets.clone()


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerFusionVTDownsampleStudentEnvCfg(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg,
):
    rma_task_id = GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerFusionVTDownsampleStudentEnv(
    _ScheduledObjectXYResetMixin,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnv,
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerFusionVTDownsampleStudentEnvCfg


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileVisionOnlyDownsampleStudentEnvCfg(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg,
):
    rma_task_id = GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_VISION_ONLY_DOWNSAMPLE_STUDENT_TASK


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileVisionOnlyDownsampleStudentEnv(
    _ScheduledObjectXYResetMixin,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnv,
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileVisionOnlyDownsampleStudentEnvCfg


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossDownsampleStudentEnvCfg(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg,
):
    rma_task_id = GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_DOWNSAMPLE_STUDENT_TASK


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossDownsampleStudentEnv(
    _ScheduledObjectXYResetMixin,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnv,
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossDownsampleStudentEnvCfg


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaDownsampleStudentEnvCfg(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg,
):
    rma_task_id = GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_DOWNSAMPLE_STUDENT_TASK


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaDownsampleStudentEnv(
    _ScheduledObjectXYResetMixin,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnv,
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaDownsampleStudentEnvCfg


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxDownsampleStudentEnvCfg(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg,
):
    rma_task_id = GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE_STUDENT_TASK


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxDownsampleStudentEnv(
    _ScheduledObjectXYResetMixin,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnv,
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxDownsampleStudentEnvCfg


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxGRUDownsampleStudentEnvCfg(
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg,
):
    rma_task_id = GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE_STUDENT_TASK


class Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxGRUDownsampleStudentEnv(
    _ScheduledObjectXYResetMixin,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnv,
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxGRUDownsampleStudentEnvCfg


__all__ = tuple(name for name in globals() if name.startswith(("GELSIGHT_", "Sim2Real")))
