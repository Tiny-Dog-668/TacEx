"""GelSight-equipped RMA variants of the Real-Alignment cube task."""

from __future__ import annotations

import gymnasium as gym

from . import agents
from .sim2real_cube_real_alignment_gelsight_rma_env import (
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightStudentEnv,
    Sim2RealCubeRealAlignmentRMAGelSightStudentEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapDREnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapEnv,
    Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg,
)
from .sim2real_cube_real_alignment_gelsight_size_buckets_env import (
    GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK,
    GELSIGHT_SIZE_BUCKETS_TEACHER_TASK,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnvCfg,
)
from .sim2real_cube_real_alignment_gelsight_x040_three_frame_env import (
    GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK,
    GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg,
)
from .sim2real_cube_real_alignment_gelsight_pulled_drawer_env import (
    GELSIGHT_PULLED_DRAWER_TEACHER_TASK,
    GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnv,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg,
)


RMA_GELSIGHT_TEACHER_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Teacher-v0"
RMA_GELSIGHT_STUDENT_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-v0"
RMA_GELSIGHT_STUDENT_HEATMAP_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-v0"
)
RMA_GELSIGHT_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-DR-v0"
)
RMA_GELSIGHT_STUDENT_HEATMAP_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-DR-v0"
)


gym.register(
    id=RMA_GELSIGHT_TEACHER_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_rma_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg,
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_cube_real_alignment_gelsight_rma_teacher.yaml"
        ),
    },
)

gym.register(
    id=GELSIGHT_SIZE_BUCKETS_TEACHER_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_size_buckets_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnvCfg,
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_cube_real_alignment_gelsight_size_buckets_teacher.yaml"
        ),
    },
)

gym.register(
    id=GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_size_buckets_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnvCfg,
    },
)

gym.register(
    id=GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_x040_three_frame_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_cube_real_alignment_gelsight_x040_dr_size_buckets_teacher.yaml"
        ),
    },
)

gym.register(
    id=GELSIGHT_PULLED_DRAWER_TEACHER_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_pulled_drawer_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg,
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:"
            "skrl_ppo_cube_real_alignment_gelsight_pulled_drawer_teacher.yaml"
        ),
    },
)

gym.register(
    id=GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_pulled_drawer_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg
        ),
    },
)

gym.register(
    id=GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_x040_three_frame_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsThreeFrameStudentDREnvCfg
        ),
    },
)

gym.register(
    id=RMA_GELSIGHT_STUDENT_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_rma_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightStudentEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAGelSightStudentEnvCfg,
    },
)

gym.register(
    id=RMA_GELSIGHT_STUDENT_HEATMAP_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_rma_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapEnvCfg,
    },
)

gym.register(
    id=RMA_GELSIGHT_STUDENT_DR_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_rma_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightStudentDREnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg,
    },
)

gym.register(
    id=RMA_GELSIGHT_STUDENT_HEATMAP_DR_TASK,
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_gelsight_rma_env:"
        "Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapDREnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAGelSightStudentHeatmapDREnvCfg,
    },
)
