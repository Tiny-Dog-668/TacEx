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
