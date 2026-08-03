"""Minimal sim-to-real Franka grasping task."""

import gymnasium as gym

from . import agents
from ..cylinder_grasping import agents as cylinder_agents
from .sim2real_cube_grasp_env import Sim2RealCubeGraspEnv, Sim2RealCubeGraspEnvCfg
from .sim2real_cube_real_alignment_env import (
    Sim2RealCubeRealAlignmentDREnv,
    Sim2RealCubeRealAlignmentDREnvCfg,
    Sim2RealCubeRealAlignmentEnv,
    Sim2RealCubeRealAlignmentEnvCfg,
)
from .sim2real_cube_real_alignment_privileged_env import (
    Sim2RealCubeRealAlignmentPrivilegedEnv,
    Sim2RealCubeRealAlignmentPrivilegedEnvCfg,
)
from .sim2real_cube_real_alignment_rma_env import (
    Sim2RealCubeRealAlignmentRMAStudentDREnv,
    Sim2RealCubeRealAlignmentRMAStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMAStudentEnv,
    Sim2RealCubeRealAlignmentRMAStudentEnvCfg,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnv,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnvCfg,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapEnv,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapEnvCfg,
    Sim2RealCubeRealAlignmentRMATeacherEnv,
    Sim2RealCubeRealAlignmentRMATeacherEnvCfg,
)
from .sim2real_grasp_env import (
    Sim2RealGraspEnv,
    Sim2RealGraspEnvCfg,
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Sim2Real-Grasp-v0 --num_envs 4 --enable_cameras
# isaaclab -p ./scripts/reinforcement_learning/skrl/play.py --task TacEx-Sim2Real-Grasp-v0 --num_envs 1 --enable_cameras
gym.register(
    id="TacEx-Sim2Real-Grasp-v0",
    entry_point=f"{__name__}.sim2real_grasp_env:Sim2RealGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealGraspEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_only_cfg_resnet18.yaml",
        "skrl_sac_cfg_entry_point": f"{cylinder_agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# RMA-style privileged teacher using the current Clean v9 contract.
gym.register(
    id="TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0",
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_rma_env:"
        "Sim2RealCubeRealAlignmentRMATeacherEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMATeacherEnvCfg,
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_cube_real_alignment_rma_teacher.yaml"
        ),
    },
)

# Camera student trained by online position and action distillation.
gym.register(
    id="TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-v0",
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_rma_env:"
        "Sim2RealCubeRealAlignmentRMAStudentEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAStudentEnvCfg,
    },
)

# Camera student with explicit projected cube-center heatmap supervision.
gym.register(
    id="TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-v0",
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_rma_env:"
        "Sim2RealCubeRealAlignmentRMAStudentHeatmapEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAStudentHeatmapEnvCfg,
    },
)

# RMA Student with full-strength camera/image/scene domain randomization from
# the first update. The Teacher stays on the matching privileged Clean physics
# contract; only the visual observation distribution differs.
gym.register(
    id="TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0",
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_rma_env:"
        "Sim2RealCubeRealAlignmentRMAStudentDREnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAStudentDREnvCfg,
    },
)

# Heatmap-supervised RMA Student with full-strength visual DR from the first update.
gym.register(
    id="TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0",
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_rma_env:"
        "Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnvCfg,
    },
)

# Real-reference-aligned Franka + D435 cube task.
# python scripts/reinforcement_learning/skrl/train.py \
#   --task TacEx-Sim2Real-Cube-Real-Alignment-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Sim2Real-Cube-Real-Alignment-v0",
    entry_point=f"{__name__}.sim2real_cube_real_alignment_env:Sim2RealCubeRealAlignmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cube_real_alignment_cfg_resnet18.yaml",
        "skrl_sac_cfg_entry_point": f"{cylinder_agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# Real-reference-aligned Cube with broad visual/scene domain randomization.
gym.register(
    id="TacEx-Sim2Real-Cube-Real-Alignment-DR-v0",
    entry_point=f"{__name__}.sim2real_cube_real_alignment_env:Sim2RealCubeRealAlignmentDREnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentDREnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cube_real_alignment_dr_cfg_resnet18.yaml",
        "skrl_sac_cfg_entry_point": f"{cylinder_agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# State-only diagnostic upper bound. It shares the scene/reward implementation
# with Clean but retains its existing 2 mm gripper and +/-10 cm XY reset
# settings; it creates no RGB camera and exposes simulator positions to Actor.
gym.register(
    id="TacEx-Sim2Real-Cube-Real-Alignment-Privileged-v0",
    entry_point=(
        f"{__name__}.sim2real_cube_real_alignment_privileged_env:"
        "Sim2RealCubeRealAlignmentPrivilegedEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeRealAlignmentPrivilegedEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cube_real_alignment_privileged_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{cylinder_agents.__name__}:skrl_sac_cfg.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Sim2Real-Cube-Grasp-v0 --num_envs 4 --enable_cameras
# python scripts/reinforcement_learning/skrl/play.py --task TacEx-Sim2Real-Cube-Grasp-v0 --num_envs 1 --enable_cameras
gym.register(
    id="TacEx-Sim2Real-Cube-Grasp-v0",
    entry_point=f"{__name__}.sim2real_cube_grasp_env:Sim2RealCubeGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Sim2RealCubeGraspEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cube_vision_only_cfg_resnet18.yaml",
        "skrl_sac_cfg_entry_point": f"{cylinder_agents.__name__}:skrl_sac_cfg.yaml",
    },
)
