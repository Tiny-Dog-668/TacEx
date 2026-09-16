"""Offline contracts for the high-cylinder four-tactile Pulled-Drawer route."""

import math

import gymnasium as gym
import pytest
import torch
import isaaclab.sim as sim_utils

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_artifacts import (
    CYLINDER_FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION,
    CYLINDER_FOUR_TACTILE_STUDENT_KIND,
    CYLINDER_FOUR_TACTILE_TEACHER_KIND,
    CYLINDER_FOUR_TACTILE_TEACHER_MANIFEST_VERSION,
    FOUR_TACTILE_STUDENT_KIND,
    FOUR_TACTILE_TEACHER_KIND,
    GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK,
    _student_artifact_identity,
    _teacher_artifact_identity,
    student_environment_contract,
    teacher_environment_contract,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_four_tactile_models import (
    FOUR_TACTILE_FUSION_DIM,
    RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent,
    RMAGelSightPulledDrawerFourTactileActorCore,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
    CYLINDER_ABSOLUTE_CONTACT_REWARD_SCALE,
    CYLINDER_DENSITY_KG_M3,
    CYLINDER_HEIGHT_BUCKETS_M,
    CYLINDER_NOMINAL_HEIGHT_M,
    CYLINDER_NOMINAL_RADIUS_M,
    CYLINDER_SCALE_BUCKETS,
    GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
    GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg,
    absolute_weighted_contact_reward,
    cylinder_dimensions_and_mass_from_scale,
    cylinder_lowest_world_height,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_pulled_drawer_env import (
    PULLED_DRAWER_PANEL_THICKNESS_M,
    balanced_cube_bucket_ids,
)


def test_cylinder_tasks_dimensions_and_model_contracts() -> None:
    teacher = Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg()
    student = Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg()
    assert gym.spec(GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK)
    assert gym.spec(GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK)
    assert isinstance(teacher.cube.spawn, sim_utils.CylinderCfg)
    assert teacher.cube.spawn.height == CYLINDER_NOMINAL_HEIGHT_M
    assert teacher.cube.spawn.radius == CYLINDER_NOMINAL_RADIUS_M
    assert teacher.cube.spawn.mass_props.density == CYLINDER_DENSITY_KG_M3
    assert teacher.cube.spawn.activate_contact_sensors is True
    assert student.cube.spawn.activate_contact_sensors is True
    assert tuple(teacher.cylinder_scale_buckets) == CYLINDER_SCALE_BUCKETS
    assert tuple(teacher.cube_size_buckets_m) == pytest.approx(CYLINDER_HEIGHT_BUCKETS_M)
    assert teacher.action_space == student.action_space == 4
    assert teacher.observation_space["rma_contact_state"].shape == (4,)
    assert student.observation_space["rma_contact_state"].shape == (4,)
    assert RMAGelSightPulledDrawerFourTactileActorCore().network[0].in_features == 32
    model = RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent(
        pretrained_backbone=False
    )
    assert model.action_head[0].in_features == FOUR_TACTILE_FUSION_DIM == 1555


def test_cylinder_scale_assignment_spawn_height_and_mass() -> None:
    ids = balanced_cube_bucket_ids(16, seed=42)
    assert sorted(ids[:8].tolist()) == list(range(8))
    assert sorted(ids[8:].tolist()) == list(range(8))
    scale = torch.tensor(CYLINDER_SCALE_BUCKETS)
    height, radius, mass = cylinder_dimensions_and_mass_from_scale(scale)
    assert height.tolist() == pytest.approx([0.090, 0.093, 0.096, 0.099, 0.101, 0.104, 0.107, 0.110])
    assert (2.0 * radius).tolist() == pytest.approx([0.045, 0.0465, 0.048, 0.0495, 0.0505, 0.052, 0.0535, 0.055])
    nominal_mass = CYLINDER_DENSITY_KG_M3 * math.pi * 0.025**2 * 0.100
    assert nominal_mass == pytest.approx(0.1963495408)
    assert mass[0].item() == pytest.approx(nominal_mass * 0.90**3)
    assert mass[-1].item() == pytest.approx(nominal_mass * 1.10**3)
    spawn_z = PULLED_DRAWER_PANEL_THICKNESS_M + 0.5 * height
    assert spawn_z.tolist() == pytest.approx([0.053, 0.0545, 0.056, 0.0575, 0.0585, 0.060, 0.0615, 0.063])


def test_cylinder_lowest_point_for_upright_horizontal_and_tilted() -> None:
    center = torch.tensor([[0.0, 0.0, 0.2]]).repeat(3, 1)
    half = math.sqrt(0.5)
    quat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half, 0.0, half, 0.0],
            [math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0],
        ]
    )
    height = torch.full((3,), 0.100)
    radius = torch.full((3,), 0.025)
    lowest = cylinder_lowest_world_height(center, quat, height, radius)
    assert lowest[0].item() == pytest.approx(0.150)
    assert lowest[1].item() == pytest.approx(0.175)
    expected_tilt_extent = half * (0.050 + 0.025)
    assert lowest[2].item() == pytest.approx(0.2 - expected_tilt_extent)


def test_cylinder_success_and_artifact_isolation() -> None:
    teacher = Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg()
    student = Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg()
    assert teacher.success_lift_delta == 0.035
    assert teacher.success_hold_steps == 5
    assert teacher.success_requires_upright is False
    assert teacher.lift_tilt_curriculum_enabled is False
    assert teacher.contact_reward_mode == "absolute_weighted_contact_state_per_step"
    assert teacher.cylinder_absolute_contact_reward_scale == 0.2
    assert student.contact_reward_mode == teacher.contact_reward_mode
    assert GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK[
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK
    ] == GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    assert _teacher_artifact_identity(
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    ) == (
        CYLINDER_FOUR_TACTILE_TEACHER_KIND,
        CYLINDER_FOUR_TACTILE_TEACHER_MANIFEST_VERSION,
    )
    assert _student_artifact_identity(
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK
    )[:2] == (
        CYLINDER_FOUR_TACTILE_STUDENT_KIND,
        CYLINDER_FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION,
    )
    assert CYLINDER_FOUR_TACTILE_TEACHER_KIND != FOUR_TACTILE_TEACHER_KIND
    assert CYLINDER_FOUR_TACTILE_STUDENT_KIND != FOUR_TACTILE_STUDENT_KIND
    teacher_contract = teacher_environment_contract(teacher)
    student_contract = student_environment_contract(student)
    assert teacher_contract["profile"] == "rma_gelsight_pulled_drawer_progress_four_tactile_cylinder_v2"
    assert teacher_contract["progress_reward"]["contact"] == teacher.contact_reward_mode
    assert teacher_contract["progress_reward"]["contact_holding_state_repeats_reward"] is True
    assert teacher_contract["progress_reward"]["contact_absolute_scale"] == 0.2
    assert teacher_contract["target_object"]["shape"] == "cylinder"
    assert teacher_contract["target_object"]["success_requires_upright"] is False
    assert "cube_size_buckets_m" not in teacher_contract
    assert student_contract["task"] == GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK


def test_cylinder_absolute_contact_reward_is_bounded_and_repeats_state_value() -> None:
    contact = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    reward = absolute_weighted_contact_reward(
        contact,
        (1.5, 1.5, 1.0, 1.0),
        scale=CYLINDER_ABSOLUTE_CONTACT_REWARD_SCALE,
    )
    assert reward.tolist() == pytest.approx([0.0, 0.3, 0.6, 1.0])
