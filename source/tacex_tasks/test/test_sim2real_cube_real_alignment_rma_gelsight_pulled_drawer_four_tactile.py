"""Offline contracts for the four-tactile Pulled-Drawer route."""

from pathlib import Path

import gymnasium as gym
import torch

import tacex_tasks  # noqa: F401
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_USD,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_four_tactile_models import (
    FOUR_TACTILE_FUSION_DIM,
    RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent,
    RMAGelSightPulledDrawerFourTactileActorCore,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_binary_tactile_models import (
    BinaryReferenceDeltaTactileEncoder,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
    FOUR_TACTILE_CONTACT_ORDER,
    FOUR_TACTILE_CONTACT_WEIGHTS,
    FOUR_TACTILE_DOWN_COLLISION_CURRICULUM_END_STEP,
    FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_END_N,
    FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_START_N,
    FOUR_TACTILE_DOWN_FACE_POINTS_HAND_M,
    FOUR_TACTILE_LOWEST_POINT_MAX_HAND_Z_M,
    GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
    GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg,
    down_tactile_collision_penalty,
    four_tactile_down_collision_threshold,
    lowest_world_point_from_hand_frame,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_size_buckets_progress_env import (
    quadratic_excess_contact_force_penalty,
)


def test_four_tactile_tasks_and_static_contracts() -> None:
    teacher = Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg()
    student = Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg()
    assert gym.spec(GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_TEACHER_TASK)
    assert gym.spec(GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK)
    assert teacher.observation_space["rma_contact_state"].shape == (4,)
    assert student.observation_space["rma_contact_state"].shape == (4,)
    assert len(teacher.rma_cube_contact_sensor.filter_prim_paths_expr) == 4
    assert tuple(teacher.rma_contact_order) == FOUR_TACTILE_CONTACT_ORDER
    assert tuple(teacher.rma_contact_reward_weights) == FOUR_TACTILE_CONTACT_WEIGHTS
    assert teacher.action_space == student.action_space == 4
    assert teacher.illegal_collision_terminates_episode is True
    assert teacher.four_tactile_down_collision_threshold_start_n == 50.0
    assert teacher.four_tactile_down_collision_threshold_end_n == 10.0
    assert teacher.four_tactile_down_collision_curriculum_end_step == 100_000
    assert teacher.gelsight_fingertip_bottom_offset_hand_m[2] == 0.18331
    assert len(teacher.four_tactile_clearance_points_hand_m) == 8
    assert Path(GELSIGHT_FOUR_TACTILE_FRANKA_ARM_VISUAL_USD).is_file()


def test_four_tactile_thresholds_and_force_penalty() -> None:
    forces = torch.tensor([[1.99, 2.0, 2.01, 0.0]])
    state = (forces >= 2.0).float()
    assert state.tolist() == [[0.0, 1.0, 1.0, 0.0]]
    weighted = state @ torch.tensor(FOUR_TACTILE_CONTACT_WEIGHTS)
    assert weighted.item() == 2.5
    penalty, excessive = quadratic_excess_contact_force_penalty(
        torch.tensor([[15.0, 15.0, 15.0, 30.0]]), threshold_n=15.0, weight=5.0
    )
    assert penalty.item() == -5.0 and excessive.item() is True
    penalties, collisions = down_tactile_collision_penalty(
        torch.tensor(
            [
                [100.0, 100.0, 49.99, 0.0],
                [0.0, 0.0, 50.0, 1.0],
                [0.0, 0.0, 50.01, 2.0],
            ]
        ),
        threshold_n=FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_START_N,
    )
    assert penalties.tolist() == [0.0, 0.0, -10.0]
    assert collisions.tolist() == [False, False, True]
    assert FOUR_TACTILE_DOWN_COLLISION_THRESHOLD_END_N == 10.0
    assert FOUR_TACTILE_DOWN_COLLISION_CURRICULUM_END_STEP == 100_000
    assert four_tactile_down_collision_threshold(-1) == 50.0
    assert four_tactile_down_collision_threshold(50_000) == 30.0
    assert four_tactile_down_collision_threshold(100_001) == 10.0


def test_four_tactile_clearance_uses_downward_face() -> None:
    points = torch.tensor(FOUR_TACTILE_DOWN_FACE_POINTS_HAND_M)
    # 180 degrees around hand X maps hand +Z toward world -Z.
    lowest = lowest_world_point_from_hand_frame(
        torch.zeros(1, 3), torch.tensor([[0.0, 1.0, 0.0, 0.0]]), points
    )
    assert torch.isclose(
        lowest[0, 2], torch.tensor(-FOUR_TACTILE_LOWEST_POINT_MAX_HAND_Z_M)
    )


def test_binary_boundary_and_four_tactile_models() -> None:
    reference = torch.zeros(1, 1, 3, 3, dtype=torch.uint8)
    current = torch.tensor([[[[4, 5, 6]]]], dtype=torch.uint8)
    assert BinaryReferenceDeltaTactileEncoder.binary_delta(current, reference).flatten().tolist() == [0.0, 0.0, 1.0]
    actor = RMAGelSightPulledDrawerFourTactileActorCore()
    assert actor.network[0].in_features == 32
    student = RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent(pretrained_backbone=False)
    assert student.action_head[0].in_features == FOUR_TACTILE_FUSION_DIM == 1555
    assert student.inner_tactile_encoder is not student.down_tactile_encoder
