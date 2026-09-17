"""Offline contracts for the high-cylinder four-tactile Pulled-Drawer route."""

import math
from copy import deepcopy

import gymnasium as gym
import pytest
import torch
import torch.nn.functional as F
import isaaclab.sim as sim_utils

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_artifacts import (
    CYLINDER_FOUR_TACTILE_STUDENT_CHECKPOINT_VERSION,
    CYLINDER_FOUR_TACTILE_STUDENT_KIND,
    CYLINDER_FOUR_TACTILE_TEACHER_KIND,
    CYLINDER_FOUR_TACTILE_TEACHER_MANIFEST_VERSION,
    FOUR_TACTILE_STUDENT_KIND,
    FOUR_TACTILE_TEACHER_KIND,
    LARGE_DRAWER_CYLINDER_FOUR_TACTILE_STUDENT_KIND,
    LARGE_DRAWER_CYLINDER_FOUR_TACTILE_TEACHER_KIND,
    LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_CHECKPOINT_VERSION,
    LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_KIND,
    LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_TEACHER_KIND,
    LARGE_DRAWER_FUSION_STUDENT_TASKS,
    LARGE_DRAWER_FUSION_REGISTRY,
    LARGE_DRAWER_FUSION_VARIANT_BY_TASK,
    GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK,
    _validate_environment_contract_for_task,
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
    sample_pulled_drawer_layouts,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK,
    LARGE_PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M,
    LARGE_PULLED_DRAWER_CABINET_NOMINAL_SIZE_M,
    LARGE_PULLED_DRAWER_TRAY_NOMINAL_SIZE_M,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileBinaryStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnvCfg,
    large_pulled_drawer_geometry_contract,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_downsample_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleTeacherEnvCfg,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_fusion_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_DOWNSAMPLE_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_DOWNSAMPLE_STUDENT_TASK,
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_VISION_ONLY_DOWNSAMPLE_STUDENT_TASK,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxDownsampleStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxGRUDownsampleStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaDownsampleStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossDownsampleStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileVisionOnlyDownsampleStudentEnvCfg,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_large_drawer_fusion_models import (
    LARGE_DRAWER_FUSION_VARIANTS,
    RMAGelSightLargeDrawerTactileCrossAlphaAuxGRUStudent,
    RMAGelSightLargeDrawerTactileCrossAlphaAuxStudent,
    RMAGelSightLargeDrawerTactileCrossAlphaStudent,
    RMAGelSightLargeDrawerTactileCrossStudent,
    VISION_ONLY_DOWNSAMPLE,
    large_drawer_fusion_model_contract,
    make_large_drawer_fusion_student,
)
from tacex_tasks.sim2real_grasp.sim2real_cube_real_alignment_env import (
    Sim2RealCubeRealAlignmentEnv,
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
    assert teacher.cube_x_pos_range == student.cube_x_pos_range == pytest.approx(0.03)
    assert teacher.cube_y_pos_range == student.cube_y_pos_range == pytest.approx(0.03)
    assert teacher.observation_space["rma_contact_state"].shape == (4,)
    assert student.observation_space["rma_contact_state"].shape == (4,)
    assert RMAGelSightPulledDrawerFourTactileActorCore().network[0].in_features == 32
    model = RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent(
        pretrained_backbone=False
    )
    assert model.action_head[0].in_features == FOUR_TACTILE_FUSION_DIM == 1555


def test_large_drawer_cylinder_tasks_use_large_geometry_and_reset_range() -> None:
    teacher = Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnvCfg()
    student = Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileBinaryStudentEnvCfg()
    assert gym.spec(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    )
    assert gym.spec(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK
    )
    assert tuple(teacher.pulled_drawer_cabinet_nominal_size_m) == pytest.approx(
        LARGE_PULLED_DRAWER_CABINET_NOMINAL_SIZE_M
    )
    assert tuple(teacher.pulled_drawer_tray_nominal_size_m) == pytest.approx(
        LARGE_PULLED_DRAWER_TRAY_NOMINAL_SIZE_M
    )
    assert teacher.action_space == student.action_space == 4
    assert teacher.cube_x_pos_range == student.cube_x_pos_range == pytest.approx(0.05)
    assert teacher.cube_y_pos_range == student.cube_y_pos_range == pytest.approx(0.05)
    base_teacher = Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg()
    assert teacher.observation_space.keys() == base_teacher.observation_space.keys()
    assert teacher.observation_space["rma_contact_state"].shape == (4,)
    geometry = large_pulled_drawer_geometry_contract()
    assert geometry["cabinet"]["nominal_outer_size_m"] == [0.40, 0.35, 0.18]
    assert geometry["tray"]["nominal_outer_size_m"] == [0.25, 0.32, 0.15]
    assert geometry["tray"]["nominal_center_xy_m"] == pytest.approx(
        [0.5275, 0.0]
    )
    assert LARGE_PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M == pytest.approx(
        (0.8525, -0.01)
    )
    sampled = sample_pulled_drawer_layouts(
        16,
        seed=42,
        cabinet_nominal_size_m=LARGE_PULLED_DRAWER_CABINET_NOMINAL_SIZE_M,
        cabinet_nominal_center_xy_m=LARGE_PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M,
        tray_nominal_size_m=LARGE_PULLED_DRAWER_TRAY_NOMINAL_SIZE_M,
        tray_reference_center_xy_m=(0.49, 0.0),
    )
    assert torch.all(sampled["side_clearance_m"] >= 0.001)
    assert torch.allclose(sampled["gap_m"], torch.zeros(16, dtype=torch.float64))
    teacher_contract = teacher_environment_contract(teacher)
    student_contract = student_environment_contract(student)
    assert teacher_contract["profile"] == (
        "rma_gelsight_large_pulled_drawer_progress_four_tactile_cylinder_v2"
    )
    assert teacher_contract["geometry"] == geometry
    assert teacher_contract["cube_reset_half_range_xy_m"] == [0.05, 0.05]
    assert student_contract["geometry"] == geometry
    assert GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK[
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK
    ] == GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    assert _teacher_artifact_identity(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    )[0] == LARGE_DRAWER_CYLINDER_FOUR_TACTILE_TEACHER_KIND
    assert _student_artifact_identity(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK
    )[0] == LARGE_DRAWER_CYLINDER_FOUR_TACTILE_STUDENT_KIND


def test_large_drawer_downsample_students_share_standard_teacher_and_disable_blur() -> None:
    teacher = Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleTeacherEnvCfg()
    student = Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg()
    assert gym.spec(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK
    )
    assert gym.spec(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK
    )
    assert teacher.action_space == student.action_space == 4
    assert student.observation_space["wrist_rgb_history"].shape == (3, 224, 224, 3)
    assert student.wrist_downsample_degradation_enabled is True
    assert student.wrist_downsample_size == 32
    assert student.wrist_blur_randomization_enabled is False
    assert GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK[
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK
    ] == GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    assert _teacher_artifact_identity(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK
    )[0] == LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_TEACHER_KIND
    assert _student_artifact_identity(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK
    )[:2] == (
        LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_KIND,
        LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_CHECKPOINT_VERSION,
    )
    assert LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_CHECKPOINT_VERSION == 2
    assert LARGE_DRAWER_CYLINDER_FOUR_TACTILE_DOWNSAMPLE_STUDENT_KIND != (
        LARGE_DRAWER_CYLINDER_FOUR_TACTILE_STUDENT_KIND
    )

    teacher_contract = teacher_environment_contract(teacher)
    _validate_environment_contract_for_task(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_TEACHER_TASK,
        teacher_contract,
        student=False,
    )
    contract = student_environment_contract(student)
    assert contract["profile"] == (
        "rma_gelsight_large_pulled_drawer_progress_four_tactile_cylinder_downsample_v2"
    )
    assert contract["wrist_rgb_degradation"] == {
        "mode": "bilinear_downsample_then_bilinear_upsample",
        "input_output_size_hw": [224, 224],
        "downsample_size_hw": [32, 32],
        "align_corners": False,
        "applied_after": "wrist_color_and_intrinsic_dr_before_sensor_noise",
        "random_gaussian_blur_enabled": False,
    }
    _validate_environment_contract_for_task(
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
        contract,
        student=True,
    )
    incompatible = deepcopy(contract)
    incompatible["wrist_rgb_degradation"]["downsample_size_hw"] = [64, 64]
    with pytest.raises(RuntimeError, match="Downsample Student visual contract"):
        _validate_environment_contract_for_task(
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
            incompatible,
            student=True,
        )


def test_six_fusion_tasks_are_registered_and_physically_paired() -> None:
    task_cfg_pairs = (
        (
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_VISION_ONLY_DOWNSAMPLE_STUDENT_TASK,
            Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileVisionOnlyDownsampleStudentEnvCfg,
        ),
        (
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
            Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileDownsampleBinaryStudentEnvCfg,
        ),
        (
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_DOWNSAMPLE_STUDENT_TASK,
            Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossDownsampleStudentEnvCfg,
        ),
        (
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_DOWNSAMPLE_STUDENT_TASK,
            Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaDownsampleStudentEnvCfg,
        ),
        (
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE_STUDENT_TASK,
            Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxDownsampleStudentEnvCfg,
        ),
        (
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE_STUDENT_TASK,
            Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxGRUDownsampleStudentEnvCfg,
        ),
    )
    assert len(LARGE_DRAWER_FUSION_STUDENT_TASKS) == 6
    assert set(LARGE_DRAWER_FUSION_REGISTRY) == LARGE_DRAWER_FUSION_STUDENT_TASKS
    assert set(LARGE_DRAWER_FUSION_VARIANT_BY_TASK.values()) == set(LARGE_DRAWER_FUSION_VARIANTS)
    identities = [_student_artifact_identity(task)[:2] for task, _ in task_cfg_pairs]
    assert len(set(identities)) == 6
    standard_teacher = Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileTeacherEnvCfg()
    expected_teacher_contract = teacher_environment_contract(standard_teacher)
    for task, cfg_class in task_cfg_pairs:
        assert gym.spec(task)
        cfg = cfg_class()
        assert cfg.action_space == 4
        assert cfg.wrist_downsample_degradation_enabled is True
        assert cfg.wrist_downsample_size == 32
        assert cfg.wrist_blur_randomization_enabled is False
        assert GELSIGHT_PULLED_DRAWER_STUDENT_TO_TEACHER_TASK[task] == (
            GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
        )
        assert teacher_environment_contract(cfg) == expected_teacher_contract
        _validate_environment_contract_for_task(
            task, student_environment_contract(cfg), student=True
        )


def test_six_fusion_model_contracts_have_distinct_runtime_interfaces() -> None:
    for variant in LARGE_DRAWER_FUSION_VARIANTS:
        model = make_large_drawer_fusion_student(variant, pretrained_backbone=False)
        assert model.contract() == large_drawer_fusion_model_contract(variant)
        assert model.action_head[-1].out_features == 4
    vision_contract = large_drawer_fusion_model_contract(VISION_ONLY_DOWNSAMPLE)
    assert all(not name.startswith("gsmini_") for name in vision_contract["runtime_input_order"])
    assert "contact_probability" not in vision_contract["runtime_output"]


def test_cross_attention_has_three_keys_and_backpropagates() -> None:
    model = RMAGelSightLargeDrawerTactileCrossStudent(pretrained_backbone=False)
    tactile = torch.randn(2, 4, 256, requires_grad=True)
    visual = torch.randn(2, 3, 256, requires_grad=True)
    fused, attention = model._cross_features(tactile, visual)
    assert fused.shape == (2, 256)
    assert attention.shape == (2, 4, 4, 3)
    assert attention.std(dim=-1).mean().item() > 0.0
    fused.square().mean().backward()
    assert tactile.grad is not None and tactile.grad.abs().sum().item() > 0.0
    assert visual.grad is not None and visual.grad.abs().sum().item() > 0.0


def test_alpha_and_aux_gate_inputs_are_deployable() -> None:
    alpha_model = RMAGelSightLargeDrawerTactileCrossAlphaStudent(
        pretrained_backbone=False
    )
    tactile = torch.randn(2, 4, 256)
    visual_tokens = torch.randn(2, 3, 256)
    proprio = torch.randn(2, 15)
    fused, alpha, attention = alpha_model._mix_features(
        tactile, visual_tokens, proprio
    )
    assert fused.shape == (2, 256)
    assert alpha.shape == (2, 1)
    assert torch.all((alpha >= 0.0) & (alpha <= 1.0))
    assert attention.shape[-1] == 3

    aux_model = RMAGelSightLargeDrawerTactileCrossAlphaAuxStudent(
        pretrained_backbone=False
    )
    zeros = torch.zeros((2, 4, 5, 3), dtype=torch.uint8)
    changed = zeros.clone()
    changed[0, :2] = 6
    ratios = aux_model._contact_area_ratios(
        changed, zeros, zeros, zeros, zeros, zeros, zeros, zeros
    )
    assert ratios.shape == (2, 4)
    assert ratios[0].tolist() == pytest.approx([0.5, 0.0, 0.0, 0.0])
    assert ratios[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_gru_reset_fill_and_environment_state_are_isolated() -> None:
    model = RMAGelSightLargeDrawerTactileCrossAlphaAuxGRUStudent(
        pretrained_backbone=False
    )
    current = torch.randn(2, 4, 256)
    history_a = torch.randn(2, 4, 9, 256)
    history_b = history_a.clone()
    history_b[0].add_(10.0)
    reset = torch.tensor([True, False])
    tokens_a, next_a = model._temporal_tactile_tokens(current, history_a, reset)
    tokens_b, next_b = model._temporal_tactile_tokens(current, history_b, reset)
    assert tokens_a.shape == (2, 4, 256)
    assert next_a.shape == (2, 4, 9, 256)
    assert torch.equal(tokens_a, tokens_b)
    assert torch.equal(next_a, next_b)
    assert torch.equal(next_a[0, :, -1], current[0])
    changed_current = current.clone()
    changed_current[0].add_(5.0)
    isolated_tokens, isolated_history = model._temporal_tactile_tokens(
        changed_current, history_a, torch.zeros(2, dtype=torch.bool)
    )
    baseline_tokens, baseline_history = model._temporal_tactile_tokens(
        current, history_a, torch.zeros(2, dtype=torch.bool)
    )
    assert torch.equal(isolated_tokens[1], baseline_tokens[1])
    assert torch.equal(isolated_history[1], baseline_history[1])


def test_wrist_downsample_matches_occluded_bilinear_reference() -> None:
    image = torch.linspace(0.0, 1.0, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
    actual = Sim2RealCubeRealAlignmentEnv._apply_wrist_downsample_degradation(
        image, 32
    )
    expected = F.interpolate(
        F.interpolate(image, size=(32, 32), mode="bilinear", align_corners=False),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    assert actual.shape == image.shape
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="must be positive"):
        Sim2RealCubeRealAlignmentEnv._apply_wrist_downsample_degradation(image, 0)


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
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_VISION_ONLY_DOWNSAMPLE_STUDENT_TASK,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxDownsampleStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaAuxGRUDownsampleStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossAlphaDownsampleStudentEnvCfg,
    Sim2RealCylinderRealAlignmentRMAGelSightLargePulledDrawerProgressFourTactileCrossDownsampleStudentEnvCfg,
