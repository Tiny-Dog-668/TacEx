"""Contracts for the GelSight X040-DR three-frame Teacher/Student route."""

from __future__ import annotations

import sys

if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import pytest
import torch
import torch.nn.functional as F
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_three_frame_artifacts import (
    GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK,
    GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
    STUDENT_CHECKPOINT_VERSION,
    STUDENT_KIND,
    load_student_checkpoint,
    student_environment_contract,
    student_input_contract,
    teacher_environment_contract,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_three_frame_models import (
    GELSIGHT_X040_THREE_FRAME_FUSION_DIM,
    RMAGelSightX040ThreeFrameStudent,
)
from tacex_tasks.sim2real_gelsight_rma.gelsight_geometry import geometry_contract
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_x040_three_frame_env import (
    _GelSightX040SafetyMixin,
    linear_collision_threshold,
)


@pytest.fixture(scope="module", autouse=True)
def close_app():
    yield
    simulation_app.close()


def test_task_registration_and_shared_x040_contract():
    assert gym.spec(GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK) is not None
    assert gym.spec(GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK) is not None
    teacher = parse_env_cfg(
        GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK, device="cuda:0", num_envs=8
    )
    student = parse_env_cfg(
        GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
        device="cuda:0",
        num_envs=8,
    )
    assert teacher_environment_contract(teacher) == teacher_environment_contract(student)
    teacher_contract = teacher_environment_contract(teacher)
    assert teacher_contract["gelsight_geometry"] == geometry_contract()
    assert teacher_contract["profile"] == "rma_gelsight_x040_dr_static_size_buckets_v3"
    assert teacher_contract["cube_size_sampling"] == "fixed_round_robin_by_environment"
    assert teacher_contract["cube_size_assignment"] == "env_id_mod_8"
    assert teacher_contract["cube_size_object_count_per_environment"] == 1
    assert teacher_contract["physics_layout"] == {
        "gpu_dynamics": True,
        "replicate_physics": False,
        "env_spacing_m": 3.5,
    }
    assert student_environment_contract(student)["wrist_rgb_history"] == {
        "shape": [3, 224, 224, 3],
        "dtype": "uint8",
        "order": "oldest_to_newest",
        "stride_policy_steps": 1,
        "policy_frequency_hz": 30,
        "reset_fill": "repeat_first_post_reset_frame",
    }
    assert tuple(student.cube.init_state.pos[:2]) == pytest.approx((0.40, 0.0))
    assert student.cube_x_pos_range == pytest.approx(0.08)
    assert student.cube_y_pos_range == pytest.approx(0.10)
    assert student.cube_position_curriculum_enabled is False
    assert teacher.enable_gpu_dynamics is True
    assert student.enable_gpu_dynamics is True
    assert teacher.scene.env_spacing == pytest.approx(3.5)
    assert student.scene.env_spacing == pytest.approx(3.5)
    assert teacher.scene.replicate_physics is False
    assert student.scene.replicate_physics is False
    assert tuple(student.observation_space["wrist_rgb_history"].shape) == (3, 224, 224, 3)
    assert tuple(student.observation_space["gsmini_left_reference_rgb"].shape) == (96, 128, 3)
    assert teacher.illegal_collision_penalty_threshold_start_n == pytest.approx(20.0)
    assert teacher.illegal_collision_penalty_threshold_end_n == pytest.approx(5.0)
    assert teacher.illegal_collision_termination_threshold_start_n == pytest.approx(200.0)
    assert teacher.illegal_collision_termination_threshold_end_n == pytest.approx(20.0)


def test_collision_curricula_are_linear_clamped_and_strict_boundaries():
    assert linear_collision_threshold(0, start_n=20.0, end_n=5.0) == pytest.approx(20.0)
    assert linear_collision_threshold(50_000, start_n=20.0, end_n=5.0) == pytest.approx(12.5)
    assert linear_collision_threshold(100_000, start_n=20.0, end_n=5.0) == pytest.approx(5.0)
    assert linear_collision_threshold(200_000, start_n=20.0, end_n=5.0) == pytest.approx(5.0)
    assert linear_collision_threshold(0, start_n=200.0, end_n=20.0) == pytest.approx(200.0)
    assert linear_collision_threshold(50_000, start_n=200.0, end_n=20.0) == pytest.approx(110.0)
    assert linear_collision_threshold(100_000, start_n=200.0, end_n=20.0) == pytest.approx(20.0)


def test_compact_reward_summary_uses_x040_unified_collision_metrics():
    log = {
        "reward/illegal_collision": torch.tensor(-1.25),
        "info/illegal_collision_penalty_threshold_n": torch.tensor(12.5),
        "info/illegal_collision_termination_threshold_n": torch.tensor(110.0),
        "info/illegal_collision_max_force_n": torch.tensor(7.5),
    }
    fields = _GelSightX040SafetyMixin._rma_collision_reward_print_fields(None, log)
    assert "illegal_collision=-1.250" in fields
    assert "illegal_threshold=12.50 N" in fields
    assert "illegal_termination_threshold=110.00 N" in fields
    assert "illegal_force_max=7.50 N" in fields
    assert "table=" not in fields


def test_three_frame_model_outputs_and_has_no_heatmap_head():
    model = RMAGelSightX040ThreeFrameStudent(pretrained_backbone=False).eval()
    history = torch.randint(0, 256, (1, 3, 224, 224, 3), dtype=torch.uint8)
    tactile = torch.randint(0, 256, (1, 96, 128, 3), dtype=torch.uint8)
    proprio = torch.zeros((1, 15))
    action_history = torch.zeros((1, 4))
    with torch.inference_mode():
        action, contact_probability, cube_position_root_m = model(
            history, proprio, action_history, tactile, tactile, tactile, tactile
        )
    assert action.shape == (1, 4)
    assert contact_probability.shape == (1, 2)
    assert cube_position_root_m.shape == (1, 3)
    assert torch.all(action.abs() <= 1.0)
    assert torch.all((contact_probability >= 0.0) & (contact_probability <= 1.0))
    assert torch.isfinite(cube_position_root_m).all()
    assert not hasattr(model, "heatmap_head")
    assert model.contract()["heatmap"] == "absent"
    assert model.contract()["actor_feature_dim"] == GELSIGHT_X040_THREE_FRAME_FUSION_DIM


def test_position_and_contact_tasks_reach_their_expected_heads():
    model = RMAGelSightX040ThreeFrameStudent(pretrained_backbone=False)
    history = torch.randint(0, 256, (1, 3, 224, 224, 3), dtype=torch.uint8)
    tactile = torch.randint(0, 256, (1, 96, 128, 3), dtype=torch.uint8)
    proprio = torch.zeros((1, 15))
    action_history = torch.zeros((1, 4))
    _, normalized_position, contact_logits = model.forward_with_training_outputs(
        history, proprio, action_history, tactile, tactile, torch.zeros_like(tactile), torch.zeros_like(tactile)
    )
    (F.mse_loss(normalized_position, torch.zeros_like(normalized_position)) + F.binary_cross_entropy_with_logits(
        contact_logits, torch.ones_like(contact_logits)
    )).backward()
    assert any(parameter.grad is not None for parameter in model.position_head.parameters())
    assert any(parameter.grad is not None for parameter in model.tactile_encoder.contact_output.parameters())
    assert not any(name.startswith("heatmap") for name, _ in model.named_parameters())


def test_position_head_can_represent_the_x040_reset_minimum():
    """The X040 x=0.32 m label must not be clipped by a bounded head."""
    model = RMAGelSightX040ThreeFrameStudent(pretrained_backbone=False).eval()
    target_position = torch.tensor([[0.32, 0.0, 0.026]])
    normalized_target = model.normalizer.normalize_position(target_position)
    final_layer = model.position_head[-1]
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.copy_(normalized_target.squeeze(0))
    history = torch.zeros((1, 3, 224, 224, 3), dtype=torch.uint8)
    tactile = torch.zeros((1, 96, 128, 3), dtype=torch.uint8)
    with torch.inference_mode():
        _, _, actual_position = model(
            history, torch.zeros((1, 15)), torch.zeros((1, 4)), tactile, tactile, tactile, tactile
        )
    torch.testing.assert_close(actual_position, target_position)


def test_torchscript_preserves_three_public_outputs():
    model = RMAGelSightX040ThreeFrameStudent(pretrained_backbone=False).eval()
    scripted = torch.jit.script(model)
    history = torch.zeros((1, 3, 224, 224, 3), dtype=torch.uint8)
    tactile = torch.zeros((1, 96, 128, 3), dtype=torch.uint8)
    args = (history, torch.zeros((1, 15)), torch.zeros((1, 4)), tactile, tactile, tactile, tactile)
    with torch.inference_mode():
        eager = model(*args)
        actual = scripted(*args)
    for expected, observed in zip(eager, actual):
        torch.testing.assert_close(observed, expected)


def test_legacy_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"kind": "tacex_rma_gelsight_size_buckets_student", "version": 3}, checkpoint)
    with pytest.raises(RuntimeError, match="GelSight X040 DR"):
        load_student_checkpoint(checkpoint)
    assert "projected_cube_center_heatmap" not in student_input_contract()["training_only_labels"]


def test_pre_static_size_bucket_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "selected_bucket_student.pt"
    torch.save(
        {
            "kind": STUDENT_KIND,
            "version": STUDENT_CHECKPOINT_VERSION - 1,
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="version mismatch"):
        load_student_checkpoint(checkpoint)


def test_teacher_and_student_eight_env_smoke():
    teacher_cfg = parse_env_cfg(
        GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK, device="cuda:0", num_envs=8
    )
    student_cfg = parse_env_cfg(
        GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
        device="cuda:0",
        num_envs=8,
    )
    teacher = gym.make(GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK, cfg=teacher_cfg)
    student = gym.make(GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK, cfg=student_cfg)
    try:
        teacher_obs, _ = teacher.reset()
        student_obs, _ = student.reset()
        teacher_base = teacher.unwrapped
        student_base = student.unwrapped
        assert teacher_base._cube.data.root_pos_w.device == teacher_base.device
        assert student_base._cube.data.root_pos_w.device == student_base.device
        assert teacher_base.sim.get_physics_context().is_gpu_dynamics_enabled()
        expected_bucket_ids = torch.arange(8, device=teacher_base.device, dtype=torch.long)
        torch.testing.assert_close(teacher_base._active_cube_bucket_ids, expected_bucket_ids)
        torch.testing.assert_close(student_base._active_cube_bucket_ids, expected_bucket_ids)
        expected_sizes = torch.tensor(
            teacher_base.cfg.cube_size_buckets_m,
            device=teacher_base.device,
            dtype=torch.float32,
        )
        torch.testing.assert_close(teacher_base.active_cube_size_m, expected_sizes)
        torch.testing.assert_close(student_base.active_cube_size_m, expected_sizes)
        assert "cube_size_buckets" not in teacher_base.scene.rigid_object_collections
        assert "cube_size_buckets" not in student_base.scene.rigid_object_collections
        teacher_actions = torch.zeros((8, 4), device=teacher_base.device)
        student_actions = torch.zeros((8, 4), device=student_base.device)
        teacher_next, teacher_rewards, teacher_terminated, teacher_truncated, _ = teacher.step(teacher_actions)
        student_next, student_rewards, student_terminated, student_truncated, _ = student.step(student_actions)
        assert teacher_obs["policy"]["rma_contact_state"].shape == (8, 2)
        assert teacher_next["policy"]["rma_cube_pos"].shape == (8, 3)
        assert student_obs["policy"]["wrist_rgb_history"].shape == (8, 3, 224, 224, 3)
        assert student_next["policy"]["gsmini_left_reference_rgb"].shape == (8, 96, 128, 3)
        assert teacher_rewards.shape == teacher_terminated.shape == teacher_truncated.shape == (8,)
        assert student_rewards.shape == student_terminated.shape == student_truncated.shape == (8,)
        fixed_teacher_sizes = teacher_base.active_cube_size_m.clone()
        teacher_base._reset_idx(torch.tensor([0], device=teacher_base.device))
        torch.testing.assert_close(teacher_base._active_cube_bucket_ids, expected_bucket_ids)
        torch.testing.assert_close(teacher_base.active_cube_size_m, fixed_teacher_sizes)
    finally:
        teacher.close()
        student.close()
