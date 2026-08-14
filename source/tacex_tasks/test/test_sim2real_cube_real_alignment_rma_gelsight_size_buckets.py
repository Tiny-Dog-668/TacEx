"""Contracts for the fixed-size GelSight reference-delta RMA route."""

from __future__ import annotations

import sys
from types import SimpleNamespace

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
from tacex.simulation_approaches.gpu_taxim.taxim_sim import TaximSimulator
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_size_buckets_artifacts import (
    STUDENT_CHECKPOINT_VERSION,
    STUDENT_MODEL_VERSION,
    environment_contract,
    infer_teacher_checkpoint_policy_step,
    load_student_checkpoint,
    student_input_contract,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_size_buckets_models import (
    GELSIGHT_STUDENT_FUSION_DIM,
    GELSIGHT_STUDENT_TACTILE_SIDE_DIM,
    RMAGelSightReferenceStudent,
    ReferenceDeltaTactileEncoder,
)
from tacex_tasks.sim2real_gelsight_rma.gelsight_geometry import geometry_contract
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_size_buckets_env import (
    GELSIGHT_SIZE_BUCKETS_M,
    GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK,
    GELSIGHT_SIZE_BUCKETS_TEACHER_TASK,
    _GelSightFixedSizeCollisionMixin,
    illegal_collision_response,
    linear_illegal_collision_penalty_threshold,
)


@pytest.fixture(scope="module", autouse=True)
def close_app():
    yield
    simulation_app.close()


def test_new_tasks_and_fixed_size_contract_are_registered():
    assert gym.spec(GELSIGHT_SIZE_BUCKETS_TEACHER_TASK) is not None
    assert gym.spec(GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK) is not None
    teacher = parse_env_cfg(GELSIGHT_SIZE_BUCKETS_TEACHER_TASK, device="cuda:0", num_envs=8)
    student = parse_env_cfg(GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK, device="cuda:0", num_envs=8)
    expected = tuple(0.04 + index * (0.02 / 7.0) for index in range(8))
    assert tuple(teacher.cube_size_buckets_m) == pytest.approx(expected)
    assert tuple(student.cube_size_buckets_m) == pytest.approx(expected)
    assert tuple(GELSIGHT_SIZE_BUCKETS_M) == pytest.approx(expected)
    assert teacher.scene.env_spacing == pytest.approx(3.5)
    assert teacher.scene.replicate_physics is False
    assert teacher.ground.spawn.visible is False
    assert student.rma_gelsight_reference_enabled is True
    assert student.observation_space["gsmini_left_reference_rgb"].shape == (96, 128, 3)
    assert student.observation_space["gsmini_right_reference_rgb"].shape == (96, 128, 3)
    assert environment_contract(teacher) == environment_contract(student)
    assert environment_contract(teacher)["gelsight_geometry"] == geometry_contract()
    assert teacher.rma_contact_force_threshold_n == pytest.approx(0.2)
    assert teacher.illegal_collision_penalty_threshold_start_n == pytest.approx(20.0)
    assert teacher.illegal_collision_penalty_threshold_end_n == pytest.approx(5.0)
    assert teacher.illegal_collision_curriculum_start_step == 0
    assert teacher.illegal_collision_curriculum_end_step == 100_000
    assert teacher.illegal_collision_termination_threshold_n == pytest.approx(10.0)


def test_illegal_collision_boundaries_are_strict_and_non_stacking():
    force = torch.tensor([0.0, 5.0, 5.0001, 10.0, 10.0001, 50.0])
    penalty, terminated = illegal_collision_response(force)
    torch.testing.assert_close(
        penalty, torch.tensor([0.0, 0.0, -10.0, -10.0, -10.0, -10.0])
    )
    assert terminated.tolist() == [False, False, False, False, True, True]


def test_illegal_collision_penalty_threshold_curriculum_is_linear_and_clamped():
    assert linear_illegal_collision_penalty_threshold(-1) == pytest.approx(20.0)
    assert linear_illegal_collision_penalty_threshold(0) == pytest.approx(20.0)
    assert linear_illegal_collision_penalty_threshold(50_000) == pytest.approx(12.5)
    assert linear_illegal_collision_penalty_threshold(100_000) == pytest.approx(5.0)
    assert linear_illegal_collision_penalty_threshold(200_000) == pytest.approx(5.0)


def test_teacher_checkpoint_policy_step_is_inferred_fail_closed(tmp_path):
    assert infer_teacher_checkpoint_policy_step("/run/checkpoints/agent_128.pt") == 128
    assert infer_teacher_checkpoint_policy_step("agent_100000.pt") == 100_000
    resumed_run = tmp_path / "resumed_run"
    checkpoint = resumed_run / "checkpoints" / "agent_128.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    params = resumed_run / "params"
    params.mkdir()
    (params / "rma_gelsight_size_buckets_manifest.json").write_text(
        '{"curriculum_policy_step_offset": 128}', encoding="utf-8"
    )
    assert infer_teacher_checkpoint_policy_step(checkpoint) == 256
    with pytest.raises(RuntimeError, match="curriculum step"):
        infer_teacher_checkpoint_policy_step("/run/checkpoints/best_agent.pt")


def test_old_bottleneck_student_checkpoint_fails_closed(tmp_path):
    checkpoint = tmp_path / "old_student.pt"
    torch.save(
        {
            "kind": "tacex_rma_gelsight_size_buckets_student",
            "version": STUDENT_CHECKPOINT_VERSION - 1,
            "model_version": STUDENT_MODEL_VERSION - 1,
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="checkpoint version"):
        load_student_checkpoint(checkpoint)
    contract = student_input_contract()
    assert contract["actor_fusion_dim"] == 1043
    assert contract["tactile_feature"] == "shared_cnn_256_per_side"


def test_non_multiple_of_eight_fails_before_scene_creation():
    dummy = SimpleNamespace(
        num_envs=4,
        cfg=SimpleNamespace(cube_size_buckets_m=GELSIGHT_SIZE_BUCKETS_M),
    )
    with pytest.raises(ValueError, match="multiple of 8"):
        _GelSightFixedSizeCollisionMixin._setup_scene(dummy)


def test_taxim_partial_reset_preserves_other_environment():
    simulator = TaximSimulator.__new__(TaximSimulator)
    simulator._num_envs = 3
    simulator._device = "cpu"
    simulator._indentation_depth = torch.tensor([1.0, 2.0, 3.0])
    simulator.background_img = torch.full((2, 2, 3), 9.0)
    simulator.tactile_rgb_img = torch.stack(
        [torch.full((2, 2, 3), float(index)) for index in range(3)]
    )
    before = simulator.tactile_rgb_img.clone()
    simulator.reset(torch.tensor([1]))
    assert simulator._indentation_depth.tolist() == [1.0, 0.0, 3.0]
    torch.testing.assert_close(simulator.tactile_rgb_img[0], before[0])
    torch.testing.assert_close(simulator.tactile_rgb_img[2], before[2])
    torch.testing.assert_close(simulator.tactile_rgb_img[1], simulator.background_img)


def test_signed_delta_and_shared_side_encoder_are_independent():
    encoder = ReferenceDeltaTactileEncoder()
    current = torch.full((2, 96, 128, 3), 200, dtype=torch.uint8)
    reference = torch.full_like(current, 100)
    delta = encoder.signed_delta(current, reference)
    assert delta.shape == (2, 3, 96, 128)
    torch.testing.assert_close(delta.mean(), torch.tensor(100.0 / 255.0))
    zero = encoder.signed_delta(reference, reference)
    assert torch.count_nonzero(zero) == 0

    with torch.no_grad():
        encoder.contact_output[-1].weight.fill_(0.1)
    zeros = torch.zeros((1, 96, 128, 3), dtype=torch.uint8)
    changed = torch.full_like(zeros, 255)
    left_a, right_a, logits_a = encoder(changed, zeros, zeros, zeros)
    left_b, right_b, logits_b = encoder(changed, changed, zeros, zeros)
    assert left_a.shape == right_a.shape == (1, GELSIGHT_STUDENT_TACTILE_SIDE_DIM)
    assert logits_a.shape == (1, 2)
    torch.testing.assert_close(left_a, left_b)
    torch.testing.assert_close(logits_a[:, 0], logits_b[:, 0])
    assert not torch.equal(right_a, right_b)


def test_continuous_visual_tactile_fusion_and_auxiliary_gradients():
    device = torch.device("cuda:0")
    model = RMAGelSightReferenceStudent(pretrained_backbone=False).to(device)
    model.unfreeze_backbone_after_layer2()
    wrist = torch.randint(
        0, 256, (1, 224, 224, 3), dtype=torch.uint8, device=device
    )
    tactile = torch.randint(
        0, 256, (1, 96, 128, 3), dtype=torch.uint8, device=device
    )
    reference = torch.zeros_like(tactile)
    proprio = torch.zeros((1, 15), device=device)
    proprio[:, -1] = 0.04
    history = torch.zeros((1, 4), device=device)
    action, position, contact_logits, heatmap = model.forward_with_auxiliary(
        wrist, proprio, history, tactile, tactile, reference, reference
    )
    action.square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.tactile_encoder.side_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.vision_encoder[7].parameters())
    assert all(
        parameter.grad is None
        for parameter in model.tactile_encoder.contact_output.parameters()
    )

    model.zero_grad(set_to_none=True)
    _, position, contact_logits, heatmap = model.forward_with_auxiliary(
        wrist, proprio, history, tactile, tactile, reference, reference
    )
    auxiliary_loss = (
        F.smooth_l1_loss(position, torch.zeros_like(position))
        + F.mse_loss(heatmap, torch.zeros_like(heatmap))
        + F.binary_cross_entropy_with_logits(
            contact_logits, torch.ones_like(contact_logits)
        )
    )
    auxiliary_loss.backward()
    assert any(parameter.grad is not None for parameter in model.position_head.parameters())
    assert any(parameter.grad is not None for parameter in model.heatmap_head.parameters())
    assert any(
        parameter.grad is not None
        for parameter in model.tactile_encoder.contact_output.parameters()
    )
    assert action.shape == (1, 4)
    assert position.shape == (1, 3)
    assert contact_logits.shape == (1, 2)
    assert heatmap.shape == (1, 1, 14, 14)
    assert torch.all(action.abs() <= 1.0)
    assert model.contract()["actor_feature_dim"] == GELSIGHT_STUDENT_FUSION_DIM


def test_fusion_student_torchscript_runtime_signature():
    model = RMAGelSightReferenceStudent(pretrained_backbone=False).eval()
    scripted = torch.jit.script(model)
    wrist = torch.zeros((1, 224, 224, 3), dtype=torch.uint8)
    proprio = torch.zeros((1, 15))
    history = torch.zeros((1, 4))
    tactile = torch.zeros((1, 96, 128, 3), dtype=torch.uint8)
    with torch.inference_mode():
        eager = model(wrist, proprio, history, tactile, tactile, tactile, tactile)
        actual = scripted(wrist, proprio, history, tactile, tactile, tactile, tactile)
    torch.testing.assert_close(actual, eager)
    assert actual.shape == (1, 4)


def test_teacher_eight_env_runtime_has_fixed_sizes_and_steps():
    cfg = parse_env_cfg(GELSIGHT_SIZE_BUCKETS_TEACHER_TASK, device="cuda:0", num_envs=8)
    env = gym.make(GELSIGHT_SIZE_BUCKETS_TEACHER_TASK, cfg=cfg)
    try:
        observations, _ = env.reset()
        base = env.unwrapped
        expected = torch.tensor(GELSIGHT_SIZE_BUCKETS_M, device=base.device)
        torch.testing.assert_close(base.active_cube_size_m, expected)
        assert observations["policy"]["rma_contact_state"].shape == (8, 2)
        assert base.cube_illegal_contact_sensor.data.force_matrix_w_history.shape[-1] == 3
        actions = torch.zeros((8, 4), device=base.device)
        for _ in range(2):
            observations, rewards, terminated, truncated, _ = env.step(actions)
        assert rewards.shape == terminated.shape == truncated.shape == (8,)
        reward_log = base.extras["log"]
        assert "reward/illegal_collision" in reward_log
        assert "reward/table_collision" not in reward_log
        expected_threshold = linear_illegal_collision_penalty_threshold(
            base.common_step_counter
        )
        assert reward_log[
            "info/illegal_collision_penalty_threshold_n"
        ].item() == pytest.approx(expected_threshold)
        print_fields = base._rma_collision_reward_print_fields(reward_log)
        assert "illegal_collision=" in print_fields
        assert "illegal_threshold=" in print_fields
        assert "table=" not in print_fields
        torch.testing.assert_close(base.active_cube_size_m, expected)
        base.episode_length_buf[:] = base.max_episode_length - 1
        terminated, truncated = base._get_dones()
        assert not terminated.any()
        assert truncated.all()
    finally:
        env.close()


def test_student_eight_env_reference_is_local_to_reset():
    cfg = parse_env_cfg(GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK, device="cuda:0", num_envs=8)
    env = gym.make(GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK, cfg=cfg)
    try:
        observations, _ = env.reset()
        base = env.unwrapped
        obs = observations["policy"]
        torch.testing.assert_close(
            obs["gsmini_left_rgb"], obs["gsmini_left_reference_rgb"]
        )
        torch.testing.assert_close(
            obs["gsmini_right_rgb"], obs["gsmini_right_reference_rgb"]
        )
        other_left = obs["gsmini_left_reference_rgb"][1:].clone()
        other_right = obs["gsmini_right_reference_rgb"][1:].clone()
        base._reset_idx(torch.tensor([0], device=base.device))
        reset_observations = base._get_observations()["policy"]
        torch.testing.assert_close(
            reset_observations["gsmini_left_reference_rgb"][1:], other_left
        )
        torch.testing.assert_close(
            reset_observations["gsmini_right_reference_rgb"][1:], other_right
        )
        torch.testing.assert_close(
            reset_observations["gsmini_left_rgb"][0],
            reset_observations["gsmini_left_reference_rgb"][0],
        )
        actions = torch.zeros((8, 4), device=base.device)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        assert observations["policy"]["gsmini_left_reference_rgb"].shape == (
            8, 96, 128, 3
        )
        assert rewards.shape == terminated.shape == truncated.shape == (8,)
    finally:
        env.close()
