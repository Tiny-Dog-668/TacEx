"""RMA model and environment contract tests."""

from __future__ import annotations

import json
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

import tacex_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tacex_tasks.sim2real_grasp.rma_artifacts import (
    RMA_STUDENT_DR_TASK,
    RMA_STUDENT_HEATMAP_DR_TASK,
    RMA_STUDENT_HEATMAP_TASK,
    RMA_STUDENT_TASK,
    RMA_STUDENT_TASKS,
    RMA_TEACHER_TASK,
    _env_contract,
    load_student_model_state,
    load_student_checkpoint,
    load_teacher_manifest,
)
from tacex_tasks.sim2real_grasp.rma_models import (
    RMA_ACTOR_FEATURE_DIM,
    RMAActorCore,
    heatmap_soft_argmax,
    make_gaussian_heatmaps,
    project_points_root_to_image,
    RMAVisualStudent,
)


TEACHER_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0"
STUDENT_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-v0"
STUDENT_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0"
STUDENT_HEATMAP_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-v0"
STUDENT_HEATMAP_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0"
CLEAN_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-v0"
DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-DR-v0"


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


def test_rma_configs_are_isolated_from_existing_tasks():
    clean = parse_env_cfg(CLEAN_TASK, device="cuda:0", num_envs=1)
    dr = parse_env_cfg(DR_TASK, device="cuda:0", num_envs=1)
    teacher = parse_env_cfg(TEACHER_TASK, device="cuda:0", num_envs=1)
    student = parse_env_cfg(STUDENT_TASK, device="cuda:0", num_envs=1)
    student_dr = parse_env_cfg(STUDENT_DR_TASK, device="cuda:0", num_envs=1)
    student_heatmap = parse_env_cfg(STUDENT_HEATMAP_TASK, device="cuda:0", num_envs=1)
    student_heatmap_dr = parse_env_cfg(STUDENT_HEATMAP_DR_TASK, device="cuda:0", num_envs=1)

    for cfg in (clean, dr, teacher, student, student_dr, student_heatmap, student_heatmap_dr):
        assert cfg.action_scale == pytest.approx(0.05)
        assert cfg.gripper_width_delta_scale == pytest.approx(0.01)
        assert cfg.cube_x_pos_range == pytest.approx(0.05)
        assert cfg.cube_y_pos_range == pytest.approx(0.05)
    assert clean.robot.actuators["panda_hand"].effort_limit_sim == pytest.approx(200.0)
    assert dr.robot.actuators["panda_hand"].effort_limit_sim == pytest.approx(200.0)
    for cfg in (teacher, student, student_dr, student_heatmap, student_heatmap_dr):
        hand = cfg.robot.actuators["panda_hand"]
        assert hand.effort_limit_sim == pytest.approx(40.0)
        assert hand.stiffness == pytest.approx(400.0)
        assert hand.damping == pytest.approx(40.0)
        assert hand.velocity_limit_sim is None
        contract = _env_contract(cfg)
        assert contract["robot_base_world_position_m"] == [0.0, 0.0, 0.02]
        assert contract["action_rate_penalty_weight"] == pytest.approx(0.05)
        assert contract["action_rate_penalty_scales"] == "environment_action_scales"
        assert contract["camera_base_position_m"] == pytest.approx(
            [1.166091088407, 0.035901608197, 0.514200335898]
        )
        assert contract["camera_world_position_m"] == pytest.approx(
            [1.166091088407, 0.035901608197, 0.534200335898]
        )
    assert teacher.vision_encoder_enabled is False
    assert student.vision_encoder_enabled is False
    assert student_dr.vision_encoder_enabled is False
    assert student_heatmap.vision_encoder_enabled is False
    assert student_heatmap_dr.vision_encoder_enabled is False
    assert (
        teacher.rma_actor_feature_dim
        == student.rma_actor_feature_dim
        == student_dr.rma_actor_feature_dim
        == student_heatmap.rma_actor_feature_dim
        == student_heatmap_dr.rma_actor_feature_dim
        == 30
    )
    assert teacher.rma_object_pose_components == "position_xyz_only"
    assert student.rma_end_effector_position_source == (
        "embedded_panda_fk_from_proprio_joint_position"
    )
    assert teacher.rma_contact_components == "left_right_cube_finger_binary"
    assert teacher.rma_action_rate_penalty_weight == pytest.approx(0.05)
    assert student.rma_action_rate_penalty_weight == pytest.approx(0.05)
    assert student_dr.rma_action_rate_penalty_weight == pytest.approx(0.05)
    assert teacher.rma_success_terminates_episode is False
    assert student.rma_success_terminates_episode is False
    assert student_dr.rma_success_terminates_episode is False
    assert student.rma_contact_force_threshold_n == pytest.approx(0.2)
    assert student_dr.rma_contact_force_threshold_n == pytest.approx(0.2)
    assert student_dr.dr_curriculum_enabled is False
    assert student.rma_heatmap_supervision_enabled is False
    assert student.rma_heatmap_loss_weight == pytest.approx(0.0)
    assert student_dr.rma_heatmap_supervision_enabled is False
    assert student_dr.rma_heatmap_loss_weight == pytest.approx(0.0)
    assert student_heatmap.rma_heatmap_supervision_enabled is True
    assert student_heatmap.rma_heatmap_loss_weight == pytest.approx(1.0)
    assert student_heatmap.rma_heatmap_sigma_px == pytest.approx(1.5)
    assert student_heatmap_dr.rma_heatmap_supervision_enabled is True
    assert student_heatmap_dr.rma_heatmap_loss_weight == pytest.approx(1.0)
    assert student_heatmap_dr.rma_heatmap_sigma_px == pytest.approx(1.5)
    assert student_dr.wrist_visual_randomization_enabled is True
    assert student_dr.camera_pose_randomization_enabled is True
    assert student_dr.camera_intrinsic_warp_enabled is True
    assert not hasattr(clean, "rma_cube_contact_sensor")
    assert not hasattr(dr, "rma_cube_contact_sensor")
    assert clean.cube.spawn.activate_contact_sensors is False
    assert dr.cube.spawn.activate_contact_sensors is False
    assert teacher.cube.spawn.activate_contact_sensors is True
    assert student.cube.spawn.activate_contact_sensors is True
    assert student_dr.cube.spawn.activate_contact_sensors is True
    assert clean.cube_position_curriculum_force_full_range is False
    assert student.cube_position_curriculum_force_full_range is True
    assert student_dr.cube_position_curriculum_force_full_range is True
    assert RMA_STUDENT_TASK == STUDENT_TASK
    assert RMA_STUDENT_DR_TASK == STUDENT_DR_TASK
    assert RMA_STUDENT_HEATMAP_TASK == STUDENT_HEATMAP_TASK
    assert RMA_STUDENT_HEATMAP_DR_TASK == STUDENT_HEATMAP_DR_TASK
    assert STUDENT_HEATMAP_TASK in RMA_STUDENT_TASKS
    assert STUDENT_HEATMAP_DR_TASK in RMA_STUDENT_TASKS


def test_student_gradients_only_update_adaptation_head():
    actor = RMAActorCore()
    assert actor.network[0].in_features == RMA_ACTOR_FEATURE_DIM == 30
    assert actor.contract()["feature_order"][-3:] == [
        "normalized_gripper_position_root_from_fk[3]",
        "normalized_cube_minus_gripper_position_root[3]",
        "left_right_cube_finger_contact[2]",
    ]
    student = RMAVisualStudent(actor, pretrained_backbone=False)
    rgb = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    proprio = torch.zeros((2, 15))
    proprio[:, -1] = 0.04
    history = torch.zeros((2, 4))
    target_position = torch.tensor([[0.45, -0.05, 0.026], [0.55, 0.05, 0.05]])
    target_contact = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    predicted, contact_logits = student.predict_adaptation(rgb)
    student_action = student.action_from_normalized_position(
        proprio, history, predicted, torch.sigmoid(contact_logits)
    )
    with torch.no_grad():
        teacher_action = actor(proprio, history, target_position, target_contact)
    loss = F.smooth_l1_loss(
        predicted,
        actor.normalizer.normalize_position(target_position),
        beta=0.1,
    ) + F.binary_cross_entropy_with_logits(
        contact_logits, target_contact
    ) + F.mse_loss(student_action, teacher_action)
    loss.backward()

    assert any(parameter.grad is not None for parameter in student.adaptation_head.parameters())
    assert all(parameter.grad is None for parameter in student.vision_encoder.parameters())
    assert all(parameter.grad is None for parameter in student.actor_core.parameters())
    assert contact_logits.shape == (2, 2)
    assert student_action.shape == (2, 4)
    assert torch.all(student_action.abs() <= 1.0)


def test_student_heatmap_head_updates_only_heatmap_branch():
    student = RMAVisualStudent(RMAActorCore(), pretrained_backbone=False)
    rgb = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    _, _, predicted_heatmap = student.predict_adaptation_and_heatmap(rgb)
    target_uv = torch.tensor([[112.0, 112.0], [64.0, 80.0]])
    valid = torch.tensor([True, True])
    target_heatmap = make_gaussian_heatmaps(target_uv, valid, sigma=1.5)

    loss = F.mse_loss(predicted_heatmap, target_heatmap)
    loss.backward()

    assert predicted_heatmap.shape == (2, 1, 14, 14)
    assert any(parameter.grad is not None for parameter in student.heatmap_head.parameters())
    assert all(parameter.grad is None for parameter in student.vision_encoder.parameters())
    assert all(parameter.grad is None for parameter in student.actor_core.parameters())


def test_student_can_unfreeze_resnet_after_layer2_only():
    student = RMAVisualStudent(RMAActorCore(), pretrained_backbone=False)
    student.unfreeze_backbone_after_layer2()

    for index, module in enumerate(student.vision_encoder):
        requires_grad = [parameter.requires_grad for parameter in module.parameters()]
        if not requires_grad:
            continue
        if index <= 5:
            assert not any(requires_grad)
        else:
            assert all(requires_grad)

    rgb = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    predicted, _, predicted_heatmap = student.predict_adaptation_and_heatmap(rgb)
    loss = predicted.square().mean() + predicted_heatmap.mean()
    loss.backward()

    for index, module in enumerate(student.vision_encoder):
        gradients = [parameter.grad for parameter in module.parameters()]
        if not gradients:
            continue
        if index <= 5:
            assert all(gradient is None for gradient in gradients)
        else:
            assert any(gradient is not None for gradient in gradients)


def test_student_state_loader_accepts_legacy_missing_heatmap_head():
    student = RMAVisualStudent(RMAActorCore(), pretrained_backbone=False)
    legacy_state = {
        key: value
        for key, value in student.state_dict().items()
        if not key.startswith("heatmap_head.")
    }

    load_student_model_state(student, legacy_state)


def test_heatmap_projection_generation_and_soft_argmax():
    points_root = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    camera_position_root = torch.zeros(3)
    camera_quaternion_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0])
    intrinsic = torch.tensor(
        [
            [100.0, 0.0, 112.0],
            [0.0, 100.0, 112.0],
            [0.0, 0.0, 1.0],
        ]
    )

    uv, valid = project_points_root_to_image(
        points_root,
        camera_position_root,
        camera_quaternion_wxyz,
        intrinsic,
    )
    torch.testing.assert_close(uv[0], torch.tensor([112.0, 112.0]))
    assert valid.tolist() == [True, False]

    heatmaps = make_gaussian_heatmaps(uv, valid, sigma=1.0)
    assert heatmaps.shape == (2, 1, 14, 14)
    assert heatmaps[0, 0].argmax().item() == 7 * 14 + 7
    assert float(heatmaps[1].sum().item()) == pytest.approx(0.0)

    predicted_uv = heatmap_soft_argmax(heatmaps)
    torch.testing.assert_close(predicted_uv[0], torch.tensor([112.0, 112.0]), atol=2.0, rtol=0.0)


def test_rma_student_environment_observation_contract():
    student_cfg = parse_env_cfg(STUDENT_TASK, device="cuda:0", num_envs=2)
    student_env = gym.make(STUDENT_TASK, cfg=student_cfg)
    try:
        observations, _ = student_env.reset()
        base_env = student_env.unwrapped
        obs = observations["policy"]
        assert "wrist_camera" in base_env.scene.sensors
        assert not hasattr(base_env, "_resnet18")
        assert obs["wrist_rgb"].shape == (2, 224, 224, 3)
        assert obs["wrist_rgb"].dtype == torch.uint8
        assert obs["rma_cube_pos"].shape == (2, 3)
        assert obs["rma_contact_state"].shape == (2, 2)
        assert torch.all((obs["rma_contact_state"] == 0.0) | (obs["rma_contact_state"] == 1.0))
        assert "rma_cube_contact_sensor" in base_env.scene.sensors
        torch.testing.assert_close(
            obs["rma_cube_pos"],
            base_env._cube.data.root_pos_w - base_env.scene.env_origins,
        )
        actor = RMAActorCore().to(base_env.device)
        fk_gripper_position = actor.kinematics(obs["proprio_obs"][:, :7])
        simulated_gripper_position = (
            base_env._compute_reach_center_world() - base_env.scene.env_origins
        )
        torch.testing.assert_close(
            fk_gripper_position,
            simulated_gripper_position,
            atol=2.0e-4,
            rtol=0.0,
        )
        actions = torch.zeros((2, 4), device=base_env.device)
        next_observations, rewards, terminated, truncated, _ = student_env.step(actions)
        assert next_observations["policy"]["wrist_rgb"].shape == (2, 224, 224, 3)
        assert rewards.shape == terminated.shape == truncated.shape == (2,)
    finally:
        student_env.close()


def test_rma_student_dr_environment_uses_full_strength_randomization():
    student_cfg = parse_env_cfg(STUDENT_DR_TASK, device="cuda:0", num_envs=2)
    student_env = gym.make(STUDENT_DR_TASK, cfg=student_cfg)
    try:
        observations, _ = student_env.reset()
        base_env = student_env.unwrapped
        obs = observations["policy"]
        assert obs["wrist_rgb"].shape == (2, 224, 224, 3)
        assert obs["wrist_rgb"].dtype == torch.uint8
        assert base_env._dr_curriculum_scale() == pytest.approx(1.0)
        assert base_env._dr_randomization_initialized is True
        assert base_env._dr_current_curriculum_scale == pytest.approx(1.0)
        actions = torch.zeros((2, 4), device=base_env.device)
        next_observations, rewards, terminated, truncated, _ = student_env.step(actions)
        assert next_observations["policy"]["wrist_rgb"].shape == (2, 224, 224, 3)
        assert rewards.shape == terminated.shape == truncated.shape == (2,)
        assert torch.isfinite(rewards).all()
    finally:
        student_env.close()


def test_student_torchscript_has_no_privileged_input(tmp_path):
    student = RMAVisualStudent(RMAActorCore(), pretrained_backbone=False).eval()
    inputs = (
        torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
        torch.zeros((1, 15), dtype=torch.float32),
        torch.zeros((1, 4), dtype=torch.float32),
    )
    traced = torch.jit.script(student)
    with torch.inference_mode():
        torch.testing.assert_close(student(*inputs), traced(*inputs))

    if torch.cuda.is_available():
        export_path = tmp_path / "rma_student.pt"
        traced.save(str(export_path))
        cuda_model = torch.jit.load(str(export_path), map_location="cuda:0").eval()
        cuda_inputs = tuple(value.cuda() for value in inputs)
        with torch.inference_mode():
            cuda_actions = cuda_model(*cuda_inputs)
        assert cuda_actions.is_cuda
        torch.testing.assert_close(traced(*inputs), cuda_actions.cpu(), atol=1e-3, rtol=0.0)


def test_rma_legacy_artifacts_are_rejected(tmp_path):
    run_dir = tmp_path / "teacher_run"
    checkpoint = run_dir / "checkpoints" / "teacher.pt"
    manifest = run_dir / "params" / "rma_manifest.json"
    checkpoint.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    torch.save({"policy": {}}, checkpoint)
    manifest.write_text(
        json.dumps(
            {
                "kind": "tacex_rma_teacher",
                "version": 6,
                "task": RMA_TEACHER_TASK,
                "model_version": 3,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Unsupported RMA teacher manifest version"):
        load_teacher_manifest(checkpoint)

    student_checkpoint = tmp_path / "student_v2.pt"
    torch.save(
        {
            "kind": "tacex_rma_student",
            "version": 4,
            "task": RMA_STUDENT_TASK,
            "model_version": 3,
        },
        student_checkpoint,
    )
    with pytest.raises(RuntimeError, match="Unsupported RMA student checkpoint version"):
        load_student_checkpoint(student_checkpoint)
