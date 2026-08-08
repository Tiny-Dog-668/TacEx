"""RMA model and environment contract tests."""

from __future__ import annotations

import argparse
import json
import importlib.util
import sys
from pathlib import Path

if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import tacex_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tacex_tasks.sim2real_grasp.rma_artifacts import (
    RMA_GELSIGHT_STUDENT_DR_TASK,
    RMA_GELSIGHT_STUDENT_HEATMAP_DR_TASK,
    RMA_GELSIGHT_STUDENT_HEATMAP_TASK,
    RMA_GELSIGHT_STUDENT_TASK,
    RMA_GELSIGHT_TEACHER_TASK,
    RMA_STUDENT_DR_TASK,
    RMA_STUDENT_HEATMAP_DR_TASK,
    RMA_STUDENT_HEATMAP_TASK,
    RMA_STUDENT_TASK,
    RMA_STUDENT_TASKS,
    RMA_TEACHER_TASK,
    RMA_TEACHER_TASKS,
    _env_contract,
    load_student_model_state,
    load_student_checkpoint,
    load_teacher_manifest,
    validate_live_env_contract,
)
from tacex_tasks.sim2real_grasp.rma_models import (
    RMA_ACTOR_FEATURE_DIM,
    RMAActorCore,
    heatmap_soft_argmax,
    make_gaussian_heatmaps,
    project_points_root_to_image,
    RMAVisualStudent,
)
from tacex_tasks.sim2real_grasp.rma_xy_models import (
    RMA_XY_ACTOR_FEATURE_DIM,
    RMAXYActorCore,
    RMAXYVisualStudent,
)
from tacex_tasks.sim2real_grasp.rma_legacy_rollout import (
    RMA_LEGACY_STUDENT_HEATMAP_DR_REPLAY_TASK,
    validate_legacy_v5_student_payload,
)


TEACHER_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0"
STUDENT_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-v0"
STUDENT_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0"
STUDENT_HEATMAP_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-v0"
STUDENT_HEATMAP_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0"
GELSIGHT_TEACHER_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Teacher-v0"
GELSIGHT_STUDENT_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-v0"
GELSIGHT_STUDENT_HEATMAP_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-v0"
)
GELSIGHT_STUDENT_DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-DR-v0"
GELSIGHT_STUDENT_HEATMAP_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-DR-v0"
)
CLEAN_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-v0"
DR_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-DR-v0"


def _rollout_collector_module():
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/reinforcement_learning/skrl/collect_rma_student_rollouts.py"
    )
    spec = importlib.util.spec_from_file_location("rma_student_rollout_collector", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


def test_rma_student_rollout_collector_parses_and_validates_runs(tmp_path):
    collector = _rollout_collector_module()
    checkpoint = tmp_path / "student.pt"
    run = collector.parse_run_spec(f"{GELSIGHT_STUDENT_TASK}={checkpoint}")
    assert run.task == GELSIGHT_STUDENT_TASK
    assert run.checkpoint == checkpoint.resolve()
    parsed = collector.build_parser().parse_args(
        ["--run", f"{GELSIGHT_STUDENT_TASK}={checkpoint}"]
    )
    assert parsed.run == [run]

    collector.validate_run_payload(
        run,
        {"task": GELSIGHT_STUDENT_TASK},
        frozenset((GELSIGHT_STUDENT_TASK,)),
    )
    with pytest.raises(RuntimeError, match="differs from the Student checkpoint task"):
        collector.validate_run_payload(
            run,
            {"task": STUDENT_TASK},
            frozenset((GELSIGHT_STUDENT_TASK, STUDENT_TASK)),
        )
    with pytest.raises(argparse.ArgumentTypeError):
        collector.parse_run_spec(GELSIGHT_STUDENT_TASK)


def test_rma_student_rollout_collector_writes_aligned_npz(tmp_path):
    collector = _rollout_collector_module()
    output = tmp_path / "episode_0000.npz"
    collector.save_episode_npz(
        output,
        {
            "reward": np.asarray([0.1, 0.2], dtype=np.float32),
            "student_action": np.zeros((2, 4), dtype=np.float32),
            "frame_step_indices": np.asarray([0], dtype=np.int32),
            "wrist_rgb": np.zeros((1, 224, 224, 3), dtype=np.uint8),
        },
    )
    assert output.is_file()
    assert not (tmp_path / ".episode_0000.npz.tmp").exists()
    with np.load(output) as rollout:
        assert rollout["reward"].shape == (2,)
        assert rollout["student_action"].shape == (2, 4)
        assert rollout["frame_step_indices"].tolist() == [0]
        assert rollout["wrist_rgb"].dtype == np.uint8


def test_rma_student_rollout_collector_clean_omits_dr_parameters():
    collector = _rollout_collector_module()

    class CleanConfig:
        wrist_visual_randomization_enabled = False

    class CleanEnvironment:
        cfg = CleanConfig()

    assert collector._episode_dr_parameters(CleanEnvironment()) == {}


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
        assert "tangential_contact_penalty_weight" not in contract
        assert "tangential_contact_deadband_n" not in contract
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
        == 26
    )
    assert teacher.rma_object_pose_components == "position_xy_only"
    assert student.rma_end_effector_position_source == (
        "embedded_panda_fk_from_proprio_joint_position"
    )
    assert teacher.rma_contact_components == "bilateral_cube_finger_force_n_gte_1N_binary"
    assert teacher.rma_action_rate_penalty_weight == pytest.approx(0.05)
    assert student.rma_action_rate_penalty_weight == pytest.approx(0.05)
    assert student_dr.rma_action_rate_penalty_weight == pytest.approx(0.05)
    assert teacher.rma_success_terminates_episode is False
    assert student.rma_success_terminates_episode is False
    assert student_dr.rma_success_terminates_episode is False
    assert student.rma_contact_force_threshold_n == pytest.approx(1.0)
    assert student_dr.rma_contact_force_threshold_n == pytest.approx(1.0)
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


def test_rma_gelsight_configs_use_separate_robot_profile():
    teacher = parse_env_cfg(TEACHER_TASK, device="cuda:0", num_envs=1)
    gelsight_teacher = parse_env_cfg(GELSIGHT_TEACHER_TASK, device="cuda:0", num_envs=1)
    gelsight_student = parse_env_cfg(GELSIGHT_STUDENT_TASK, device="cuda:0", num_envs=1)
    gelsight_student_heatmap = parse_env_cfg(
        GELSIGHT_STUDENT_HEATMAP_TASK, device="cuda:0", num_envs=1
    )
    gelsight_student_dr = parse_env_cfg(GELSIGHT_STUDENT_DR_TASK, device="cuda:0", num_envs=1)
    gelsight_student_heatmap_dr = parse_env_cfg(
        GELSIGHT_STUDENT_HEATMAP_DR_TASK, device="cuda:0", num_envs=1
    )

    assert RMA_GELSIGHT_TEACHER_TASK == GELSIGHT_TEACHER_TASK
    assert RMA_GELSIGHT_STUDENT_TASK == GELSIGHT_STUDENT_TASK
    assert RMA_GELSIGHT_STUDENT_HEATMAP_TASK == GELSIGHT_STUDENT_HEATMAP_TASK
    assert RMA_GELSIGHT_STUDENT_DR_TASK == GELSIGHT_STUDENT_DR_TASK
    assert RMA_GELSIGHT_STUDENT_HEATMAP_DR_TASK == GELSIGHT_STUDENT_HEATMAP_DR_TASK
    assert GELSIGHT_TEACHER_TASK in RMA_TEACHER_TASKS
    assert GELSIGHT_STUDENT_TASK in RMA_STUDENT_TASKS
    assert GELSIGHT_STUDENT_HEATMAP_TASK in RMA_STUDENT_TASKS
    assert GELSIGHT_STUDENT_DR_TASK in RMA_STUDENT_TASKS
    assert GELSIGHT_STUDENT_HEATMAP_DR_TASK in RMA_STUDENT_TASKS

    for cfg in (
        gelsight_teacher,
        gelsight_student,
        gelsight_student_heatmap,
        gelsight_student_dr,
        gelsight_student_heatmap_dr,
    ):
        assert cfg.action_space == 4
        assert cfg.rma_actor_feature_dim == 30
        assert cfg.rma_gelsight_enabled is True
        assert cfg.rma_gelsight_actor_observation == "none"
        assert cfg.rma_robot_profile == "franka_gsmini_gripper_rigid_left_right"
        assert tuple(cfg.rma_gelsight_sensor_names) == ("gsmini_left", "gsmini_right")
        assert tuple(cfg.rma_gelsight_sensor_prims) == (
            "/World/envs/env_.*/Robot/gelsight_mini_case_left",
            "/World/envs/env_.*/Robot/gelsight_mini_case_right",
        )
        assert tuple(cfg.rma_gelsight_contact_filter_prims) == (
            "/World/envs/env_.*/Robot/gelpad_left",
            "/World/envs/env_.*/Robot/gelpad_right",
        )
        assert tuple(cfg.rma_cube_contact_sensor.filter_prim_paths_expr) == (
            "/World/envs/env_.*/Robot/gelpad_left",
            "/World/envs/env_.*/Robot/gelpad_right",
        )
        assert cfg.gsmini_left.data_types == ["tactile_rgb"]
        assert cfg.gsmini_right.data_types == ["tactile_rgb"]
        assert cfg.gsmini_left.marker_motion_sim_cfg is None
        assert cfg.gsmini_right.marker_motion_sim_cfg is None
        assert cfg.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.02))
        assert cfg.robot.actuators["panda_hand"].effort_limit_sim == pytest.approx(40.0)
        assert cfg.robot.actuators["panda_hand"].stiffness == pytest.approx(400.0)
        assert cfg.robot.actuators["panda_hand"].damping == pytest.approx(40.0)

    assert gelsight_teacher.rma_gelsight_tactile_sensor_enabled is False
    for cfg in (
        gelsight_student,
        gelsight_student_heatmap,
        gelsight_student_dr,
        gelsight_student_heatmap_dr,
    ):
        assert cfg.rma_gelsight_tactile_sensor_enabled is True
        assert cfg.rma_student_contact_observation_source == "gelsight_tactile_rgb"
        assert cfg.observation_space["gsmini_left_rgb"].shape == (96, 128, 3)
        assert cfg.observation_space["gsmini_right_rgb"].shape == (96, 128, 3)

    gelsight_contract = _env_contract(gelsight_teacher)
    student_contract = _env_contract(gelsight_student)
    legacy_contract = _env_contract(teacher)
    assert gelsight_contract == student_contract
    assert gelsight_contract["robot_profile"] == "franka_gsmini_gripper_rigid_left_right"
    assert gelsight_contract["gelsight_enabled"] is True
    assert gelsight_contract["contact_filter_prim_paths"] == [
        "/World/envs/env_.*/Robot/gelpad_left",
        "/World/envs/env_.*/Robot/gelpad_right",
    ]
    assert legacy_contract["robot_profile"] == "franka_panda_hand"
    assert legacy_contract["gelsight_enabled"] is False
    assert legacy_contract["contact_filter_prim_paths"] == [
        "/World/envs/env_.*/Robot/panda_leftfinger",
        "/World/envs/env_.*/Robot/panda_rightfinger",
    ]
    assert gelsight_contract != legacy_contract

    validate_live_env_contract(gelsight_student, {"environment_contract": gelsight_contract})
    with pytest.raises(RuntimeError, match="Live RMA environment differs"):
        validate_live_env_contract(gelsight_student, {"environment_contract": legacy_contract})
    with pytest.raises(RuntimeError, match="Live RMA environment differs"):
        validate_live_env_contract(teacher, {"environment_contract": gelsight_contract})

    assert gelsight_student_heatmap.rma_heatmap_supervision_enabled is True
    assert gelsight_student_dr.dr_curriculum_enabled is False
    assert gelsight_student_heatmap_dr.rma_heatmap_supervision_enabled is True
    assert gelsight_student_heatmap_dr.dr_curriculum_enabled is False


def test_rma_gelsight_teacher_runtime_uses_gelpad_contact_filter():
    cfg = parse_env_cfg(GELSIGHT_TEACHER_TASK, device="cuda:0", num_envs=1)
    env = gym.make(GELSIGHT_TEACHER_TASK, cfg=cfg)
    try:
        env.reset()
        base_env = env.unwrapped
        assert "gsmini_left" not in base_env.scene.sensors
        assert "gsmini_right" not in base_env.scene.sensors
        assert "rma_cube_contact_sensor" in base_env.scene.sensors
        assert tuple(cfg.rma_cube_contact_sensor.filter_prim_paths_expr) == (
            "/World/envs/env_.*/Robot/gelpad_left",
            "/World/envs/env_.*/Robot/gelpad_right",
        )
        assert base_env.rma_cube_contact_sensor.data.force_matrix_w_history.shape == (
            1,
            2,
            1,
            2,
            3,
        )

        actions = torch.zeros((1, int(cfg.action_space)), device=base_env.device)
        reach_center = base_env._compute_reach_center_world().detach().clone()
        candidates = [
            reach_center,
            reach_center + torch.tensor([[0.0, -0.02, 0.0]], device=base_env.device),
            reach_center + torch.tensor([[0.0, 0.02, 0.0]], device=base_env.device),
        ]
        max_forces = torch.zeros((1, 2), device=base_env.device)
        max_contact_state = torch.zeros((1, 2), device=base_env.device)
        max_contact_reward = torch.zeros((1,), device=base_env.device)
        for position in candidates:
            cube_state = base_env._cube.data.root_state_w.clone()
            cube_state[:, :3] = position
            cube_state[:, 3:7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=base_env.device
            )
            cube_state[:, 7:] = 0.0
            base_env._cube.write_root_state_to_sim(cube_state)
            for _ in range(12):
                observations, _, _, _, _ = env.step(actions)
                max_forces = torch.maximum(max_forces, base_env._last_rma_contact_forces)
                max_contact_state = torch.maximum(
                    max_contact_state,
                    observations["policy"]["rma_contact_state"],
                )
                max_contact_reward = torch.maximum(
                    max_contact_reward,
                    base_env._last_rma_contact_reward,
                )

        assert torch.max(max_forces).item() >= cfg.rma_contact_force_threshold_n
        assert torch.max(max_contact_state).item() == pytest.approx(1.0)
        assert torch.max(max_contact_reward).item() > 0.0
    finally:
        env.close()


def test_rma_gelsight_student_environment_exposes_tactile_contact_inputs():
    cfg = parse_env_cfg(GELSIGHT_STUDENT_TASK, device="cuda:0", num_envs=1)
    env = gym.make(GELSIGHT_STUDENT_TASK, cfg=cfg)
    try:
        observations, _ = env.reset()
        base_env = env.unwrapped
        obs = observations["policy"]
        assert "gsmini_left" in base_env.scene.sensors
        assert "gsmini_right" in base_env.scene.sensors
        assert obs["wrist_rgb"].shape == (1, 224, 224, 3)
        assert obs["gsmini_left_rgb"].shape == (1, 96, 128, 3)
        assert obs["gsmini_right_rgb"].shape == (1, 96, 128, 3)
        assert obs["gsmini_left_rgb"].dtype == torch.uint8
        assert obs["gsmini_right_rgb"].dtype == torch.uint8
        actions = torch.zeros((1, 4), device=base_env.device)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        assert next_observations["policy"]["gsmini_left_rgb"].shape == (1, 96, 128, 3)
        assert rewards.shape == terminated.shape == truncated.shape == (1,)
    finally:
        env.close()


def test_student_gradients_only_update_adaptation_head():
    actor = RMAXYActorCore()
    assert actor.network[0].in_features == RMA_XY_ACTOR_FEATURE_DIM == 26
    assert actor.contract()["feature_order"][-3:] == [
        "normalized_gripper_xy_root_from_fk[2]",
        "normalized_cube_minus_gripper_xy_root[2]",
        "grasped_bilateral_cube_finger_force_n_gte_1N[1]",
    ]
    student = RMAXYVisualStudent(actor, pretrained_backbone=False)
    rgb = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    proprio = torch.zeros((2, 15))
    proprio[:, -1] = 0.04
    history = torch.zeros((2, 4))
    target_position = torch.tensor([[0.45, -0.05], [0.55, 0.05]])
    target_force = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    predicted = student.predict_adaptation(rgb)
    student_action = student.action_from_normalized_position(
        proprio, history, predicted, target_force
    )
    with torch.no_grad():
        teacher_action = actor(proprio, history, target_position, target_force)
    loss = F.smooth_l1_loss(
        predicted,
        actor.normalizer.normalize_position(target_position),
        beta=0.1,
    ) + F.mse_loss(student_action, teacher_action)
    loss.backward()

    assert any(parameter.grad is not None for parameter in student.adaptation_head.parameters())
    assert all(parameter.grad is None for parameter in student.vision_encoder.parameters())
    assert all(parameter.grad is None for parameter in student.actor_core.parameters())
    assert predicted.shape == (2, 2)
    assert student_action.shape == (2, 4)
    assert torch.all(student_action.abs() <= 1.0)


def test_xy_actor_uses_one_bilateral_grasp_feature():
    """A single force above threshold must not be exposed as a grasp feature."""
    actor = RMAXYActorCore()
    captured_features: list[torch.Tensor] = []
    hook = actor.network[0].register_forward_pre_hook(
        lambda _module, inputs: captured_features.append(inputs[0].detach().clone())
    )
    try:
        proprio = torch.zeros((3, 15))
        proprio[:, -1] = 0.04
        actor(
            proprio,
            torch.zeros((3, 4)),
            torch.tensor([[0.50, 0.00], [0.50, 0.00], [0.50, 0.00]]),
            torch.tensor([[1.00, 0.999], [1.00, 1.00], [2.00, 1.00]]),
        )
    finally:
        hook.remove()

    assert len(captured_features) == 1
    assert captured_features[0].shape == (3, 26)
    torch.testing.assert_close(captured_features[0][:, -1], torch.tensor([0.0, 1.0, 1.0]))


def test_legacy_rollout_policy_accepts_only_archived_v5_contract():
    assert gym.spec(RMA_LEGACY_STUDENT_HEATMAP_DR_REPLAY_TASK) is not None
    payload = {
        "kind": "tacex_rma_student",
        "version": 5,
        "task": "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0",
        "model_version": 3,
        "teacher_manifest": {
            "model_version": 3,
            "actor_contract": {"feature_dim": 30},
            "actor_inputs": {
                "proprio_obs": 15,
                "action_history": 4,
                "rma_cube_pos": 3,
                "rma_contact_state": 2,
            },
        },
    }
    validate_legacy_v5_student_payload(payload)
    payload["model_version"] = 2
    with pytest.raises(RuntimeError, match="Legacy replay requires"):
        validate_legacy_v5_student_payload(payload)


def test_gelsight_student_contact_head_uses_tactile_rgb():
    actor = RMAActorCore()
    student = RMAVisualStudent(
        actor,
        pretrained_backbone=False,
        use_tactile_contact=True,
    )
    rgb = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    tactile_left = torch.randint(0, 256, (2, 96, 128, 3), dtype=torch.uint8)
    tactile_right = torch.randint(0, 256, (2, 96, 128, 3), dtype=torch.uint8)
    proprio = torch.zeros((2, 15))
    proprio[:, -1] = 0.04
    history = torch.zeros((2, 4))
    target_position = torch.tensor([[0.45, -0.05, 0.026], [0.55, 0.05, 0.05]])
    target_contact = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    predicted, contact_logits = student.predict_adaptation(
        rgb,
        tactile_left,
        tactile_right,
    )
    student_action = student.action_from_normalized_position(
        proprio,
        history,
        predicted,
        torch.sigmoid(contact_logits),
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

    assert contact_logits.shape == (2, 2)
    assert student_action.shape == (2, 4)
    assert any(parameter.grad is not None for parameter in student.tactile_contact_head.parameters())
    assert all(parameter.grad is None for parameter in student.vision_encoder.parameters())
    assert all(parameter.grad is None for parameter in student.actor_core.parameters())


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
        assert obs["rma_cube_xy"].shape == (2, 2)
        assert obs["rma_contact_force"].shape == (2, 2)
        assert torch.all(obs["rma_contact_force"] >= 0.0)
        assert "rma_cube_contact_sensor" in base_env.scene.sensors
        torch.testing.assert_close(
            obs["rma_cube_xy"],
            (base_env._cube.data.root_pos_w - base_env.scene.env_origins)[..., :2],
        )
        actor = RMAXYActorCore().to(base_env.device)
        fk_gripper_position = actor.kinematics(obs["proprio_obs"][:, :7])
        simulated_gripper_position = (
            base_env._compute_reach_center_world() - base_env.scene.env_origins
        )
        torch.testing.assert_close(
            fk_gripper_position[..., :2],
            simulated_gripper_position[..., :2],
            atol=2.0e-4,
            rtol=0.0,
        )
        actions = torch.zeros((2, 4), device=base_env.device)
        next_observations, rewards, terminated, truncated, _ = student_env.step(actions)
        assert next_observations["policy"]["wrist_rgb"].shape == (2, 224, 224, 3)
        assert rewards.shape == terminated.shape == truncated.shape == (2,)
        log = base_env.extras["log"]
        assert "reward/rma_tangential_contact" not in log
        assert "info/rma_tangential_contact_force_n" not in log
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
    student = RMAXYVisualStudent(RMAXYActorCore(), pretrained_backbone=False).eval()
    inputs = (
        torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
        torch.zeros((1, 15), dtype=torch.float32),
        torch.zeros((1, 4), dtype=torch.float32),
        torch.tensor([[1.0, 1.0]], dtype=torch.float32),
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


def test_gelsight_student_torchscript_uses_tactile_contact_inputs():
    student = RMAVisualStudent(
        RMAActorCore(),
        pretrained_backbone=False,
        use_tactile_contact=True,
    ).eval()
    inputs = (
        torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
        torch.zeros((1, 15), dtype=torch.float32),
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros((1, 96, 128, 3), dtype=torch.uint8),
        torch.zeros((1, 96, 128, 3), dtype=torch.uint8),
    )
    traced = torch.jit.script(student)
    with torch.inference_mode():
        torch.testing.assert_close(student(*inputs), traced(*inputs))


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
            "version": 5,
            "task": RMA_STUDENT_TASK,
            "model_version": 3,
        },
        student_checkpoint,
    )
    with pytest.raises(RuntimeError, match="Unsupported RMA student checkpoint version"):
        load_student_checkpoint(student_checkpoint)
