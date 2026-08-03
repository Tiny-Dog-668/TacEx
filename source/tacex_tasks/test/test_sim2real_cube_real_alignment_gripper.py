"""Focused checks for the Real-Alignment Cube control and visual-DR contracts."""

from __future__ import annotations

import copy
import sys

if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


import gymnasium as gym
import pytest
import torch

import tacex_tasks  # noqa: F401
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from scripts.reinforcement_learning.skrl.vision_encoder_artifact import (
    ACTION_HISTORY_CONTRACT_V1,
    ACTION_HISTORY_CONTRACT_V2,
    ACTION_HISTORY_CONTRACT_V3,
    POLICY_CONTRACT_VERSION,
    REAL_ALIGNMENT_CUBE_TASKS,
    _build_policy_contract,
    validate_live_env_against_policy_contract,
    validate_sim2real_policy_contract,
)


TASK_ID = "TacEx-Sim2Real-Cube-Real-Alignment-v0"
DR_TASK_ID = "TacEx-Sim2Real-Cube-Real-Alignment-DR-v0"
LEGACY_CUBE_TASK_ID = "TacEx-Sim2Real-Cube-Grasp-v0"


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


@pytest.fixture(scope="module")
def real_alignment_env():
    env_cfg = parse_env_cfg(DR_TASK_ID, device="cuda:0", num_envs=4)
    env = gym.make(DR_TASK_ID, cfg=env_cfg)
    env.reset()
    try:
        yield env.unwrapped
    finally:
        env.close()


def test_clean_and_dr_randomization_profiles_are_separated():
    base_cube_cfg = parse_env_cfg(LEGACY_CUBE_TASK_ID, device="cuda:0", num_envs=1)
    clean_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=1)
    dr_cfg = parse_env_cfg(DR_TASK_ID, device="cuda:0", num_envs=1)

    assert {TASK_ID, DR_TASK_ID}.issubset(REAL_ALIGNMENT_CUBE_TASKS)
    assert clean_cfg.sim.dt == pytest.approx(1.0 / 60.0)
    assert clean_cfg.decimation == dr_cfg.decimation == 2
    assert clean_cfg.wrist_camera.update_period == dr_cfg.wrist_camera.update_period == pytest.approx(1.0 / 30.0)
    assert clean_cfg.episode_length_s == dr_cfg.episode_length_s == pytest.approx(5.0)
    assert clean_cfg.episode_success_rate_window_steps == 200
    assert dr_cfg.episode_success_rate_window_steps == 200
    policy_dt = clean_cfg.sim.dt * clean_cfg.decimation
    assert policy_dt == pytest.approx(1.0 / 30.0)
    assert round(clean_cfg.episode_length_s / policy_dt) == 150
    assert clean_cfg.cube_x_pos_range == dr_cfg.cube_x_pos_range == 0.05
    assert clean_cfg.cube_y_pos_range == dr_cfg.cube_y_pos_range == 0.05
    assert clean_cfg.cube_rot_range == dr_cfg.cube_rot_range == 0.0
    for cfg in (base_cube_cfg, clean_cfg, dr_cfg):
        assert cfg.lift_reference_mode == "center_of_mass"
        assert cfg.lift_reward_start_delta == pytest.approx(0.0)
        assert cfg.success_lift_delta == pytest.approx(0.035)
        assert cfg.privileged_dz_gate_enabled is False
        assert cfg.lift_reward_requires_upright is False
    assert base_cube_cfg.success_requires_upright is True
    assert base_cube_cfg.lift_tilt_curriculum_enabled is True
    for cfg in (clean_cfg, dr_cfg):
        # Environment commands use the physical increments stored in action
        # history.  RMA model checkpoints may deliberately normalize that
        # history with their own legacy 0.025 / 0.005 scales.
        assert cfg.action_scale == pytest.approx(0.05)
        assert cfg.gripper_width_delta_scale == pytest.approx(0.01)
        assert cfg.success_requires_upright is False
        assert cfg.lift_tilt_curriculum_enabled is False
        assert cfg.plate_thickness_m == pytest.approx(0.001)
        assert cfg.plate_top_height_m == pytest.approx(0.001)
        assert cfg.cube.init_state.pos == pytest.approx((0.5, 0.0, 0.026))
        assert cfg.cube_position_curriculum_initial_x_range == pytest.approx(0.02)
        assert cfg.cube_position_curriculum_initial_y_range == pytest.approx(0.02)
        assert cfg.cube_position_curriculum_start_step == 20_000
        assert cfg.cube_position_curriculum_end_step == 100_000
        assert cfg.table_collision_force_threshold_n == pytest.approx(1.0)
        assert cfg.table_collision_penalty == pytest.approx(-10.0)
        assert cfg.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.02))
        assert cfg.robot_base_world_position_m == pytest.approx((0.0, 0.0, 0.02))
        assert cfg.camera_base_position_m == pytest.approx(
            (1.166091088407, 0.035901608197, 0.514200335898)
        )
        assert cfg.wrist_camera.offset.convention == "ros"
        assert cfg.wrist_camera.offset.pos == pytest.approx(
            (1.166091088407, 0.035901608197, 0.534200335898)
        )
        assert cfg.wrist_camera.offset.rot == pytest.approx(
            (0.378248136306, -0.604227000834, -0.586824979121, 0.384024117374)
        )
        assert cfg.camera_crop_roi_xywh == (100, 34, 400, 398)
        assert cfg.camera_model_intrinsic_matrix == pytest.approx(
            (
                338.742544,
                0.0,
                123.748857,
                0.0,
                340.550811,
                120.393372,
                0.0,
                0.0,
                1.0,
            )
        )
        assert cfg.camera_native_render_intrinsic_matrix == pytest.approx(
            (300.0, 0.0, 112.0, 0.0, 300.0, 112.0, 0.0, 0.0, 1.0)
        )
        assert cfg.camera_nominal_intrinsic_compensation_enabled is True
        assert cfg.deployment_camera_serial == "215322076207"
    for cfg in (base_cube_cfg,):
        assert cfg.lift_tilt_curriculum_enabled is True
        assert cfg.lift_tilt_curriculum_start_deg == pytest.approx(40.0)
        assert cfg.lift_tilt_curriculum_end_deg == pytest.approx(10.0)
        assert cfg.lift_tilt_curriculum_start_step == 0
        assert cfg.lift_tilt_curriculum_end_step == 120_000
        assert cfg.lift_tilt_curriculum_step_offset == 0
        assert cfg.lift_upright_tilt_threshold_deg == pytest.approx(10.0)

    randomization_flags = (
        "camera_pose_randomization_enabled",
        "camera_intrinsic_warp_enabled",
        "wrist_visual_randomization_enabled",
        "wrist_brightness_randomization_enabled",
        "wrist_gamma_randomization_enabled",
        "wrist_contrast_randomization_enabled",
        "wrist_saturation_randomization_enabled",
        "wrist_hue_randomization_enabled",
        "wrist_white_balance_randomization_enabled",
        "wrist_blur_randomization_enabled",
        "wrist_gaussian_noise_randomization_enabled",
        "light_randomization_enabled",
        "plate_color_randomization_enabled",
        "backdrop_color_randomization_enabled",
    )
    assert all(getattr(clean_cfg, name, False) is False for name in randomization_flags)
    assert all(getattr(dr_cfg, name) is True for name in randomization_flags)
    assert clean_cfg.ground_color_randomization_enabled is False
    assert dr_cfg.ground_color_randomization_enabled is False

    assert clean_cfg.robot_joint_pos_noise == 0.0
    assert clean_cfg.robot_joint_vel_noise == 0.0
    assert clean_cfg.action_noise_scale == 0.0
    assert clean_cfg.reset_jitter_max_steps == 1

    assert dr_cfg.dr_curriculum_start_step == 100_000
    assert dr_cfg.dr_curriculum_end_step == 220_000
    assert dr_cfg.dr_curriculum_initial_scale == pytest.approx(0.0)
    assert dr_cfg.camera_position_delta_max_m == (0.0015, 0.0015, 0.0015)
    assert dr_cfg.camera_rotation_delta_max_deg == (0.5, 0.5, 0.5)
    assert dr_cfg.camera_focal_scale_range == (0.99, 1.01)
    assert dr_cfg.camera_principal_point_shift_max_px == (1.0, 1.0)
    assert dr_cfg.wrist_brightness_range == (0.90, 1.10)
    assert dr_cfg.wrist_contrast_range == (0.90, 1.10)
    assert dr_cfg.wrist_saturation_range == (0.90, 1.05)
    assert dr_cfg.wrist_gamma_range == (0.92, 1.08)
    assert dr_cfg.wrist_hue_max_deg == pytest.approx(2.0)
    assert dr_cfg.wrist_white_balance_shift_max == pytest.approx(0.04)
    assert dr_cfg.wrist_blur_probability == pytest.approx(0.08)
    assert dr_cfg.wrist_blur_kernel_sizes == (3,)
    assert dr_cfg.wrist_gaussian_noise_std_range == (0.0, 0.006)
    assert dr_cfg.light_nominal_intensity == pytest.approx(1600.0)
    assert dr_cfg.light_intensity_range == (1300.0, 1900.0)
    assert dr_cfg.light_color_temperature_range == (4800.0, 6200.0)
    assert dr_cfg.light_nominal_color == pytest.approx((0.72, 0.72, 0.72))
    assert dr_cfg.plate_color_center == pytest.approx(
        clean_cfg.plate.spawn.visual_material.diffuse_color
    )
    assert dr_cfg.backdrop_color_center == pytest.approx(
        clean_cfg.backdrop.spawn.visual_material.diffuse_color
    )
    assert dr_cfg.plate_color_min == pytest.approx((0.003, 0.003, 0.003))
    assert dr_cfg.plate_color_max == pytest.approx((0.012, 0.012, 0.012))
    assert dr_cfg.backdrop_color_min == pytest.approx((0.002, 0.002, 0.002))
    assert dr_cfg.backdrop_color_max == pytest.approx((0.008, 0.008, 0.008))


def test_nominal_intrinsic_compensation_places_native_optical_axis_at_calibrated_principal_point(
    real_alignment_env,
):
    env = real_alignment_env
    native = env.cfg.camera_native_render_intrinsic_matrix
    target = env.cfg.camera_model_intrinsic_matrix
    image = torch.zeros((1, 1, 224, 224), device=env.device)
    image[0, 0, round(native[5]), round(native[2])] = 1.0

    warped = env._apply_nominal_camera_intrinsic_compensation(image)
    peak_index = int(torch.argmax(warped).item())
    peak_y, peak_x = divmod(peak_index, 224)

    assert peak_x == round(target[2])
    assert peak_y == round(target[5])
    assert torch.isfinite(warped).all()


def test_center_lift_reward_ignores_tilt_and_step_average(real_alignment_env):
    env = real_alignment_env
    env._ensure_cube_lift_reference_height()
    expected_reference = env._cylinder_spawn_height_per_env + 0.001
    torch.testing.assert_close(env._cube_lift_reference_height_per_env, expected_reference)

    original_step = int(env.common_step_counter)
    try:
        upright_cos = torch.ones(env.num_envs, device=env.device)
        lift_deltas = torch.tensor([0.0, 0.0175, 0.035, 0.050], device=env.device)
        center_heights = env._cube_lift_reference_height_per_env + lift_deltas

        measured_delta, lift_reward, success = env._compute_cube_lift_terms(
            center_heights,
            upright_cos,
        )
        torch.testing.assert_close(measured_delta, lift_deltas)
        torch.testing.assert_close(
            lift_reward,
            torch.tensor([0.0, 0.5, 1.0, 1.0], device=env.device),
        )
        assert success.tolist() == [False, False, True, True]

        ninety_degree_cos = torch.full(
            (env.num_envs,),
            0.0,
            device=env.device,
        )
        _, tilted_reward, tilted_success = env._compute_cube_lift_terms(
            center_heights,
            ninety_degree_cos,
        )
        torch.testing.assert_close(
            tilted_reward,
            torch.tensor([0.0, 0.5, 1.0, 1.0], device=env.device),
        )
        assert tilted_success.tolist() == [False, False, True, True]

        env._reset_reward_print_window()
        first_average = env._update_reward_print_window(
            torch.tensor([1.0, 3.0, 1.0, 3.0], device=env.device)
        )
        second_average = env._update_reward_print_window(
            torch.tensor([3.0, 5.0, 3.0, 5.0], device=env.device)
        )
        torch.testing.assert_close(first_average, torch.tensor(2.0, device=env.device))
        torch.testing.assert_close(second_average, torch.tensor(3.0, device=env.device))
        assert env._reward_print_window_count == 2
        env._reset_reward_print_window()
        assert env._reward_print_window_count == 0
    finally:
        env.common_step_counter = original_step


def test_position_curriculum_and_full_range_override(real_alignment_env):
    env = real_alignment_env
    original_step = int(env.common_step_counter)
    original_force_full = bool(env.cfg.cube_position_curriculum_force_full_range)
    try:
        env.cfg.cube_position_curriculum_force_full_range = False
        env.common_step_counter = 0
        assert env._current_cube_xy_half_ranges() == pytest.approx((0.02, 0.02))
        env.common_step_counter = 20_000
        assert env._current_cube_xy_half_ranges() == pytest.approx((0.02, 0.02))
        env.common_step_counter = 60_000
        assert env._current_cube_xy_half_ranges() == pytest.approx((0.035, 0.035))
        env.common_step_counter = 100_000
        assert env._current_cube_xy_half_ranges() == pytest.approx((0.05, 0.05))

        env.cfg.cube_position_curriculum_force_full_range = True
        env.common_step_counter = 0
        assert env._current_cube_xy_half_ranges() == pytest.approx((0.05, 0.05))
        samples = env._sample_cube_xy_offsets(4096)
        assert torch.all(samples[:, 0] >= -0.05)
        assert torch.all(samples[:, 0] <= 0.05)
        assert torch.all(samples[:, 1] >= -0.05)
        assert torch.all(samples[:, 1] <= 0.05)
    finally:
        env.common_step_counter = original_step
        env.cfg.cube_position_curriculum_force_full_range = original_force_full


def test_episode_success_rate_uses_last_200_policy_steps(real_alignment_env):
    env = real_alignment_env
    env._reset_episode_success_statistics()
    assert env._episode_success_window_steps == 200

    # The production done path must advance the time window and publish fields
    # even when no episode happens to finish on this policy step.
    env._get_dones()
    assert env._episode_success_window_step_count.item() == 1
    assert "episode_success_rate_window" in env.extras["log"]
    env._reset_episode_success_statistics()

    # Step 1: four episodes finish and one succeeds.
    env._record_episode_outcomes_for_step(
        torch.tensor(4, device=env.device),
        torch.tensor(1, device=env.device),
    )
    # Steps 2--199: no completed episodes, but time still advances.
    for _ in range(198):
        env._record_episode_outcomes_for_step(
            torch.tensor(0, device=env.device),
            torch.tensor(0, device=env.device),
        )
    # Step 200: two completed episodes and both succeed.
    env._record_episode_outcomes_for_step(
        torch.tensor(2, device=env.device),
        torch.tensor(2, device=env.device),
    )
    metrics = env._episode_success_statistics()
    assert metrics["window_rate"].item() == pytest.approx(3.0 / 6.0)
    assert metrics["cumulative_rate"].item() == pytest.approx(3.0 / 6.0)
    assert metrics["window_completed_count"].item() == pytest.approx(6.0)
    assert metrics["window_success_count"].item() == pytest.approx(3.0)
    assert metrics["window_step_count"].item() == pytest.approx(200.0)

    # Step 201 evicts step 1. The last 200 steps now contain only the two
    # successful episodes from step 200, while the cumulative rate is unchanged.
    env._record_episode_outcomes_for_step(
        torch.tensor(0, device=env.device),
        torch.tensor(0, device=env.device),
    )
    metrics = env._episode_success_statistics()
    assert metrics["window_rate"].item() == pytest.approx(1.0)
    assert metrics["cumulative_rate"].item() == pytest.approx(0.5)
    assert metrics["window_completed_count"].item() == pytest.approx(2.0)
    assert metrics["window_success_count"].item() == pytest.approx(2.0)

    env._publish_episode_success_statistics()
    log = env.extras["log"]
    assert log["recent_success_rate"].item() == pytest.approx(1.0)
    assert log["episode_success_rate_window"].item() == pytest.approx(1.0)
    assert log["info/episode_success_rate_cumulative"].item() == pytest.approx(0.5)
    assert log["info/episode_completed_count_window"].item() == pytest.approx(2.0)
    assert "episode_success_window=1.000 (2/2, last 200 policy steps)" in (
        env._additional_reward_print_fields()
    )
    env._reset_episode_success_statistics()


def test_table_collision_penalty_threshold(real_alignment_env, monkeypatch):
    env = real_alignment_env
    forces = torch.tensor([0.0, 1.0, 1.0001, 25.0], device=env.device)
    monkeypatch.setattr(env, "_compute_table_robot_contact_force", lambda: forces)
    penalty, logs = env._compute_additional_reward()
    torch.testing.assert_close(
        penalty,
        torch.tensor([0.0, 0.0, -10.0, -10.0], device=env.device),
    )
    assert env._last_table_collision.tolist() == [False, False, True, True]
    assert logs["info/table_collision_fraction"].item() == pytest.approx(0.5)
    assert logs["info/table_collision_max_force_n"].item() == pytest.approx(25.0)
    assert logs["info/dr_curriculum_scale"].item() == pytest.approx(env._dr_curriculum_scale())


def test_cube_pre_step_does_not_read_privileged_gate_geometry(real_alignment_env, monkeypatch):
    env = real_alignment_env

    def fail_if_called():
        raise AssertionError("disabled privileged dz gate read simulator geometry")

    monkeypatch.setattr(env, "_compute_action_gate_finger_tip_z", fail_if_called)
    monkeypatch.setattr(env, "_compute_action_gate_center_xy", fail_if_called)
    actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    actions[:, :3] = torch.tensor(
        [[1.0, -1.0, 0.5], [-0.5, 0.25, -1.0], [0.0, 0.0, 0.0], [2.0, -2.0, 1.0]],
        device=env.device,
    )
    env._pre_physics_step(actions)
    expected = torch.clamp(actions[:, :3], -1.0, 1.0) * 0.025
    torch.testing.assert_close(env.processed_actions[:, :3], expected)
    torch.testing.assert_close(env.action_history[:, :3], expected)


def test_total_width_delta_is_integrated_once_per_policy_step(real_alignment_env):
    env = real_alignment_env
    assert env.step_dt == pytest.approx(1.0 / 30.0)
    assert env.max_episode_length == 150
    initial_width = env._robot.data.default_joint_pos[:, env._finger_joint_ids].sum(dim=-1)
    torch.testing.assert_close(
        initial_width,
        torch.full_like(initial_width, 0.040001507848501206),
    )
    torch.testing.assert_close(env._desired_gripper_width, initial_width)

    actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    gripper_actions = torch.tensor([1.0, -0.5, 0.0, 0.25], device=env.device)
    actions[:, -1] = gripper_actions
    env._pre_physics_step(actions)

    expected_delta = 0.005 * gripper_actions
    expected_width = torch.clamp(initial_width + expected_delta, 0.0, env.cfg.max_gripper_opening_width)
    torch.testing.assert_close(env.processed_actions[:, -1], expected_delta)
    torch.testing.assert_close(env.action_history[:, -1], expected_delta)
    torch.testing.assert_close(env._desired_gripper_width, expected_width)

    env._apply_action()
    first_target = env._robot.data.joint_pos_target[:, env._finger_joint_ids].clone()
    width_after_first_apply = env._desired_gripper_width.clone()
    env._apply_action()
    second_target = env._robot.data.joint_pos_target[:, env._finger_joint_ids].clone()

    expected_finger_target = (0.5 * expected_width).unsqueeze(-1).expand_as(first_target)
    torch.testing.assert_close(first_target, expected_finger_target)
    torch.testing.assert_close(second_target, expected_finger_target)
    torch.testing.assert_close(env._desired_gripper_width, width_after_first_apply)

    target_before_nonfinite = env._desired_gripper_width.clone()
    actions.zero_()
    actions[:, -1] = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0],
        device=env.device,
    )
    env._pre_physics_step(actions)
    torch.testing.assert_close(env._desired_gripper_width, target_before_nonfinite)
    torch.testing.assert_close(env.action_history[:, -1], torch.zeros(env.num_envs, device=env.device))

    env._desired_gripper_width[:] = torch.tensor([0.079, 0.001, 0.08, 0.0], device=env.device)
    actions.zero_()
    actions[:, -1] = torch.tensor([1.0, -1.0, 1.0, -1.0], device=env.device)
    env._pre_physics_step(actions)
    torch.testing.assert_close(
        env._desired_gripper_width,
        torch.tensor([0.08, 0.0, 0.08, 0.0], device=env.device),
    )
    torch.testing.assert_close(
        env.action_history[:, -1],
        torch.tensor([0.005, -0.005, 0.005, -0.005], device=env.device),
    )

    env._desired_gripper_width[:] = torch.tensor([0.08, 0.06, 0.05, 0.03], device=env.device)
    env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
    torch.testing.assert_close(env._desired_gripper_width[0], initial_width[0])
    torch.testing.assert_close(env._desired_gripper_width[1], torch.tensor(0.06, device=env.device))
    torch.testing.assert_close(env._desired_gripper_width[2], torch.tensor(0.05, device=env.device))
    torch.testing.assert_close(env._desired_gripper_width[3], torch.tensor(0.03, device=env.device))


def test_dr_curriculum_partial_reset_and_gpu_postprocess(real_alignment_env):
    env = real_alignment_env
    assert env.num_envs == 4
    all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    original_step = int(env.common_step_counter)
    try:
        env.common_step_counter = 0
        assert env._dr_curriculum_scale() == pytest.approx(0.0)
        env.common_step_counter = 100_000
        assert env._dr_curriculum_scale() == pytest.approx(0.0)
        env.common_step_counter = 160_000
        assert env._dr_curriculum_scale() == pytest.approx(0.5)
        env.common_step_counter = 220_000
        assert env._dr_curriculum_scale() == pytest.approx(1.0)

        env.common_step_counter = 0
        env._randomize_scene_visuals(all_env_ids)
        assert env._dr_current_curriculum_scale == pytest.approx(0.0)
        torch.testing.assert_close(
            env._dr_camera_delta_pos,
            torch.zeros_like(env._dr_camera_delta_pos),
        )
        torch.testing.assert_close(
            env._dr_camera_delta_rpy_rad,
            torch.zeros_like(env._dr_camera_delta_rpy_rad),
        )
        torch.testing.assert_close(env._dr_focal_scale, torch.ones_like(env._dr_focal_scale))
        torch.testing.assert_close(
            env._dr_principal_shift_px,
            torch.zeros_like(env._dr_principal_shift_px),
        )
        torch.testing.assert_close(env._dr_brightness, torch.ones_like(env._dr_brightness))
        torch.testing.assert_close(env._dr_contrast, torch.ones_like(env._dr_contrast))
        torch.testing.assert_close(env._dr_saturation, torch.ones_like(env._dr_saturation))
        torch.testing.assert_close(env._dr_gamma, torch.ones_like(env._dr_gamma))
        torch.testing.assert_close(env._dr_hue_rad, torch.zeros_like(env._dr_hue_rad))
        torch.testing.assert_close(env._dr_white_balance, torch.ones_like(env._dr_white_balance))
        torch.testing.assert_close(env._dr_noise_std, torch.zeros_like(env._dr_noise_std))
        assert not env._dr_blur_mask.any()
        torch.testing.assert_close(
            env._dr_plate_colors,
            torch.tensor(env.cfg.plate_color_center, device=env.device).expand(env.num_envs, -1),
        )
        torch.testing.assert_close(
            env._dr_backdrop_colors,
            torch.tensor(env.cfg.backdrop_color_center, device=env.device).expand(env.num_envs, -1),
        )
        assert env._dr_global_light_intensity == pytest.approx(env.cfg.light_nominal_intensity)
        assert env._dr_global_light_color_temperature == pytest.approx(
            env.cfg.light_nominal_color_temperature
        )
        assert env._dr_global_light_color == pytest.approx(env.cfg.light_nominal_color)

        # With zero curriculum scale, the DR image path must reduce to the Clean
        # calibrated-K path rather than applying even a mild visual perturbation.
        probe = torch.rand((env.num_envs, 3, 224, 224), device=env.device)
        clean_equivalent = env._apply_nominal_camera_intrinsic_compensation(probe)
        dr_zero_scale = env._apply_wrist_visual_randomization(probe)
        torch.testing.assert_close(dr_zero_scale, clean_equivalent, rtol=1e-5, atol=1e-6)

        untouched_brightness = env._dr_brightness[1:].clone()
        untouched_camera_pos = env._dr_camera_pos_w[1:].clone()
        untouched_plate_colors = env._dr_plate_colors[1:].clone()
        global_light = (
            env._dr_global_light_intensity,
            env._dr_global_light_color_temperature,
            env._dr_global_light_color,
        )
        env._randomize_scene_visuals(torch.tensor([0], device=env.device))
        torch.testing.assert_close(env._dr_brightness[1:], untouched_brightness)
        torch.testing.assert_close(env._dr_camera_pos_w[1:], untouched_camera_pos)
        torch.testing.assert_close(env._dr_plate_colors[1:], untouched_plate_colors)
        assert (
            env._dr_global_light_intensity,
            env._dr_global_light_color_temperature,
            env._dr_global_light_color,
        ) == global_light

        # Exercise the complete batched GPU path with deterministic cached
        # parameters. A horizontal/vertical gradient makes focal/shift changes
        # observable while preserving an exact shape check.
        axis = torch.linspace(0.0, 1.0, 224, device=env.device)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        image = torch.stack([grid_x, grid_y, 0.5 * (grid_x + grid_y)], dim=0)
        images = image.unsqueeze(0).repeat(env.num_envs, 1, 1, 1)
        env._dr_current_curriculum_scale = 1.0
        env._dr_brightness[:] = 1.0
        env._dr_contrast[:] = 1.0
        env._dr_saturation[:] = 1.0
        env._dr_gamma[:] = 1.0
        env._dr_hue_rad[:] = 0.0
        env._dr_white_balance[:] = 1.0
        env._dr_noise_std[:] = 0.0
        env._dr_blur_mask[:] = False
        env._dr_focal_scale[:] = torch.tensor([1.0, 1.01, 0.99, 1.0], device=env.device)
        env._dr_principal_shift_px[:] = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 0.0]],
            device=env.device,
        )
        env._dr_brightness[3] = 0.9
        processed = env._apply_wrist_visual_randomization(images)
        assert processed.shape == images.shape
        assert torch.isfinite(processed).all()
        assert torch.all((processed >= 0.0) & (processed <= 1.0))
        assert not torch.allclose(processed[0], processed[1])
        assert not torch.allclose(processed[0], processed[3])
    finally:
        env.common_step_counter = original_step
        env._randomize_scene_visuals(all_env_ids)


def test_reach_center_uses_fingertip_midpoint_and_tcp_uses_0p1034(real_alignment_env):
    env = real_alignment_env
    assert env.cfg.arm_ik_tcp_source == "panda_hand_fixed_offset"
    assert env.cfg.arm_ik_tcp_offset_m == (0.0, 0.0, 0.1034)
    assert env.cfg.reach_center_source == "mean_of_left_and_right_fingertip_centers"
    assert env.cfg.fingertip_local_offset_m == (0.0, 0.0, 0.045)

    expected_tcp_offset = torch.tensor(
        env.cfg.arm_ik_tcp_offset_m,
        device=env.device,
        dtype=env._offset_pos.dtype,
    ).expand_as(env._offset_pos)
    torch.testing.assert_close(env._offset_pos, expected_tcp_offset)

    left_tip, right_tip = env._compute_fingertip_positions_world()
    midpoint = 0.5 * (left_tip + right_tip)
    torch.testing.assert_close(env._compute_reach_center_world(), midpoint)
    torch.testing.assert_close(env._compute_action_gate_center_xy(), midpoint[:, :2])
    torch.testing.assert_close(env._compute_action_gate_finger_tip_z(), midpoint[:, 2])

    hand_pos = env._robot.data.body_link_pos_w[:, env._body_idx]
    hand_quat = env._robot.data.body_link_quat_w[:, env._body_idx]
    tcp_pos, _ = math_utils.combine_frame_transforms(
        hand_pos,
        hand_quat,
        env._offset_pos,
        env._offset_rot,
    )
    torch.testing.assert_close(midpoint, tcp_pos, atol=2.0e-5, rtol=0.0)

    observations = env._get_observations()["policy"]
    target_pos = env._cube.data.root_pos_w - midpoint
    torch.testing.assert_close(observations["critic_gripper_pos"], midpoint)
    torch.testing.assert_close(observations["critic_target_pos"], target_pos)
    torch.testing.assert_close(
        observations["critic_target_distance"],
        torch.norm(target_pos, dim=-1, keepdim=True),
    )

    env._get_rewards()
    expected_reach_distance = torch.norm(target_pos, dim=-1).mean()
    torch.testing.assert_close(env.extras["log"]["info/reach_distance"], expected_reach_distance)


def test_reach_reward_peaks_when_cube_center_matches_fingertip_midpoint(real_alignment_env):
    """Evaluate the real reward path with the cube at and away from the grasp center."""
    env = real_alignment_env
    assert env.num_envs == 4
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    original_pose = env._cube.data.root_state_w[:, :7].clone()
    original_lift_weight = float(env.cfg.lift_weight)
    original_success_weight = float(env.cfg.success_reward_weight)
    original_collision_penalty = float(env.cfg.table_collision_penalty)
    try:
        left_tip, right_tip = env._compute_fingertip_positions_world()
        midpoint = 0.5 * (left_tip + right_tip)
        initial_reach_distance = torch.norm(original_pose[:, :3] - midpoint, dim=-1)
        initial_reach_reward = 1.0 - torch.tanh(
            initial_reach_distance / float(env.cfg.reach_sigma)
        )
        hand_pos = env._robot.data.body_link_pos_w[:, env._body_idx]
        hand_quat = env._robot.data.body_link_quat_w[:, env._body_idx]
        offsets = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.025, 0.0, 0.0],
                [0.0, 0.050, 0.0],
                [0.0, 0.0, 0.0],
            ],
            device=env.device,
            dtype=midpoint.dtype,
        )
        offsets[3] = hand_pos[3] - midpoint[3]
        diagnostic_pose = original_pose.clone()
        diagnostic_pose[:, :3] = midpoint + offsets
        env._cube.write_root_pose_to_sim(diagnostic_pose, env_ids=env_ids)

        # Isolate the reach component while still calling the environment's
        # production _get_rewards() implementation.
        env.cfg.lift_weight = 0.0
        env.cfg.success_reward_weight = 0.0
        env.cfg.table_collision_penalty = 0.0
        total_reward = env._get_rewards()
        reach_reward = total_reward / float(env.cfg.reach_weight)
        reach_distance = torch.norm(env._cube.data.root_pos_w - midpoint, dim=-1)

        expected_distance = torch.linalg.norm(offsets, dim=-1)
        expected_reward = 1.0 - torch.tanh(
            expected_distance / float(env.cfg.reach_sigma)
        )
        torch.testing.assert_close(reach_distance, expected_distance, atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(reach_reward, expected_reward, atol=1.0e-6, rtol=0.0)
        assert int(torch.argmax(reach_reward).item()) == 0
        assert reach_reward[0].item() == pytest.approx(1.0)
        assert torch.all(reach_reward[1:] < reach_reward[0])

        tcp_pos, _ = math_utils.combine_frame_transforms(
            hand_pos,
            hand_quat,
            env._offset_pos,
            env._offset_rot,
        )
        root_midpoint = midpoint - env.scene.env_origins
        root_tcp = tcp_pos - env.scene.env_origins
        print(
            "\n[reach geometry diagnostic]"
            f"\n  left_tip_root_m={left_tip[0] - env.scene.env_origins[0]}"
            f"\n  right_tip_root_m={right_tip[0] - env.scene.env_origins[0]}"
            f"\n  midpoint_root_m={root_midpoint[0]}"
            f"\n  tcp_root_m={root_tcp[0]}"
            f"\n  midpoint_tcp_error_m={torch.norm(midpoint - tcp_pos, dim=-1)}"
            f"\n  initial_cube_root_m={original_pose[:, :3] - env.scene.env_origins}"
            f"\n  initial_reach_distance_m={initial_reach_distance}"
            f"\n  initial_reach_reward={initial_reach_reward}"
            "\n  candidates=[fingertip_midpoint, +25mm_x, +50mm_y, panda_hand_origin]"
            f"\n  offsets_m={offsets}"
            f"\n  reach_distance_m={reach_distance}"
            f"\n  reach_reward={reach_reward}"
        )
    finally:
        env.cfg.lift_weight = original_lift_weight
        env.cfg.success_reward_weight = original_success_weight
        env.cfg.table_collision_penalty = original_collision_penalty
        env._cube.write_root_pose_to_sim(original_pose, env_ids=env_ids)
        env._reset_idx(env_ids)


def test_policy_contract_v9_and_legacy_compatibility(real_alignment_env, tmp_path):
    assert real_alignment_env.cfg.wrist_camera.height == 224
    assert real_alignment_env.cfg.wrist_camera.width == 224
    assert tuple(real_alignment_env.wrist_camera.data.output["rgb"].shape[1:3]) == (224, 224)

    for filename in ("agent.yaml", "agent.pkl", "env.yaml", "env.pkl"):
        (tmp_path / filename).write_bytes(f"test:{filename}".encode())

    contract = _build_policy_contract(
        real_alignment_env,
        tmp_path,
        DR_TASK_ID,
        "tanh(ACTIONS)",
    )
    assert contract["version"] == POLICY_CONTRACT_VERSION
    assert contract["action_scales"] == [0.025, 0.025, 0.025, 0.005]
    assert contract["deployment_history_scales"] == [0.025, 0.025, 0.025, 0.005]
    assert contract["gripper_control_mode"] == "total_width_delta_cached_target"
    assert contract["gripper_width_delta_scale"] == 0.005
    assert contract["gripper_width_bounds"] == [0.0, 0.08]
    assert contract["gripper_width_delta_updates_per_policy_step"] == 1
    assert contract["gripper_joint_target_mapping"] == "symmetric_half_total_width"
    assert contract["gripper_target_reapplications_per_policy_step"] == 2
    assert contract["sim_dt"] == pytest.approx(1.0 / 60.0)
    assert contract["decimation"] == 2
    assert contract["nominal_policy_frequency_hz"] == pytest.approx(30.0)
    assert contract["camera_update_period_s"] == pytest.approx(1.0 / 30.0)
    assert contract["nominal_camera_frequency_hz"] == pytest.approx(30.0)
    assert contract["episode_length_s"] == pytest.approx(5.0)
    assert contract["max_episode_length_steps"] == 150
    assert contract["camera_height"] == 224
    assert contract["camera_width"] == 224
    assert contract["action_history"] == ACTION_HISTORY_CONTRACT_V3
    assert contract["privileged_dz_gate"] == "disabled"
    assert contract["lift_reference_mode"] == "center_of_mass"
    assert contract["lift_reward_start_delta_m"] == pytest.approx(0.0)
    assert contract["success_lift_delta_m"] == pytest.approx(0.035)
    assert contract["lift_reward_requires_upright"] is False
    assert contract["success_requires_upright"] is False
    assert contract["lift_tilt_curriculum_enabled"] is False
    assert contract["arm_ik_tcp_source"] == "panda_hand_fixed_offset"
    assert contract["arm_ik_tcp_offset_m"] == [0.0, 0.0, 0.1034]
    assert contract["reach_center_source"] == "mean_of_left_and_right_fingertip_centers"
    assert contract["fingertip_local_offset_m"] == [0.0, 0.0, 0.045]
    assert contract["critic_gripper_position_source"] == contract["reach_center_source"]
    assert contract["action_gate_center_source"] == contract["reach_center_source"]
    assert contract["camera_pose_convention"] == "ros"
    assert contract["camera_crop_roi_xywh"] == [100, 34, 400, 398]
    assert contract["camera_model_intrinsic_matrix"] == pytest.approx(
        [
            338.742544,
            0.0,
            123.748857,
            0.0,
            340.550811,
            120.393372,
            0.0,
            0.0,
            1.0,
        ]
    )
    assert contract["camera_native_render_intrinsic_matrix"] == pytest.approx(
        [300.0, 0.0, 112.0, 0.0, 300.0, 112.0, 0.0, 0.0, 1.0]
    )
    assert contract["camera_nominal_intrinsic_compensation_enabled"] is True
    assert contract["deployment_camera_serial"] == "215322076207"
    assert contract["cube_nominal_xy_m"] == pytest.approx([0.5, 0.0])
    assert contract["cube_full_xy_bounds_m"] == pytest.approx([0.45, 0.55, -0.05, 0.05])
    assert contract["cube_position_curriculum_initial_xy_range_m"] == pytest.approx(
        [0.02, 0.02]
    )
    assert contract["cube_position_curriculum_start_step"] == 20_000
    assert contract["cube_position_curriculum_end_step"] == 100_000
    assert contract["plate_thickness_m"] == pytest.approx(0.001)
    assert contract["table_collision_force_threshold_n"] == pytest.approx(1.0)
    assert contract["table_collision_penalty"] == pytest.approx(-10.0)

    manifest = {"policy_contract": contract}
    validate_sim2real_policy_contract(manifest, task=DR_TASK_ID, params_dir=tmp_path)
    validate_live_env_against_policy_contract(real_alignment_env, contract)

    compatible_v8_contract = copy.deepcopy(contract)
    compatible_v8_contract["version"] = 8
    compatible_v8_contract["gripper_width_delta_scale"] = 0.002
    compatible_v8_contract["action_scales"] = [0.025, 0.025, 0.025, 0.002]
    compatible_v8_contract["deployment_history_scales"] = [0.025, 0.025, 0.025, 0.002]
    compatible_v8_contract["cube_full_xy_bounds_m"] = [0.4, 0.6, -0.1, 0.1]
    validate_sim2real_policy_contract(
        {"policy_contract": compatible_v8_contract},
        task=DR_TASK_ID,
        params_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="Live environment violates saved policy contract"):
        validate_live_env_against_policy_contract(real_alignment_env, compatible_v8_contract)

    compatible_v7_contract = copy.deepcopy(contract)
    compatible_v7_contract["version"] = 7
    compatible_v7_contract["action_scale"] = 0.01
    compatible_v7_contract["gripper_width_delta_scale"] = 0.002
    compatible_v7_contract["action_scales"] = [0.01, 0.01, 0.01, 0.002]
    compatible_v7_contract["deployment_history_scales"] = [0.01, 0.01, 0.01, 0.002]
    compatible_v7_contract["cube_full_xy_bounds_m"] = [0.4, 0.6, -0.1, 0.1]
    validate_sim2real_policy_contract(
        {"policy_contract": compatible_v7_contract},
        task=DR_TASK_ID,
        params_dir=tmp_path,
    )

    obsolete_v5_action_contract = copy.deepcopy(contract)
    obsolete_v5_action_contract["version"] = 5
    obsolete_v5_action_contract["action_scales"] = [0.005, 0.005, 0.005, 0.002]
    obsolete_v5_action_contract["deployment_history_scales"] = [
        0.005,
        0.005,
        0.005,
        0.002,
    ]
    with pytest.raises(RuntimeError, match="obsolete 5 mm XYZ action scale"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_v5_action_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    obsolete_v6_crop_contract = copy.deepcopy(contract)
    obsolete_v6_crop_contract["version"] = 6
    obsolete_v6_crop_contract["camera_crop_roi_xywh"] = [80, 0, 480, 480]
    with pytest.raises(RuntimeError, match="obsolete 480x480 center crop"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_v6_crop_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    obsolete_v2_lift_contract = copy.deepcopy(contract)
    obsolete_v2_lift_contract["version"] = 2
    obsolete_v2_lift_contract["action_history"] = ACTION_HISTORY_CONTRACT_V2
    with pytest.raises(RuntimeError, match="obsolete lowest-corner/non-curriculum lift semantics"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_v2_lift_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    obsolete_v3_gate_contract = copy.deepcopy(contract)
    obsolete_v3_gate_contract.update(
        {
            "version": 3,
            "action_history": ACTION_HISTORY_CONTRACT_V2,
            "privileged_dz_gate": "ground_truth_object_xy_applied_after_action_history",
        }
    )
    with pytest.raises(RuntimeError, match="obsolete ground-truth object-XY dz gate"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_v3_gate_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    obsolete_v4_alignment_contract = copy.deepcopy(contract)
    obsolete_v4_alignment_contract["version"] = 4
    with pytest.raises(RuntimeError, match="contract v4 predates"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_v4_alignment_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    obsolete_lift_contract = copy.deepcopy(contract)
    obsolete_lift_contract.pop("lift_reference_mode")
    with pytest.raises(RuntimeError, match="obsolete lift-reward semantics"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_lift_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    obsolete_center_contract = copy.deepcopy(contract)
    obsolete_center_contract.pop("reach_center_source")
    with pytest.raises(RuntimeError, match="obsolete grasp-center semantics"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_center_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    obsolete_timing_contract = copy.deepcopy(contract)
    obsolete_timing_contract.update(
        {
            "decimation": 1,
            "nominal_policy_frequency_hz": 60.0,
            "episode_length_s": 2.5,
        }
    )
    with pytest.raises(RuntimeError, match="obsolete timing semantics"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_timing_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    obsolete_alignment_contract = copy.deepcopy(contract)
    obsolete_alignment_contract.update(
        {
            "version": 1,
            "action_history": ACTION_HISTORY_CONTRACT_V1,
            "action_scale": 0.005,
            "deployment_history_scale": 0.005,
        }
    )
    with pytest.raises(RuntimeError, match="obsolete per-finger gripper increment semantics"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_alignment_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    obsolete_dr_contract = copy.deepcopy(obsolete_alignment_contract)
    obsolete_dr_contract["task"] = DR_TASK_ID
    with pytest.raises(RuntimeError, match="obsolete per-finger gripper increment semantics"):
        validate_sim2real_policy_contract(
            {"policy_contract": obsolete_dr_contract},
            task=DR_TASK_ID,
            params_dir=tmp_path,
        )

    legacy_cube_contract = copy.deepcopy(obsolete_alignment_contract)
    legacy_cube_contract["task"] = LEGACY_CUBE_TASK_ID
    with pytest.raises(RuntimeError, match="obsolete action-history and privileged dz-gate semantics"):
        validate_sim2real_policy_contract(
            {"policy_contract": legacy_cube_contract},
            task=LEGACY_CUBE_TASK_ID,
            params_dir=tmp_path,
        )

    legacy_non_cube_contract = copy.deepcopy(obsolete_alignment_contract)
    legacy_non_cube_contract["task"] = "TacEx-Sim2Real-Grasp-v0"
    validated = validate_sim2real_policy_contract(
        {"policy_contract": legacy_non_cube_contract},
        task="TacEx-Sim2Real-Grasp-v0",
        params_dir=tmp_path,
    )
    assert validated["version"] == 1
