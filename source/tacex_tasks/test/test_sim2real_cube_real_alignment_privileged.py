"""Smoke checks for the no-camera privileged Real-Alignment Cube task."""

from __future__ import annotations

import sys

if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app


import gymnasium as gym
import pytest
import torch

import tacex_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


TASK_ID = "TacEx-Sim2Real-Cube-Real-Alignment-Privileged-v0"


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


def test_privileged_task_has_no_camera_and_consistent_position_observations():
    env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=2)
    assert env_cfg.camera_sensor_enabled is False
    assert env_cfg.vision_encoder_enabled is False
    assert "wrist_resnet" not in env_cfg.observation_space
    assert env_cfg.gripper_width_delta_scale == pytest.approx(0.002)
    assert env_cfg.cube_x_pos_range == pytest.approx(0.10)
    assert env_cfg.cube_y_pos_range == pytest.approx(0.10)

    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        observations, _ = env.reset()
        base_env = env.unwrapped
        policy_obs = observations["policy"]

        assert "wrist_camera" not in base_env.scene.sensors
        assert base_env._use_resnet18 is False
        assert not hasattr(base_env, "_resnet18")
        assert policy_obs["proprio_obs"].shape == (2, 15)
        assert policy_obs["action_history"].shape == (2, 4)
        assert policy_obs["privileged_cube_pos"].shape == (2, 3)
        assert policy_obs["privileged_gripper_pos"].shape == (2, 3)
        assert policy_obs["privileged_target_pos"].shape == (2, 3)

        torch.testing.assert_close(
            policy_obs["privileged_target_pos"],
            policy_obs["privileged_cube_pos"] - policy_obs["privileged_gripper_pos"],
        )
        torch.testing.assert_close(
            policy_obs["privileged_cube_pos"],
            base_env._cube.data.root_pos_w - base_env.scene.env_origins,
        )

        actions = torch.zeros((2, env_cfg.action_space), device=base_env.device)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        assert next_observations["policy"]["privileged_cube_pos"].shape == (2, 3)
        assert rewards.shape == terminated.shape == truncated.shape == (2,)
        assert torch.isfinite(rewards).all()
    finally:
        env.close()
