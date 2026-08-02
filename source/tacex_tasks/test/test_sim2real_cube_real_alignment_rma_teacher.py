"""No-camera environment smoke test for the RMA Teacher."""

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
from tacex_tasks.sim2real_grasp.rma_models import RMAActorCore


TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0"


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


def test_rma_teacher_observation_and_step_contract():
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=2)
    env = gym.make(TASK, cfg=cfg)
    try:
        observations, _ = env.reset()
        obs = observations["policy"]
        assert "wrist_camera" not in env.unwrapped.scene.sensors
        assert obs["proprio_obs"].shape == (2, 15)
        assert obs["action_history"].shape == (2, 4)
        assert obs["rma_cube_pos"].shape == (2, 3)
        assert obs["rma_contact_state"].shape == (2, 2)
        assert torch.all((obs["rma_contact_state"] == 0.0) | (obs["rma_contact_state"] == 1.0))
        assert "rma_cube_contact_sensor" in env.unwrapped.scene.sensors
        assert env.unwrapped.rma_cube_contact_sensor.data.force_matrix_w_history.shape == (
            2,
            2,
            1,
            2,
            3,
        )
        assert "privileged_gripper_pos" not in obs
        assert "privileged_target_pos" not in obs
        actor = RMAActorCore().to(env.unwrapped.device)
        fk_gripper_position = actor.kinematics(obs["proprio_obs"][:, :7])
        simulated_gripper_position = (
            env.unwrapped._compute_reach_center_world() - env.unwrapped.scene.env_origins
        )
        torch.testing.assert_close(
            fk_gripper_position,
            simulated_gripper_position,
            atol=2.0e-4,
            rtol=0.0,
        )
        actions = torch.zeros((2, 4), device=env.unwrapped.device)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        assert next_observations["policy"]["rma_cube_pos"].shape == (2, 3)
        assert next_observations["policy"]["rma_contact_state"].shape == (2, 2)
        assert rewards.shape == terminated.shape == truncated.shape == (2,)
        assert torch.isfinite(rewards).all()

        base_env = env.unwrapped
        base_env._ensure_cube_lift_reference_height()
        lifted_state = base_env._cube.data.root_state_w.clone()
        lifted_state[:, :3] = base_env._cube.data.root_pos_w
        lifted_state[:, 2] = (
            base_env._cube_lift_reference_height_per_env
            + cfg.success_lift_delta
            + 0.01
        )
        lifted_state[:, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=base_env.device
        )
        lifted_state[:, 7:] = 0.0
        base_env._cube.write_root_state_to_sim(lifted_state)
        for _ in range(cfg.success_hold_steps):
            dones, _ = base_env._get_dones()
        assert torch.all(
            base_env._success_hold_counter >= cfg.success_hold_steps
        )
        assert torch.all(~dones)
        assert torch.all(base_env._last_rma_success_nonterminal)
        assert torch.all(base_env._rma_episode_success_ever)

        cube_state = base_env._cube.data.root_state_w.clone()
        cube_state[:, :3] = base_env._compute_reach_center_world()
        cube_state[:, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=base_env.device
        )
        cube_state[:, 7:] = 0.0
        base_env._cube.write_root_state_to_sim(cube_state)

        max_forces = torch.zeros((2, 2), device=base_env.device)
        contact_state = torch.zeros((2, 2), device=base_env.device)
        max_contact_reward = torch.zeros((2,), device=base_env.device)
        for _ in range(4):
            observations, _, _, _, _ = env.step(actions)
            max_forces = torch.maximum(max_forces, base_env._last_rma_contact_forces)
            contact_state = torch.maximum(
                contact_state, observations["policy"]["rma_contact_state"]
            )
            max_contact_reward = torch.maximum(
                max_contact_reward, base_env._last_rma_contact_reward
            )

        assert torch.all(max_forces >= cfg.rma_contact_force_threshold_n)
        torch.testing.assert_close(contact_state, torch.ones_like(contact_state))
        torch.testing.assert_close(
            max_contact_reward,
            torch.full((2,), cfg.rma_contact_reward_weight, device=base_env.device),
        )
    finally:
        env.close()
