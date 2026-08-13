"""Runtime checks for the X040-Wide 4--6 cm size-bucket Student task."""

from __future__ import annotations

import pytest
import torch
from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from tacex_tasks.sim2real_grasp.rma_x040_wide_artifacts import (  # noqa: E402
    RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK,
)
from tacex_tasks.sim2real_grasp.sim2real_cube_real_alignment_rma_x040_wide_size_buckets_env import (  # noqa: E402
    _CubeSizeBucketMixin,
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnvCfg,
)


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


def test_size_bucket_student_switches_one_physical_target_per_reset():
    cfg = parse_env_cfg(
        RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK,
        device="cuda:0",
        num_envs=4,
    )
    cfg.seed = 42
    env = gym.make(RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK, cfg=cfg)
    try:
        observations, _ = env.reset()
        base = env.unwrapped
        seen_bucket_ids = set(base._active_cube_bucket_ids.detach().cpu().tolist())
        sampled_arm_noise = [base._last_arm_joint_reset_noise_rad.detach().clone()]
        for _ in range(4):
            observations, _ = env.reset()
            seen_bucket_ids.update(base._active_cube_bucket_ids.detach().cpu().tolist())
            sampled_arm_noise.append(base._last_arm_joint_reset_noise_rad.detach().clone())

        assert len(seen_bucket_ids) > 1
        assert base.rma_cube_contact_sensor.num_bodies == 1
        assert base._rma_contact_force_history().shape == (4, 2, 1, 2, 3)
        assert torch.all((base.active_cube_size_m >= 0.04) & (base.active_cube_size_m <= 0.06))

        all_arm_noise = torch.stack(sampled_arm_noise)
        assert torch.any(all_arm_noise != 0.0)
        assert torch.all(all_arm_noise.abs() <= 0.03 + 1.0e-6)
        assert not torch.equal(all_arm_noise[0], all_arm_noise[-1])
        collection_pos = base._cube_size_bucket_collection.data.object_pos_w
        max_camera_world_x = (
            base.scene.env_origins[:, 0].max()
            + float(cfg.wrist_camera.offset.pos[0])
            + abs(float(cfg.camera_position_delta_max_m[0]))
        ).to(device=collection_pos.device, dtype=collection_pos.dtype)
        rows = torch.arange(base.num_envs, device=base.device)
        inactive_mask = torch.ones(
            (base.num_envs, len(cfg.cube_size_buckets_m)),
            device=base.device,
            dtype=torch.bool,
        )
        inactive_mask[rows, base._active_cube_bucket_ids] = False
        assert torch.all(
            collection_pos[..., 0][inactive_mask]
            >= max_camera_world_x + float(cfg.cube_bucket_parking_camera_margin_m) - 1.0e-5
        )
        torch.testing.assert_close(
            base._robot.data.joint_pos[:, :7],
            base._robot.data.default_joint_pos[:, :7] + sampled_arm_noise[-1],
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            base._robot.data.joint_pos[:, 7:],
            base._robot.data.default_joint_pos[:, 7:],
            atol=1.0e-6,
            rtol=0.0,
        )

        selected_pos = collection_pos[rows, base._active_cube_bucket_ids]
        torch.testing.assert_close(selected_pos, base._cube.data.root_pos_w)
        expected_center_z = float(cfg.plate_top_height_m) + 0.5 * base.active_cube_size_m
        torch.testing.assert_close(
            observations["policy"]["rma_cube_pos"][:, 2],
            expected_center_z,
            atol=1.0e-5,
            rtol=0.0,
        )

        _, rewards, terminated, truncated, _ = env.step(
            torch.zeros((4, 4), device=base.device)
        )
        assert torch.isfinite(rewards).all()
        assert terminated.shape == truncated.shape == (4,)
        assert "info/cube_size_mean_m" in base.extras["log"]
        assert "info/arm_joint_reset_noise_abs_mean_rad" in base.extras["log"]
    finally:
        env.close()


def test_size_bucket_reset_disables_gpu_dynamics_before_scene_creation():
    """Guard the Direct-GPU-incompatible reset path for the eight dynamic cubes."""
    import inspect

    cfg = Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnvCfg()
    source = inspect.getsource(_CubeSizeBucketMixin._setup_scene)
    assert cfg.enable_gpu_dynamics is False
    assert source.index("self._configure_size_bucket_physics()") < source.index("super()._setup_scene()")
