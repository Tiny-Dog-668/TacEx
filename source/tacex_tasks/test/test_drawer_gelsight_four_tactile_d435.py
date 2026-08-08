"""Contract checks for the D435-calibrated four-GelSight drawer task."""

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

import tacex_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tacex_tasks.drawer_gelsight_four_tactile import (
    DRAWER_GELSIGHT_FOUR_TACTILE_D435_TASK,
)
from tacex_tasks.drawer_gelsight_four_tactile.drawer_gelsight_four_tactile_d435_env import (
    DrawerGelSightFourTactileD435Env,
)
from tacex_tasks.third_view_grasping.vt_box import OccludedGraspingVisionFourTactileBoxEnv


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


def test_task_registration_and_observation_contract():
    spec = gym.spec(DRAWER_GELSIGHT_FOUR_TACTILE_D435_TASK)
    assert spec.entry_point.endswith("DrawerGelSightFourTactileD435Env")

    cfg = parse_env_cfg(DRAWER_GELSIGHT_FOUR_TACTILE_D435_TASK, device="cuda:0", num_envs=4)
    assert cfg.action_space == 5
    assert cfg.observation_space["third_resnet"] == 512
    for key in (
        "tactile_left_depth_resnet",
        "tactile_right_depth_resnet",
        "tactile_left_down_depth_resnet",
        "tactile_right_down_depth_resnet",
    ):
        assert cfg.observation_space[key] == 256
    assert cfg.gsmini_left.prim_path.endswith("gelsight_mini_case_left")
    assert cfg.gsmini_right.prim_path.endswith("gelsight_mini_case_right")
    assert cfg.gsmini_left_down.prim_path.endswith("gelsight_mini_case_left_down")
    assert cfg.gsmini_right_down.prim_path.endswith("gelsight_mini_case_right_down")


def test_d435_camera_contract_and_intrinsic_compensation():
    cfg = parse_env_cfg(DRAWER_GELSIGHT_FOUR_TACTILE_D435_TASK, device="cuda:0", num_envs=1)
    assert (cfg.third_person_camera.width, cfg.third_person_camera.height) == (224, 224)
    assert cfg.third_person_camera.offset.convention == "ros"
    assert cfg.third_person_camera.offset.pos == pytest.approx(
        (1.166091088407, 0.035901608197, 0.534200335898)
    )
    assert cfg.third_person_camera.offset.rot == pytest.approx(
        (0.378248136306, -0.604227000834, -0.586824979121, 0.384024117374)
    )
    assert cfg.camera_raw_resolution == (640, 480)
    assert cfg.camera_crop_roi_xywh == (100, 34, 400, 398)
    assert cfg.camera_model_intrinsic_matrix == pytest.approx(
        (338.742544, 0.0, 123.748857, 0.0, 340.550811, 120.393372, 0.0, 0.0, 1.0)
    )
    assert cfg.camera_native_render_intrinsic_matrix == pytest.approx(
        (300.0, 0.0, 112.0, 0.0, 300.0, 112.0, 0.0, 0.0, 1.0)
    )

    env = object.__new__(DrawerGelSightFourTactileD435Env)
    env.cfg = cfg
    env._d435_intrinsic_grid = None
    env._d435_intrinsic_grid_key = None
    image = torch.zeros((1, 1, 224, 224), device="cuda:0")
    image[0, 0, round(cfg.camera_native_render_intrinsic_matrix[5]), round(cfg.camera_native_render_intrinsic_matrix[2])] = 1.0

    warped = env._preprocess_third_person_rgb(image)
    peak_y, peak_x = divmod(int(torch.argmax(warped).item()), 224)
    assert (peak_x, peak_y) == (
        round(cfg.camera_model_intrinsic_matrix[2]),
        round(cfg.camera_model_intrinsic_matrix[5]),
    )
    assert torch.isfinite(warped).all()


def test_four_env_camera_smoke_preserves_policy_feature_shapes():
    cfg = parse_env_cfg(DRAWER_GELSIGHT_FOUR_TACTILE_D435_TASK, device="cuda:0", num_envs=4)
    env = gym.make(DRAWER_GELSIGHT_FOUR_TACTILE_D435_TASK, cfg=cfg)
    try:
        observations, _ = env.reset()
        policy = observations["policy"]
        assert policy["third_resnet"].shape == (4, 512)
        for key in (
            "tactile_left_depth_resnet",
            "tactile_right_depth_resnet",
            "tactile_left_down_depth_resnet",
            "tactile_right_down_depth_resnet",
        ):
            assert policy[key].shape == (4, 256)

        actions = torch.zeros((4, cfg.action_space), device=env.unwrapped.device)
        next_observations, _, _, _, _ = env.step(actions)
        assert next_observations["policy"]["third_resnet"].shape == (4, 512)
    finally:
        env.close()


def test_original_third_view_preprocess_hook_is_identity():
    env = object.__new__(OccludedGraspingVisionFourTactileBoxEnv)
    image = torch.rand((2, 3, 96, 128))
    assert env._preprocess_third_person_rgb(image) is image
