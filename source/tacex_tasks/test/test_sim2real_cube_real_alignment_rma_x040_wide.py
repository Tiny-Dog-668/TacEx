"""Contract tests for the independent X040-Wide XYZ/no-contact RMA profile."""

from __future__ import annotations

import colorsys
import pytest
import torch
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils  # noqa: E402
from pxr import UsdShade  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from tacex_tasks.sim2real_grasp.rma_x040_wide_artifacts import (  # noqa: E402
    RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK,
    RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK,
    RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK,
    RMA_X040_WIDE_STUDENT_KIND,
    RMA_X040_WIDE_STUDENT_VERSION,
    direct_action_input_contract,
    load_student_checkpoint,
    state_dict_sha256,
    student_environment_contract,
)
from tacex_tasks.sim2real_grasp.rma_x040_wide_models import (  # noqa: E402
    RMA_X040_WIDE_DIRECT_STUDENT_MODEL_VERSION,
    RMAX040WideActorCore,
    RMAX040WideDirectActionVisualStudent,
)


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


def test_x040_wide_model_and_torchscript_keep_three_runtime_inputs():
    model = RMAX040WideDirectActionVisualStudent().train()
    inputs = (
        torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8),
        torch.zeros((2, 15), dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float32),
    )
    inputs[1][:, -1] = 0.04
    actions, position = model.forward_with_position(*inputs)
    assert actions.shape == (2, 4)
    assert position.shape == (2, 3)
    assert torch.all(actions.abs() <= 1.0)
    (actions.square().mean() + position.square().mean()).backward()
    assert all(parameter.grad is None for index, module in enumerate(model.vision_encoder) if index < 6 for parameter in module.parameters())
    assert any(parameter.grad is not None for index, module in enumerate(model.vision_encoder) if index >= 6 for parameter in module.parameters())
    scripted = torch.jit.script(model.eval())
    with torch.inference_mode():
        torch.testing.assert_close(model(*inputs), scripted(*inputs))
        assert scripted(*(value[:1] for value in inputs)).shape == (1, 4)


def test_x040_wide_teacher_has_28_features_and_no_contact_argument():
    teacher = RMAX040WideActorCore()
    proprio = torch.zeros((2, 15)); proprio[:, -1] = 0.04
    actions = teacher(proprio, torch.zeros((2, 4)), torch.tensor([[0.4, 0.0, 0.026], [0.48, 0.1, 0.026]]))
    assert actions.shape == (2, 4)
    assert teacher.contract()["feature_dim"] == 28
    assert teacher.contract()["contact_components"] == "none"


def test_x040_wide_artifact_rejects_old_direct_student(tmp_path):
    old = tmp_path / "old.pt"
    torch.save({"kind": "tacex_rma_direct_action_student", "version": 1}, old)
    with pytest.raises(RuntimeError, match="not an X040-Wide"):
        load_student_checkpoint(old)
    model = RMAX040WideDirectActionVisualStudent()
    payload = {
        "kind": RMA_X040_WIDE_STUDENT_KIND, "version": RMA_X040_WIDE_STUDENT_VERSION,
        "model_version": RMA_X040_WIDE_DIRECT_STUDENT_MODEL_VERSION,
        "task": RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK, "model": model.state_dict(),
        "student_input_contract": direct_action_input_contract(), "student_model_contract": model.contract(),
        "normalization": model.normalizer.contract(), "encoder_init_checkpoint_sha256": "a" * 64,
        "encoder_init_state_dict_sha256": "b" * 64,
        "vision_encoder_state_dict_sha256": state_dict_sha256(model.vision_encoder.state_dict()),
    }
    torch.save(payload, old)
    assert load_student_checkpoint(old)["task"] == RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK
    payload["task"] = RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK
    torch.save(payload, old)
    with pytest.raises(RuntimeError, match="Size-Buckets Student rendering contract mismatch"):
        load_student_checkpoint(old)
    payload["task"] = RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK
    payload["version"] = RMA_X040_WIDE_STUDENT_VERSION - 1
    torch.save(payload, old)
    with pytest.raises(RuntimeError, match="Unsupported X040-Wide direct-action Student version"):
        load_student_checkpoint(old)


def test_x040_wide_task_configs_isolate_contact_and_set_requested_ranges():
    teacher_task = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Teacher-v0"
    assert gym.spec(teacher_task) is not None
    assert gym.spec(RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK) is not None
    teacher = parse_env_cfg(teacher_task, device="cuda:0", num_envs=1)
    student = parse_env_cfg(RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK, device="cuda:0", num_envs=1)
    for cfg in (teacher, student):
        assert cfg.cube.init_state.pos == pytest.approx((0.4, 0.0, 0.026))
        assert cfg.cube_x_pos_range == pytest.approx(0.08)
        assert cfg.cube_y_pos_range == pytest.approx(0.10)
        assert cfg.cube_position_curriculum_enabled is False
        assert cfg.tcp_table_clearance_min_z_m == pytest.approx(0.011)
        assert cfg.tcp_table_clearance_penalty == pytest.approx(-10.0)
        assert "rma_contact_state" not in cfg.observation_space
        assert "rma_contact_force" not in cfg.observation_space
    assert student.camera_position_delta_max_m == pytest.approx((0.01, 0.01, 0.01))
    assert student.camera_rotation_delta_max_deg == pytest.approx((2.0, 2.0, 2.0))
    assert student.student_base_led_subset_path == "panda_link0/visuals/panda_link0/subset_5"
    assert student.student_base_led_emissive_color == pytest.approx((0.0, 1.0, 0.0))
    assert student.student_base_led_hue_deg_range == pytest.approx((105.0, 135.0))
    assert student.student_base_led_saturation_range == pytest.approx((0.75, 1.0))
    assert student.student_base_led_value_range == pytest.approx((0.35, 1.0))
    assert student_environment_contract(student)["student_base_led"] == {
        "subset_path": "panda_link0/visuals/panda_link0/subset_5",
        "nominal_emissive_color": [0.0, 1.0, 0.0],
        "sampling": "per_environment_per_episode_hsv",
        "hue_deg_range": [105.0, 135.0],
        "saturation_range": [0.75, 1.0],
        "value_range": [0.35, 1.0],
    }


def test_x040_wide_size_bucket_configs_are_separate_and_contract_bound():
    assert gym.spec(RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK) is not None
    assert gym.spec(RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK) is not None
    teacher = parse_env_cfg(
        RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK,
        device="cuda:0",
        num_envs=1,
    )
    student = parse_env_cfg(
        RMA_X040_WIDE_SIZE_BUCKETS_DIRECT_STUDENT_DR_TASK,
        device="cuda:0",
        num_envs=1,
    )
    expected_sizes = np.linspace(0.04, 0.06, 8)
    for cfg in (teacher, student):
        assert cfg.cube_size_buckets_m == pytest.approx(expected_sizes)
        assert cfg.cube_size_sampling == "uniform_discrete_per_environment_per_reset"
        assert cfg.cube_bucket_parking_strategy == "beyond_positive_x_env_extent_behind_all_cameras"
        assert cfg.cube_bucket_parking_camera_margin_m == pytest.approx(0.5)
        assert cfg.arm_joint_reset_noise_distribution == "normal_clipped_per_environment_per_reset"
        assert cfg.arm_joint_reset_noise_std_rad == pytest.approx(0.01)
        assert cfg.arm_joint_reset_noise_clip_rad == pytest.approx(0.03)
        assert len(cfg.cube_bucket_parking_xy_m) == 8
        assert cfg.rma_cube_contact_sensor.filter_prim_paths_expr == [
            "/World/envs/env_.*/Robot/panda_leftfinger",
            "/World/envs/env_.*/Robot/panda_rightfinger",
        ]
        assert cfg.action_space == 4
        assert cfg.rma_actor_feature_dim == 28
        assert "rma_contact_state" not in cfg.observation_space
    contract = student_environment_contract(student)
    assert contract["profile"] == "rma_x040_wide_xyz_no_contact_size_buckets_v2"
    assert contract["cube_size_buckets_m"] == pytest.approx(expected_sizes)
    assert contract["cube_center_z_buckets_root_m"] == pytest.approx(
        0.001 + 0.5 * expected_sizes
    )
    assert contract["arm_joint_reset_noise"] == {
        "joint_count": 7,
        "distribution": "normal_clipped_per_environment_per_reset",
        "mean_rad": 0.0,
        "std_rad": 0.01,
        "clip_abs_rad": 0.03,
        "finger_joint_noise": "none",
        "joint_velocity_reset_rad_s": 0.0,
    }
    assert contract["cube_bucket_rendering"] == {
        "selection": "selected_bucket_per_environment_per_reset",
        "inactive_bucket_exclusion": "parked_behind_all_cameras_using_env_x_extent",
        "usd_visibility_mutation": "none",
        "parking_strategy": "beyond_positive_x_env_extent_behind_all_cameras",
        "camera_margin_m": 0.5,
    }


def test_x040_wide_student_runtime_exposes_only_xyz_training_label():
    cfg = parse_env_cfg(RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK, device="cuda:0", num_envs=2)
    env = gym.make(RMA_X040_WIDE_DIRECT_STUDENT_DR_TASK, cfg=cfg)
    try:
        observations, _ = env.reset()
        obs = observations["policy"]
        assert set(("proprio_obs", "action_history", "wrist_rgb", "rma_cube_pos")).issubset(obs)
        assert "rma_contact_state" not in obs
        assert "rma_contact_force" not in obs
        observations, rewards, terminated, truncated, _ = env.step(
            torch.zeros((2, 4), device=env.unwrapped.device)
        )
        assert torch.isfinite(rewards).all()
        assert terminated.shape == truncated.shape == (2,)
        log = env.unwrapped.extras["log"]
        assert "reward/tcp_table_clearance" in log
        assert "info/tcp_table_clearance_violation_fraction" in log
        stage = sim_utils.stage_utils.get_current_stage()
        base_led_paths = [
            f"/World/envs/env_{env_id}/Robot/panda_link0/visuals/panda_link0/subset_5"
            for env_id in range(2)
        ]
        base_led_prims = [stage.GetPrimAtPath(path) for path in base_led_paths]
        wrist_led = stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/panda_link6/visuals/panda_link6/subset_5"
        )
        assert all(prim.IsValid() and not prim.IsInstanceProxy() for prim in base_led_prims)
        assert wrist_led.IsValid()
        assert wrist_led.IsInstanceProxy()
        base_materials = [
            UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            for prim in base_led_prims
        ]
        wrist_material, _ = UsdShade.MaterialBindingAPI(wrist_led).ComputeBoundMaterial()
        assert [material.GetPath().pathString for material in base_materials] == [
            f"/World/Looks/X040WideStudentBaseVerticalLed_env_{env_id}"
            for env_id in range(2)
        ]
        assert wrist_material.GetPath().pathString.endswith("/Looks/EmissiveBlue")
        sampled_rgb = []
        for material in base_materials:
            shader = stage.GetPrimAtPath(f"{material.GetPath()}/Shader")
            emissive = tuple(shader.GetAttribute("inputs:emissiveColor").Get())
            diffuse = tuple(shader.GetAttribute("inputs:diffuseColor").Get())
            assert diffuse == pytest.approx(emissive)
            hue, saturation, value = colorsys.rgb_to_hsv(*emissive)
            assert 105.0 <= hue * 360.0 <= 135.0
            assert 0.75 <= saturation <= 1.0
            assert 0.35 <= value <= 1.0
            sampled_rgb.append(emissive)
        assert sampled_rgb[0] != pytest.approx(sampled_rgb[1])

        env.unwrapped._randomize_scene_visuals(
            torch.tensor([0], device=env.unwrapped.device, dtype=torch.long)
        )
        resampled_rgb = [
            tuple(
                stage.GetPrimAtPath(f"{material.GetPath()}/Shader")
                .GetAttribute("inputs:emissiveColor")
                .Get()
            )
            for material in base_materials
        ]
        assert resampled_rgb[0] != pytest.approx(sampled_rgb[0])
        assert resampled_rgb[1] == pytest.approx(sampled_rgb[1])
    finally:
        env.close()


def test_x040_wide_camera_extremes_keep_every_cube_center_in_view():
    """Projection check for all reset/pose/intrinsic extreme combinations.

    Full 5 cm cube visibility is intentionally not required at every extreme:
    the approved profile clips only a small lower border subset. The deployable
    target center must nevertheless remain in the calibrated 224x224 image.
    """
    fx, fy, cx, cy = 338.742544, 340.550811, 123.748857, 120.393372
    camera_position = np.array([1.166091088407, 0.035901608197, 0.534200335898])
    camera_rotation = Rotation.from_quat(
        [-0.604227000834, -0.586824979121, 0.384024117374, 0.378248136306]
    )
    center_visible = 0
    fully_visible = 0
    corners = list(itertools.product((-0.025, 0.025), repeat=3))
    for (x, y), delta_pos, delta_rpy, focal, principal in itertools.product(
        itertools.product((0.32, 0.48), (-0.10, 0.10)),
        itertools.product((-0.01, 0.01), repeat=3),
        itertools.product((-2.0, 2.0), repeat=3),
        (0.99, 1.01),
        itertools.product((-1.0, 1.0), repeat=2),
    ):
        rotation = camera_rotation * Rotation.from_euler("xyz", delta_rpy, degrees=True)
        position = camera_position + camera_rotation.apply(np.asarray(delta_pos))
        center_camera = rotation.inv().apply(np.array([x, y, 0.026]) - position)
        u = fx * focal * center_camera[0] / center_camera[2] + cx + principal[0]
        v = fy * focal * center_camera[1] / center_camera[2] + cy + principal[1]
        center_visible += int(center_camera[2] > 0.0 and 0.0 <= u < 224.0 and 0.0 <= v < 224.0)
        all_corners_visible = True
        for corner in corners:
            point = rotation.inv().apply(np.array([x, y, 0.026]) + corner - position)
            u = fx * focal * point[0] / point[2] + cx + principal[0]
            v = fy * focal * point[1] / point[2] + cy + principal[1]
            all_corners_visible &= point[2] > 0.0 and 0.0 <= u < 224.0 and 0.0 <= v < 224.0
        fully_visible += int(all_corners_visible)
    assert center_visible == 2048
    assert fully_visible == 1932
