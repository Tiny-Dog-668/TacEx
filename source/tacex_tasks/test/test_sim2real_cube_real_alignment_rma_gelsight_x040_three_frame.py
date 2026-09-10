"""Contracts for the GelSight X040-DR three-frame Teacher/Student route."""

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
import isaaclab.sim as sim_utils
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from pxr import Usd, UsdGeom, UsdShade

import tacex_tasks  # noqa: F401
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_USD,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_three_frame_artifacts import (
    GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK,
    GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
    LEGACY_STUDENT_CHECKPOINT_VERSION,
    PRE_GREEN_BASE_LED_STUDENT_CHECKPOINT_VERSION,
    STUDENT_CHECKPOINT_VERSION,
    STUDENT_KIND,
    TEACHER_MANIFEST_VERSION,
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
    assert PRE_GREEN_BASE_LED_STUDENT_CHECKPOINT_VERSION == 5
    assert TEACHER_MANIFEST_VERSION == 7
    assert STUDENT_CHECKPOINT_VERSION == 10
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
    assert teacher_contract["gelsight_geometry"]["finger_and_gelsight_extension_m"] == (
        pytest.approx(0.021)
    )
    assert teacher_contract["gelsight_geometry"]["center_offset_hand_m"] == pytest.approx(
        [0.0, 0.0, 0.1392]
    )
    assert teacher_contract["gelsight_geometry"]["lowest_point_offset_hand_m"] == (
        pytest.approx([0.0, 0.0, 0.1563])
    )
    assert teacher_contract["profile"] == "rma_gelsight_x040_dr_static_size_buckets_v7"
    assert teacher_contract["robot_asset_filename"] == (
        "franka_gsmini_standard_arm_visuals_v7.usd"
    )
    assert teacher_contract["robot_base_world_position_m"] == [0.0, 0.0, 0.015]
    assert teacher_contract["camera_world_position_m"] == pytest.approx(
        [1.166091088407, 0.035901608197, 0.529200335898]
    )
    assert teacher_contract["cube_size_sampling"] == "fixed_round_robin_by_environment"
    assert teacher_contract["cube_size_assignment"] == "env_id_mod_8"
    assert teacher_contract["cube_size_object_count_per_environment"] == 1
    assert teacher_contract["compliant_grasp"]["contact_force_threshold_n"] == 2.0
    assert teacher_contract["compliant_grasp"]["excess_contact_force_threshold_n"] == 15.0
    assert teacher_contract["compliant_grasp"]["drop_penalty"] == -10.0
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
    assert student_environment_contract(student)["render_timing"] == {
        "physics_decimation": 2,
        "render_interval": 2,
    }
    assert student_environment_contract(student)["visual_alignment"] == {
        "shared_ground_visible": True,
        "shared_ground_color_rgb": [0.0, 0.0, 0.0],
        "franka_body_visual_profile": (
            "gelsight_physics_isaaclab_panda_arm_link0_7_visuals_green_base_led_"
            "finger_offset_21mm_usd_v7"
        ),
        "base_status_led": {
            "subset_path": "panda_link0/standard_visuals/panda_link0/subset_5",
            "color_rgb": [0.0, 1.0, 0.0],
            "material": "UsdPreviewSurface_emissive",
        },
        "gelsight_case_and_gelpad_visuals": "preserved_from_gelsight_asset",
        "floor_panel_size_m": [3.5, 3.5, 0.001],
        "backdrop_size_m": [0.02, 3.5, 2.5],
    }
    assert teacher.ground.spawn.visible is True
    assert student.ground.spawn.visible is True
    assert student.robot.spawn.func.__name__ == "spawn_from_usd"
    assert student.robot.spawn.usd_path == GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_USD
    assert teacher.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.015))
    assert student.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.015))
    assert student_environment_contract(student)["gelsight_depth_camera"] == {
        "left_update_latest_camera_pose": False,
        "right_update_latest_camera_pose": False,
        "tactile_rgb_float_to_uint8_scale": 255.0,
    }
    assert teacher.sim.render_interval == teacher.decimation == 2
    assert student.sim.render_interval == student.decimation == 2
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
    assert teacher.illegal_collision_terminates_episode is False
    assert teacher_contract["illegal_collision_terminates_episode"] is False
    assert "illegal_collision_termination_threshold_n" not in teacher_contract


def test_collision_curricula_are_linear_clamped_and_strict_boundaries():
    assert linear_collision_threshold(0, start_n=20.0, end_n=5.0) == pytest.approx(20.0)
    assert linear_collision_threshold(50_000, start_n=20.0, end_n=5.0) == pytest.approx(12.5)
    assert linear_collision_threshold(100_000, start_n=20.0, end_n=5.0) == pytest.approx(5.0)
    assert linear_collision_threshold(200_000, start_n=20.0, end_n=5.0) == pytest.approx(5.0)


def test_compact_reward_summary_uses_x040_unified_collision_metrics():
    log = {
        "reward/illegal_collision": torch.tensor(-1.25),
        "info/illegal_collision_penalty_threshold_n": torch.tensor(12.5),
        "info/illegal_collision_max_force_n": torch.tensor(7.5),
    }
    fields = _GelSightX040SafetyMixin._rma_collision_reward_print_fields(None, log)
    assert "illegal_collision=-1.250" in fields
    assert "illegal_threshold=12.50 N" in fields
    assert "illegal_force_max=7.50 N" in fields
    assert "table=" not in fields


def test_force_collision_does_not_terminate_x040_episode():
    dummy = SimpleNamespace(
        episode_length_buf=torch.tensor([149, 0]),
        max_episode_length=150,
        cfg=SimpleNamespace(success_hold_steps=5, ground_height=0.0),
        _cube=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.4, 0.0, 0.03], [0.4, 0.0, 0.03]]),
                root_quat_w=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                ),
            )
        ),
        _robot=SimpleNamespace(
            data=SimpleNamespace(
                body_link_pos_w=torch.tensor(
                    [
                        [[0.0, 0.0, 0.10], [0.0, 0.0, 0.05]],
                        [[0.0, 0.0, 0.10], [0.0, 0.0, -0.001]],
                    ]
                )
            )
        ),
        _success_hold_counter=torch.zeros(2, dtype=torch.long),
        _rma_episode_success_ever=torch.zeros(2, dtype=torch.bool),
    )
    dummy._compute_cube_upright_cos = lambda quat: torch.ones(quat.shape[0])
    dummy._compute_cube_lift_terms = lambda height, upright: (
        torch.zeros_like(height),
        torch.zeros_like(height),
        torch.zeros_like(height, dtype=torch.bool),
    )
    dummy._compute_illegal_collision_force = lambda: (_ for _ in ()).throw(
        AssertionError("force-based termination must not be queried")
    )
    dummy._record_episode_outcomes_for_step = lambda **kwargs: None
    dummy._publish_episode_success_statistics = lambda: None

    terminated, truncated = _GelSightX040SafetyMixin._get_dones(dummy)

    assert terminated.tolist() == [False, True]
    assert truncated.tolist() == [True, False]


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


def test_position_normalization_matches_x040_reset_box():
    model = RMAGelSightX040ThreeFrameStudent(pretrained_backbone=False)
    positions = torch.tensor(
        [[0.32, -0.10, 0.026], [0.40, 0.00, 0.026], [0.48, 0.10, 0.026]]
    )
    expected = torch.tensor(
        [[-1.0, -1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    normalized = model.normalizer.normalize_position(positions)
    torch.testing.assert_close(normalized, expected)
    torch.testing.assert_close(
        model.normalizer.denormalize_position(normalized), positions
    )
    assert model.contract()["position_normalization"] == "x040_wide_robot_root_xyz"


def test_legacy_position_normalization_remains_constructible_for_replay():
    model = RMAGelSightX040ThreeFrameStudent(
        pretrained_backbone=False, legacy_position_normalization=True
    )
    assert model.contract()["model_version"] == 2
    assert "position_normalization" not in model.contract()
    torch.testing.assert_close(
        model.normalizer.cube_position_center, torch.tensor([0.50, 0.00, 0.026])
    )


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
            "version": LEGACY_STUDENT_CHECKPOINT_VERSION - 1,
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="version mismatch"):
        load_student_checkpoint(checkpoint)


def test_pre_21mm_v7_asset_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "pre_21mm_asset_student.pt"
    torch.save({"kind": STUDENT_KIND, "version": 8}, checkpoint)
    with pytest.raises(RuntimeError, match="version mismatch"):
        load_student_checkpoint(checkpoint)


def test_student_eight_env_visual_alignment_smoke():
    student_cfg = parse_env_cfg(
        GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
        device="cuda:0",
        num_envs=8,
    )
    expected_bucket_ids = torch.arange(8, device=student_cfg.sim.device, dtype=torch.long)
    expected_sizes = torch.tensor(
        student_cfg.cube_size_buckets_m,
        device=student_cfg.sim.device,
        dtype=torch.float32,
    )

    student = gym.make(GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK, cfg=student_cfg)
    try:
        student_obs, _ = student.reset()
        student_base = student.unwrapped
        assert str(student_base._cube.data.root_pos_w.device) == str(student_base.device)
        assert student_base.gsmini_left.camera.cfg.update_latest_camera_pose is False
        assert student_base.gsmini_right.camera.cfg.update_latest_camera_pose is False
        torch.testing.assert_close(student_base._active_cube_bucket_ids, expected_bucket_ids)
        torch.testing.assert_close(student_base.active_cube_size_m, expected_sizes)
        assert "cube_size_buckets" not in student_base.scene.rigid_object_collections

        stage = sim_utils.stage_utils.get_current_stage()
        ground = stage.GetPrimAtPath(student_base.cfg.ground.prim_path)
        assert UsdGeom.Imageable(ground).ComputeVisibility() == UsdGeom.Tokens.inherited
        assert stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/gelsight_mini_case_left"
        ).IsValid()
        assert stage.GetPrimAtPath("/World/envs/env_0/Robot/gelpad_left").IsValid()
        aligned_visual = stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/panda_link0/standard_visuals"
        )
        assert aligned_visual.GetTypeName() == "Xform"
        assert not aligned_visual.IsInstanceable()
        aligned_mesh_names = {
            prim.GetName()
            for prim in Usd.PrimRange(aligned_visual)
            if prim.GetTypeName() == "Mesh"
        }
        assert aligned_mesh_names == {"panda_link0"}
        base_led = stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/panda_link0/standard_visuals/"
            "panda_link0/subset_5"
        )
        assert base_led.IsValid()
        assert not base_led.IsInstanceProxy()
        base_led_material, _ = UsdShade.MaterialBindingAPI(
            base_led
        ).ComputeBoundMaterial()
        assert base_led_material.GetPath().pathString.endswith(
            "/Looks/TacExBaseStatusGreen"
        )
        base_led_shader = stage.GetPrimAtPath(
            f"{base_led_material.GetPath()}/Shader"
        )
        assert tuple(
            base_led_shader.GetAttribute("inputs:diffuseColor").Get()
        ) == pytest.approx((0.0, 1.0, 0.0))
        assert tuple(
            base_led_shader.GetAttribute("inputs:emissiveColor").Get()
        ) == pytest.approx((0.0, 1.0, 0.0))
        wrist_led = stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/panda_link6/standard_visuals/"
            "panda_link6/subset_5"
        )
        wrist_led_material, _ = UsdShade.MaterialBindingAPI(
            wrist_led
        ).ComputeBoundMaterial()
        assert wrist_led_material.GetPath().pathString.endswith("/Looks/EmissiveBlue")
        original_arm_visual = stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/panda_link0/visuals"
        )
        assert original_arm_visual.IsActive() is False
        hand_visual = stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/panda_hand/visuals"
        )
        finger_visual = stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/panda_leftfinger/visuals"
        )
        assert hand_visual.GetTypeName() == ""
        assert finger_visual.GetTypeName() == ""
        assert student_base.cfg.robot.spawn.articulation_props.enabled_self_collisions is True

        student_actions = torch.zeros((8, 4), device=student_base.device)
        student_next, student_rewards, student_terminated, student_truncated, _ = student.step(student_actions)
        assert student_obs["policy"]["wrist_rgb_history"].shape == (8, 3, 224, 224, 3)
        assert student_next["policy"]["gsmini_left_reference_rgb"].shape == (8, 96, 128, 3)
        assert student_next["policy"]["gsmini_left_rgb"].dtype == torch.uint8
        assert student_next["policy"]["gsmini_right_rgb"].dtype == torch.uint8
        assert student_rewards.shape == student_terminated.shape == student_truncated.shape == (8,)
    finally:
        student.close()
