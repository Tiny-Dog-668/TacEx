"""Contract/runtime tests for the X040-Wide Size-Buckets three-frame Student."""

from __future__ import annotations

import copy

import pytest
import torch
from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from pxr import UsdGeom, UsdShade  # noqa: E402
from tacex_tasks.sim2real_grasp.rma_x040_wide_artifacts import (  # noqa: E402
    RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK,
    _shared_environment_contract,
    validate_live_teacher_contract,
)
from tacex_tasks.sim2real_grasp.rma_x040_wide_three_frame_artifacts import (  # noqa: E402
    RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK,
    RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK,
    RMA_X040_WIDE_THREE_FRAME_STUDENT_KIND,
    RMA_X040_WIDE_THREE_FRAME_STUDENT_VERSION,
    appearance_randomization_contract,
    load_three_frame_student_checkpoint,
    state_dict_sha256,
    three_frame_input_contract,
    three_frame_student_environment_contract,
)
from tacex_tasks.sim2real_grasp.sim2real_cube_real_alignment_rma_x040_wide_three_frame_independent_appearance_env import (  # noqa: E402
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnvCfg,
)
from tacex_tasks.sim2real_grasp.sim2real_cube_real_alignment_rma_x040_wide_three_frame_appearance_env import (  # noqa: E402
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameAppearanceStudentDREnvCfg,
)
from tacex_tasks.sim2real_grasp.rma_x040_wide_three_frame_models import (  # noqa: E402
    RMA_X040_WIDE_THREE_FRAME_DIRECT_STUDENT_MODEL_VERSION,
    RMAX040WideThreeFrameDirectActionVisualStudent,
)


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


def test_three_frame_model_keeps_fused_visual_feature_at_512():
    model = RMAX040WideThreeFrameDirectActionVisualStudent().train()
    images = torch.randint(
        0,
        256,
        (1, 3, 224, 224, 3),
        dtype=torch.uint8,
    )
    proprio = torch.zeros((1, 15), dtype=torch.float32)
    proprio[:, -1] = 0.04
    history = torch.zeros((1, 4), dtype=torch.float32)
    actions, position = model.forward_with_position(images, proprio, history)
    assert actions.shape == (1, 4)
    assert position.shape == (1, 3)
    assert model.temporal_fusion[0].in_features == 1536
    assert model.temporal_fusion[0].out_features == 512
    assert model.action_head[0].in_features == 531
    assert model.contract()["visual_feature_dim"] == 512
    assert model.contract()["frame_order"] == "oldest_to_newest"

    (actions.square().mean() + position.square().mean()).backward()
    assert model.temporal_fusion[0].weight.grad is not None
    assert all(
        parameter.grad is None
        for index, module in enumerate(model.vision_encoder)
        if index < 6
        for parameter in module.parameters()
    )
    assert any(
        parameter.grad is not None
        for index, module in enumerate(model.vision_encoder)
        if index >= 6
        for parameter in module.parameters()
    )


def test_three_frame_artifact_rejects_single_frame_contract(tmp_path):
    cfg = parse_env_cfg(
        RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK,
        device="cuda:0",
        num_envs=1,
    )
    model = RMAX040WideThreeFrameDirectActionVisualStudent()
    model_state = model.state_dict()
    vision_state = {
        key[len("vision_encoder.") :]: value
        for key, value in model_state.items()
        if key.startswith("vision_encoder.")
    }
    payload = {
        "kind": RMA_X040_WIDE_THREE_FRAME_STUDENT_KIND,
        "version": RMA_X040_WIDE_THREE_FRAME_STUDENT_VERSION,
        "model_version": RMA_X040_WIDE_THREE_FRAME_DIRECT_STUDENT_MODEL_VERSION,
        "task": RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK,
        "model": model_state,
        "model_state_dict_sha256": state_dict_sha256(model_state),
        "student_input_contract": three_frame_input_contract(),
        "student_model_contract": model.contract(),
        "normalization": model.normalizer.contract(),
        "student_environment_contract": three_frame_student_environment_contract(cfg),
        "vision_encoder_state_dict_sha256": state_dict_sha256(vision_state),
        "teacher_manifest": {"task": RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK},
    }
    checkpoint = tmp_path / "student.pt"
    torch.save(payload, checkpoint)
    assert load_three_frame_student_checkpoint(checkpoint)["task"] == payload["task"]

    payload["student_input_contract"] = {
        "wrist_rgb": [224, 224, 3],
        "proprio_obs": [15],
        "action_history": [4],
    }
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="task/input mismatch"):
        load_three_frame_student_checkpoint(checkpoint)


def test_appearance_task_has_an_independent_fail_closed_contract(tmp_path):
    assert gym.spec(
        RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK
    )
    legacy_cfg = parse_env_cfg(
        RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK,
        device="cuda:0",
        num_envs=1,
    )
    appearance_cfg = parse_env_cfg(
        RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK,
        device="cuda:0",
        num_envs=1,
    )
    assert "appearance_randomization" not in three_frame_student_environment_contract(
        legacy_cfg
    )
    contract = three_frame_student_environment_contract(appearance_cfg)
    assert contract["appearance_randomization"] == appearance_randomization_contract(
        Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnvCfg()
    )
    appearance = contract["appearance_randomization"]
    assert appearance["profile"] == "x040_three_frame_independent_preview_surface_v5"
    assert appearance["sampling_frequency"] == "per_environment_on_actual_done_reset"
    assert appearance["sampling_scope"] == "per_environment"
    assert appearance["material_instance_scope"] == "per_environment_unique_prim"
    assert appearance["timeout_termination_enabled"] is True
    assert appearance["full_strength_from_step"] == 0
    assert appearance["categories_sampled_independently"] is True
    assert appearance["scene_layout"] == {
        "profile": "per_environment_non_overlapping_v1",
        "env_spacing_m": 3.5,
        "plate_size_m": [3.0, 3.0, 0.001],
        "backdrop_size_m": [0.02, 2.0, 2.5],
        "min_inter_env_clearance_m": 0.5,
        "global_ground_visible": False,
    }
    assert contract["profile"] == (
        "rma_x040_wide_xyz_no_contact_static_size_buckets_v3"
    )
    assert contract["cube_size_sampling"] == "fixed_round_robin_by_environment"
    assert contract["cube_size_assignment"] == "env_id_mod_bucket_count"
    assert contract["cube_size_object_count_per_environment"] == 1
    assert contract["cube_size_requires_num_envs_multiple_of_bucket_count"] is True
    assert contract["cube_bucket_rendering"] == {
        "selection": "fixed_bucket_by_environment",
        "assignment": "env_id_mod_bucket_count",
        "object_count_per_environment": 1,
        "requires_num_envs_multiple_of_bucket_count": True,
        "usd_visibility_mutation": "none",
    }
    teacher_manifest = {
        "environment_contract": _shared_environment_contract(legacy_cfg)
    }
    with pytest.raises(RuntimeError, match="differs from Teacher contract"):
        validate_live_teacher_contract(appearance_cfg, teacher_manifest)
    validate_live_teacher_contract(
        appearance_cfg,
        teacher_manifest,
        allow_static_size_bucket_assignment=True,
    )
    for category in ("plate", "backdrop", "cube"):
        assert sum(entry["weight"] for entry in appearance[category]) == pytest.approx(1.0)
        assert len({entry["id"] for entry in appearance[category]}) == len(
            appearance[category]
        )

    model = RMAX040WideThreeFrameDirectActionVisualStudent()
    model_state = model.state_dict()
    vision_state = {
        key[len("vision_encoder.") :]: value
        for key, value in model_state.items()
        if key.startswith("vision_encoder.")
    }
    payload = {
        "kind": RMA_X040_WIDE_THREE_FRAME_STUDENT_KIND,
        "version": RMA_X040_WIDE_THREE_FRAME_STUDENT_VERSION,
        "model_version": RMA_X040_WIDE_THREE_FRAME_DIRECT_STUDENT_MODEL_VERSION,
        "task": RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK,
        "model": model_state,
        "model_state_dict_sha256": state_dict_sha256(model_state),
        "student_input_contract": three_frame_input_contract(),
        "student_model_contract": model.contract(),
        "normalization": model.normalizer.contract(),
        "student_environment_contract": three_frame_student_environment_contract(legacy_cfg),
        "vision_encoder_state_dict_sha256": state_dict_sha256(vision_state),
        "teacher_manifest": {"task": RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK},
    }
    checkpoint = tmp_path / "student.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint task mismatch"):
        load_three_frame_student_checkpoint(
            checkpoint,
            expected_task=(
                RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK
            ),
        )

    payload["task"] = (
        RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK
    )
    payload["student_environment_contract"] = contract
    torch.save(payload, checkpoint)
    assert (
        load_three_frame_student_checkpoint(
            checkpoint,
            expected_task=(
                RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK
            ),
        )["task"]
        == RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK
    )

    legacy_appearance_payload = copy.deepcopy(payload)
    legacy_appearance_payload["student_environment_contract"] = (
        three_frame_student_environment_contract(
            Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameAppearanceStudentDREnvCfg()
        )
    )
    legacy_appearance_payload["student_environment_contract"]["appearance_randomization"][
        "profile"
    ] = "x040_three_frame_realistic_material_v1"
    legacy_appearance_payload["student_environment_contract"][
        "appearance_randomization"
    ].pop("sampling_scope")
    legacy_appearance_payload["student_environment_contract"][
        "appearance_randomization"
    ].pop("material_instance_scope")
    torch.save(legacy_appearance_payload, checkpoint)
    with pytest.raises(RuntimeError, match="task environment contract mismatch"):
        load_three_frame_student_checkpoint(
            checkpoint,
            expected_task=(
                RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK
            ),
        )
    assert (
        load_three_frame_student_checkpoint(
            checkpoint,
            expected_task=(
                RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK
            ),
            allow_appearance_v1_evaluation=True,
        )["student_environment_contract"]["appearance_randomization"]["profile"]
        == "x040_three_frame_realistic_material_v1"
    )


def test_three_frame_environment_orders_frames_and_refills_on_reset():
    assert gym.spec(RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK)
    cfg = parse_env_cfg(
        RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK,
        device="cuda:0",
        num_envs=2,
    )
    assert "wrist_rgb" not in cfg.observation_space
    assert cfg.observation_space["wrist_rgb_history"].shape == (3, 224, 224, 3)
    assert cfg.action_space == 4
    assert cfg.wrist_rgb_history_stride_policy_steps == 1

    env = gym.make(
        RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_DIRECT_STUDENT_DR_TASK,
        cfg=cfg,
    )
    try:
        observations, _ = env.reset()
        first = observations["policy"]["wrist_rgb_history"].clone()
        assert first.shape == (2, 3, 224, 224, 3)
        assert first.dtype == torch.uint8
        torch.testing.assert_close(first[:, 0], first[:, 1])
        torch.testing.assert_close(first[:, 1], first[:, 2])
        assert "wrist_rgb" not in observations["policy"]

        observations, rewards, terminated, truncated, _ = env.step(
            torch.zeros((2, 4), device=env.unwrapped.device)
        )
        second = observations["policy"]["wrist_rgb_history"].clone()
        torch.testing.assert_close(second[:, 0], first[:, 1])
        torch.testing.assert_close(second[:, 1], first[:, 2])
        assert torch.isfinite(rewards).all()
        assert terminated.shape == truncated.shape == (2,)

        observations, _ = env.reset()
        reset_history = observations["policy"]["wrist_rgb_history"]
        torch.testing.assert_close(reset_history[:, 0], reset_history[:, 1])
        torch.testing.assert_close(reset_history[:, 1], reset_history[:, 2])
    finally:
        env.close()


def test_appearance_environment_owns_materials_and_only_changes_reset_env(monkeypatch):
    cfg = parse_env_cfg(
        RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK,
        device="cuda:0",
        num_envs=8,
    )
    assert cfg.plate_color_randomization_enabled is False
    assert cfg.backdrop_color_randomization_enabled is False
    assert cfg.observation_space["wrist_rgb_history"].shape == (3, 224, 224, 3)
    assert cfg.action_space == 4

    env = gym.make(
        RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK,
        cfg=cfg,
    )
    try:
        observations, _ = env.reset()
        base = env.unwrapped
        assert observations["policy"]["wrist_rgb_history"].shape == (
            8,
            3,
            224,
            224,
            3,
        )
        for values in (
            base._appearance_plate_material_ids,
            base._appearance_backdrop_material_ids,
            base._appearance_cube_material_ids,
        ):
            assert torch.all(values >= 0)

        torch.manual_seed(1234)
        first = base._sample_color_ids("cube", 32)
        torch.manual_seed(1234)
        second = base._sample_color_ids("cube", 32)
        torch.testing.assert_close(first, second)

        sampled_categories = []
        original_sampler = base._sample_color_ids

        def record_sampler(category, count):
            sampled_categories.append(category)
            return original_sampler(category, count)

        monkeypatch.setattr(base, "_sample_color_ids", record_sampler)
        base._randomize_scene_visuals(
            torch.tensor([0], device=base.device, dtype=torch.long)
        )
        assert sampled_categories == ["plate", "backdrop", "cube"]

        stage = sim_utils.stage_utils.get_current_stage()
        env_zero = torch.tensor([0], device=base.device, dtype=torch.long)
        assert base.sim.get_physics_context().is_gpu_dynamics_enabled()
        assert base.cfg.scene.env_spacing == pytest.approx(3.5)
        assert base.cfg.ground.spawn.visible is False
        assert UsdGeom.Imageable(
            stage.GetPrimAtPath(base.cfg.ground.prim_path)
        ).ComputeVisibility() == UsdGeom.Tokens.invisible
        assert base._plate.root_physx_view.prim_paths == [
            f"/World/envs/env_{env_id}/floor_panel" for env_id in range(8)
        ]
        assert base._backdrop.root_physx_view.prim_paths == [
            f"/World/envs/env_{env_id}/real_alignment_backdrop"
            for env_id in range(8)
        ]
        assert sim_utils.find_matching_prim_paths(
            "/World/envs/env_.*/floor_panel"
        ) == base._plate.root_physx_view.prim_paths
        assert sim_utils.find_matching_prim_paths(
            "/World/envs/env_.*/real_alignment_backdrop"
        ) == base._backdrop.root_physx_view.prim_paths
        origins_xy = base.scene.env_origins[:, :2].detach().cpu()
        plate_xy = torch.tensor(base.cfg.plate.spawn.size[:2])
        clearance = float(base.cfg.appearance_min_inter_env_clearance_m)
        for left in range(8):
            for right in range(left + 1, 8):
                separation = (origins_xy[left] - origins_xy[right]).abs()
                assert bool(
                    (separation[0] >= plate_xy[0] + clearance)
                    or (separation[1] >= plate_xy[1] + clearance)
                )
        torch.testing.assert_close(
            base._active_cube_bucket_ids,
            torch.arange(8, device=base.device, dtype=torch.long),
        )
        fixed_sizes = base.active_cube_size_m.clone()
        expected_local_z = (
            float(base.cfg.plate_top_height_m) + 0.5 * fixed_sizes
        )
        actual_local_z = (
            base._cube.data.root_pos_w[:, 2]
            - base.scene.env_origins.to(base.device)[:, 2]
        )
        torch.testing.assert_close(actual_local_z, expected_local_z, atol=1.0e-5, rtol=0.0)
        for env_id in range(8):
            assert stage.GetPrimAtPath(f"/World/envs/env_{env_id}/cube").IsValid()
            assert all(
                not stage.GetPrimAtPath(
                    f"/World/envs/env_{env_id}/cube_size_bucket_{bucket_id}"
                ).IsValid()
                for bucket_id in range(8)
            )
            scale = stage.GetPrimAtPath(
                f"/World/envs/env_{env_id}/cube"
            ).GetAttribute("xformOp:scale").Get()
            expected_scale = float(fixed_sizes[env_id].item()) / 0.05
            assert tuple(scale) == pytest.approx((expected_scale,) * 3)
        for category, paths in base._independent_material_paths.items():
            assert len(set(paths)) == len(paths)
            assert all(
                path.startswith(
                    f"/World/Looks/X040IndependentAppearance/env_{env_id}/"
                )
                for env_id, path in enumerate(paths)
            )

        def bound_material_path(category, env_id):
            if category == "plate":
                root = base._plate.root_physx_view.prim_paths[env_id]
            elif category == "backdrop":
                root = base._backdrop.root_physx_view.prim_paths[env_id]
            else:
                root = base._active_cube_root_prim_path(env_id)
            target = base._resolve_visual_target_prim(root)
            material, _ = UsdShade.MaterialBindingAPI(
                stage.GetPrimAtPath(target)
            ).ComputeBoundMaterial()
            return material.GetPath().pathString

        all_bindings = {
            (category, env_id): bound_material_path(category, env_id)
            for category in ("plate", "backdrop", "cube")
            for env_id in range(8)
        }
        assert all_bindings == {
            (category, env_id): base._independent_material_paths[category][env_id]
            for category in ("plate", "backdrop", "cube")
            for env_id in range(8)
        }
        env_one_bound_materials = {
            category: bound_material_path(category, 1)
            for category in ("plate", "backdrop", "cube")
        }

        episode_material_ids = (
            base._appearance_plate_material_ids.clone(),
            base._appearance_backdrop_material_ids.clone(),
            base._appearance_cube_material_ids.clone(),
            base._appearance_cube_bucket_ids.clone(),
        )
        episode_bindings = {
            (category, env_id): bound_material_path(category, env_id)
            for category in ("plate", "backdrop", "cube")
            for env_id in range(8)
        }
        _, _, terminated, truncated, _ = env.step(
            torch.zeros((8, 4), device=base.device)
        )
        assert not torch.any(terminated)
        assert not torch.any(truncated)
        for actual, expected in zip(
            (
                base._appearance_plate_material_ids,
                base._appearance_backdrop_material_ids,
                base._appearance_cube_material_ids,
                base._appearance_cube_bucket_ids,
            ),
            episode_material_ids,
        ):
            torch.testing.assert_close(actual, expected)
        assert {
            (category, env_id): bound_material_path(category, env_id)
            for category in ("plate", "backdrop", "cube")
            for env_id in range(8)
        } == episode_bindings

        untouched = (
            base._appearance_plate_material_ids[1].clone(),
            base._appearance_backdrop_material_ids[1].clone(),
            base._appearance_cube_material_ids[1].clone(),
            base._appearance_cube_bucket_ids[1].clone(),
        )
        untouched_cube_state = torch.cat(
            (
                base._cube.data.root_pos_w[1],
                base._cube.data.root_quat_w[1],
                base._cube.data.root_lin_vel_w[1],
                base._cube.data.root_ang_vel_w[1],
            )
        ).clone()
        base._apply_independent_colors(
            env_zero,
            torch.tensor([1], device=base.device),
            torch.tensor([1], device=base.device),
            torch.tensor([1], device=base.device),
        )
        torch.testing.assert_close(base._appearance_plate_material_ids[1], untouched[0])
        torch.testing.assert_close(base._appearance_backdrop_material_ids[1], untouched[1])
        torch.testing.assert_close(base._appearance_cube_material_ids[1], untouched[2])
        torch.testing.assert_close(base._appearance_cube_bucket_ids[1], untouched[3])
        assert {
            category: bound_material_path(category, 1)
            for category in ("plate", "backdrop", "cube")
        } == env_one_bound_materials
        base._reset_idx(env_zero)
        torch.testing.assert_close(base.active_cube_size_m, fixed_sizes)
        torch.testing.assert_close(
            base._active_cube_bucket_ids,
            torch.arange(8, device=base.device, dtype=torch.long),
        )
        torch.testing.assert_close(base._appearance_plate_material_ids[1], untouched[0])
        torch.testing.assert_close(base._appearance_backdrop_material_ids[1], untouched[1])
        torch.testing.assert_close(base._appearance_cube_material_ids[1], untouched[2])
        torch.testing.assert_close(base._appearance_cube_bucket_ids[1], untouched[3])
        torch.testing.assert_close(
            torch.cat(
                (
                    base._cube.data.root_pos_w[1],
                    base._cube.data.root_quat_w[1],
                    base._cube.data.root_lin_vel_w[1],
                    base._cube.data.root_ang_vel_w[1],
                )
            ),
            untouched_cube_state,
        )
        assert {
            category: bound_material_path(category, 1)
            for category in ("plate", "backdrop", "cube")
        } == env_one_bound_materials

        base._apply_independent_colors(
            env_zero,
            torch.tensor([0], device=base.device),
            torch.tensor([0], device=base.device),
            torch.tensor([0], device=base.device),
        )
        dark_observations, *_ = env.step(torch.zeros((8, 4), device=base.device))
        dark_rgb = dark_observations["policy"]["wrist_rgb_history"][0, -1].clone()
        base._apply_independent_colors(
            env_zero,
            torch.tensor([0], device=base.device),
            torch.tensor([2], device=base.device),
            torch.tensor([0], device=base.device),
        )
        green_observations, *_ = env.step(torch.zeros((8, 4), device=base.device))
        green_rgb = green_observations["policy"]["wrist_rgb_history"][0, -1]
        mean_rgb_delta = (dark_rgb.float() - green_rgb.float()).abs().mean()
        assert mean_rgb_delta > 0.5

        # Only env 0 reaches the common 150-step horizon. DirectRLEnv must
        # truncate/reset/recolor that env while all other episode clocks and
        # appearance IDs remain untouched.
        stable_ids = (
            base._appearance_plate_material_ids.clone(),
            base._appearance_backdrop_material_ids.clone(),
            base._appearance_cube_material_ids.clone(),
            base._appearance_cube_bucket_ids.clone(),
        )
        material_buffers = {
            "plate": base._appearance_plate_material_ids,
            "backdrop": base._appearance_backdrop_material_ids,
            "cube": base._appearance_cube_material_ids,
        }

        def sample_different_color(category, count):
            next_id = int(material_buffers[category][0].item()) + 1
            next_id %= int(base._appearance_palettes[category].shape[0])
            return torch.full(
                (count,), next_id, device=base.device, dtype=torch.long
            )

        monkeypatch.setattr(base, "_sample_color_ids", sample_different_color)
        base.episode_length_buf.zero_()
        base.episode_length_buf[0] = base.max_episode_length - 2
        _, _, terminated, truncated, _ = env.step(
            torch.zeros((8, 4), device=base.device)
        )
        assert not torch.any(terminated)
        assert truncated.tolist() == [True, False, False, False, False, False, False, False]
        assert base.episode_length_buf.tolist() == [0, 1, 1, 1, 1, 1, 1, 1]
        for actual, expected in zip(
            material_buffers.values(), stable_ids[:3]
        ):
            assert actual[0] != expected[0]
            torch.testing.assert_close(actual[1:], expected[1:])
        torch.testing.assert_close(
            base._appearance_cube_bucket_ids,
            stable_ids[3],
        )
    finally:
        env.close()
