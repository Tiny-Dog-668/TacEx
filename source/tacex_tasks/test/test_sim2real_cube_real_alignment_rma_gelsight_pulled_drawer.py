from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
import torch
from pxr import Usd

import tacex_tasks  # noqa: F401
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M,
    GELSIGHT_PULLED_DRAWER_FRANKA_USD,
    GELSIGHT_LEGACY_STANDARD_FRANKA_ARM_VISUAL_USD,
    GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_USD,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_artifacts import (
    STUDENT_KIND,
    TEACHER_KIND,
    make_student_model_for_checkpoint,
    student_environment_contract,
    teacher_environment_contract,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_models import (
    PULLED_DRAWER_LINK7_TO_GELPAD_MIDPOINT_M,
    RMAGelSightPulledDrawerHandKinematics,
    RMAGelSightPulledDrawerObservationNormalizer,
    RMAGelSightPulledDrawerThreeFrameStudent,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_three_frame_artifacts import (
    GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK,
    GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
    STUDENT_KIND as X040_STUDENT_KIND,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_x040_three_frame_env import (
    Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_pulled_drawer_env import (
    GELSIGHT_PULLED_DRAWER_TEACHER_TASK,
    GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK,
    PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M,
    PULLED_DRAWER_CABINET_NOMINAL_SIZE_M,
    PULLED_DRAWER_GELSIGHT_BOTTOM_OFFSET_HAND_M,
    PULLED_DRAWER_GELSIGHT_CENTER_OFFSET_HAND_M,
    PULLED_DRAWER_MIN_SIDE_CLEARANCE_M,
    PULLED_DRAWER_PANDA_FINGER_JOINT_LOCAL_POS0_Z_M,
    PULLED_DRAWER_PANEL_CONTACT_OFFSET_M,
    PULLED_DRAWER_PANEL_REST_OFFSET_M,
    PULLED_DRAWER_PANEL_THICKNESS_M,
    PULLED_DRAWER_POSITION_DELTA_XY_M,
    PULLED_DRAWER_SEAM_GAP_M,
    PULLED_DRAWER_SIZE_SCALE_RANGE,
    PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M,
    PULLED_DRAWER_TRAY_NOMINAL_SIZE_M,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg,
    balanced_cube_bucket_ids,
    pulled_drawer_geometry_contract,
    sample_pulled_drawer_layouts,
)


def test_pulled_drawer_task_registration_and_runtime_contract() -> None:
    teacher = Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg()
    student = Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg()
    x040_teacher = Sim2RealCubeRealAlignmentRMAGelSightX040DRSizeBucketsTeacherEnvCfg()
    assert gym.spec(GELSIGHT_PULLED_DRAWER_TEACHER_TASK) is not None
    assert gym.spec(GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK) is not None
    assert gym.spec(GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK) is not None
    assert gym.spec(GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK) is not None
    assert GELSIGHT_PULLED_DRAWER_TEACHER_TASK != GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK
    assert GELSIGHT_PULLED_DRAWER_TEACHER_TASK != GELSIGHT_X040_DR_SIZE_BUCKETS_TEACHER_TASK
    assert (
        GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK
        != GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK
    )
    assert teacher.rma_task_id == GELSIGHT_PULLED_DRAWER_TEACHER_TASK
    assert student.rma_task_id == GELSIGHT_PULLED_DRAWER_THREE_FRAME_STUDENT_TASK
    assert teacher.action_space == student.action_space == 4
    assert teacher.observation_space["proprio_obs"] == 15
    assert teacher.observation_space["action_history"] == 4
    assert teacher.observation_space["rma_cube_pos"] == 3
    assert teacher.observation_space["rma_contact_state"] == 2
    assert student.observation_space["wrist_rgb_history"].shape == (3, 224, 224, 3)
    assert student.observation_space["proprio_obs"] == 15
    assert student.observation_space["action_history"] == 4
    assert student.observation_space["gsmini_left_rgb"].shape == (96, 128, 3)
    assert student.observation_space["gsmini_right_reference_rgb"].shape == (96, 128, 3)
    assert teacher.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.015))
    assert Path(teacher.robot.spawn.usd_path) == Path(GELSIGHT_PULLED_DRAWER_FRANKA_USD)
    assert Path(teacher.robot.spawn.usd_path).is_file()
    assert Path(x040_teacher.robot.spawn.usd_path) == Path(
        GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_USD
    )
    assert teacher.robot.spawn.usd_path == x040_teacher.robot.spawn.usd_path
    assert teacher.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.015))
    assert x040_teacher.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.015))
    assert teacher.pulled_drawer_finger_extension_local_z_m == pytest.approx(0.021)
    assert teacher.rma_contact_force_threshold_n == pytest.approx(2.0)
    assert teacher.arm_ik_tcp_offset_m == pytest.approx((0.0, 0.0, 0.1392))
    assert teacher.gelsight_center_offset_hand_m == pytest.approx((0.0, 0.0, 0.1392))
    assert teacher.gelsight_fingertip_bottom_offset_hand_m == pytest.approx(
        (0.0, 0.0, 0.1563)
    )
    assert teacher.robot.spawn.articulation_props.solver_position_iteration_count == 32
    assert teacher.robot.spawn.articulation_props.solver_velocity_iteration_count == 4
    assert student.robot.spawn.articulation_props.solver_position_iteration_count == 32
    assert student.robot.spawn.articulation_props.solver_velocity_iteration_count == 4
    assert x040_teacher.robot.spawn.articulation_props.solver_position_iteration_count == 8
    assert x040_teacher.robot.spawn.articulation_props.solver_velocity_iteration_count == 0
    assert teacher.illegal_collision_penalty_threshold_start_n == pytest.approx(20.0)
    assert teacher.illegal_collision_penalty_threshold_end_n == pytest.approx(5.0)
    assert teacher.illegal_collision_termination_threshold_start_n == pytest.approx(200.0)
    assert teacher.illegal_collision_termination_threshold_end_n == pytest.approx(20.0)
    assert teacher.illegal_collision_terminates_episode is True
    assert teacher.cube.init_state.pos == pytest.approx((0.5275, 0.0, 0.033))
    assert not hasattr(teacher, "pulled_drawer_cabinet_bottom")
    assert hasattr(teacher, "pulled_drawer_tray_floor")
    assert hasattr(teacher, "pulled_drawer_tray_front")
    assert hasattr(teacher, "pulled_drawer_tray_back")
    assert hasattr(teacher, "pulled_drawer_tray_left")
    assert hasattr(teacher, "pulled_drawer_tray_right")
    assert not teacher.pulled_drawer_appearance_randomization_enabled
    assert student.pulled_drawer_appearance_randomization_enabled
    assert not teacher.pulled_drawer_opacity_randomization_enabled
    assert not student.pulled_drawer_opacity_randomization_enabled
    assert teacher.pulled_drawer_tray_opacity == pytest.approx(1.0)
    assert student.pulled_drawer_tray_opacity == pytest.approx(1.0)
    for panel_name in (
        "pulled_drawer_cabinet_left",
        "pulled_drawer_cabinet_right",
        "pulled_drawer_cabinet_back",
        "pulled_drawer_cabinet_top",
        "pulled_drawer_tray_floor",
        "pulled_drawer_tray_front",
        "pulled_drawer_tray_back",
        "pulled_drawer_tray_left",
        "pulled_drawer_tray_right",
    ):
        panel = getattr(teacher, panel_name)
        assert panel.spawn.activate_contact_sensors is True
        assert panel.spawn.rigid_props.kinematic_enabled is True
        assert panel.spawn.collision_props.collision_enabled is True
        assert panel.spawn.collision_props.contact_offset == pytest.approx(
            PULLED_DRAWER_PANEL_CONTACT_OFFSET_M
        )
        assert panel.spawn.collision_props.rest_offset == pytest.approx(
            PULLED_DRAWER_PANEL_REST_OFFSET_M
        )
        assert panel.spawn.physics_material.static_friction == pytest.approx(1.0)
        assert panel.spawn.physics_material.dynamic_friction == pytest.approx(0.8)


def test_pulled_drawer_geometry_sampling_is_fixed_deterministic_and_bounded() -> None:
    first = sample_pulled_drawer_layouts(64, seed=42)
    second = sample_pulled_drawer_layouts(64, seed=42)
    third = sample_pulled_drawer_layouts(64, seed=43)
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    assert not torch.equal(first["cabinet_size_m"], third["cabinet_size_m"])

    for key, nominal in (
        ("cabinet_size_m", PULLED_DRAWER_CABINET_NOMINAL_SIZE_M),
        ("tray_size_m", PULLED_DRAWER_TRAY_NOMINAL_SIZE_M),
    ):
        normalized = first[key] / torch.tensor(nominal, dtype=torch.float64)
        assert torch.all(normalized >= PULLED_DRAWER_SIZE_SCALE_RANGE[0])
        assert torch.all(normalized <= PULLED_DRAWER_SIZE_SCALE_RANGE[1])
        torch.testing.assert_close(
            normalized,
            first["shared_size_scale"].unsqueeze(1).expand_as(normalized),
        )
    cabinet_delta = torch.abs(
        first["cabinet_center_xy_m"]
        - torch.tensor(PULLED_DRAWER_CABINET_NOMINAL_CENTER_XY_M, dtype=torch.float64)
    )
    assert torch.all(
        cabinet_delta
        <= torch.tensor(PULLED_DRAWER_POSITION_DELTA_XY_M, dtype=torch.float64)
    )
    tray_y_delta = torch.abs(
        first["tray_center_xy_m"][:, 1] - PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M[1]
    )
    assert torch.all(tray_y_delta <= PULLED_DRAWER_POSITION_DELTA_XY_M[1])
    cabinet_front_x = (
        first["cabinet_center_xy_m"][:, 0] - 0.5 * first["cabinet_size_m"][:, 0]
    )
    tray_back_x = first["tray_center_xy_m"][:, 0] + 0.5 * first["tray_size_m"][:, 0]
    torch.testing.assert_close(cabinet_front_x, tray_back_x, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        first["gap_m"],
        torch.full_like(first["gap_m"], PULLED_DRAWER_SEAM_GAP_M),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert torch.all(first["side_clearance_m"] >= PULLED_DRAWER_MIN_SIDE_CLEARANCE_M)


def test_pulled_drawer_balanced_cube_assignment_and_safe_top_margin() -> None:
    bucket_ids = balanced_cube_bucket_ids(32, seed=42)
    for group in bucket_ids.reshape(-1, 8):
        assert sorted(group.tolist()) == list(range(8))
    assert not torch.equal(bucket_ids, balanced_cube_bucket_ids(32, seed=43))
    with pytest.raises(ValueError, match="multiple of 8"):
        balanced_cube_bucket_ids(10, seed=42)

    # Minimum 20.25 cm tray, 8 mm walls, maximum 6 cm Cube and +/-3 cm reset.
    minimum_margin = (
        0.5
        * (
            0.90 * PULLED_DRAWER_TRAY_NOMINAL_SIZE_M[0]
            - 2.0 * PULLED_DRAWER_PANEL_THICKNESS_M
        )
        - 0.5 * 0.06
        - 0.03
    )
    assert minimum_margin == pytest.approx(0.03325)


def test_pulled_drawer_normalization_and_fail_closed_contracts() -> None:
    normalizer = RMAGelSightPulledDrawerObservationNormalizer()
    expected_center = torch.tensor([0.5275, 0.0, 0.033])
    expected_scale = torch.tensor([0.11, 0.08, 0.10])
    torch.testing.assert_close(normalizer.cube_position_center, expected_center)
    torch.testing.assert_close(normalizer.cube_position_scale, expected_scale)
    positions = torch.tensor([[0.4175, -0.08, 0.028], [0.6375, 0.08, 0.038]])
    torch.testing.assert_close(
        normalizer.denormalize_position(normalizer.normalize_position(positions)), positions
    )
    assert STUDENT_KIND == "tacex_rma_gelsight_pulled_drawer_three_frame_student"
    assert TEACHER_KIND == "tacex_rma_gelsight_pulled_drawer_teacher"
    assert normalizer.contract()["profile"] == (
        "pulled_drawer_contiguous_rigid_tray_robot_root_xyz_v3"
    )
    kinematics = RMAGelSightPulledDrawerHandKinematics()
    assert kinematics.link7_to_fingertip_midpoint[2].item() == pytest.approx(0.2462)
    assert PULLED_DRAWER_LINK7_TO_GELPAD_MIDPOINT_M == pytest.approx(0.2462)
    assert kinematics.contract()["panda_hand_to_finger_gelsight_extension_m"] == (
        pytest.approx(0.021)
    )
    assert kinematics.contract()["panda_hand_to_fingertip_midpoint_m"] == pytest.approx(
        0.1392
    )
    with pytest.raises(RuntimeError, match="Unsupported Pulled-Drawer"):
        make_student_model_for_checkpoint({"kind": X040_STUDENT_KIND, "version": 6})


def test_pulled_drawer_student_seven_inputs_and_three_outputs() -> None:
    model = RMAGelSightPulledDrawerThreeFrameStudent(pretrained_backbone=False).eval()
    with torch.inference_mode():
        action, contact_probability, cube_position = model(
            torch.zeros((1, 3, 224, 224, 3), dtype=torch.uint8),
            torch.zeros((1, 15)),
            torch.zeros((1, 4)),
            torch.zeros((1, 96, 128, 3), dtype=torch.uint8),
            torch.zeros((1, 96, 128, 3), dtype=torch.uint8),
            torch.zeros((1, 96, 128, 3), dtype=torch.uint8),
            torch.zeros((1, 96, 128, 3), dtype=torch.uint8),
        )
    assert action.shape == (1, 4)
    assert contact_probability.shape == (1, 2)
    assert cube_position.shape == (1, 3)
    assert torch.isfinite(action).all()
    assert torch.logical_and(contact_probability >= 0.0, contact_probability <= 1.0).all()


def test_pulled_drawer_environment_contract_records_geometry_and_collisions() -> None:
    teacher = Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerTeacherEnvCfg()
    student = Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerThreeFrameStudentEnvCfg()
    teacher_contract = teacher_environment_contract(teacher)
    student_contract = student_environment_contract(student)
    assert teacher_contract["profile"] == (
        "rma_gelsight_pulled_drawer_contiguous_rigid_tray_v11"
    )
    assert teacher_contract["geometry"] == pulled_drawer_geometry_contract()
    assert teacher_contract["robot_base_world_position_m"] == [0.0, 0.0, 0.015]
    assert teacher_contract["cube_reset_half_range_xy_m"] == [0.03, 0.03]
    assert any(
        "pulled_drawer_cabinet_top" in path
        for path in teacher_contract["cube_illegal_filters"]
    )
    assert "pulled_drawer_tray_floor" in teacher_contract["surface_robot_collision_filters"]
    assert "pulled_drawer_tray_front" in teacher_contract["surface_robot_collision_filters"]
    assert teacher_contract["geometry"]["tray"]["shape"] == (
        "open_top_five_panel_pulled_drawer"
    )
    assert teacher_contract["geometry"]["sampling"]["seam_gap_m"] == 0.0
    assert teacher_contract["geometry"]["sampling"]["size_scale_coupling"] == (
        "same_scalar_for_cabinet_and_tray_xyz"
    )
    assert teacher_contract["geometry"]["collision"]["collision_enabled"] is True
    assert teacher_contract["geometry"]["collision"]["panel_type"] == (
        "kinematic_rigid_body"
    )
    assert teacher_contract["physics_layout"]["robot_articulation_solver"] == {
        "position_iterations": 32,
        "velocity_iterations": 4,
    }
    assert teacher_contract["geometry"]["collision"]["panel_contact_offset_m"] == (
        pytest.approx(PULLED_DRAWER_PANEL_CONTACT_OFFSET_M)
    )
    assert student_contract["appearance_randomization_enabled"] is True
    assert student_contract["opacity_randomization_enabled"] is False
    assert student_contract["tray_opacity"] == pytest.approx(1.0)
    assert teacher_contract["geometry"]["appearance"]["tray_opacity"] == pytest.approx(1.0)
    assert teacher_contract["geometry"]["appearance"]["tray_nominal_rgb"] == pytest.approx(
        [0.34, 0.40, 0.43]
    )
    assert teacher_contract["geometry"]["appearance"]["tray_hsv"] == {
        "h": pytest.approx([0.52, 0.62]),
        "s": pytest.approx([0.08, 0.20]),
        "v": pytest.approx([0.30, 0.55]),
    }
    assert teacher_contract["geometry"]["appearance"]["tray_roughness"] == pytest.approx(
        0.18
    )
    assert teacher_contract["geometry"]["appearance"]["opacity_randomization_enabled"] is False
    assert teacher_contract["compliant_grasp"]["contact_force_threshold_n"] == 2.0
    assert teacher_contract["compliant_grasp"]["excess_contact_force_penalty_per_policy_step"] == -5.0
    assert teacher_contract["compliant_grasp"]["drop_penalty"] == -10.0
    assert student_contract["wrist_rgb_history"]["shape"] == [3, 224, 224, 3]


def test_pulled_drawer_asset_extends_fingers_without_moving_panda_hand() -> None:
    base_stage = Usd.Stage.Open(GELSIGHT_LEGACY_STANDARD_FRANKA_ARM_VISUAL_USD)
    pulled_stage = Usd.Stage.Open(GELSIGHT_PULLED_DRAWER_FRANKA_USD)
    assert base_stage is not None
    assert pulled_stage is not None

    def _translate(stage: Usd.Stage, path: str) -> tuple[float, float, float]:
        value = stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Get()
        return tuple(float(component) for component in value)

    for body_name in ("panda_link8", "panda_hand"):
        path = f"/panda/{body_name}"
        assert _translate(pulled_stage, path) == pytest.approx(_translate(base_stage, path))

    for body_name in (
        "panda_leftfinger",
        "panda_rightfinger",
        "gelsight_mini_case_left",
        "gelsight_mini_case_right",
        "gelpad_left",
        "gelpad_right",
        "panda_fingertip_centered",
    ):
        path = f"/panda/{body_name}"
        base = _translate(base_stage, path)
        pulled = _translate(pulled_stage, path)
        assert pulled == pytest.approx(
            (base[0], base[1], base[2] - GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M)
        )

    hand_joint = pulled_stage.GetPrimAtPath("/panda/panda_link8/panda_hand_joint")
    assert tuple(hand_joint.GetAttribute("physics:localPos0").Get()) == pytest.approx(
        (0.0, 0.0, 0.0)
    )
    for joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
        joint = pulled_stage.GetPrimAtPath(f"/panda/panda_hand/{joint_name}")
        assert tuple(joint.GetAttribute("physics:localPos0").Get()) == pytest.approx(
            (0.0, 0.0, PULLED_DRAWER_PANDA_FINGER_JOINT_LOCAL_POS0_Z_M)
        )
    centered_joint = pulled_stage.GetPrimAtPath(
        "/panda/panda_hand/panda_fingertip_centered"
    )
    assert centered_joint.GetAttribute("physics:localPos0").Get()[2] == pytest.approx(
        -0.15633872
    )

    geometry = pulled_drawer_geometry_contract()["robot_end_effector"]
    assert geometry["panda_link8_to_hand_fixed_joint_local_pos0_m"] == [0.0, 0.0, 0.0]
    assert geometry["finger_and_gelsight_extension_from_base_asset_m"] == pytest.approx(
        GELSIGHT_PULLED_DRAWER_FINGER_EXTENSION_M
    )
    assert geometry["panda_hand_to_gelpad_midpoint_m"] == pytest.approx(
        PULLED_DRAWER_GELSIGHT_CENTER_OFFSET_HAND_M
    )
    assert geometry["panda_hand_to_fingertip_bottom_m"] == pytest.approx(
        PULLED_DRAWER_GELSIGHT_BOTTOM_OFFSET_HAND_M
    )
