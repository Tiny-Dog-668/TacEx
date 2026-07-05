# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_inv, sample_uniform, subtract_frame_transforms

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINT_NAME = "finger_joint"
GRIPPER_COUPLING_RULES = (
    # (joint_name, multiplier_to_finger, offset_deg)
    ("right_outer_knuckle_joint", 1.0, 0.0),
    ("right_inner_finger_joint", -1.0, 0.0),
    ("right_inner_finger_knuckle_joint", -1.0, 0.0),
    ("left_inner_finger_knuckle_joint", -1.0, 0.0),
    ("left_inner_finger_joint", -1.0, 0.0),
)
GRIP_BODY_CANDIDATE_GROUPS = (
    ("left_inner_finger", "right_inner_finger"),
    ("left_outer_finger", "right_outer_finger"),
)
GRIP_BODY_FALLBACKS = ("base_link", "ee_link", "wrist_3_link")
GRIPPER_LINKAGE_JOINT_NAMES = (
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
)
EE_BODY_NAME_CANDIDATES = ("ee_link", "wrist_3_link", "left_inner_finger", "right_inner_finger")

_TACEX_ROOT = Path(__file__).resolve().parents[5]
_UR_ROBOTIQ_ASSET_DIR = _TACEX_ROOT / "source" / "tacex_assets" / "tacex_assets" / "data" / "Robots" / "URRobotiq"

DEFAULT_UR10_ROBOTIQ_2F85_USD_PATH = str(_UR_ROBOTIQ_ASSET_DIR / "ur10_robotiq_2f85.usda")
UR10_ROBOTIQ_2F85_USD_PATH = os.environ.get("UR10_ROBOTIQ_2F85_USD_PATH", DEFAULT_UR10_ROBOTIQ_2F85_USD_PATH)

OBJECT_SIZE = (0.055, 0.055, 0.055)
GRIPPER_ACTUATOR_EFFORT_LIMIT_SIM = 400.0
GRIPPER_ACTUATOR_VELOCITY_LIMIT_SIM = 40.0
GRIPPER_ACTUATOR_STIFFNESS = 800.0
GRIPPER_ACTUATOR_DAMPING = 60.0
GRIPPER_LINKAGE_EFFORT_LIMIT_SIM = 200.0
GRIPPER_LINKAGE_VELOCITY_LIMIT_SIM = 40.0
GRIPPER_LINKAGE_STIFFNESS = 500.0
GRIPPER_LINKAGE_DAMPING = 40.0


@configclass
class UR10Robotiq2F85ThirdPersonPickPlaceEnvCfg(DirectRLEnvCfg):
    """Standalone UR10 + Robotiq 2F85 pick-place environment with a third-person RGB camera."""

    episode_length_s = 3.33
    decimation = 2
    action_space = 4
    observation_space = 27
    state_space = 0
    robot_usd_path = UR10_ROBOTIQ_2F85_USD_PATH
    gripper_coupling_rules = GRIPPER_COUPLING_RULES
    grip_body_candidate_groups = GRIP_BODY_CANDIDATE_GROUPS
    grip_body_fallbacks = GRIP_BODY_FALLBACKS
    # Additional offset (in world frame, meters) applied to the averaged grip-body position
    # used by rewards/metrics.
    grip_center_world_offset = (0.0, 0.0, -0.1)
    gripper_linkage_joint_names = GRIPPER_LINKAGE_JOINT_NAMES
    ee_body_name_candidates = EE_BODY_NAME_CANDIDATES
    object_height = OBJECT_SIZE[2]

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=True,
    )
    enable_gpu_dynamics = True
    # For GPU dynamics path, use an implicit PhysX-side PD actuator for finger_joint.
    gripper_actuator_effort_limit_sim = GRIPPER_ACTUATOR_EFFORT_LIMIT_SIM
    gripper_actuator_velocity_limit_sim = GRIPPER_ACTUATOR_VELOCITY_LIMIT_SIM
    gripper_actuator_stiffness = GRIPPER_ACTUATOR_STIFFNESS
    gripper_actuator_damping = GRIPPER_ACTUATOR_DAMPING
    gripper_linkage_effort_limit_sim = GRIPPER_LINKAGE_EFFORT_LIMIT_SIM
    gripper_linkage_velocity_limit_sim = GRIPPER_LINKAGE_VELOCITY_LIMIT_SIM
    gripper_linkage_stiffness = GRIPPER_LINKAGE_STIFFNESS
    gripper_linkage_damping = GRIPPER_LINKAGE_DAMPING

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": -0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
                "finger_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm_main": IdealPDActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                effort_limit=150.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "arm_wrist": IdealPDActuatorCfg(
                joint_names_expr=["wrist_[1-3]_joint"],
                effort_limit=28.0,
                stiffness=400.0,
                damping=20.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=GRIPPER_ACTUATOR_EFFORT_LIMIT_SIM,
                velocity_limit_sim=GRIPPER_ACTUATOR_VELOCITY_LIMIT_SIM,
                stiffness=GRIPPER_ACTUATOR_STIFFNESS,
                damping=GRIPPER_ACTUATOR_DAMPING,
            ),
            "gripper_linkage": ImplicitActuatorCfg(
                joint_names_expr=list(gripper_linkage_joint_names),
                effort_limit_sim=GRIPPER_LINKAGE_EFFORT_LIMIT_SIM,
                velocity_limit_sim=GRIPPER_LINKAGE_VELOCITY_LIMIT_SIM,
                stiffness=GRIPPER_LINKAGE_STIFFNESS,
                damping=GRIPPER_LINKAGE_DAMPING,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, 0.0, 0.2475)),
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.25, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    ee_action_scale = 0.01
    ee_ik_lambda = 0.02
    ee_pos_min = (0.20, -0.80, 0.02)
    ee_pos_max = (1.30, 0.80, 1.20)
    # Additive action noise in normalized action space (before action scaling).
    # noisy_action = clamp(action + N(0, std), -1, 1)
    ee_action_noise_std = 0.05
    gripper_action_noise_std = 0.02
    # Incremental gripper control in degrees:
    # gripper_target_deg += action[-1] * gripper_action_scale
    # and then clamp to [open_target_deg, close_target_deg].
    gripper_action_scale = 3
    # Optional down-sampling for gripper target updates.
    gripper_update_interval = 1
    arm_reset_noise = 0.05
    open_target_deg = 0.0
    close_target_deg = 35.0
    table_size = (0.60, 0.80, 0.04)
    table_pos = (0.8, 0.0, 0.02)
    object_spawn_center = (0.8, 0.0, 0.0)
    object_xy_noise = 0.1
    lift_height = 0.07
    reach_radius = 0.08
    fall_height = -0.02

    reward_reach_weight = 1.0
    reward_lift_weight = 1.5
    reward_success_weight = 5.0
    reward_print_interval = 200

    third_person_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                287.0,
                0.0,
                112.89374542236316,
                0.0,
                382.6666666666667,
                121.33333333333333,
                0.0,
                0.0,
                1.0,
            ],
            width=224,
            height=224,
            focal_length=1.0,
            focus_distance=400.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.60000, 0.000000, 1.50000),
            rot=(0.683818, 0.179979, 0.179979, 0.683818),
            convention="opengl",
        ),
    )


class UR10Robotiq2F85ThirdPersonPickPlaceEnv(DirectRLEnv):
    cfg: UR10Robotiq2F85ThirdPersonPickPlaceEnvCfg

    def __init__(self, cfg: UR10Robotiq2F85ThirdPersonPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        joint_name_to_id = {joint_name: idx for idx, joint_name in enumerate(self._robot.joint_names)}
        missing_arm_joints = [joint_name for joint_name in ARM_JOINT_NAMES if joint_name not in joint_name_to_id]
        if missing_arm_joints:
            raise RuntimeError(f"Missing required UR10 arm joints: {missing_arm_joints}")
        if GRIPPER_JOINT_NAME not in joint_name_to_id:
            raise RuntimeError(f"Missing required gripper joint '{GRIPPER_JOINT_NAME}'.")

        self._arm_joint_ids = [joint_name_to_id[joint_name] for joint_name in ARM_JOINT_NAMES]
        self._finger_joint_id = joint_name_to_id[GRIPPER_JOINT_NAME]
        coupling_joint_ids: list[int] = []
        coupling_multipliers: list[float] = []
        coupling_offsets_rad: list[float] = []
        missing_coupling_joints: list[str] = []
        for joint_name, multiplier, offset_deg in self.cfg.gripper_coupling_rules:
            if joint_name not in joint_name_to_id:
                missing_coupling_joints.append(joint_name)
                continue
            coupling_joint_ids.append(joint_name_to_id[joint_name])
            coupling_multipliers.append(float(multiplier))
            coupling_offsets_rad.append(float(torch.deg2rad(torch.tensor(offset_deg)).item()))
        if missing_coupling_joints:
            print(f"[WARN] Missing gripper coupling joints: {missing_coupling_joints}", flush=True)
        self._gripper_coupling_joint_ids = coupling_joint_ids
        self._gripper_coupling_multipliers = torch.tensor(
            coupling_multipliers, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        self._gripper_coupling_offsets = torch.tensor(
            coupling_offsets_rad, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        self._grip_body_ids, self._grip_body_names = self._resolve_grip_body_ids()
        grip_center_world_offset = torch.tensor(self.cfg.grip_center_world_offset, device=self.device, dtype=torch.float32)
        if grip_center_world_offset.shape[0] != 3:
            raise ValueError("Expected grip_center_world_offset to be a 3D tuple (x, y, z).")
        self._grip_center_world_offset = grip_center_world_offset
        self._ee_body_id, self._ee_body_name = self._resolve_ee_body()
        if self._robot.is_fixed_base:
            if self._ee_body_id <= 0:
                raise RuntimeError(
                    f"Invalid EE body '{self._ee_body_name}' with body_id={self._ee_body_id} for a fixed-base robot."
                )
            self._ee_jacobi_idx = self._ee_body_id - 1
        else:
            self._ee_jacobi_idx = self._ee_body_id
        print(f"[INFO] Using IK body: {self._ee_body_name} (body_id={self._ee_body_id})", flush=True)

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": float(self.cfg.ee_ik_lambda)},
        )
        self._ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        self._target_span = max(float(self.cfg.close_target_deg - self.cfg.open_target_deg), 1.0e-6)
        self._object_half_height = float(self.cfg.object_height) * 0.5

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._arm_joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 0].clone()
        self._arm_joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 1].clone()
        self._arm_default_joint_pos = self._robot.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._arm_targets = self._arm_default_joint_pos.clone()
        ee_pos_min = torch.tensor(self.cfg.ee_pos_min, device=self.device, dtype=torch.float32)
        ee_pos_max = torch.tensor(self.cfg.ee_pos_max, device=self.device, dtype=torch.float32)
        if ee_pos_min.shape[0] != 3 or ee_pos_max.shape[0] != 3:
            raise ValueError("Expected ee_pos_min and ee_pos_max to be 3D tuples (x, y, z).")
        self._ee_pos_min = ee_pos_min
        self._ee_pos_max = ee_pos_max
        ee_pos_b, ee_quat_b = self._compute_ee_pose_local()
        self._ee_pos_targets = ee_pos_b.clone()
        self._ee_quat_targets = ee_quat_b.clone()

        self._gripper_targets = torch.full(
            (self.num_envs, 1), float(self.cfg.open_target_deg), device=self.device, dtype=torch.float32
        )
        # Kept as a derived metric for logs/debugging: target above mid-span is considered "closed".
        self._gripper_closed = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self._open_targets = self._gripper_targets.clone()
        self._close_targets = torch.full_like(self._gripper_targets, float(self.cfg.close_target_deg))

        self._table_top_z = float(self.cfg.table_pos[2]) + 0.5 * float(self.cfg.table_size[2])

        object_spawn_nominal = torch.tensor(self.cfg.object_spawn_center, device=self.device, dtype=torch.float32)
        object_spawn_nominal[2] = self._table_top_z + self._object_half_height
        self._object_spawn_nominal = object_spawn_nominal

        self._object_spawn_positions = self._object_spawn_nominal.repeat(self.num_envs, 1)
        self.step_count = 0
        self.reward_print_interval = int(self.cfg.reward_print_interval)
        self._control_step_count = 0
        self._gripper_update_interval = max(int(self.cfg.gripper_update_interval), 1)
        print(
            f"[INFO] Gripper target is updated every {self._gripper_update_interval} control step(s).",
            flush=True,
        )

    def _setup_scene(self):
        self._configure_gpu_dynamics()

        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        spawn_ground_plane(prim_path="/World/defaultGroundPlane", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)

        table_cfg = sim_utils.CuboidCfg(
            size=self.cfg.table_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35), roughness=0.9),
        )
        table_cfg.func("/World/envs/env_.*/Table", table_cfg, translation=self.cfg.table_pos)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

        self.third_person_camera = TiledCamera(self.cfg.third_person_camera)
        self.scene.sensors["third_person_camera"] = self.third_person_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _configure_gpu_dynamics(self):
        if "cuda" not in str(self.device):
            print("[INFO] GPU dynamics disabled because the environment device is not CUDA.", flush=True)
            return

        physics_context = self.sim.get_physics_context()
        if hasattr(physics_context, "enable_gpu_dynamics"):
            physics_context.enable_gpu_dynamics(bool(self.cfg.enable_gpu_dynamics))
            if self.cfg.enable_gpu_dynamics:
                print("[INFO] Enabled GPU dynamics for this environment.", flush=True)
            else:
                print(
                    "[WARN] Disabled GPU dynamics for this environment. "
                    "This may reduce GPU utilization but can help avoid Direct GPU API articulation issues.",
                    flush=True,
                )

    def _resolve_grip_body_ids(self) -> tuple[list[int], list[str]]:
        for body_group in self.cfg.grip_body_candidate_groups:
            try:
                body_ids, body_names = self._robot.find_bodies(list(body_group), preserve_order=True)
            except Exception:
                continue
            if len(body_ids) == len(body_group):
                print(f"[INFO] Using gripper bodies: {body_names}", flush=True)
                return body_ids, body_names

        for body_name in self.cfg.grip_body_fallbacks:
            try:
                body_ids, body_names = self._robot.find_bodies(body_name)
            except Exception:
                continue
            if len(body_ids) >= 1:
                print(f"[INFO] Using fallback grip body: {body_names[0]}", flush=True)
                return [body_ids[0]], [body_names[0]]

        raise RuntimeError(
            "Failed to find a usable gripper body from the configured candidate groups and fallbacks."
        )

    def _resolve_ee_body(self) -> tuple[int, str]:
        for body_name in self.cfg.ee_body_name_candidates:
            try:
                body_ids, body_names = self._robot.find_bodies(body_name)
            except Exception:
                continue
            if len(body_ids) >= 1:
                return body_ids[0], body_names[0]
        raise RuntimeError(f"Failed to find IK body from candidates: {self.cfg.ee_body_name_candidates}")

    def _compute_ee_pose_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id]
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_body_id]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return ee_pos_b, ee_quat_b

    def _compute_ee_jacobian_local(self) -> torch.Tensor:
        jacobian_w = self._robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._arm_joint_ids]
        base_rot_mat = matrix_from_quat(quat_inv(self._robot.data.root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, 0:3, :] = torch.bmm(base_rot_mat, jacobian_w[:, 0:3, :])
        jacobian_b[:, 3:6, :] = torch.bmm(base_rot_mat, jacobian_w[:, 3:6, :])
        return jacobian_b

    def _build_ik_command(self, ee_pos_b: torch.Tensor) -> torch.Tensor:
        if self._ik_controller.action_dim == 3:
            return self._ee_pos_targets
        if self._ik_controller.action_dim == 6:
            delta_pos = self._ee_pos_targets - ee_pos_b
            delta_rot = torch.zeros_like(delta_pos)
            return torch.cat((delta_pos, delta_rot), dim=1)
        if self._ik_controller.action_dim == 7:
            return torch.cat((self._ee_pos_targets, self._ee_quat_targets), dim=1)
        raise RuntimeError(f"Unsupported IK command dimension: {self._ik_controller.action_dim}")

    def _compute_grip_center_world(self) -> torch.Tensor:
        return self._robot.data.body_pos_w[:, self._grip_body_ids].mean(dim=1) + self._grip_center_world_offset

    def _compute_grip_center_local(self) -> torch.Tensor:
        return self._compute_grip_center_world() - self.scene.env_origins

    def _compute_task_terms(self):
        grip_pos = self._compute_grip_center_local()
        object_pos = self._object.data.root_pos_w - self.scene.env_origins
        object_linvel = self._object.data.root_lin_vel_w

        grip_to_object = object_pos - grip_pos
        grip_to_object_dist = torch.linalg.norm(grip_to_object, dim=1)

        lift_amount = object_pos[:, 2] - self._object_spawn_positions[:, 2]
        lifted = lift_amount > float(self.cfg.lift_height)
        success = lifted

        return {
            "grip_pos": grip_pos,
            "object_pos": object_pos,
            "object_linvel": object_linvel,
            "grip_to_object": grip_to_object,
            "grip_to_object_dist": grip_to_object_dist,
            "lift_amount": lift_amount,
            "lifted": lifted,
            "success": success,
        }

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        ee_noise_std = float(self.cfg.ee_action_noise_std)
        gripper_noise_std = float(self.cfg.gripper_action_noise_std)
        if ee_noise_std > 0.0:
            self.actions[:, :3] += torch.randn_like(self.actions[:, :3]) * ee_noise_std
        if gripper_noise_std > 0.0:
            self.actions[:, -1:] += torch.randn_like(self.actions[:, -1:]) * gripper_noise_std
        self.actions = self.actions.clamp(-1.0, 1.0)

        ee_actions = self.actions[:, :3]
        self._ee_pos_targets += ee_actions * float(self.cfg.ee_action_scale)
        self._ee_pos_targets = torch.clamp(self._ee_pos_targets, min=self._ee_pos_min, max=self._ee_pos_max)

        should_update_gripper = (self._control_step_count % self._gripper_update_interval) == 0
        if should_update_gripper:
            gripper_actions = self.actions[:, -1:]
            self._gripper_targets += gripper_actions * float(self.cfg.gripper_action_scale)
            self._gripper_targets = torch.clamp(
                self._gripper_targets, min=float(self.cfg.open_target_deg), max=float(self.cfg.close_target_deg)
            )
            mid_target = 0.5 * float(self.cfg.open_target_deg + self.cfg.close_target_deg)
            self._gripper_closed = self._gripper_targets >= mid_target
        self._control_step_count += 1

    def _apply_action(self):
        ee_pos_b, ee_quat_b = self._compute_ee_pose_local()
        jacobian_b = self._compute_ee_jacobian_local()
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        ik_command = self._build_ik_command(ee_pos_b)
        self._ik_controller.set_command(ik_command, ee_pos_b, ee_quat_b)
        arm_joint_targets = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
        self._arm_targets = torch.clamp(arm_joint_targets, self._arm_joint_lower_limits, self._arm_joint_upper_limits)

        self._robot.set_joint_position_target(self._arm_targets, joint_ids=self._arm_joint_ids)
        finger_target_rad = torch.deg2rad(self._gripper_targets)
        self._robot.set_joint_position_target(finger_target_rad, joint_ids=[self._finger_joint_id])
        if len(self._gripper_coupling_joint_ids) > 0:
            coupled_targets_rad = (
                finger_target_rad * self._gripper_coupling_multipliers + self._gripper_coupling_offsets
            )
            self._robot.set_joint_position_target(coupled_targets_rad, joint_ids=self._gripper_coupling_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        task_terms = self._compute_task_terms()

        arm_joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        arm_joint_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]
        finger_joint_pos_deg = torch.rad2deg(self._robot.data.joint_pos[:, self._finger_joint_id]).unsqueeze(-1)
        finger_joint_vel_deg = torch.rad2deg(self._robot.data.joint_vel[:, self._finger_joint_id]).unsqueeze(-1)

        arm_pos_norm = 2.0 * (arm_joint_pos - self._arm_joint_lower_limits) / (
            self._arm_joint_upper_limits - self._arm_joint_lower_limits + 1.0e-6
        ) - 1.0
        arm_vel_norm = torch.tanh(arm_joint_vel / 4.0)
        gripper_pos_norm = 2.0 * (finger_joint_pos_deg - float(self.cfg.open_target_deg)) / self._target_span - 1.0
        gripper_vel_norm = torch.tanh(finger_joint_vel_deg / 180.0)

        obs = torch.cat(
            (
                arm_pos_norm,
                arm_vel_norm,
                gripper_pos_norm,
                gripper_vel_norm,
                task_terms["grip_to_object"],
                task_terms["object_pos"],
                task_terms["object_linvel"],
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        task_terms = self._compute_task_terms()

        reach_reward = 1.0 - torch.tanh(6.0 * task_terms["grip_to_object_dist"])
        lift_reward = torch.clamp(task_terms["lift_amount"] / float(self.cfg.lift_height), min=0.0, max=1.0)
        success_reward = task_terms["success"].float()

        rewards = (
            float(self.cfg.reward_reach_weight) * reach_reward
            + float(self.cfg.reward_lift_weight) * lift_reward
            + float(self.cfg.reward_success_weight) * success_reward
        )

        if self.reward_print_interval > 0 and (self.step_count + 1) % self.reward_print_interval == 0:
            print(
                f"[奖励] step {self.step_count + 1}: "
                f"reach={reach_reward.mean().item():.3f} "
                f"(w={self.cfg.reward_reach_weight}), "
                f"lift={lift_reward.mean().item():.3f} "
                f"(w={self.cfg.reward_lift_weight}), "
                f"success={success_reward.mean().item():.3f} "
                f"(w={self.cfg.reward_success_weight}), "
                f"dist={task_terms['grip_to_object_dist'].mean().item():.3f}, "
                f"lift_amount={task_terms['lift_amount'].mean().item():.3f}, "
                f"object_height={task_terms['object_pos'][:, 2].mean().item():.3f}, "
                f"gripper_target_deg={self._gripper_targets[:, 0].mean().item():.3f}, "
                f"total={rewards.mean().item():.3f}",
                flush=True,
            )

        self.extras["log"] = {
            "reward/total": rewards.mean().detach(),
            "reward/reach": reach_reward.mean().detach(),
            "reward/lift": lift_reward.mean().detach(),
            "reward/success": success_reward.mean().detach(),
            "metric/grip_to_object_dist": task_terms["grip_to_object_dist"].mean().detach(),
            "metric/object_height": task_terms["object_pos"][:, 2].mean().detach(),
            "metric/lifted_ratio": task_terms["lifted"].float().mean().detach(),
            "metric/success_ratio": task_terms["success"].float().mean().detach(),
            "metric/gripper_target_deg": self._gripper_targets[:, 0].mean().detach(),
            "metric/gripper_closed_ratio": self._gripper_closed.float().mean().detach(),
        }
        self.step_count += 1
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_terms = self._compute_task_terms()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        invalid_state = torch.isnan(self._robot.data.joint_pos[:, self._finger_joint_id]) | torch.isnan(
            self._object.data.root_pos_w[:, 2]
        )
        object_fell = task_terms["object_pos"][:, 2] < float(self.cfg.fall_height)
        terminated = invalid_state | object_fell | task_terms["success"]
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if env_ids.numel() == self.num_envs:
            self._control_step_count = 0

        num_resets = len(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        arm_noise = sample_uniform(-1.0, 1.0, (num_resets, len(self._arm_joint_ids)), device=self.device)
        arm_joint_pos = self._arm_default_joint_pos[env_ids] + float(self.cfg.arm_reset_noise) * arm_noise
        arm_joint_pos = torch.clamp(arm_joint_pos, self._arm_joint_lower_limits, self._arm_joint_upper_limits)

        joint_pos[:, self._arm_joint_ids] = arm_joint_pos
        joint_pos[:, self._finger_joint_id] = torch.deg2rad(
            torch.full((num_resets,), float(self.cfg.open_target_deg), device=self.device)
        )
        joint_vel.zero_()

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._arm_targets[env_ids] = arm_joint_pos
        self._robot.set_joint_position_target(arm_joint_pos, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        ee_pos_b, ee_quat_b = self._compute_ee_pose_local()
        self._ee_pos_targets[env_ids] = ee_pos_b[env_ids]
        self._ee_quat_targets[env_ids] = ee_quat_b[env_ids]
        self._ik_controller.reset(env_ids)

        self._gripper_targets[env_ids] = self._open_targets[env_ids]
        self._gripper_closed[env_ids] = False
        finger_target_rad = torch.deg2rad(self._gripper_targets[env_ids])
        self._robot.set_joint_position_target(
            finger_target_rad, joint_ids=[self._finger_joint_id], env_ids=env_ids
        )
        if len(self._gripper_coupling_joint_ids) > 0:
            coupled_targets_rad = (
                finger_target_rad * self._gripper_coupling_multipliers + self._gripper_coupling_offsets
            )
            self._robot.set_joint_position_target(
                coupled_targets_rad, joint_ids=self._gripper_coupling_joint_ids, env_ids=env_ids
            )

        object_spawn_pos = self._object_spawn_nominal.repeat(num_resets, 1)
        object_spawn_pos[:, :2] += sample_uniform(
            -float(self.cfg.object_xy_noise), float(self.cfg.object_xy_noise), (num_resets, 2), device=self.device
        )

        self._object_spawn_positions[env_ids] = object_spawn_pos

        object_default_state = self._object.data.default_root_state[env_ids].clone()
        object_default_state[:, 0:3] = object_spawn_pos + self.scene.env_origins[env_ids]
        object_default_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
        object_default_state[:, 7:] = 0.0
        self._object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self._object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
