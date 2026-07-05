# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import isaacsim.core.utils.stage as stage_utils
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms
from pxr import UsdPhysics

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINT_NAME = "finger_joint"
EE_BODY_NAME_CANDIDATES = ("ee_link", "wrist_3_link", "left_inner_finger", "right_inner_finger")
_TACEX_ROOT = Path(__file__).resolve().parents[5]
_UR_ROBOTIQ_ASSET_DIR = _TACEX_ROOT / "source" / "tacex_assets" / "tacex_assets" / "data" / "Robots" / "URRobotiq"
UR10_ROBOTIQ_USD_PATH = str(_UR_ROBOTIQ_ASSET_DIR / "ur10_robotiq_f140.usda")


@configclass
class UR10RobotiqGripperCloseEnvCfg(DirectRLEnvCfg):
    episode_length_s = 2.0
    decimation = 2
    action_space = 4
    observation_space = 3
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=UR10_ROBOTIQ_USD_PATH,
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
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
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
        },
        soft_joint_pos_limit_factor=1.0,
    )

    open_target_deg = 0.0
    close_target_deg = 45.0
    gripper_action_scale = 3.0
    ee_action_scale = 0.01
    ee_ik_lambda = 0.02
    ee_pos_min = (0.20, -0.80, 0.02)
    ee_pos_max = (1.30, 0.80, 1.20)
    ee_body_name_candidates = EE_BODY_NAME_CANDIDATES
    usd_drive_stiffness = 50000.0
    usd_drive_damping = 500.0
    usd_drive_max_force = 10000.0
    success_tolerance_deg = 2.0
    close_reward_weight = 1.0
    velocity_penalty_weight = 0.05
    success_bonus = 1.0


class UR10RobotiqGripperCloseEnv(DirectRLEnv):
    cfg: UR10RobotiqGripperCloseEnvCfg

    def __init__(self, cfg: UR10RobotiqGripperCloseEnvCfg, render_mode: str | None = None, **kwargs):
        self._finger_drive_target_attrs = []
        self._finger_drive_state_pos_attrs = []
        super().__init__(cfg, render_mode, **kwargs)

        joint_name_to_id = {joint_name: idx for idx, joint_name in enumerate(self._robot.joint_names)}
        missing_arm_joints = [joint_name for joint_name in ARM_JOINT_NAMES if joint_name not in joint_name_to_id]
        if missing_arm_joints:
            raise RuntimeError(f"Missing required UR10 arm joints: {missing_arm_joints}")
        if GRIPPER_JOINT_NAME not in joint_name_to_id:
            raise RuntimeError(f"Missing required gripper joint '{GRIPPER_JOINT_NAME}'.")

        self._arm_joint_ids = [joint_name_to_id[joint_name] for joint_name in ARM_JOINT_NAMES]
        self._finger_joint_ids = [joint_name_to_id[GRIPPER_JOINT_NAME]]
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
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": float(self.cfg.ee_ik_lambda)},
        )
        self._ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._arm_joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 0].clone()
        self._arm_joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 1].clone()
        self._gripper_targets = torch.full(
            (self.num_envs, 1),
            float(self.cfg.open_target_deg),
            device=self.device,
            dtype=torch.float32,
        )
        self._open_targets = self._gripper_targets.clone()
        self._close_targets = torch.full_like(self._gripper_targets, float(self.cfg.close_target_deg))
        self._target_span = max(float(self.cfg.close_target_deg - self.cfg.open_target_deg), 1.0e-6)
        self._arm_default_joint_pos = self._robot.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._arm_targets = self._arm_default_joint_pos.clone()
        ee_pos_min = torch.tensor(self.cfg.ee_pos_min, device=self.device, dtype=torch.float32)
        ee_pos_max = torch.tensor(self.cfg.ee_pos_max, device=self.device, dtype=torch.float32)
        if ee_pos_min.shape[0] != 3 or ee_pos_max.shape[0] != 3:
            raise ValueError("Expected ee_pos_min and ee_pos_max to be 3D tuples (x, y, z).")
        self._ee_pos_min = ee_pos_min
        self._ee_pos_max = ee_pos_max
        ee_pos_b, _ = self._compute_ee_pose_local()
        self._ee_pos_targets = ee_pos_b.clone()

    def _setup_scene(self):
        self._apply_gui_gpu_workaround()

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        spawn_ground_plane(prim_path="/World/defaultGroundPlane", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self._cache_gripper_drive_targets()

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_gui_gpu_workaround(self):
        if "cuda" not in str(self.device):
            return

        physics_context = self.sim.get_physics_context()
        if hasattr(physics_context, "enable_gpu_dynamics"):
            physics_context.enable_gpu_dynamics(False)
            print(
                "[WARN] Disabled GPU dynamics for this drive control to avoid Direct GPU API articulation errors.",
                flush=True,
            )

    def _find_revolute_joint_prim(self, root_prim_path: str, joint_name: str):
        matching_joints = []
        stage = stage_utils.get_current_stage()
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(root_prim_path):
                continue
            if prim.GetName() != joint_name:
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                matching_joints.append(prim)
        if len(matching_joints) != 1:
            raise RuntimeError(
                f"Expected exactly one revolute joint named '{joint_name}' under '{root_prim_path}', "
                f"found {len(matching_joints)}."
            )
        return matching_joints[0]

    def _cache_gripper_drive_targets(self):
        drive_target_attrs = []
        drive_state_pos_attrs = []

        for env_prim_path in self.scene.env_prim_paths:
            robot_prim_path = f"{env_prim_path}/Robot"
            joint_prim = self._find_revolute_joint_prim(robot_prim_path, GRIPPER_JOINT_NAME)

            drive_api = UsdPhysics.DriveAPI(joint_prim, "angular")
            target_attr = drive_api.GetTargetPositionAttr()
            if not target_attr.IsValid():
                target_attr = drive_api.CreateTargetPositionAttr(float(self.cfg.open_target_deg))

            drive_api.CreateStiffnessAttr(float(self.cfg.usd_drive_stiffness))
            drive_api.CreateDampingAttr(float(self.cfg.usd_drive_damping))
            drive_api.CreateMaxForceAttr(float(self.cfg.usd_drive_max_force))
            drive_api.CreateTargetVelocityAttr(0.0)

            state_pos_attr = joint_prim.GetAttribute("state:angular:physics:position")
            if not state_pos_attr.IsValid():
                raise RuntimeError(
                    f"Joint '{joint_prim.GetPath()}' does not expose state:angular:physics:position."
                )

            drive_target_attrs.append(target_attr)
            drive_state_pos_attrs.append(state_pos_attr)

            if len(drive_target_attrs) == 1:
                print(f"[INFO] Using finger_joint drive target attr at: {joint_prim.GetPath()}", flush=True)

        if len(drive_target_attrs) != self.num_envs:
            raise RuntimeError(
                f"Expected {self.num_envs} finger drive targets, found {len(drive_target_attrs)}."
            )

        self._finger_drive_target_attrs = drive_target_attrs
        self._finger_drive_state_pos_attrs = drive_state_pos_attrs

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

    def _write_gripper_drive_targets(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            indices = range(self.num_envs)
            targets_deg = self._gripper_targets[:, 0]
        else:
            env_ids = env_ids.to(dtype=torch.long, device=self.device)
            indices = env_ids.detach().cpu().tolist()
            targets_deg = self._gripper_targets[env_ids, 0]

        for env_idx, target_deg in zip(indices, targets_deg.detach().cpu().tolist()):
            self._finger_drive_target_attrs[env_idx].Set(float(target_deg))

    def _read_gripper_drive_state_deg(self) -> torch.Tensor:
        values = []
        for attr in self._finger_drive_state_pos_attrs:
            value = attr.Get()
            if value is None:
                value = 0.0
            values.append(float(value))
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

        ee_actions = self.actions[:, :3]
        self._ee_pos_targets += ee_actions * float(self.cfg.ee_action_scale)
        self._ee_pos_targets = torch.clamp(self._ee_pos_targets, min=self._ee_pos_min, max=self._ee_pos_max)

        gripper_actions = self.actions[:, 3:4]
        self._gripper_targets += gripper_actions * float(self.cfg.gripper_action_scale)
        self._gripper_targets = torch.clamp(
            self._gripper_targets, min=float(self.cfg.open_target_deg), max=float(self.cfg.close_target_deg)
        )

    def _apply_action(self):
        ee_pos_b, ee_quat_b = self._compute_ee_pose_local()
        jacobian_b = self._compute_ee_jacobian_local()
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        self._ik_controller.set_command(self._ee_pos_targets, ee_pos_b, ee_quat_b)
        arm_joint_targets = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
        self._arm_targets = torch.clamp(arm_joint_targets, self._arm_joint_lower_limits, self._arm_joint_upper_limits)
        self._robot.set_joint_position_target(self._arm_targets, joint_ids=self._arm_joint_ids)
        self._write_gripper_drive_targets()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        finger_joint_pos_deg = torch.rad2deg(self._robot.data.joint_pos[:, self._finger_joint_ids[0]])
        finger_joint_vel_deg = torch.rad2deg(self._robot.data.joint_vel[:, self._finger_joint_ids[0]])
        target_deg = self._gripper_targets[:, 0]

        pos_norm = 2.0 * (finger_joint_pos_deg - float(self.cfg.open_target_deg)) / self._target_span - 1.0
        vel_norm = torch.tanh(finger_joint_vel_deg / 180.0)
        target_norm = 2.0 * (target_deg - float(self.cfg.open_target_deg)) / self._target_span - 1.0

        obs = torch.stack((pos_norm, vel_norm, target_norm), dim=-1)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        finger_joint_pos_deg = torch.rad2deg(self._robot.data.joint_pos[:, self._finger_joint_ids[0]])
        finger_joint_vel_deg = torch.abs(torch.rad2deg(self._robot.data.joint_vel[:, self._finger_joint_ids[0]]))
        drive_state_deg = self._read_gripper_drive_state_deg()

        error_deg = torch.abs(finger_joint_pos_deg - self._close_targets[:, 0])
        close_reward = 1.0 - torch.clamp(error_deg / self._target_span, min=0.0, max=1.0)
        velocity_penalty = torch.clamp(finger_joint_vel_deg / 180.0, min=0.0, max=1.0)
        success_bonus = (error_deg < float(self.cfg.success_tolerance_deg)).float() * float(self.cfg.success_bonus)

        rewards = (
            float(self.cfg.close_reward_weight) * close_reward
            - float(self.cfg.velocity_penalty_weight) * velocity_penalty
            + success_bonus
        )

        self.extras["log"] = {
            "reward/close": close_reward.mean().detach(),
            "reward/total": rewards.mean().detach(),
            "info/gripper_target_deg": self._gripper_targets[:, 0].mean().detach(),
            "info/gripper_joint_pos_deg": finger_joint_pos_deg.mean().detach(),
            "info/gripper_drive_state_deg": drive_state_deg.mean().detach(),
            "info/gripper_error_deg": error_deg.mean().detach(),
        }
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.isnan(self._robot.data.joint_pos[:, self._finger_joint_ids[0]])
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self._finger_joint_ids[0]] = torch.deg2rad(
            torch.full((len(env_ids),), float(self.cfg.open_target_deg), device=self.device)
        )

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._arm_targets[env_ids] = self._arm_default_joint_pos[env_ids]
        self._robot.set_joint_position_target(self._arm_targets[env_ids], joint_ids=self._arm_joint_ids, env_ids=env_ids)
        ee_pos_b, _ = self._compute_ee_pose_local()
        self._ee_pos_targets[env_ids] = ee_pos_b[env_ids]
        self._ik_controller.reset(env_ids)

        self._gripper_targets[env_ids] = self._open_targets[env_ids]
        self._write_gripper_drive_targets(env_ids)
