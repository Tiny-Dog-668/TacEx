"""Minimal sim-to-real cube grasping scene using Franka and a fixed third-person camera."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
import isaacsim.core.utils.prims as prim_utils

from ..cylinder_grasping.cylinder_grasping_vision_only_resnet18 import CylinderGraspingVisionOnlyEnv
from .sim2real_grasp_env import Sim2RealGraspEnv, Sim2RealGraspEnvCfg


@configclass
class Sim2RealCubeGraspEnvCfg(Sim2RealGraspEnvCfg):
    """Cube grasping scene simplified for a clean sim-to-real baseline."""

    cube_size = (0.06, 0.06, 0.06)

    wrist_camera: TiledCameraCfg = TiledCameraCfg(
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
            pos=(1.190000, 0.000000, 1.260000),
            rot=(0.683818, 0.179979, 0.179979, 0.683818),
            convention="opengl",
        ),
    )

    observation_space = {
        "proprio_obs": 15,
        "action_history": 4,
        "wrist_resnet": 512,
        "critic_cube_pos": 3,
        "critic_cube_quat": 4,
        "critic_cube_lin_vel": 3,
        "critic_cube_ang_vel": 3,
        "critic_gripper_pos": 3,
        "critic_gripper_quat": 4,
        "critic_gripper_lin_vel": 3,
        "critic_gripper_ang_vel": 3,
        "critic_target_pos": 3,
        "critic_target_distance": 1,
    }

    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0.0, 0.01 + 0.5 * cube_size[2]]),
        spawn=sim_utils.CuboidCfg(
            size=cube_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=1.0,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0005,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.95, 0.94),
                metallic=0.0,
                roughness=0.88,
            ),
        ),
    )

    cube_x_pos_range = 0.10
    cube_y_pos_range = 0.05
    cube_rot_range = 0.0
    cube_half_xy_extent = 0.5 * max(cube_size[0], cube_size[1])

    # Legacy absolute thresholds are retained for saved-config compatibility.
    # Cube reward/done use the reset-relative center-of-mass delta fields below.
    lift_reward_start_lowest_height = 0.007
    success_lowest_height = 0.05
    lift_reference_mode = "center_of_mass"
    lift_reward_start_delta = 0.0
    success_lift_delta = 0.035
    success_hold_steps = 5
    # A real deployment has no simulator ground-truth object XY after Actor
    # inference, so all sim2real Cube variants execute the requested xyz action
    # without the inherited privileged near-table dz gate.
    privileged_dz_gate_enabled = False
    # Lift is a center-height achievement and remains dense even when the cube
    # is tilted. The tilt curriculum gates success only.
    lift_reward_requires_upright = False
    success_requires_upright = True

    # Start permissive so the policy can first learn grasp/lift, then tighten
    # the success-quality requirement. DirectRLEnv.common_step_counter advances
    # once per outer/policy step and is independent of num_envs.
    lift_tilt_curriculum_enabled = True
    lift_tilt_curriculum_start_deg = 40.0
    lift_tilt_curriculum_end_deg = 10.0
    lift_tilt_curriculum_start_step = 0
    lift_tilt_curriculum_end_step = 120_000
    lift_tilt_curriculum_step_offset = 0
    lift_upright_tilt_threshold_deg = 10.0

    # Retained for inherited object geometry/reset helpers.
    cylinder_radius = cube_half_xy_extent


class Sim2RealCubeGraspEnv(Sim2RealGraspEnv):
    """Minimal third-person Franka cube grasping environment for sim-to-real tests."""

    cfg: Sim2RealCubeGraspEnvCfg

    def _setup_scene(self):
        """Setup the scene with a single cube target."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cube = RigidObject(self.cfg.cube)
        self._cylinder = self._cube  # Compatibility for inherited action gate and visual encoder helpers.
        self.scene.rigid_objects["cube"] = self._cube

        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        self.scene.clone_environments(copy_from_source=False)

        if prim_utils.is_prim_path_valid("/World/defaultDomeLight"):
            prim_utils.delete_prim("/World/defaultDomeLight")

        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.107)),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _sync_cap_visual(self, env_ids: torch.Tensor | None = None):
        """Cube target has no separate cap visual."""
        return

    def _sample_cube_xy_offsets(self, count: int) -> torch.Tensor:
        offsets = torch.empty((count, 2), device=self.device, dtype=torch.float32)
        offsets[:, 0].uniform_(-float(self.cfg.cube_x_pos_range), float(self.cfg.cube_x_pos_range))
        offsets[:, 1].uniform_(-float(self.cfg.cube_y_pos_range), float(self.cfg.cube_y_pos_range))
        return offsets

    def _initialize_cylinder_positions(self):
        """Initialize cube positions with sim2real x/y randomization and yaw randomization."""
        cube_state = self._cube.data.default_root_state.clone()
        cube_state[:, :2] += self._sample_cube_xy_offsets(self.num_envs)

        rand_yaw = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        rand_yaw.uniform_(-float(self.cfg.cube_rot_range), float(self.cfg.cube_rot_range))
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        cube_state[:, 3:7] = rand_quat

        # Input state/origins: [N,13] / [N,3]. With CPU PhysX dynamics,
        # InteractiveScene may retain origins on CPU while state tensors stay
        # on the configured CUDA policy device.
        env_origins = self.scene.env_origins.to(
            device=cube_state.device,
            dtype=cube_state.dtype,
        )
        cube_state[:, :3] += env_origins
        self._cube.write_root_state_to_sim(cube_state, torch.arange(self.num_envs, device=self.device))

    def _compute_cube_upright_cos(self, cube_quat: torch.Tensor) -> torch.Tensor:
        local_z_axis = self._cylinder_local_z_axis[: cube_quat.shape[0]]
        cube_up_axis = math_utils.quat_apply(cube_quat, local_z_axis)
        return torch.clamp(cube_up_axis[:, 2], min=-1.0, max=1.0)

    def _compute_cube_lowest_height(self, cube_pos: torch.Tensor, cube_quat: torch.Tensor) -> torch.Tensor:
        """Compute the cube's lowest world-space corner height."""
        sx, sy, sz = (float(v) for v in self.cfg.cube_size)
        corners = torch.tensor(
            [
                [-0.5 * sx, -0.5 * sy, -0.5 * sz],
                [-0.5 * sx, -0.5 * sy, 0.5 * sz],
                [-0.5 * sx, 0.5 * sy, -0.5 * sz],
                [-0.5 * sx, 0.5 * sy, 0.5 * sz],
                [0.5 * sx, -0.5 * sy, -0.5 * sz],
                [0.5 * sx, -0.5 * sy, 0.5 * sz],
                [0.5 * sx, 0.5 * sy, -0.5 * sz],
                [0.5 * sx, 0.5 * sy, 0.5 * sz],
            ],
            device=self.device,
            dtype=cube_pos.dtype,
        )
        env_corners = corners.unsqueeze(0).expand(cube_pos.shape[0], -1, -1).reshape(-1, 3)
        env_quats = cube_quat.unsqueeze(1).expand(-1, corners.shape[0], -1).reshape(-1, 4)
        rotated = math_utils.quat_apply(env_quats, env_corners).view(cube_pos.shape[0], corners.shape[0], 3)
        world_corners = rotated + cube_pos.unsqueeze(1)
        return torch.min(world_corners[:, :, 2], dim=1).values

    def _cube_rest_separation(self) -> float:
        """Return the configured cube/table rest-offset separation in meters."""

        def rest_offset(asset_cfg: RigidObjectCfg) -> float:
            collision_props = getattr(asset_cfg.spawn, "collision_props", None)
            value = getattr(collision_props, "rest_offset", None)
            return 0.0 if value is None else float(value)

        return rest_offset(self.cfg.cube) + rest_offset(self.cfg.plate)

    def _current_lift_tilt_threshold_deg(self) -> float:
        """Return the curriculum-scaled maximum allowed cube tilt."""
        if not bool(getattr(self.cfg, "lift_tilt_curriculum_enabled", False)):
            return float(self.cfg.lift_upright_tilt_threshold_deg)

        start_step = int(self.cfg.lift_tilt_curriculum_start_step)
        end_step = max(int(self.cfg.lift_tilt_curriculum_end_step), start_step + 1)
        step = int(getattr(self, "common_step_counter", 0)) + int(
            self.cfg.lift_tilt_curriculum_step_offset
        )
        progress = min(max((step - start_step) / float(end_step - start_step), 0.0), 1.0)
        start_deg = float(self.cfg.lift_tilt_curriculum_start_deg)
        end_deg = float(self.cfg.lift_tilt_curriculum_end_deg)
        return start_deg + progress * (end_deg - start_deg)

    def _ensure_cube_lift_reference_height(self) -> None:
        """Create the per-env settled center-height reference when first needed."""
        if hasattr(self, "_cube_lift_reference_height_per_env"):
            return
        self._cube_lift_reference_height_per_env = (
            self._cylinder_spawn_height_per_env + self._cube_rest_separation()
        ).detach().clone()

    def _compute_cube_lift_terms(
        self,
        current_center_height: torch.Tensor,
        upright_cos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute settled-center-relative lift progress and success."""
        self._ensure_cube_lift_reference_height()
        lift_delta = current_center_height - self._cube_lift_reference_height_per_env

        lift_start_delta = max(0.0, float(self.cfg.lift_reward_start_delta))
        success_lift_delta = max(float(self.cfg.success_lift_delta), lift_start_delta + 1e-6)
        lift_span = success_lift_delta - lift_start_delta
        lift_progress = torch.clamp(
            (lift_delta - lift_start_delta) / lift_span,
            min=0.0,
            max=1.0,
        )

        tilt_threshold_deg = self._current_lift_tilt_threshold_deg()
        upright_cos_threshold = math.cos(math.radians(tilt_threshold_deg))
        upright = upright_cos >= upright_cos_threshold
        lift_reward = self._shape_lift_reward(lift_progress, upright)
        # Treat the configured millimeter boundary as inclusive despite normal
        # float32 subtraction error in current_height - reference_height.
        success = lift_delta >= success_lift_delta - 1e-6
        if bool(getattr(self.cfg, "success_requires_upright", True)):
            success &= upright
        return lift_delta, lift_reward, success

    def _shape_lift_reward(
        self, lift_progress: torch.Tensor, upright: torch.Tensor
    ) -> torch.Tensor:
        """Map normalized absolute lift progress ``[N]`` to a reward signal."""
        return (
            lift_progress * upright.float()
            if bool(self.cfg.lift_reward_requires_upright)
            else lift_progress
        )

    def _shape_success_reward(self, success: torch.Tensor) -> torch.Tensor:
        """Map the instantaneous success predicate ``[N]`` to reward values."""
        return success.float()

    def _shape_reach_reward(self, reach_proximity: torch.Tensor) -> torch.Tensor:
        """Map normalized reach proximity ``[N]`` to a reward signal."""
        return reach_proximity

    def _compute_additional_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return task-specific additive reward terms and scalar log values."""
        return torch.zeros(self.num_envs, device=self.device), {}

    def _additional_reward_print_fields(self) -> str:
        """Return task-specific fields appended to the periodic reward line."""
        return ""

    def _update_reward_print_window(self, rewards: torch.Tensor) -> torch.Tensor:
        """Accumulate the mean per-env reward over the current print window."""
        if not hasattr(self, "_reward_print_window_sum"):
            self._reward_print_window_sum = torch.zeros((), device=self.device, dtype=rewards.dtype)
            self._reward_print_window_count = 0
        self._reward_print_window_sum.add_(rewards.mean().detach())
        self._reward_print_window_count += 1
        return self._reward_print_window_sum / max(self._reward_print_window_count, 1)

    def _reset_reward_print_window(self) -> None:
        if hasattr(self, "_reward_print_window_sum"):
            self._reward_print_window_sum.zero_()
            self._reward_print_window_count = 0

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return clean cube critic keys while reusing the sim2real visual/proprio path."""
        observations = super()._get_observations()
        obs = observations["policy"]
        rename_map = {
            "critic_cylinder_pos": "critic_cube_pos",
            "critic_cylinder_quat": "critic_cube_quat",
            "critic_cylinder_lin_vel": "critic_cube_lin_vel",
            "critic_cylinder_ang_vel": "critic_cube_ang_vel",
        }
        for old_key, new_key in rename_map.items():
            if old_key in obs:
                obs[new_key] = obs.pop(old_key)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Use reach + settled-center lift shaping + sustained success reward."""
        cube_pos = self._cube.data.root_pos_w
        ee_pos = self._compute_reach_center_world()

        sigma = max(self.cfg.reach_sigma, 1e-6)
        reach_distance = torch.norm(cube_pos - ee_pos, dim=-1)
        reach_proximity = 1.0 - torch.tanh(reach_distance / sigma)
        reach_reward = self._shape_reach_reward(reach_proximity)

        upright_cos = self._compute_cube_upright_cos(self._cube.data.root_quat_w)
        upright_tilt_deg = torch.rad2deg(torch.acos(upright_cos))

        current_height = cube_pos[:, 2]
        current_lowest_height = self._compute_cube_lowest_height(cube_pos, self._cube.data.root_quat_w)
        lift_delta, lift_reward, success = self._compute_cube_lift_terms(current_height, upright_cos)
        success_reward = self._shape_success_reward(success)
        tilt_threshold_deg = self._current_lift_tilt_threshold_deg()

        rewards = (
            self.cfg.reach_weight * reach_reward
            + self.cfg.lift_weight * lift_reward
            + self.cfg.success_reward_weight * success_reward
        )
        additional_reward, additional_log = self._compute_additional_reward()
        rewards = rewards + additional_reward
        average_step_reward = self._update_reward_print_window(rewards)

        log = self.extras.setdefault("log", {})
        log["reward/reach"] = reach_reward.mean().detach()
        log["info/reach_proximity"] = reach_proximity.mean().detach()
        log["reward/lift"] = lift_reward.mean().detach()
        log["reward/success"] = success_reward.mean().detach()
        log["reward/total"] = rewards.mean().detach()
        log["info/reach_distance"] = reach_distance.mean().detach()
        log["info/cube_avg_height"] = current_height.mean().detach()
        log["info/cube_lowest_height"] = current_lowest_height.mean().detach()
        log["info/cube_lift_delta"] = lift_delta.mean().detach()
        log["info/cube_upright_cos"] = upright_cos.mean().detach()
        log["info/cube_tilt_deg"] = upright_tilt_deg.mean().detach()
        log["info/lift_tilt_threshold_deg"] = torch.tensor(tilt_threshold_deg, device=self.device)
        log["info/success_requires_upright"] = torch.tensor(
            float(bool(getattr(self.cfg, "success_requires_upright", True))),
            device=self.device,
        )
        log["info/success_hold_steps"] = self._success_hold_counter.to(torch.float32).mean().detach()
        log.update(additional_log)

        if self.reward_print_interval > 0 and (self.step_count + 1) % self.reward_print_interval == 0:
            tilt_limit_text = (
                f"{tilt_threshold_deg:.2f} deg"
                if bool(getattr(self.cfg, "success_requires_upright", True))
                else "disabled"
            )
            print(
                f"[奖励] step {self.step_count + 1}: "
                f"reach={reach_reward.mean().item():.3f} (w={self.cfg.reach_weight}), "
                f"lift={lift_reward.mean().item():.3f} (w={self.cfg.lift_weight}), "
                f"success={success_reward.mean().item():.3f} (w={self.cfg.success_reward_weight}), "
                f"tilt={upright_tilt_deg.mean().item():.2f} deg, "
                f"tilt_limit={tilt_limit_text}, "
                f"hold={self._success_hold_counter.float().mean().item():.2f}/{self.cfg.success_hold_steps}, "
                f"center_z={current_height.mean().item():.4f} m, "
                f"lowest_z={current_lowest_height.mean().item():.4f} m, "
                f"lift_delta={lift_delta.mean().item():.4f} m, "
                f"{self._additional_reward_print_fields()}"
                f"total={rewards.mean().item():.3f}, "
                f"avg_step_total={average_step_reward.item():.3f} "
                f"(last {self._reward_print_window_count} policy steps)"
            )
            self._reset_reward_print_window()

        action_curr = torch.where(
            torch.isfinite(self.processed_actions),
            self.processed_actions,
            torch.zeros_like(self.processed_actions),
        )
        self.prev_actions = action_curr.detach().clone()
        self.step_count += 1

        return rewards

    def _compute_reach_center_world(self) -> torch.Tensor:
        """Return the fixed hand-offset reach point used by legacy Cube tasks."""
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )
        return ee_pos

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate on timeout, severe ground penetration, or sustained cube lift success."""
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).bool()

        current_center_height = self._cube.data.root_pos_w[:, 2]
        upright_cos = self._compute_cube_upright_cos(self._cube.data.root_quat_w)
        _, _, above_success_height = self._compute_cube_lift_terms(current_center_height, upright_cos)
        self._success_hold_counter = torch.where(
            above_success_height,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        success = self._success_hold_counter >= self.cfg.success_hold_steps

        joint_positions = self._robot.data.body_link_pos_w
        joint_z_positions = joint_positions[:, :, 2]
        collision_with_ground = torch.any(joint_z_positions < self.cfg.ground_height, dim=1).bool()

        dones = time_out | collision_with_ground | success
        return dones, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset selected environments and sample cube poses."""
        super(CylinderGraspingVisionOnlyEnv, self)._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros((len(env_ids), joint_pos.shape[1]), device=self.device, dtype=joint_pos.dtype)

        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        cube_state = self._cube.data.default_root_state[env_ids].clone()
        cube_state[:, :2] += self._sample_cube_xy_offsets(len(env_ids))

        rand_yaw = torch.empty((len(env_ids),), device=self.device, dtype=torch.float32)
        rand_yaw.uniform_(-float(self.cfg.cube_rot_range), float(self.cfg.cube_rot_range))
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        cube_state[:, 3:7] = rand_quat

        # Input state/origins: [K,13] / [K,3], aligned to the state device.
        env_origins = self.scene.env_origins.to(
            device=cube_state.device,
            dtype=cube_state.dtype,
        )
        cube_state[:, :3] += env_origins[env_ids]
        self._cube.write_root_state_to_sim(cube_state, env_ids=env_ids)

        self._cylinder_spawn_height_per_env[env_ids] = cube_state[:, 2].clone()
        self._lift_target_height_per_env[env_ids] = self._cylinder_spawn_height_per_env[env_ids] + self.cfg.lift_height
        self._ensure_cube_lift_reference_height()
        self._cube_lift_reference_height_per_env[env_ids] = (
            self._cylinder_spawn_height_per_env[env_ids] + self._cube_rest_separation()
        )

        self.actions[env_ids] = 0
        self.processed_actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.action_history[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self._success_hold_counter[env_ids] = 0
        self._randomize_scene_visuals(env_ids)
