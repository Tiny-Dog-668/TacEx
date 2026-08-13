"""Three-frame Student with one fixed-size Cube rigid body per environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import configclass
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils

from .sim2real_cube_real_alignment_rma_env import _CRITIC_OBSERVATION_SPACE
from .sim2real_cube_real_alignment_rma_x040_wide_env import (
    Sim2RealCubeRealAlignmentRMAX040WideStudentDREnv,
    Sim2RealCubeRealAlignmentRMAX040WideStudentDREnvCfg,
)


_STATIC_CUBE_SIZE_BUCKETS_M = tuple(
    0.04 + index * (0.02 / 7.0) for index in range(8)
)


def _make_isolated_scene_cfg():
    scene_cfg = Sim2RealCubeRealAlignmentRMAX040WideStudentDREnvCfg().scene.copy()
    scene_cfg.env_spacing = 3.5
    return scene_cfg


def _make_invisible_global_ground_cfg():
    ground_cfg = Sim2RealCubeRealAlignmentRMAX040WideStudentDREnvCfg().ground.copy()
    ground_cfg.spawn = ground_cfg.spawn.copy()
    ground_cfg.spawn.visible = False
    return ground_cfg


class _StaticCubeSizeBucketMixin:
    """Assign ``env_id % bucket_count`` once and retain one Cube per env."""

    def _setup_scene(self) -> None:
        bucket_count = len(self.cfg.cube_size_buckets_m)
        if bucket_count == 0:
            raise ValueError("cube_size_buckets_m must not be empty")
        if self.num_envs % bucket_count != 0:
            raise ValueError(
                f"Static Size-Buckets requires num_envs to be a multiple of "
                f"{bucket_count}, got {self.num_envs}"
            )
        self._validate_isolated_environment_layout()
        if "cuda" in str(self.device):
            physics_context = self.sim.get_physics_context()
            if not hasattr(physics_context, "enable_gpu_dynamics"):
                raise RuntimeError(
                    "Static Size-Buckets requires PhysicsContext.enable_gpu_dynamics()"
                )
            physics_context.enable_gpu_dynamics(True)
            print(
                "[INFO] Static Size-Buckets PhysX GPU dynamics enabled; "
                "one fixed Cube rigid body is assigned to each environment.",
                flush=True,
            )

        super()._setup_scene()
        ground_prim = sim_utils.stage_utils.get_current_stage().GetPrimAtPath(
            self.cfg.ground.prim_path
        )
        if not ground_prim.IsValid():
            raise RuntimeError(
                f"Shared GroundPlane prim is missing: {self.cfg.ground.prim_path}"
            )
        # Isaac Lab 2.1.1's spawn_ground_plane() does not route through the
        # generic spawner visibility wrapper, so enforce the configured hidden
        # state explicitly before simulation starts. USD visibility never
        # disables the Plane collision schema.
        ground_imageable = UsdGeom.Imageable(ground_prim)
        ground_imageable.MakeInvisible()
        if ground_imageable.ComputeVisibility() != UsdGeom.Tokens.invisible:
            raise RuntimeError("Failed to hide the shared Appearance GroundPlane")
        self._author_fixed_cube_scales()
        # replicate_physics=False is required by heterogeneous sizes, so the
        # cloner cannot install inter-environment collision filtering for us.
        self.scene.filter_collisions(
            global_prim_paths=[self.cfg.ground.prim_path]
        )

    def _validate_isolated_environment_layout(self) -> None:
        """Fail closed when an env-owned board can overlap a neighboring env."""
        spacing_m = float(self.cfg.scene.env_spacing)
        clearance_m = float(self.cfg.appearance_min_inter_env_clearance_m)
        plate_xy_m = tuple(float(value) for value in self.cfg.plate.spawn.size[:2])
        backdrop_xy_m = tuple(
            float(value) for value in self.cfg.backdrop.spawn.size[:2]
        )
        max_horizontal_extent_m = max(*plate_xy_m, *backdrop_xy_m)
        required_spacing_m = max_horizontal_extent_m + clearance_m
        if spacing_m < required_spacing_m:
            raise ValueError(
                "Appearance env assets are not spatially isolated: "
                f"env_spacing={spacing_m:.3f} m, required>={required_spacing_m:.3f} m"
            )
        if bool(self.cfg.ground.spawn.visible):
            raise ValueError(
                "Appearance isolated layout requires the shared GroundPlane to be invisible"
            )

    def _setup_cube_target(self) -> None:
        # Shape [N]. Assignment is deterministic and never mutated by reset.
        self._active_cube_bucket_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        ) % len(self.cfg.cube_size_buckets_m)
        self._cube_size_bucket_values = torch.tensor(
            self.cfg.cube_size_buckets_m,
            device=self.device,
            dtype=torch.float32,
        )
        super()._setup_cube_target()

    def _author_fixed_cube_scales(self) -> None:
        """Author per-env USD scales before PhysX parses collision geometry."""
        if bool(self.cfg.scene.replicate_physics):
            raise RuntimeError(
                "Static Size-Buckets requires scene.replicate_physics=False for "
                "heterogeneous per-environment collision geometry"
            )
        prim_paths = sim_utils.find_matching_prim_paths(self.cfg.cube.prim_path)
        if len(prim_paths) != self.num_envs:
            raise RuntimeError(
                f"Expected one Cube prim per environment, found {len(prim_paths)} "
                f"for {self.num_envs} environments"
            )

        nominal_size = float(self.cfg.cube.spawn.size[0])
        if nominal_size <= 0.0:
            raise ValueError("Nominal Cube size must be positive")
        stage = sim_utils.stage_utils.get_current_stage()
        with Sdf.ChangeBlock():
            for env_id, prim_path in enumerate(prim_paths):
                size_m = float(
                    self.cfg.cube_size_buckets_m[
                        env_id % len(self.cfg.cube_size_buckets_m)
                    ]
                )
                scale = size_m / nominal_size
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
                scale_spec = prim_spec.GetAttributeAtPath(
                    prim_path + ".xformOp:scale"
                )
                has_scale_attr = scale_spec is not None
                if not has_scale_attr:
                    scale_spec = Sdf.AttributeSpec(
                        prim_spec,
                        prim_path + ".xformOp:scale",
                        Sdf.ValueTypeNames.Double3,
                    )
                scale_spec.default = Gf.Vec3d(scale, scale, scale)
                if not has_scale_attr:
                    order_spec = prim_spec.GetAttributeAtPath(
                        prim_path + ".xformOpOrder"
                    )
                    if order_spec is None:
                        order_spec = Sdf.AttributeSpec(
                            prim_spec,
                            UsdGeom.Tokens.xformOpOrder,
                            Sdf.ValueTypeNames.TokenArray,
                        )
                    order_spec.default = Vt.TokenArray(
                        ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
                    )

    @property
    def active_cube_size_m(self) -> torch.Tensor:
        """Fixed per-environment physical edge length, shape ``[N]``."""
        return self._cube_size_bucket_values[self._active_cube_bucket_ids]

    def _active_cube_root_prim_path(self, env_id: int) -> str:
        return f"/World/envs/env_{env_id}/cube"

    def _set_default_cube_spawn_heights(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        # default_root_state is [N,13] in each environment's local frame.
        self._cube.data.default_root_state[env_ids, 2] = (
            float(self.cfg.plate_top_height_m)
            + 0.5 * self.active_cube_size_m[env_ids]
        )

    def _initialize_cylinder_positions(self) -> None:
        self._set_default_cube_spawn_heights()
        super()._initialize_cylinder_positions()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._set_default_cube_spawn_heights(env_ids)
        super()._reset_idx(env_ids)

        # Sample [K,7] arm-joint offsets in radians. Finger positions and all
        # joint velocities retain the inherited reset contract.
        std_rad = float(self.cfg.arm_joint_reset_noise_std_rad)
        clip_rad = float(self.cfg.arm_joint_reset_noise_clip_rad)
        if std_rad < 0.0 or clip_rad < 0.0:
            raise ValueError("Arm-joint reset noise std/clip must be non-negative")
        noise = torch.randn(
            (env_ids.numel(), 7),
            device=self.device,
            dtype=self._robot.data.default_joint_pos.dtype,
        ) * std_rad
        noise.clamp_(min=-clip_rad, max=clip_rad)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        arm_lower = self._robot.data.soft_joint_pos_limits[env_ids, :7, 0]
        arm_upper = self._robot.data.soft_joint_pos_limits[env_ids, :7, 1]
        noisy_arm_pos = torch.clamp(
            joint_pos[:, :7] + noise,
            min=arm_lower,
            max=arm_upper,
        )
        joint_pos[:, :7] = noisy_arm_pos
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if not hasattr(self, "_last_arm_joint_reset_noise_rad"):
            self._last_arm_joint_reset_noise_rad = torch.zeros(
                (self.num_envs, 7),
                device=self.device,
                dtype=joint_pos.dtype,
            )
        self._last_arm_joint_reset_noise_rad[env_ids] = (
            noisy_arm_pos - self._robot.data.default_joint_pos[env_ids, :7]
        )

    def _compute_cube_lowest_height(
        self,
        cube_pos: torch.Tensor,
        cube_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Return the fixed-size cube's lowest world corner for ``[N,3]/[N,4]``."""
        if cube_pos.shape[0] != self.num_envs:
            raise ValueError("Static cube geometry expects one pose per environment")
        unit_corners = torch.tensor(
            [
                [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
            ],
            device=cube_pos.device,
            dtype=cube_pos.dtype,
        )
        # [N,8,3] local corners -> [N*8,3] rotation -> [N,8,3].
        local_corners = unit_corners.unsqueeze(0) * self.active_cube_size_m[:, None, None]
        env_corners = local_corners.reshape(-1, 3)
        env_quats = cube_quat.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)
        from isaaclab.utils import math as math_utils

        rotated = math_utils.quat_apply(env_quats, env_corners).view(
            self.num_envs, 8, 3
        )
        return (rotated + cube_pos.unsqueeze(1))[:, :, 2].amin(dim=1)

    def _compute_additional_reward(self):
        reward, log = super()._compute_additional_reward()
        log["info/cube_size_mean_m"] = self.active_cube_size_m.mean().detach()
        log["info/cube_size_min_m"] = self.active_cube_size_m.min().detach()
        log["info/cube_size_max_m"] = self.active_cube_size_m.max().detach()
        if hasattr(self, "_last_arm_joint_reset_noise_rad"):
            reset_noise_abs = self._last_arm_joint_reset_noise_rad.abs()
            log["info/arm_joint_reset_noise_abs_mean_rad"] = (
                reset_noise_abs.mean().detach()
            )
            log["info/arm_joint_reset_noise_abs_max_rad"] = (
                reset_noise_abs.max().detach()
            )
        return reward, log


@configclass
class Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsStudentDREnvCfg(
    Sim2RealCubeRealAlignmentRMAX040WideStudentDREnvCfg
):
    """One fixed Cube per env, evenly assigned across eight size buckets."""

    appearance_scene_layout_profile = "per_environment_non_overlapping_v1"
    appearance_min_inter_env_clearance_m = 0.5
    appearance_global_ground_visible = False
    scene = _make_isolated_scene_cfg()
    ground = _make_invisible_global_ground_cfg()

    cube_size_buckets_m = _STATIC_CUBE_SIZE_BUCKETS_M
    cube_size_sampling = "fixed_round_robin_by_environment"
    cube_size_assignment = "env_id_mod_bucket_count"
    cube_size_object_count_per_environment = 1
    cube_size_requires_num_envs_multiple_of_bucket_count = True
    enable_gpu_dynamics = True
    arm_joint_reset_noise_distribution = "normal_clipped_per_environment_per_reset"
    arm_joint_reset_noise_std_rad = 0.01
    arm_joint_reset_noise_clip_rad = 0.03


class Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsStudentDREnv(
    _StaticCubeSizeBucketMixin,
    Sim2RealCubeRealAlignmentRMAX040WideStudentDREnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsStudentDREnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsThreeFrameStudentDREnvCfg(
    Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsStudentDREnvCfg
):
    """Static Size-Buckets Student exposing the latest three wrist RGB frames."""

    wrist_rgb_history_length = 3
    wrist_rgb_history_stride_policy_steps = 1
    wrist_rgb_history_order = "oldest_to_newest"
    wrist_rgb_history_reset_fill = "repeat_first_post_reset_frame"
    observation_space = {
        "proprio_obs": 15,
        "action_history": 4,
        "wrist_rgb_history": gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 224, 224, 3),
            dtype=np.uint8,
        ),
        "rma_cube_pos": 3,
        **_CRITIC_OBSERVATION_SPACE,
    }


class Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsThreeFrameStudentDREnv(
    Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsStudentDREnv
):
    """Maintain a reset-safe ``[oldest, middle, newest]`` uint8 frame history."""

    cfg: Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsThreeFrameStudentDREnvCfg

    def _ensure_wrist_rgb_history(self) -> None:
        expected_shape = (self.num_envs, 3, 224, 224, 3)
        if (
            not hasattr(self, "_wrist_rgb_history")
            or self._wrist_rgb_history.shape != expected_shape
        ):
            self._wrist_rgb_history = torch.zeros(
                expected_shape,
                device=self.device,
                dtype=torch.uint8,
            )
            self._wrist_rgb_history_reset_pending = torch.ones(
                (self.num_envs,),
                device=self.device,
                dtype=torch.bool,
            )

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self._ensure_wrist_rgb_history()
        self._wrist_rgb_history_reset_pending[
            env_ids.to(device=self.device, dtype=torch.long)
        ] = True

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        policy = observations["policy"]
        current_rgb = policy.pop("wrist_rgb")
        self._ensure_wrist_rgb_history()

        # Input [N,224,224,3]; output [N,3,224,224,3], oldest to newest.
        self._wrist_rgb_history[:, 0].copy_(self._wrist_rgb_history[:, 1])
        self._wrist_rgb_history[:, 1].copy_(self._wrist_rgb_history[:, 2])
        self._wrist_rgb_history[:, 2].copy_(current_rgb)
        reset_ids = torch.nonzero(
            self._wrist_rgb_history_reset_pending,
            as_tuple=False,
        ).squeeze(-1)
        if reset_ids.numel() > 0:
            repeated = current_rgb[reset_ids].unsqueeze(1).expand(
                -1, 3, -1, -1, -1
            )
            self._wrist_rgb_history[reset_ids] = repeated
            self._wrist_rgb_history_reset_pending[reset_ids] = False
        policy["wrist_rgb_history"] = self._wrist_rgb_history
        return observations


__all__ = (
    "Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsStudentDREnv",
    "Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsThreeFrameStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsThreeFrameStudentDREnv",
)
