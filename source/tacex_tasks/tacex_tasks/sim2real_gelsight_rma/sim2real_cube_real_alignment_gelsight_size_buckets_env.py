"""Fixed-size-per-env GelSight RMA task with reset-local tactile references."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils

from .sim2real_cube_real_alignment_gelsight_rma_env import (
    _GELSIGHT_REFERENCE_OBSERVATION_SPACE,
    _GELSIGHT_TACTILE_OBSERVATION_SPACE,
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnv,
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg,
)


GELSIGHT_SIZE_BUCKETS_M = tuple(0.04 + index * (0.02 / 7.0) for index in range(8))
GELSIGHT_SIZE_BUCKETS_TEACHER_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Size-Buckets-Teacher-v0"
)
GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Size-Buckets-Student-DR-v0"
)

_CUBE_ILLEGAL_ROBOT_BODY_NAMES = (
    "panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4",
    "panda_link5", "panda_link6", "panda_link7", "panda_link8", "panda_hand",
    "panda_leftfinger", "panda_rightfinger", "panda_fingertip_centered",
    "gelsight_mini_case_left", "gelsight_mini_case_right",
)
_TABLE_ILLEGAL_ROBOT_BODY_NAMES = (
    "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5",
    "panda_link6", "panda_link7", "panda_link8", "panda_hand", "panda_leftfinger",
    "panda_rightfinger", "panda_fingertip_centered", "gelsight_mini_case_left",
    "gelsight_mini_case_right", "gelpad_left", "gelpad_right",
)


def _isolated_scene_cfg(base_cfg_cls: type):
    scene = base_cfg_cls().scene.copy()
    scene.env_spacing = 3.5
    scene.replicate_physics = False
    return scene


def _invisible_ground_cfg(base_cfg_cls: type):
    ground = base_cfg_cls().ground.copy()
    ground.spawn = ground.spawn.copy()
    ground.spawn.visible = False
    return ground


def _visible_ground_cfg(base_cfg_cls: type):
    ground = base_cfg_cls().ground.copy()
    ground.spawn = ground.spawn.copy()
    ground.spawn.visible = True
    return ground


def _full_cell_plate_cfg(base_cfg_cls: type, *, cell_size_m: float = 3.5):
    """Tile one floor panel across an entire environment cell without changing its height."""
    base = base_cfg_cls()
    plate = base.plate.copy()
    plate.init_state = plate.init_state.copy()
    plate.spawn = plate.spawn.copy()
    thickness_m = float(plate.spawn.size[2])
    backdrop_x_m = float(base.backdrop.init_state.pos[0])
    plate.init_state.pos = (
        backdrop_x_m + 0.5 * float(cell_size_m),
        float(plate.init_state.pos[1]),
        float(plate.init_state.pos[2]),
    )
    plate.spawn.size = (float(cell_size_m), float(cell_size_m), thickness_m)
    return plate


def _full_cell_backdrop_cfg(base_cfg_cls: type, *, cell_size_m: float = 3.5):
    """Extend the visual-only backdrop across the full environment-cell width."""
    backdrop = base_cfg_cls().backdrop.copy()
    backdrop.spawn = backdrop.spawn.copy()
    backdrop.spawn.size = (
        float(backdrop.spawn.size[0]),
        float(cell_size_m),
        float(backdrop.spawn.size[2]),
    )
    return backdrop


def _cube_illegal_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/cube",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        filter_prim_paths_expr=[
            f"/World/envs/env_.*/Robot/{name}" for name in _CUBE_ILLEGAL_ROBOT_BODY_NAMES
        ],
    )


def _table_illegal_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/floor_panel",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        filter_prim_paths_expr=[
            f"/World/envs/env_.*/Robot/{name}" for name in _TABLE_ILLEGAL_ROBOT_BODY_NAMES
        ],
    )


def illegal_collision_response(
    force_n: torch.Tensor,
    *,
    penalty_threshold_n: float = 5.0,
    termination_threshold_n: float = 10.0,
    penalty_value: float = -10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map per-env maximum illegal force ``[N]`` to reward and termination."""
    penalized = force_n > penalty_threshold_n
    penalty = penalized.to(force_n.dtype) * penalty_value
    terminated = force_n > termination_threshold_n
    return penalty, terminated


def linear_illegal_collision_penalty_threshold(
    step: int,
    *,
    start_n: float = 20.0,
    end_n: float = 5.0,
    start_step: int = 0,
    end_step: int = 100_000,
) -> float:
    """Linearly tighten the penalty threshold and clamp outside the schedule."""
    if end_step <= start_step:
        raise ValueError("illegal collision curriculum end_step must exceed start_step")
    progress = min(max((int(step) - start_step) / float(end_step - start_step), 0.0), 1.0)
    return start_n + progress * (end_n - start_n)


class _GelSightFixedSizeCubeMixin:
    """Assign one fixed physical Cube size to each environment before PhysX starts."""

    def _setup_scene(self) -> None:
        bucket_count = len(self.cfg.cube_size_buckets_m)
        if bucket_count != 8:
            raise ValueError(f"GelSight Size-Buckets requires exactly 8 sizes, got {bucket_count}")
        if self.num_envs % bucket_count != 0:
            raise ValueError(
                f"GelSight Size-Buckets requires num_envs to be a multiple of 8, got {self.num_envs}"
            )
        if float(self.cfg.scene.env_spacing) < 3.5:
            raise ValueError("GelSight Size-Buckets requires env_spacing >= 3.5 m")
        if bool(self.cfg.scene.replicate_physics):
            raise ValueError("Heterogeneous fixed Cube sizes require replicate_physics=False")
        if "cuda" in str(self.device):
            physics_context = self.sim.get_physics_context()
            if not hasattr(physics_context, "enable_gpu_dynamics"):
                raise RuntimeError("GelSight Size-Buckets requires GPU dynamics support")
            physics_context.enable_gpu_dynamics(True)

        self._active_cube_bucket_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long
        ) % bucket_count
        self._cube_size_bucket_values = torch.tensor(
            self.cfg.cube_size_buckets_m, device=self.device, dtype=torch.float32
        )
        super()._setup_scene()
        self._set_shared_ground_visibility(bool(self.cfg.ground.spawn.visible))
        self._author_fixed_cube_scales()
        self.scene.filter_collisions(global_prim_paths=[self.cfg.ground.prim_path])

    @property
    def active_cube_size_m(self) -> torch.Tensor:
        """Fixed edge length for each environment, shape ``[N]``."""
        return self._cube_size_bucket_values[self._active_cube_bucket_ids]

    def _set_shared_ground_visibility(self, visible: bool) -> None:
        prim = sim_utils.stage_utils.get_current_stage().GetPrimAtPath(self.cfg.ground.prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Shared ground is missing: {self.cfg.ground.prim_path}")
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _author_fixed_cube_scales(self) -> None:
        prim_paths = sim_utils.find_matching_prim_paths(self.cfg.cube.prim_path)
        if len(prim_paths) != self.num_envs:
            raise RuntimeError(f"Expected one Cube per env, found {len(prim_paths)}")
        nominal_size = float(self.cfg.cube.spawn.size[0])
        stage = sim_utils.stage_utils.get_current_stage()
        with Sdf.ChangeBlock():
            for env_id, prim_path in enumerate(prim_paths):
                scale = float(self.cfg.cube_size_buckets_m[env_id % 8]) / nominal_size
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                had_scale = scale_spec is not None
                if not had_scale:
                    scale_spec = Sdf.AttributeSpec(
                        prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3
                    )
                scale_spec.default = Gf.Vec3d(scale, scale, scale)
                if not had_scale:
                    order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                    if order_spec is None:
                        order_spec = Sdf.AttributeSpec(
                            prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                        )
                    order_spec.default = Vt.TokenArray(
                        ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
                    )

    def _set_fixed_cube_spawn_height(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._cube.data.default_root_state[env_ids, 2] = (
            float(self.cfg.plate_top_height_m) + 0.5 * self.active_cube_size_m[env_ids]
        )

    def _initialize_cylinder_positions(self) -> None:
        self._set_fixed_cube_spawn_height()
        super()._initialize_cylinder_positions()

    def _compute_cube_lowest_height(
        self, cube_pos: torch.Tensor, cube_quat: torch.Tensor
    ) -> torch.Tensor:
        """Return the lowest rotated corner for variable-size ``[N,3]/[N,4]``."""
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
        local = unit_corners.unsqueeze(0) * self.active_cube_size_m[:, None, None]
        quaternions = cube_quat[:, None, :].expand(-1, 8, -1).reshape(-1, 4)
        rotated = math_utils.quat_apply(quaternions, local.reshape(-1, 3)).view(
            self.num_envs, 8, 3
        )
        return (rotated + cube_pos[:, None, :])[:, :, 2].amin(dim=1)


class _GelSightFixedSizeCollisionMixin(_GelSightFixedSizeCubeMixin):
    """Add the fixed-size GelSight collision contract to the shared Cube layout."""

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.cube_illegal_contact_sensor = ContactSensor(
            self.cfg.cube_illegal_contact_sensor
        )
        self.scene.sensors["cube_illegal_contact_sensor"] = self.cube_illegal_contact_sensor

    @staticmethod
    def _max_filtered_contact_force(sensor: ContactSensor) -> torch.Tensor:
        history = sensor.data.force_matrix_w_history
        if history is None:
            raise RuntimeError("Filtered contact sensor has no force history")
        # Input [N,H,source_bodies,filters,xyz] -> output [N].
        return torch.linalg.vector_norm(history, dim=-1).flatten(start_dim=1).amax(dim=1)

    def _compute_illegal_collision_force(self) -> torch.Tensor:
        cube_force = self._max_filtered_contact_force(self.cube_illegal_contact_sensor)
        table_force = self._compute_table_robot_contact_force()
        return torch.maximum(cube_force, table_force)

    def _current_illegal_collision_penalty_threshold_n(self) -> float:
        step = int(getattr(self, "common_step_counter", 0)) + int(
            self.cfg.illegal_collision_curriculum_step_offset
        )
        return linear_illegal_collision_penalty_threshold(
            step,
            start_n=float(self.cfg.illegal_collision_penalty_threshold_start_n),
            end_n=float(self.cfg.illegal_collision_penalty_threshold_end_n),
            start_step=int(self.cfg.illegal_collision_curriculum_start_step),
            end_step=int(self.cfg.illegal_collision_curriculum_end_step),
        )

    def _compute_additional_reward(self):
        reward, log = super()._compute_additional_reward()
        illegal_force = self._compute_illegal_collision_force()
        penalty_threshold_n = self._current_illegal_collision_penalty_threshold_n()
        penalty, _ = illegal_collision_response(
            illegal_force,
            penalty_threshold_n=penalty_threshold_n,
            termination_threshold_n=float(self.cfg.illegal_collision_termination_threshold_n),
            penalty_value=float(self.cfg.illegal_collision_penalty),
        )
        collision = penalty != 0
        # The inherited table-only penalty is replaced, not stacked.
        inherited_table_penalty = self._last_table_collision_penalty
        reward = reward - inherited_table_penalty + penalty
        self._last_illegal_collision_force = illegal_force.detach().clone()
        self._last_illegal_collision = collision.detach().clone()
        self._last_illegal_collision_penalty = penalty.detach().clone()
        self._last_illegal_collision_penalty_threshold_n = penalty_threshold_n
        log.pop("reward/table_collision", None)
        log.update({
            "reward/illegal_collision": penalty.mean().detach(),
            "info/illegal_collision_fraction": collision.float().mean().detach(),
            "info/illegal_collision_max_force_n": illegal_force.max().detach(),
            "info/illegal_collision_penalty_threshold_n": torch.tensor(
                penalty_threshold_n, device=self.device, dtype=illegal_force.dtype
            ),
            "info/cube_size_mean_m": self.active_cube_size_m.mean().detach(),
        })
        return reward, log

    def _rma_collision_reward_print_fields(
        self, log: dict[str, torch.Tensor]
    ) -> str:
        return (
            f"illegal_collision={log['reward/illegal_collision'].item():.3f}, "
            f"illegal_ratio={log['info/illegal_collision_fraction'].item():.3f}, "
            f"illegal_threshold={log['info/illegal_collision_penalty_threshold_n'].item():.2f} N, "
            f"illegal_force_max={log['info/illegal_collision_max_force_n'].item():.2f} N, "
        )

    def _additional_reward_print_fields(self) -> str:
        fields = super()._additional_reward_print_fields()
        if not hasattr(self, "_last_illegal_collision"):
            return fields
        return (
            fields
            + f"illegal_collision={self._last_illegal_collision.float().mean().item():.3f}, "
            + f"illegal_force_max={self._last_illegal_collision_force.max().item():.2f} N, "
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).bool()
        current_height = self._cube.data.root_pos_w[:, 2]
        upright_cos = self._compute_cube_upright_cos(self._cube.data.root_quat_w)
        _, _, above_success_height = self._compute_cube_lift_terms(current_height, upright_cos)
        self._success_hold_counter = torch.where(
            above_success_height,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        success = self._success_hold_counter >= int(self.cfg.success_hold_steps)
        self._rma_episode_success_ever |= success
        ground_collision = torch.any(
            self._robot.data.body_link_pos_w[:, :, 2] < float(self.cfg.ground_height), dim=1
        )
        _, illegal_termination = illegal_collision_response(
            self._compute_illegal_collision_force(),
            penalty_threshold_n=self._current_illegal_collision_penalty_threshold_n(),
            termination_threshold_n=float(self.cfg.illegal_collision_termination_threshold_n),
            penalty_value=float(self.cfg.illegal_collision_penalty),
        )
        terminated = ground_collision | illegal_termination
        completed = terminated | time_out
        self._record_episode_outcomes_for_step(
            completed_count=completed.long().sum(),
            success_count=(completed & self._rma_episode_success_ever).long().sum(),
        )
        self._publish_episode_success_statistics()
        self._last_rma_success_nonterminal = success.detach().clone()
        self._last_illegal_collision_termination = illegal_termination.detach().clone()
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._set_fixed_cube_spawn_height(env_ids)
        super()._reset_idx(env_ids)
        self.cube_illegal_contact_sensor.reset(env_ids)


def _new_student_observation_space() -> dict:
    return {
        **Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg().observation_space,
        **_GELSIGHT_TACTILE_OBSERVATION_SPACE,
        **_GELSIGHT_REFERENCE_OBSERVATION_SPACE,
    }


@configclass
class _GelSightSizeBucketsCfgMixin:
    cube_size_buckets_m = GELSIGHT_SIZE_BUCKETS_M
    cube_size_sampling = "fixed_round_robin_by_environment"
    cube_size_assignment = "env_id_mod_8"
    cube_size_object_count_per_environment = 1
    cube_size_requires_num_envs_multiple_of_bucket_count = True
    cube_illegal_contact_sensor = _cube_illegal_sensor_cfg()
    table_contact_sensor = _table_illegal_sensor_cfg()
    illegal_collision_scope = "cube_non_gelpad_robot_or_nonbase_robot_table"
    illegal_collision_penalty_threshold_start_n = 20.0
    illegal_collision_penalty_threshold_end_n = 5.0
    illegal_collision_curriculum_start_step = 0
    illegal_collision_curriculum_end_step = 100_000
    illegal_collision_curriculum_step_offset = 0
    illegal_collision_curriculum_schedule = "linear_clamped_global_policy_step"
    illegal_collision_penalty = -10.0
    illegal_collision_termination_threshold_n = 10000.0
    table_collision_force_threshold_n = 5.0
    table_collision_penalty = -10.0

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnvCfg(
    _GelSightSizeBucketsCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg,
):
    rma_task_id = GELSIGHT_SIZE_BUCKETS_TEACHER_TASK
    scene = _isolated_scene_cfg(Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg)
    ground = _invisible_ground_cfg(Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg)


class Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnv(
    _GelSightFixedSizeCollisionMixin,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnvCfg(
    _GelSightSizeBucketsCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg,
):
    rma_task_id = GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK
    scene = _isolated_scene_cfg(Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg)
    ground = _invisible_ground_cfg(Sim2RealCubeRealAlignmentRMAGelSightStudentDREnvCfg)
    rma_gelsight_reference_enabled = True
    rma_student_contact_observation_source = "gelsight_signed_current_minus_reference"
    observation_space = _new_student_observation_space()


class Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnv(
    _GelSightFixedSizeCollisionMixin,
    Sim2RealCubeRealAlignmentRMAGelSightStudentDREnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnvCfg


__all__ = (
    "GELSIGHT_SIZE_BUCKETS_M",
    "GELSIGHT_SIZE_BUCKETS_TEACHER_TASK",
    "GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK",
    "illegal_collision_response",
    "linear_illegal_collision_penalty_threshold",
    "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsTeacherEnv",
    "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAGelSightSizeBucketsStudentDREnv",
)
