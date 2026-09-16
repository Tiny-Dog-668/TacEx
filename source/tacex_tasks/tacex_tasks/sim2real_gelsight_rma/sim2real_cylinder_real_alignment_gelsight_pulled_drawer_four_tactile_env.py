"""High-cylinder four-tactile Pulled-Drawer Teacher and binary Student environments."""

from __future__ import annotations

import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from pxr import Sdf

from .sim2real_cube_real_alignment_gelsight_pulled_drawer_env import (
    PULLED_DRAWER_PANEL_THICKNESS_M,
    PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M,
    balanced_cube_bucket_ids,
)
from .sim2real_cube_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnv,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg,
)


GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Pulled-Drawer-Progress-"
    "Four-Tactile-Teacher-v0"
)
GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK = (
    "TacEx-Sim2Real-Cylinder-Real-Alignment-RMA-GelSight-Pulled-Drawer-Progress-"
    "Four-Tactile-Three-Frame-Binary-Direct-Action-Student-DR-v0"
)

CYLINDER_NOMINAL_HEIGHT_M = 0.100
CYLINDER_NOMINAL_RADIUS_M = 0.025
CYLINDER_DENSITY_KG_M3 = 1000.0
CYLINDER_SCALE_BUCKETS = (0.90, 0.93, 0.96, 0.99, 1.01, 1.04, 1.07, 1.10)
CYLINDER_HEIGHT_BUCKETS_M = tuple(
    CYLINDER_NOMINAL_HEIGHT_M * scale for scale in CYLINDER_SCALE_BUCKETS
)
CYLINDER_ABSOLUTE_CONTACT_REWARD_SCALE = 0.2


def absolute_weighted_contact_reward(
    contact_state: torch.Tensor,
    per_sensor_weights: tuple[float, ...],
    *,
    scale: float = CYLINDER_ABSOLUTE_CONTACT_REWARD_SCALE,
) -> torch.Tensor:
    """Return scaled absolute contact reward, ``[N,S] -> [N]``."""
    if contact_state.ndim != 2 or contact_state.shape[-1] != len(per_sensor_weights):
        raise ValueError(
            "contact_state sensor count must match per_sensor_weights, got "
            f"{tuple(contact_state.shape)} and {len(per_sensor_weights)} weights"
        )
    if scale < 0.0:
        raise ValueError("absolute contact reward scale must be non-negative")
    weights = torch.as_tensor(
        per_sensor_weights,
        device=contact_state.device,
        dtype=contact_state.dtype,
    )
    return float(scale) * (contact_state * weights).sum(dim=-1)


def cylinder_dimensions_and_mass_from_scale(
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return height [m], radius [m], and mass [kg] for isotropic ``scale [N]``."""
    height = scale * CYLINDER_NOMINAL_HEIGHT_M
    radius = scale * CYLINDER_NOMINAL_RADIUS_M
    mass = CYLINDER_DENSITY_KG_M3 * math.pi * radius.square() * height
    return height, radius, mass


def cylinder_lowest_world_height(
    center_position_w: torch.Tensor,
    quaternion_w: torch.Tensor,
    height_m: torch.Tensor,
    radius_m: torch.Tensor,
) -> torch.Tensor:
    """Return exact lowest Z of finite cylinders; tensors are ``[N,3]/[N,4]/[N]/[N]``.

    The cylinder axis is local +Z. Its vertical support extent is the sum of
    the axial half-height projection and the circular radial projection.
    """
    if center_position_w.ndim != 2 or center_position_w.shape[-1] != 3:
        raise ValueError("center_position_w must have shape [N,3]")
    count = center_position_w.shape[0]
    if quaternion_w.shape != (count, 4):
        raise ValueError("quaternion_w must have shape [N,4]")
    if height_m.shape != (count,) or radius_m.shape != (count,):
        raise ValueError("height_m and radius_m must have shape [N]")
    local_axis = torch.zeros_like(center_position_w)
    local_axis[:, 2] = 1.0
    axis_z = math_utils.quat_apply(quaternion_w, local_axis)[:, 2].clamp(-1.0, 1.0)
    radial_projection = torch.sqrt(torch.clamp(1.0 - axis_z.square(), min=0.0))
    support_extent_z = 0.5 * height_m * axis_z.abs() + radius_m * radial_projection
    return center_position_w[:, 2] - support_extent_z


def _cylinder_cfg_from(base_cfg) -> RigidObjectCfg:
    """Build the cylinder in the legacy target slot ``/cube`` for interface compatibility."""
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                *PULLED_DRAWER_TRAY_NOMINAL_CENTER_XY_M,
                PULLED_DRAWER_PANEL_THICKNESS_M + 0.5 * CYLINDER_NOMINAL_HEIGHT_M,
            )
        ),
        spawn=sim_utils.CylinderCfg(
            radius=CYLINDER_NOMINAL_RADIUS_M,
            height=CYLINDER_NOMINAL_HEIGHT_M,
            rigid_props=base_cfg.cube.spawn.rigid_props.copy(),
            collision_props=base_cfg.cube.spawn.collision_props.copy(),
            mass_props=sim_utils.MassPropertiesCfg(density=CYLINDER_DENSITY_KG_M3),
            # Four filtered contact sensors use the legacy ``/cube`` source
            # prim, so the replacement shape must retain ContactReporterAPI.
            activate_contact_sensors=bool(
                base_cfg.cube.spawn.activate_contact_sensors
            ),
            visual_material=base_cfg.cube.spawn.visual_material.copy(),
            visual_material_path=base_cfg.cube.spawn.visual_material_path,
            physics_material=(
                None
                if base_cfg.cube.spawn.physics_material is None
                else base_cfg.cube.spawn.physics_material.copy()
            ),
            physics_material_path=base_cfg.cube.spawn.physics_material_path,
        ),
    )


@configclass
class _CylinderTargetCfgMixin:
    """Cylinder-only physical and task contract; legacy Cube field names stay internal."""

    cylinder_shape = "cylinder"
    cylinder_internal_compatibility_slot = "cube"
    cylinder_nominal_height_m = CYLINDER_NOMINAL_HEIGHT_M
    cylinder_nominal_radius_m = CYLINDER_NOMINAL_RADIUS_M
    cylinder_density_kg_m3 = CYLINDER_DENSITY_KG_M3
    cylinder_scale_buckets = CYLINDER_SCALE_BUCKETS
    cylinder_scale_assignment = "seeded_balanced_permutation_per_group_of_8"
    cylinder_scale_lifetime = "fixed_per_environment"
    contact_reward_mode = "absolute_weighted_contact_state_per_step"
    cylinder_absolute_contact_reward_scale = CYLINDER_ABSOLUTE_CONTACT_REWARD_SCALE

    # The inherited fixed-size machinery indexes this field through
    # ``active_cube_size_m``. In this route the value means cylinder height.
    cube_size_buckets_m = CYLINDER_HEIGHT_BUCKETS_M
    cube_size_object_count_per_environment = 1
    cube_size = (
        2.0 * CYLINDER_NOMINAL_RADIUS_M,
        2.0 * CYLINDER_NOMINAL_RADIUS_M,
        CYLINDER_NOMINAL_HEIGHT_M,
    )
    cube_half_xy_extent = CYLINDER_NOMINAL_RADIUS_M * max(CYLINDER_SCALE_BUCKETS)
    cylinder_radius = cube_half_xy_extent

    success_lift_delta = 0.035
    success_hold_steps = 5
    success_requires_upright = False
    lift_tilt_curriculum_enabled = False


class _CylinderTargetMixin:
    """Replace Cube geometry operations while retaining trainer-facing names."""

    @property
    def active_cylinder_height_m(self) -> torch.Tensor:
        return self.active_cube_size_m

    @property
    def active_cylinder_scale(self) -> torch.Tensor:
        return self.active_cylinder_height_m / CYLINDER_NOMINAL_HEIGHT_M

    @property
    def active_cylinder_radius_m(self) -> torch.Tensor:
        return self.active_cylinder_scale * CYLINDER_NOMINAL_RADIUS_M

    @property
    def active_cylinder_mass_kg(self) -> torch.Tensor:
        _, _, mass = cylinder_dimensions_and_mass_from_scale(
            self.active_cylinder_scale
        )
        return mass

    def _author_fixed_cube_scales(self) -> None:
        seed = int(getattr(self.cfg, "seed", 42) or 42)
        bucket_ids = balanced_cube_bucket_ids(self.num_envs, seed)
        self._active_cube_bucket_ids = bucket_ids.to(self.device)
        prim_paths = sim_utils.find_matching_prim_paths(self.cfg.cube.prim_path)
        if len(prim_paths) != self.num_envs:
            raise RuntimeError(f"Expected one Cylinder per env, found {len(prim_paths)}")
        stage = sim_utils.stage_utils.get_current_stage()
        with Sdf.ChangeBlock():
            for env_id, prim_path in enumerate(prim_paths):
                scale = float(CYLINDER_SCALE_BUCKETS[int(bucket_ids[env_id].item())])
                self._author_transform_spec(
                    stage, prim_path, None, (scale, scale, scale)
                )

    def _set_fixed_cube_spawn_height(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        tray_xy = self._pulled_drawer_layout["tray_center_xy_m"][env_ids]
        self._cube.data.default_root_state[env_ids, :2] = tray_xy
        self._cube.data.default_root_state[env_ids, 2] = (
            PULLED_DRAWER_PANEL_THICKNESS_M
            + 0.5 * self.active_cylinder_height_m[env_ids]
        )

    def _compute_cube_lowest_height(
        self, cube_pos: torch.Tensor, cube_quat: torch.Tensor
    ) -> torch.Tensor:
        return cylinder_lowest_world_height(
            cube_pos,
            cube_quat,
            self.active_cylinder_height_m,
            self.active_cylinder_radius_m,
        )

    def _compute_additional_reward(self):
        reward, log = super()._compute_additional_reward()
        # The shared Progress route has already replaced the four-pad contact
        # state with its signed delta. Cylinder training instead pays a bounded
        # absolute state reward every step: 0.2 * [1.5, 1.5, 1.0, 1.0], so
        # stable four-pad contact is worth at most +1.0 per policy step.
        inherited_contact_progress = self._last_rma_contact_reward
        absolute_contact = absolute_weighted_contact_reward(
            self._last_rma_contact_state,
            tuple(float(value) for value in self.cfg.rma_contact_reward_weights),
            scale=float(self.cfg.cylinder_absolute_contact_reward_scale),
        )
        reward = reward - inherited_contact_progress + absolute_contact
        self._last_rma_contact_reward = absolute_contact.detach().clone()
        log["reward/rma_contact"] = absolute_contact.mean().detach()
        for key in tuple(log):
            if key.startswith("info/cube_size_"):
                log.pop(key)
        values = {
            ("scale", ""): self.active_cylinder_scale,
            ("height", "_m"): self.active_cylinder_height_m,
            ("radius", "_m"): self.active_cylinder_radius_m,
            ("mass", "_kg"): self.active_cylinder_mass_kg,
            ("spawn_center_z", "_m"): (
                PULLED_DRAWER_PANEL_THICKNESS_M
                + 0.5 * self.active_cylinder_height_m
            ),
        }
        for (name, unit), value in values.items():
            log[f"info/cylinder_{name}_mean{unit}"] = value.mean().detach()
            log[f"info/cylinder_{name}_min{unit}"] = value.min().detach()
            log[f"info/cylinder_{name}_max{unit}"] = value.max().detach()
        return reward, log


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg(
    _CylinderTargetCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg,
):
    rma_task_id = GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_TEACHER_TASK
    cube = _cylinder_cfg_from(
        Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg()
    )


class Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnv(
    _CylinderTargetMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnv,
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileTeacherEnvCfg


@configclass
class Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg(
    _CylinderTargetCfgMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg,
):
    rma_task_id = GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK
    cube = _cylinder_cfg_from(
        Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg()
    )


class Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnv(
    _CylinderTargetMixin,
    Sim2RealCubeRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnv,
):
    cfg: Sim2RealCylinderRealAlignmentRMAGelSightPulledDrawerProgressFourTactileBinaryStudentEnvCfg


__all__ = tuple(
    name
    for name in globals()
    if name.startswith(("CYLINDER_", "GELSIGHT_", "Sim2Real", "cylinder_"))
)
