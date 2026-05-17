"""VT-Alpha variant with tactile temporal windows for GRU policies."""

from __future__ import annotations

import torch
import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .vt_box import (
    BOX_FLOOR_THICKNESS,
    CAN_COLLISION_REST_OFFSET,
    OccludedGraspingVTAlphaBoxCfg,
    OccludedGraspingVTAlphaBoxEnv,
)


CUBE_SIDE = 0.05
CUBE_RESET_CLEARANCE = 0.001
CUBE_RESET_ROOT_Z = BOX_FLOOR_THICKNESS + 0.5 * CUBE_SIDE + CUBE_RESET_CLEARANCE
CUBOID_SIZE = (0.05, 0.05, 0.06)
CUBOID_RESET_CLEARANCE = 0.001
CUBOID_RESET_ROOT_Z = BOX_FLOOR_THICKNESS + 0.5 * CUBOID_SIZE[2] + CUBOID_RESET_CLEARANCE


def _configure_cube_object(cfg) -> None:
    """Replace the default cylinder object with a 5 cm cube."""
    side = float(getattr(cfg, "cube_side", CUBE_SIDE))
    reset_z = float(getattr(cfg, "can_reset_root_z", CUBE_RESET_ROOT_Z))
    cube_spawn = sim_utils.CuboidCfg(
        size=(side, side, side),
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
            rest_offset=CAN_COLLISION_REST_OFFSET,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.18, 0.55, 0.88),
            roughness=0.5,
            metallic=0.0,
        ),
    )
    def _replace_asset_with_cube(asset_cfg):
        pos = list(asset_cfg.init_state.pos)
        pos[2] = reset_z
        return asset_cfg.replace(
            spawn=cube_spawn,
            init_state=asset_cfg.init_state.replace(pos=pos),
        )

    if hasattr(cfg, "can"):
        cfg.can = _replace_asset_with_cube(cfg.can)
    if hasattr(cfg, "cylinder"):
        cfg.cylinder = _replace_asset_with_cube(cfg.cylinder)


def _configure_cuboid_object(cfg) -> None:
    """Replace the default cylinder object with an upright 5 cm x 5 cm x 6 cm cuboid."""
    size = tuple(float(v) for v in getattr(cfg, "cuboid_size", CUBOID_SIZE))
    reset_z = float(getattr(cfg, "can_reset_root_z", CUBOID_RESET_ROOT_Z))
    cuboid_spawn = sim_utils.CuboidCfg(
        size=size,
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
            rest_offset=CAN_COLLISION_REST_OFFSET,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.22, 0.62, 0.38),
            roughness=0.5,
            metallic=0.0,
        ),
    )

    def _replace_asset_with_cuboid(asset_cfg):
        pos = list(asset_cfg.init_state.pos)
        pos[2] = reset_z
        return asset_cfg.replace(
            spawn=cuboid_spawn,
            init_state=asset_cfg.init_state.replace(pos=pos),
        )

    if hasattr(cfg, "can"):
        cfg.can = _replace_asset_with_cuboid(cfg.can)
    if hasattr(cfg, "cylinder"):
        cfg.cylinder = _replace_asset_with_cuboid(cfg.cylinder)


@configclass
class OccludedGraspingVTAlphaGRUBoxCfg(OccludedGraspingVTAlphaBoxCfg):
    """Configuration for VT-Alpha with per-tactile temporal windows."""

    tactile_gru_window = 10

    def _compute_tactile_obs_dim(self) -> int:
        base_dim = int(super()._compute_tactile_obs_dim())
        window = max(1, int(getattr(self, "tactile_gru_window", 10)))
        return base_dim * window

    def __post_init__(self):
        super().__post_init__()


@configclass
class OccludedGraspingVTAlphaGRUDrawerOcclusionBoxCfg(OccludedGraspingVTAlphaGRUBoxCfg):
    """Alpha-GRU task with drawer walls and outer cabinet panels enabled."""

    include_drawer_walls = True
    include_outer_cabinet_panels = True


@configclass
class OccludedGraspingVTAlphaGRUSelfOcclusionBoxCfg(OccludedGraspingVTAlphaGRUBoxCfg):
    """Alpha-GRU task without drawer/cabinet occluders; only self-occlusion remains."""

    include_drawer_walls = False
    include_outer_cabinet_panels = False


@configclass
class _OccludedGraspingVTAlphaGRUCubeObjectCfg:
    """Mixin-style config fields for the 5 cm cube object."""

    cube_side = CUBE_SIDE
    can_radius = CUBE_SIDE * 0.5
    can_reset_root_z = CUBE_RESET_ROOT_Z
    lift_reward_start_height = CUBE_RESET_ROOT_Z
    success_height = CUBE_RESET_ROOT_Z + CUBE_SIDE
    drop_after_success_penalty_weight = 150.0


@configclass
class OccludedGraspingVTAlphaGRUDrawerOcclusionCubeCfg(
    _OccludedGraspingVTAlphaGRUCubeObjectCfg,
    OccludedGraspingVTAlphaGRUDrawerOcclusionBoxCfg,
):
    """Drawer-occlusion Alpha-GRU task using a 5 cm cube object."""

    def __post_init__(self):
        super().__post_init__()
        _configure_cube_object(self)


@configclass
class OccludedGraspingVTAlphaGRUSelfOcclusionCubeCfg(
    _OccludedGraspingVTAlphaGRUCubeObjectCfg,
    OccludedGraspingVTAlphaGRUSelfOcclusionBoxCfg,
):
    """Self-occlusion Alpha-GRU task using a 5 cm cube object."""

    def __post_init__(self):
        super().__post_init__()
        _configure_cube_object(self)


class OccludedGraspingVTAlphaGRUBoxEnv(OccludedGraspingVTAlphaBoxEnv):
    """Alpha-box env that exposes windowed tactile features for GRU heads."""

    cfg: OccludedGraspingVTAlphaGRUBoxCfg

    def __init__(self, cfg: OccludedGraspingVTAlphaGRUBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._tactile_gru_window = max(1, int(getattr(self.cfg, "tactile_gru_window", 10)))
        self._tactile_temporal_keys = (
            "tactile_left_depth_resnet",
            "tactile_right_depth_resnet",
            "tactile_left_down_depth_resnet",
            "tactile_right_down_depth_resnet",
        )
        self._tactile_temporal_history: dict[str, torch.Tensor] = {}
        self._tactile_temporal_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._sync_runtime_tactile_observation_space()

    def _sync_runtime_tactile_observation_space(self) -> None:
        dim = int(self.cfg._compute_tactile_obs_dim()) if hasattr(self.cfg, "_compute_tactile_obs_dim") else 2560

        def replace_key(container, key: str) -> None:
            if isinstance(container, dict):
                if key in container:
                    container[key] = dim
                return
            spaces = getattr(container, "spaces", None)
            if spaces is None:
                return
            if key in spaces:
                spaces[key] = gym.spaces.Box(-float("inf"), float("inf"), shape=(dim,), dtype=np.float32)
            elif "policy" in spaces:
                replace_key(spaces["policy"], key)

        self.cfg.observation_space = dict(self.cfg.observation_space)
        for key in self._tactile_temporal_keys:
            self.cfg.observation_space[key] = dim
            for attr in ("observation_space", "single_observation_space"):
                space = getattr(self, attr, None)
                if space is not None:
                    replace_key(space, key)

    def _refresh_tactile_temporal_feature(self, obs: dict[str, torch.Tensor], key: str) -> None:
        value = obs.get(key)
        if value is None:
            return
        value = value.to(device=self.device, dtype=torch.float32)
        if value.shape[0] != self.num_envs and value.numel() % self.num_envs == 0:
            obs[key] = value.reshape(self.num_envs, -1)
            return
        feature = value.reshape(self.num_envs, -1)
        history = self._tactile_temporal_history.get(key)
        if history is None or history.shape[-1] != feature.shape[-1]:
            history = torch.zeros(
                (self.num_envs, self._tactile_gru_window, feature.shape[-1]), dtype=torch.float32, device=self.device
            )
            self._tactile_temporal_history[key] = history
            self._tactile_temporal_pending[:] = True

        pending_ids = self._tactile_temporal_pending.nonzero(as_tuple=False).squeeze(-1)
        if pending_ids.numel() > 0:
            history[pending_ids] = feature[pending_ids].unsqueeze(1).expand(-1, self._tactile_gru_window, -1)

        history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = feature
        obs[key] = history.reshape(self.num_envs, -1)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]
        for key in self._tactile_temporal_keys:
            self._refresh_tactile_temporal_feature(obs, key)
        for key, value in list(obs.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] != self.num_envs and value.numel() % self.num_envs == 0:
                obs[key] = value.reshape(self.num_envs, -1)
        if torch.any(self._tactile_temporal_pending):
            self._tactile_temporal_pending[:] = False
        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        self._tactile_temporal_pending[env_ids] = True
        for key, history in self._tactile_temporal_history.items():
            history[env_ids] = 0.0
