"""Tactile-cross-alpha aux task with temporal tactile windows for GRU policies."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils import configclass

from .vt_tactile_cross_alpha_aux_box import (
    OccludedGraspingVTTactileCrossAlphaAuxBoxEnv,
    OccludedGraspingVTTactileCrossAlphaAuxDownsampleBoxCfg,
    _enable_predict_occlusion_aux_observations,
)


@configclass
class OccludedGraspingVTTactileCrossAlphaAuxGRUDownsampleBoxCfg(
    OccludedGraspingVTTactileCrossAlphaAuxDownsampleBoxCfg
):
    """Downsample tactile-cross-alpha aux task exposing windowed tactile features."""

    tactile_gru_window = 10

    def _compute_tactile_obs_dim(self) -> int:
        base_dim = int(super()._compute_tactile_obs_dim())
        window = max(1, int(getattr(self, "tactile_gru_window", 10)))
        return base_dim * window

    def __post_init__(self):
        super().__post_init__()
        _enable_predict_occlusion_aux_observations(self)

    def _post_configure_scene_object(self):
        _enable_predict_occlusion_aux_observations(self)


class OccludedGraspingVTTactileCrossAlphaAuxGRUBoxEnv(OccludedGraspingVTTactileCrossAlphaAuxBoxEnv):
    """Append aux labels and expose four tactile feature histories for GRU heads."""

    cfg: OccludedGraspingVTTactileCrossAlphaAuxGRUDownsampleBoxCfg

    def __init__(
        self,
        cfg: OccludedGraspingVTTactileCrossAlphaAuxGRUDownsampleBoxCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
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
                (self.num_envs, self._tactile_gru_window, feature.shape[-1]),
                dtype=torch.float32,
                device=self.device,
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
        for history in self._tactile_temporal_history.values():
            history[env_ids] = 0.0
