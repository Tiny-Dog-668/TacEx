"""VT-Alpha variant with temporal windows for both vision and tactile GRU policies."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vt_alpha_gru_box import OccludedGraspingVTAlphaGRUBoxCfg, OccludedGraspingVTAlphaGRUBoxEnv


@configclass
class OccludedGraspingVTAlphaDualGRUBoxCfg(OccludedGraspingVTAlphaGRUBoxCfg):
    """Configuration for VT-Alpha with windowed vision and tactile observations."""

    vision_gru_window = 5

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = dict(self.observation_space)
        window = max(1, int(getattr(self, "vision_gru_window", 5)))
        if "third_resnet" in self.observation_space:
            self.observation_space["third_resnet"] = int(self.observation_space["third_resnet"]) * window


class OccludedGraspingVTAlphaDualGRUBoxEnv(OccludedGraspingVTAlphaGRUBoxEnv):
    """Alpha-box env exposing temporal windows for both vision and tactile features."""

    cfg: OccludedGraspingVTAlphaDualGRUBoxCfg

    def __init__(self, cfg: OccludedGraspingVTAlphaDualGRUBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._vision_gru_window = max(1, int(getattr(self.cfg, "vision_gru_window", 5)))
        self._vision_temporal_key = "third_resnet"
        self._vision_temporal_history: torch.Tensor | None = None
        self._vision_temporal_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _refresh_vision_temporal_feature(self, obs: dict[str, torch.Tensor]) -> None:
        value = obs.get(self._vision_temporal_key)
        if value is None:
            return
        feature = value.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)

        history = self._vision_temporal_history
        if history is None or history.shape[-1] != feature.shape[-1]:
            history = torch.zeros(
                (self.num_envs, self._vision_gru_window, feature.shape[-1]), dtype=torch.float32, device=self.device
            )
            self._vision_temporal_history = history
            self._vision_temporal_pending[:] = True

        pending_ids = self._vision_temporal_pending.nonzero(as_tuple=False).squeeze(-1)
        if pending_ids.numel() > 0:
            history[pending_ids] = feature[pending_ids].unsqueeze(1).expand(-1, self._vision_gru_window, -1)

        history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = feature
        obs[self._vision_temporal_key] = history.reshape(self.num_envs, -1)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]
        self._refresh_vision_temporal_feature(obs)
        if torch.any(self._vision_temporal_pending):
            self._vision_temporal_pending[:] = False
        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        self._vision_temporal_pending[env_ids] = True
        if self._vision_temporal_history is not None:
            self._vision_temporal_history[env_ids] = 0.0
