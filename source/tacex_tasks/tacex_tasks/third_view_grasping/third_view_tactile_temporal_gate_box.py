"""Third-view env with 5-frame tactile temporal fusion and 5-frame tactile gate."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .third_view_tactile_concat_gate_box import (
    OccludedGraspingThirdViewTactileConcatGateBoxCfg,
    OccludedGraspingThirdViewTactileConcatGateBoxEnv,
    _TACTILE_SENSOR_ATTRS,
)
from .vt_box import OccludedGraspingVisionFourTactileBoxEnv

_TACTILE_FEATURE_KEYS: dict[str, str] = {
    "left": "tactile_left_depth_resnet",
    "right": "tactile_right_depth_resnet",
    "left_down": "tactile_left_down_depth_resnet",
    "right_down": "tactile_right_down_depth_resnet",
}


@configclass
class OccludedGraspingThirdViewTactileTemporalGateBoxCfg(OccludedGraspingThirdViewTactileConcatGateBoxCfg):
    """Configuration for 5-frame tactile temporal fusion with temporal gate."""

    tactile_gate_history_len = 5
    tactile_feature_history_len = 5


class OccludedGraspingThirdViewTactileTemporalGateBoxEnv(OccludedGraspingThirdViewTactileConcatGateBoxEnv):
    """Use 5-frame tactile feature histories and keep tactile output size unchanged."""

    cfg: OccludedGraspingThirdViewTactileTemporalGateBoxCfg

    def __init__(self, cfg: OccludedGraspingThirdViewTactileTemporalGateBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._tactile_feature_history_len = max(1, int(getattr(self.cfg, "tactile_feature_history_len", 5)))
        self._tactile_feature_history = {
            key: torch.zeros(
                (self.num_envs, self._tactile_feature_history_len, self._tactile_feature_dim),
                dtype=torch.float32,
                device=self.device,
            )
            for key in _TACTILE_SENSOR_ATTRS
        }
        self._pending_tactile_feature_history_refresh = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

    def _compress_temporal_tactile_history(self, history: torch.Tensor) -> torch.Tensor:
        """Concatenate 5-frame histories and reduce back to 256 dims by fixed temporal averaging."""
        flattened = history.reshape(self.num_envs, -1)
        return flattened.view(self.num_envs, self._tactile_feature_history_len, self._tactile_feature_dim).mean(dim=1)

    def _update_tactile_feature_histories(
        self, current_features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        pending = self._pending_tactile_feature_history_refresh
        pending_ids = pending.nonzero(as_tuple=False).squeeze(-1) if torch.any(pending) else None

        if pending_ids is not None and pending_ids.numel() > 0:
            for key, feature in current_features.items():
                self._tactile_feature_history[key][pending_ids] = feature[pending_ids].to(dtype=torch.float32).unsqueeze(1).expand(
                    -1, self._tactile_feature_history_len, -1
                )
            pending[pending_ids] = False

        active_ids = (~pending).nonzero(as_tuple=False).squeeze(-1)
        if active_ids.numel() > 0:
            for key, feature in current_features.items():
                history = self._tactile_feature_history[key]
                history[active_ids, :-1] = history[active_ids, 1:].clone()
                history[active_ids, -1] = feature[active_ids].to(dtype=torch.float32)

        return {
            key: self._compress_temporal_tactile_history(history)
            for key, history in self._tactile_feature_history.items()
        }

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = OccludedGraspingVisionFourTactileBoxEnv._get_observations(self)
        obs = observations["policy"]

        current_features = {
            key: obs[feature_key]
            for key, feature_key in _TACTILE_FEATURE_KEYS.items()
        }
        temporal_features = self._update_tactile_feature_histories(current_features)

        sensor_depths = self._get_tactile_sensor_depths()
        contact_ratios, contact_bits, global_gate = self._update_tactile_global_gate(sensor_depths)

        tactile_concat = torch.cat(
            [
                temporal_features["left"],
                temporal_features["right"],
                temporal_features["left_down"],
                temporal_features["right_down"],
            ],
            dim=-1,
        )
        obs["tactile_concat"] = self._gate_tactile_branch(tactile_concat, global_gate)
        obs["tactile_contact_bits"] = contact_bits
        obs["tactile_valid"] = global_gate.to(dtype=torch.float32)

        self._log_tactile_gate_stats(contact_ratios, global_gate)
        log = self.extras.setdefault("log", {})
        log["info/tactile_feature_history_len"] = torch.tensor(
            float(self._tactile_feature_history_len), device=self.device
        )

        return observations

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._pending_tactile_feature_history_refresh[env_ids] = True
        for history in self._tactile_feature_history.values():
            history[env_ids] = 0.0
