"""GRU VT policy with an extra current-tactile feature branch."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_gru_policy import OccludedGraspingVTGRUPolicy


class OccludedGraspingVTGRUExtraTactilePolicy(OccludedGraspingVTGRUPolicy):
    """Concatenate GRU tactile history and current tactile features.

    Fusion:
        vision: 256 -> fused_dim
        tactile_gru: four tactile windows -> GRU latents -> fused_dim
        tactile_current: last frame from each tactile window -> 4 * 256 -> fused_dim
        final = concat(vision, tactile_gru, tactile_current, proprio)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        **kwargs: Any,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            mlp_layers=mlp_layers,
            mlp_activation=mlp_activation,
            **kwargs,
        )

        current_tactile_dim = sum(int(self._sensor_feat_dims[key]) for key in self.tactile_keys)
        self.current_tactile_proj = nn.Sequential(
            nn.LayerNorm(current_tactile_dim),
            nn.Linear(current_tactile_dim, self.fused_dim),
            nn.ELU(),
        )

        if mlp_layers is None:
            mlp_layers = (256, 128, 64)
        activation_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        vector_dim = sum(int(self._obs_dims[key]) for key in self.vector_keys)
        mlp_input_dim = (3 * self.fused_dim) + vector_dim
        mlp: list[nn.Module] = []
        last_dim = mlp_input_dim
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(activation_cls())
            last_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)

    def _extract_current_tactile(self, obs: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Return the last frame from each tactile window and concatenate sensors."""
        chunks = []
        for key in self.tactile_keys:
            feature_dim = int(self._sensor_feat_dims[key])
            flat = self._flatten_obs(obs, key, batch_size)
            if flat.shape[-1] == feature_dim:
                current = flat
            else:
                current = flat.reshape(batch_size, self.tactile_time_window, feature_dim)[:, -1, :]
            chunks.append(current)
        return torch.cat(chunks, dim=-1)

    def compute(self, inputs: Dict[str, Any], role: str = ""):
        states = inputs.get("states")
        if isinstance(states, dict):
            obs = states
        else:
            obs = unflatten_tensorized_space(self.observation_space, states)

        batch_size = None
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        if batch_size is None:
            raise ValueError("No tensor observations found in inputs['states']")

        vision = self.vision_proj(self._flatten_obs(obs, self.vision_key, batch_size))
        tactile_gru = self.tactile_proj(self._encode_tactile(obs, batch_size))
        tactile_current = self.current_tactile_proj(self._extract_current_tactile(obs, batch_size))
        set_latest_alpha(None)

        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision, tactile_gru, tactile_current, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}
