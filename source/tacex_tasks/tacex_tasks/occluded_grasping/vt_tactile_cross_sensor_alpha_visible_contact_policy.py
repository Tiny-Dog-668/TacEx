"""Tactile-cross policy with per-sensor visibility/contact alpha gates."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_tactile_cross_alpha_visible_contact_policy import (
    OccludedGraspingVTTactileCrossAlphaVisibleContactPolicy,
)


class OccludedGraspingVTTactileCrossSensorAlphaVisibleContactPolicy(
    OccludedGraspingVTTactileCrossAlphaVisibleContactPolicy
):
    """Mix each tactile token with its visual-context token using its own alpha."""

    def __init__(self, *args, gate_hidden_dim: int = 64, **kwargs: Any):
        mlp_layers = kwargs.get("mlp_layers", None)
        mlp_activation = str(kwargs.get("mlp_activation", "elu")).lower()
        super().__init__(*args, gate_hidden_dim=gate_hidden_dim, **kwargs)
        self.sensor_alpha_gate = nn.Sequential(
            nn.Linear(2, int(gate_hidden_dim)),
            nn.ELU(),
            nn.Linear(int(gate_hidden_dim), 1),
            nn.Sigmoid(),
        )
        if mlp_layers is None:
            mlp_layers = (256, 128, 64)
        activation_cls = nn.ELU if mlp_activation == "elu" else nn.ReLU
        vector_dim = sum(self._obs_dims[key] for key in self.vector_keys)
        last_dim = (2 * self.fused_dim) + vector_dim
        mlp: list[nn.Module] = []
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(activation_cls())
            last_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)

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

        vision = self.vision_proj(self._flatten_obs(obs, self.vision_key, batch_size)).unsqueeze(1)
        tactile_tokens = torch.stack(
            [self.tactile_token_proj[key](self._flatten_obs(obs, key, batch_size)) for key in self.tactile_keys],
            dim=1,
        )
        sensor_ids = torch.arange(len(self.tactile_keys), device=self.device)
        tactile_tokens = tactile_tokens + self.tactile_sensor_embed(sensor_ids).unsqueeze(0)

        query = self.query_norm(tactile_tokens)
        context = self.context_norm(vision)
        attn_out, _ = self.tactile_visual_attn(query, context, context, need_weights=False)
        tactile_cross_tokens = query + attn_out
        tactile_cross_tokens = tactile_cross_tokens + self.attn_ff(self.attn_ff_norm(tactile_cross_tokens))

        visible_ratio = self._flatten_obs(obs, self.visible_ratio_key, batch_size).clamp(0.0, 1.0)
        contact_ratio = self._flatten_obs(obs, self.tactile_contact_ratio_key, batch_size).clamp(0.0, 1.0)
        visible_by_sensor = visible_ratio.unsqueeze(1).expand(-1, len(self.tactile_keys), -1)
        alpha_input = torch.cat([visible_by_sensor, contact_ratio.unsqueeze(-1)], dim=-1)
        sensor_alpha = self.sensor_alpha_gate(alpha_input)

        mixed_tokens = sensor_alpha * tactile_tokens + (1.0 - sensor_alpha) * tactile_cross_tokens
        tactile_mixed = self.tactile_cross_proj(mixed_tokens.reshape(batch_size, -1))

        set_latest_alpha(sensor_alpha.mean(dim=1))
        vision_feature = vision.squeeze(1)
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision_feature, tactile_mixed, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {
            "alpha": sensor_alpha.mean(),
            "sensor_alpha": sensor_alpha,
        }
