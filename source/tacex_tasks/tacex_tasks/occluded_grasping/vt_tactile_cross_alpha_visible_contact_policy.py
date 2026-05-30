"""Tactile-cross-alpha policy gated by visibility and tactile contact ratios."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_tactile_cross_alpha_policy import OccludedGraspingVTTactileCrossAlphaPolicy


class OccludedGraspingVTTactileCrossAlphaVisibleContactPolicy(OccludedGraspingVTTactileCrossAlphaPolicy):
    """Compute alpha from visible ratio and four tactile contact ratios."""

    def __init__(
        self,
        *args,
        visible_ratio_key: str = "visual_visible_ratio",
        tactile_contact_ratio_key: str = "tactile_contact_ratio",
        gate_hidden_dim: int = 64,
        **kwargs: Any,
    ):
        super().__init__(*args, gate_hidden_dim=gate_hidden_dim, **kwargs)
        self.visible_ratio_key = str(visible_ratio_key)
        self.tactile_contact_ratio_key = str(tactile_contact_ratio_key)

        for key, expected_dim in ((self.visible_ratio_key, 1), (self.tactile_contact_ratio_key, 4)):
            if not self._has_obs_key(self.observation_space, key):
                raise ValueError(f"Missing alpha gate observation key '{key}'")
            shape = self._get_obs_shape(self.observation_space, key)
            if shape is None or int(shape[-1]) != expected_dim:
                raise ValueError(f"Expected '{key}' dim={expected_dim}, got {shape}")
            self._obs_shapes[key] = shape
            self._obs_dims[key] = expected_dim

        self.alpha_gate = nn.Sequential(
            nn.Linear(5, int(gate_hidden_dim)),
            nn.ELU(),
            nn.Linear(int(gate_hidden_dim), 1),
            nn.Sigmoid(),
        )

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
        tactile_flat = torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.tactile_keys], dim=-1)
        tactile = self.tactile_base_proj(tactile_flat)
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
        tactile_cross = self.tactile_cross_proj(tactile_cross_tokens.reshape(batch_size, -1))

        visible_ratio = self._flatten_obs(obs, self.visible_ratio_key, batch_size).clamp(0.0, 1.0)
        tactile_contact_ratio = self._flatten_obs(obs, self.tactile_contact_ratio_key, batch_size).clamp(0.0, 1.0)
        alpha_input = torch.cat([visible_ratio, tactile_contact_ratio], dim=-1)
        alpha = self.alpha_gate(alpha_input)
        set_latest_alpha(alpha)
        tactile_mixed = alpha * tactile + (1.0 - alpha) * tactile_cross

        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([tactile_mixed, *vector_features], dim=-1) if vector_features else tactile_mixed

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {"alpha": alpha.mean()}
