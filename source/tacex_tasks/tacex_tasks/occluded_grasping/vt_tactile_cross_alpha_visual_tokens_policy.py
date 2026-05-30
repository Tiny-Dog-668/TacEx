"""Tactile-cross-alpha policy using 49 visual tokens and direct visual concat."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_tactile_cross_alpha_visual_policy import OccludedGraspingVTTactileCrossAlphaVisualPolicy


class OccludedGraspingVTTactileCrossAlphaVisualTokensPolicy(OccludedGraspingVTTactileCrossAlphaVisualPolicy):
    """Use four tactile queries over 49 visual tokens, then concatenate pooled visual feature."""

    def __init__(
        self,
        *args,
        vision_tokens_key: str = "vision_tokens",
        **kwargs: Any,
    ):
        super().__init__(*args, vision_key=vision_tokens_key, **kwargs)
        self.vision_tokens_key = str(vision_tokens_key)
        vision_shape = self._obs_shapes[self.vision_key]
        if len(vision_shape) != 2:
            raise ValueError(f"Expected visual token observation shape [N, D], got {vision_shape}")
        self.vision_token_count = int(vision_shape[0])
        self.vision_token_dim = int(vision_shape[1])

        act_cls = nn.ELU if str(kwargs.get("mlp_activation", "elu")).lower() == "elu" else nn.ReLU
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.vision_token_dim),
            nn.Linear(self.vision_token_dim, self.fused_dim),
            act_cls(),
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

        vision_tokens = self._flatten_obs(obs, self.vision_key, batch_size).reshape(
            batch_size, self.vision_token_count, self.vision_token_dim
        )
        vision_tokens = self.vision_proj(vision_tokens)
        vision_feature = vision_tokens.mean(dim=1)

        tactile_flat = torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.tactile_keys], dim=-1)
        tactile = self.tactile_base_proj(tactile_flat)
        tactile_tokens = torch.stack(
            [self.tactile_token_proj[key](self._flatten_obs(obs, key, batch_size)) for key in self.tactile_keys],
            dim=1,
        )
        sensor_ids = torch.arange(len(self.tactile_keys), device=self.device)
        tactile_tokens = tactile_tokens + self.tactile_sensor_embed(sensor_ids).unsqueeze(0)

        query = self.query_norm(tactile_tokens)
        context = self.context_norm(vision_tokens)
        attn_out, _ = self.tactile_visual_attn(query, context, context, need_weights=False)
        tactile_cross_tokens = query + attn_out
        tactile_cross_tokens = tactile_cross_tokens + self.attn_ff(self.attn_ff_norm(tactile_cross_tokens))
        tactile_cross = self.tactile_cross_proj(tactile_cross_tokens.reshape(batch_size, -1))

        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        alpha = self.alpha_gate(proprio)
        set_latest_alpha(alpha)
        tactile_mixed = alpha * tactile + (1.0 - alpha) * tactile_cross

        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision_feature, tactile_mixed, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}
