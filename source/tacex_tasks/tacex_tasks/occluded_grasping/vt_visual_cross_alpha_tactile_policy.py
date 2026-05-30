"""Visual-cross-alpha policy with an additional tactile branch."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_visual_cross_alpha_policy import OccludedGraspingVTVisualCrossAlphaPolicy


class OccludedGraspingVTVisualCrossAlphaTactilePolicy(OccludedGraspingVTVisualCrossAlphaPolicy):
    """Compute V_mix as visual-cross-alpha, then concatenate aggregated tactile features."""

    def __init__(self, *args, **kwargs: Any):
        mlp_layers = kwargs.get("mlp_layers", None)
        mlp_activation = str(kwargs.get("mlp_activation", "elu")).lower()
        super().__init__(*args, **kwargs)

        tactile_total_dim = sum(self._obs_dims[key] for key in self.tactile_keys)
        activation_cls = nn.ELU if mlp_activation == "elu" else nn.ReLU
        self.tactile_base_proj = nn.Sequential(
            nn.LayerNorm(tactile_total_dim),
            nn.Linear(tactile_total_dim, self.fused_dim),
            activation_cls(),
        )

        if mlp_layers is None:
            mlp_layers = (512, 256, 128, 64)
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

        vision = self.vision_proj(self._flatten_obs(obs, self.vision_key, batch_size))
        tactile_flat = torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.tactile_keys], dim=-1)
        tactile = self.tactile_base_proj(tactile_flat)
        tactile_tokens = torch.stack(
            [self.tactile_proj[key](self._flatten_obs(obs, key, batch_size)) for key in self.tactile_keys],
            dim=1,
        )
        sensor_ids = torch.arange(len(self.tactile_keys), device=self.device)
        tactile_tokens = tactile_tokens + self.tactile_sensor_embed(sensor_ids).unsqueeze(0)

        query = self.query_norm(vision).unsqueeze(1)
        context = self.context_norm(tactile_tokens)
        attn_out, _ = self.visual_tactile_attn(query, context, context, need_weights=False)
        vision_cross = (query + attn_out).squeeze(1)
        vision_cross = vision_cross + self.attn_ff(self.attn_ff_norm(vision_cross))

        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        alpha = self.alpha_gate(proprio)
        set_latest_alpha(alpha)
        vision_mixed = (1.0 - alpha) * vision + alpha * vision_cross

        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision_mixed, tactile, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}
