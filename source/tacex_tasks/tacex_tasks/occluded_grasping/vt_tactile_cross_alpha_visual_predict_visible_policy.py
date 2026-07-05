"""Tactile-cross-alpha visual policy gated by visual visible prediction and tactile contact."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_tactile_cross_alpha_visual_policy import OccludedGraspingVTTactileCrossAlphaVisualPolicy


class OccludedGraspingVTTactileCrossAlphaVisualPredictVisiblePolicy(
    OccludedGraspingVTTactileCrossAlphaVisualPolicy
):
    """Predict visible ratio from vision, supervise it with pseudo labels, and gate fusion with contact ratios."""

    def __init__(
        self,
        *args,
        pseudo_visible_ratio_key: str = "pseudo_visible_ratio",
        tactile_contact_ratio_key: str = "tactile_contact_ratio",
        visible_hidden_dim: int = 128,
        visible_aux_weight: float = 1.0,
        gate_hidden_dim: int = 64,
        **kwargs: Any,
    ):
        mlp_activation = str(kwargs.get("mlp_activation", "elu")).lower()
        super().__init__(*args, gate_hidden_dim=gate_hidden_dim, **kwargs)
        self.pseudo_visible_ratio_key = str(pseudo_visible_ratio_key)
        self.tactile_contact_ratio_key = str(tactile_contact_ratio_key)
        self.visible_aux_weight = float(visible_aux_weight)

        for key, expected_dim in ((self.pseudo_visible_ratio_key, 1), (self.tactile_contact_ratio_key, 4)):
            if not self._has_obs_key(self.observation_space, key):
                raise ValueError(f"Missing observation key '{key}' for predict-visible policy")
            shape = self._get_obs_shape(self.observation_space, key)
            if shape is None or int(shape[-1]) != expected_dim:
                raise ValueError(f"Expected '{key}' dim={expected_dim}, got {shape}")
            self._obs_shapes[key] = shape
            self._obs_dims[key] = expected_dim

        act_cls = nn.ELU if mlp_activation == "elu" else nn.ReLU
        vision_dim = self._obs_dims[self.vision_key]
        self.visible_head = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, int(visible_hidden_dim)),
            act_cls(),
            nn.Linear(int(visible_hidden_dim), 1),
            nn.Sigmoid(),
        )
        self.alpha_gate = nn.Sequential(
            nn.Linear(5, int(gate_hidden_dim)),
            act_cls(),
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

        vision_raw = self._flatten_obs(obs, self.vision_key, batch_size)
        vision_feature = self.vision_proj(vision_raw)
        vision = vision_feature.unsqueeze(1)
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

        visible_pred = self.visible_head(vision_raw)
        tactile_contact_ratio = self._flatten_obs(obs, self.tactile_contact_ratio_key, batch_size)[:, :4].clamp(0.0, 1.0)
        alpha = self.alpha_gate(torch.cat([visible_pred, tactile_contact_ratio], dim=-1))
        set_latest_alpha(alpha)
        tactile_mixed = alpha * tactile + (1.0 - alpha) * tactile_cross

        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision_feature, tactile_mixed, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)

        pseudo_visible_ratio = self._flatten_obs(obs, self.pseudo_visible_ratio_key, batch_size)[:, :1].clamp(0.0, 1.0)
        visible_loss = F.mse_loss(visible_pred, pseudo_visible_ratio)
        aux_loss = self.visible_aux_weight * visible_loss

        return mu, self.log_std_parameter, {
            "aux_loss": aux_loss,
            "aux_reliability_loss": visible_loss,
            "alpha": alpha.mean(),
            "g_probe": visible_pred.mean(),
            "g_grasp": tactile_contact_ratio.mean(),
        }
