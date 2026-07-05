"""Tactile-cross-alpha policy with visually predicted occlusion for alpha."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_tactile_cross_alpha_policy import OccludedGraspingVTTactileCrossAlphaPolicy


class OccludedGraspingVTTactileCrossAlphaAuxPolicy(OccludedGraspingVTTactileCrossAlphaPolicy):
    """Predict occlusion from vision, supervise it with pseudo GT, and use it in alpha."""

    def __init__(
        self,
        *args,
        aux_occlusion_gt_key: str = "aux_occlusion_gt",
        aux_tactile_contact_gt_key: str = "aux_tactile_contact_gt",
        occlusion_hidden_dim: int = 128,
        occlusion_aux_weight: float = 1.0,
        gate_hidden_dim: int = 64,
        mlp_activation: str = "elu",
        **kwargs: Any,
    ):
        super().__init__(*args, gate_hidden_dim=gate_hidden_dim, mlp_activation=mlp_activation, **kwargs)
        self.aux_occlusion_gt_key = str(aux_occlusion_gt_key)
        self.aux_tactile_contact_gt_key = str(aux_tactile_contact_gt_key)
        self.occlusion_aux_weight = float(occlusion_aux_weight)

        for key, expected_dim in ((self.aux_occlusion_gt_key, 1), (self.aux_tactile_contact_gt_key, 4)):
            if not self._has_obs_key(self.observation_space, key):
                raise ValueError(f"Missing observation key '{key}' for tactile-cross-alpha-aux policy")
            shape = self._get_obs_shape(self.observation_space, key)
            if shape is None or int(shape[-1]) != expected_dim:
                raise ValueError(f"Expected '{key}' dim={expected_dim}, got {shape}")
            self._obs_shapes[key] = shape
            self._obs_dims[key] = expected_dim

        proprio_dim = self._obs_dims[self.proprio_key]
        vision_dim = self._obs_dims[self.vision_key]
        act_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        self.occlusion_head = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, int(occlusion_hidden_dim)),
            act_cls(),
            nn.Linear(int(occlusion_hidden_dim), 1),
            nn.Sigmoid(),
        )
        self.alpha_gate = nn.Sequential(
            nn.Linear(1 + 4 + proprio_dim, int(gate_hidden_dim)),
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
        vision = self.vision_proj(vision_raw).unsqueeze(1)
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

        occlusion_pred = self.occlusion_head(vision_raw)
        occlusion_gt = self._flatten_obs(obs, self.aux_occlusion_gt_key, batch_size)[:, :1].clamp(0.0, 1.0)
        tactile_contact = self._flatten_obs(obs, self.aux_tactile_contact_gt_key, batch_size)[:, :4].clamp(0.0, 1.0)
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        alpha = self.alpha_gate(torch.cat([occlusion_pred, tactile_contact, proprio], dim=-1))
        set_latest_alpha(alpha)
        tactile_mixed = alpha * tactile + (1.0 - alpha) * tactile_cross

        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([tactile_mixed, *vector_features], dim=-1) if vector_features else tactile_mixed

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        occlusion_loss = F.mse_loss(occlusion_pred, occlusion_gt)
        aux_loss = self.occlusion_aux_weight * occlusion_loss
        return mu, self.log_std_parameter, {
            "aux_loss": aux_loss,
            "aux_reliability_loss": occlusion_loss,
            "alpha": alpha.mean(),
            "alpha_occlusion": occlusion_pred.mean(),
            "alpha_occlusion_gt": occlusion_gt.mean(),
            "alpha_tactile_contact": tactile_contact.mean(),
            "occlusion_pred_values": occlusion_pred.detach(),
            "occlusion_gt_values": occlusion_gt.detach(),
            "alpha_values": alpha.detach(),
            "alpha_tactile_contact_values": tactile_contact.detach(),
        }
