"""GelFusion-style PPO actor using a frozen ViT-B/16 visual feature."""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .vt_gelfusion_policy import OccludedGraspingVTGelFusionPolicy


class OccludedGraspingVTGelFusionViTPolicy(OccludedGraspingVTGelFusionPolicy):
    """Vision-led cross-attention actor for GelFusion-ViT RL experiments.

    The environment supplies `third_vit_cls`, a frozen ViT-B/16 CLS feature. This
    actor keeps the GelFusion fusion path: visual query over visual + four tactile
    tokens, then concat visual, attended context, tactile dynamics, and proprio.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        vision_key: str = "third_vit_cls",
        tactile_keys: Iterable[str] | None = None,
        dynamic_key: str = "tactile_dynamic_stats",
        vector_keys: Iterable[str] | None = None,
        proprio_key: str = "proprio_obs",
        fused_dim: int = 768,
        num_heads: int = 8,
        attention_mlp_dim: int = 1536,
        sequence_horizon: int | None = None,
        mlp_layers: Iterable[int] | None = None,
        **kwargs: Any,
    ):
        if mlp_layers is None:
            mlp_layers = (512, 256, 128)
        self._requested_mlp_layers = tuple(int(v) for v in mlp_layers)
        self._requested_mlp_activation = str(kwargs.get("mlp_activation", "elu")).lower()
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            vision_key=vision_key,
            tactile_keys=tactile_keys,
            dynamic_key=dynamic_key,
            vector_keys=vector_keys,
            proprio_key=proprio_key,
            fused_dim=fused_dim,
            num_heads=num_heads,
            attention_mlp_dim=attention_mlp_dim,
            mlp_layers=mlp_layers,
            **kwargs,
        )
        self.sequence_horizon = int(sequence_horizon or self._infer_sequence_horizon(self._obs_shapes[self.vision_key]))
        dynamic_dim = self._obs_dims[self.dynamic_key]
        vector_dim = sum(self._obs_dims[key] for key in self.vector_keys)
        last_dim = self.sequence_horizon * ((2 * self.fused_dim) + vector_dim) + dynamic_dim
        activation_cls = nn.ELU if self._requested_mlp_activation == "elu" else nn.ReLU
        mlp: list[nn.Module] = []
        for hidden_dim in self._requested_mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(activation_cls())
            last_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)

    @staticmethod
    def _infer_sequence_horizon(shape: tuple[int, ...] | None) -> int:
        if shape is None or len(shape) < 2:
            return 1
        return int(shape[0])

    def _sequence_obs(self, obs: dict[str, Any], key: str, batch_size: int) -> torch.Tensor:
        value = obs.get(key)
        dim = self._obs_dims[key]
        horizon = self._infer_sequence_horizon(self._obs_shapes[key])
        if value is None:
            return torch.zeros((batch_size, horizon, dim), device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(device=self.device, dtype=torch.float32)
        return value.reshape(batch_size, horizon, dim)

    def compute(self, inputs: dict[str, Any], role: str = ""):
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

        horizon = self.sequence_horizon
        visual_seq = self._sequence_obs(obs, self.vision_key, batch_size)
        visual = self.vision_proj(visual_seq.reshape(batch_size * horizon, -1)).reshape(
            batch_size, horizon, self.fused_dim
        )

        tactile_tokens = []
        for key in self.tactile_keys:
            tactile_seq = self._sequence_obs(obs, key, batch_size)
            tactile = self.tactile_proj[key](tactile_seq.reshape(batch_size * horizon, -1))
            tactile_tokens.append(tactile.reshape(batch_size, horizon, self.fused_dim))
        tactile_tokens = torch.stack(tactile_tokens, dim=2)

        context = torch.cat([visual.unsqueeze(2), tactile_tokens], dim=2)
        modality_ids = torch.arange(context.shape[2], device=self.device)
        context = context + self.modality_embed(modality_ids).view(1, 1, -1, self.fused_dim)
        context = context.reshape(batch_size * horizon, context.shape[2], self.fused_dim)

        query = self.query_norm(visual.reshape(batch_size * horizon, 1, self.fused_dim))
        context = self.context_norm(context)
        attended, _ = self.attn(query, context, context, need_weights=False)
        attended = attended + self.attn_ff(self.attn_ff_norm(attended))
        attended = attended.reshape(batch_size, horizon, self.fused_dim)

        vector_features = [self._sequence_obs(obs, key, batch_size) for key in self.vector_keys]
        dynamic = self._flatten_obs(obs, self.dynamic_key, batch_size)
        fused = torch.cat(
            [
                visual.reshape(batch_size, -1),
                attended.reshape(batch_size, -1),
                *[feature.reshape(batch_size, -1) for feature in vector_features],
                dynamic,
            ],
            dim=-1,
        )

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}
