"""Dual-cross-alpha policy with supervised occlusion/contact heads for alpha."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha


class OccludedGraspingVTDualCrossAlphaAuxPolicy(GaussianMixin, Model):
    """Dual cross-attention policy whose alpha is predicted from task-head outputs."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        initial_log_std: float = 0.0,
        fixed_log_std: bool = False,
        vision_key: str = "third_resnet",
        tactile_keys: Iterable[str] | None = None,
        vector_keys: Iterable[str] | None = None,
        proprio_key: str = "proprio_obs",
        aux_occlusion_gt_key: str = "aux_occlusion_gt",
        aux_tactile_contact_gt_key: str = "aux_tactile_contact_gt",
        fused_dim: int = 256,
        num_heads: int = 4,
        attention_mlp_dim: int = 512,
        dropout: float = 0.0,
        task_hidden_dim: int = 128,
        gate_hidden_dim: int = 64,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        aux_occlusion_weight: float = 1.0,
        aux_tactile_contact_weight: float = 1.0,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        if tactile_keys is None:
            tactile_keys = (
                "tactile_left_depth_resnet",
                "tactile_right_depth_resnet",
                "tactile_left_down_depth_resnet",
                "tactile_right_down_depth_resnet",
            )
        if vector_keys is None:
            vector_keys = (proprio_key,)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.aux_occlusion_gt_key = str(aux_occlusion_gt_key)
        self.aux_tactile_contact_gt_key = str(aux_tactile_contact_gt_key)
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        self.vector_keys = tuple(key for key in vector_keys if self._has_obs_key(observation_space, key))
        self.fused_dim = int(fused_dim)
        self.aux_occlusion_weight = float(aux_occlusion_weight)
        self.aux_tactile_contact_weight = float(aux_tactile_contact_weight)

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self.tactile_keys:
            raise ValueError("No tactile observation keys found for dual-cross-alpha-aux policy")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")

        tracked_keys = [self.vision_key, *self.tactile_keys, *self.vector_keys]
        for key in (self.aux_occlusion_gt_key, self.aux_tactile_contact_gt_key):
            if self._has_obs_key(observation_space, key):
                tracked_keys.append(key)
        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in tracked_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }

        vision_dim = self._obs_dims[self.vision_key]
        vector_dim = sum(self._obs_dims[key] for key in self.vector_keys)
        tactile_total_dim = sum(self._obs_dims[key] for key in self.tactile_keys)
        act_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU

        self.vision_proj = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, self.fused_dim), act_cls())
        self.tactile_base_proj = nn.Sequential(
            nn.LayerNorm(tactile_total_dim),
            nn.Linear(tactile_total_dim, self.fused_dim),
            act_cls(),
        )
        self.tactile_token_proj = nn.ModuleDict(
            {
                key: nn.Sequential(nn.LayerNorm(self._obs_dims[key]), nn.Linear(self._obs_dims[key], self.fused_dim), act_cls())
                for key in self.tactile_keys
            }
        )
        self.tactile_sensor_embed = nn.Embedding(len(self.tactile_keys), self.fused_dim)

        self.vision_query_norm = nn.LayerNorm(self.fused_dim)
        self.tactile_context_norm = nn.LayerNorm(self.fused_dim)
        self.vision_from_tactile_attn = nn.MultiheadAttention(
            embed_dim=self.fused_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.vision_ff_norm = nn.LayerNorm(self.fused_dim)
        self.vision_ff = self._make_ff(attention_mlp_dim, dropout)

        self.tactile_query_norm = nn.LayerNorm(self.fused_dim)
        self.vision_context_norm = nn.LayerNorm(self.fused_dim)
        self.tactile_from_vision_attn = nn.MultiheadAttention(
            embed_dim=self.fused_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.tactile_ff_norm = nn.LayerNorm(self.fused_dim)
        self.tactile_ff = self._make_ff(attention_mlp_dim, dropout)
        self.tactile_cross_proj = nn.Sequential(
            nn.LayerNorm(len(self.tactile_keys) * self.fused_dim),
            nn.Linear(len(self.tactile_keys) * self.fused_dim, self.fused_dim),
            act_cls(),
        )

        self.occlusion_head = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, int(task_hidden_dim)),
            act_cls(),
            nn.Linear(int(task_hidden_dim), 1),
            nn.Sigmoid(),
        )
        self.tactile_contact_head = nn.Sequential(
            nn.LayerNorm(tactile_total_dim),
            nn.Linear(tactile_total_dim, int(task_hidden_dim)),
            act_cls(),
            nn.Linear(int(task_hidden_dim), len(self.tactile_keys)),
            nn.Sigmoid(),
        )
        self.alpha_gate = nn.Sequential(
            nn.Linear(2, int(gate_hidden_dim)),
            act_cls(),
            nn.Linear(int(gate_hidden_dim), 1),
            nn.Sigmoid(),
        )

        if mlp_layers is None:
            mlp_layers = (512, 256, 128, 64)
        mlp: list[nn.Module] = []
        last_dim = (2 * self.fused_dim) + vector_dim
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(act_cls())
            last_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(initial_log_std)),
            requires_grad=not bool(fixed_log_std),
        )

    def _make_ff(self, attention_mlp_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.fused_dim, int(attention_mlp_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(attention_mlp_dim), self.fused_dim),
            nn.Dropout(float(dropout)),
        )

    def _has_obs_key(self, obs_space, key: str) -> bool:
        if hasattr(obs_space, "spaces"):
            return key in obs_space.spaces
        return isinstance(obs_space, dict) and key in obs_space

    def _get_obs_shape(self, obs_space, key: str) -> Tuple[int, ...] | None:
        if hasattr(obs_space, "spaces") and key in obs_space.spaces:
            shape = getattr(obs_space.spaces[key], "shape", None)
            return tuple(shape) if shape is not None else None
        if isinstance(obs_space, dict) and key in obs_space:
            spec = obs_space[key]
            if isinstance(spec, int):
                return (int(spec),)
            if isinstance(spec, (tuple, list)):
                return tuple(int(s) for s in spec)
            shape = getattr(spec, "shape", None)
            if shape is not None:
                return tuple(shape)
        return None

    def _flatten_obs(self, obs: Dict[str, Any], key: str, batch_size: int) -> torch.Tensor:
        value = obs.get(key)
        if value is None:
            return torch.zeros((batch_size, int(self._obs_dims.get(key, 0))), device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(device=self.device, dtype=torch.float32)
        return value.reshape(batch_size, -1)

    def _build_aux_losses(
        self,
        obs: Dict[str, Any],
        batch_size: int,
        occlusion_pred: torch.Tensor,
        tactile_contact_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = occlusion_pred.new_zeros(())
        occlusion_loss = zero
        if self._has_obs_key(self.observation_space, self.aux_occlusion_gt_key):
            gt = self._flatten_obs(obs, self.aux_occlusion_gt_key, batch_size)[:, :1].clamp(0.0, 1.0)
            occlusion_loss = F.mse_loss(occlusion_pred, gt)

        contact_loss = zero
        if self._has_obs_key(self.observation_space, self.aux_tactile_contact_gt_key):
            gt = self._flatten_obs(obs, self.aux_tactile_contact_gt_key, batch_size)
            gt = gt[:, : len(self.tactile_keys)].clamp(0.0, 1.0)
            contact_loss = F.mse_loss(tactile_contact_pred, gt)

        aux_loss = self.aux_occlusion_weight * occlusion_loss + self.aux_tactile_contact_weight * contact_loss
        return aux_loss, occlusion_loss, contact_loss

    def compute(self, inputs: Dict[str, Any], role: str = ""):
        states = inputs.get("states")
        if isinstance(states, dict):
            obs = states
        else:
            obs = unflatten_tensorized_space(self.observation_space, states)

        batch_size = None
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                batch_size = int(value.shape[0])
                break
        if batch_size is None:
            raise ValueError("No tensor observations found in inputs['states']")

        vision_raw = self._flatten_obs(obs, self.vision_key, batch_size)
        vision = self.vision_proj(vision_raw)
        tactile_flat = torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.tactile_keys], dim=-1)
        tactile = self.tactile_base_proj(tactile_flat)
        tactile_tokens = torch.stack(
            [self.tactile_token_proj[key](self._flatten_obs(obs, key, batch_size)) for key in self.tactile_keys],
            dim=1,
        )
        sensor_ids = torch.arange(len(self.tactile_keys), device=self.device)
        tactile_tokens = tactile_tokens + self.tactile_sensor_embed(sensor_ids).unsqueeze(0)

        vision_query = self.vision_query_norm(vision).unsqueeze(1)
        tactile_context = self.tactile_context_norm(tactile_tokens)
        vision_attn, _ = self.vision_from_tactile_attn(vision_query, tactile_context, tactile_context, need_weights=False)
        vision_cross = (vision_query + vision_attn).squeeze(1)
        vision_cross = vision_cross + self.vision_ff(self.vision_ff_norm(vision_cross))

        tactile_query = self.tactile_query_norm(tactile_tokens)
        vision_context = self.vision_context_norm(vision).unsqueeze(1)
        tactile_attn, _ = self.tactile_from_vision_attn(tactile_query, vision_context, vision_context, need_weights=False)
        tactile_cross_tokens = tactile_query + tactile_attn
        tactile_cross_tokens = tactile_cross_tokens + self.tactile_ff(self.tactile_ff_norm(tactile_cross_tokens))
        tactile_cross = self.tactile_cross_proj(tactile_cross_tokens.reshape(batch_size, -1))

        occlusion_pred = self.occlusion_head(vision_raw)
        tactile_contact_pred = self.tactile_contact_head(tactile_flat)
        tactile_contact_mean = tactile_contact_pred.mean(dim=-1, keepdim=True)
        alpha = self.alpha_gate(torch.cat([occlusion_pred, tactile_contact_mean], dim=-1))
        set_latest_alpha(alpha)

        vision_mixed = (1.0 - alpha) * vision + alpha * vision_cross
        tactile_mixed = alpha * tactile + (1.0 - alpha) * tactile_cross
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision_mixed, tactile_mixed, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)

        aux_loss, occlusion_loss, contact_loss = self._build_aux_losses(
            obs=obs,
            batch_size=batch_size,
            occlusion_pred=occlusion_pred,
            tactile_contact_pred=tactile_contact_pred,
        )
        return mu, self.log_std_parameter, {
            "aux_loss": aux_loss,
            "aux_reliability_loss": occlusion_loss,
            "aux_probe_loss": contact_loss,
            "alpha": alpha.mean(),
            "g_probe": occlusion_pred.mean(),
            "g_grasp": tactile_contact_mean.mean(),
        }
