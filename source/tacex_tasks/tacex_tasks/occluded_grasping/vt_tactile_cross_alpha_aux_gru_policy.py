"""Tactile-cross-alpha aux policy with per-sensor tactile GRU encoders."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha


class OccludedGraspingVTTactileCrossAlphaAuxGRUPolicy(GaussianMixin, Model):
    """Encode tactile histories with GRUs, predict occlusion, and alpha-mix T and T'."""

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
        tactile_time_window: int = 10,
        tactile_gru_hidden_dim: int = 64,
        tactile_gru_layers: int = 2,
        tactile_latent_dim: int = 128,
        fused_dim: int = 256,
        num_heads: int = 4,
        attention_mlp_dim: int = 512,
        dropout: float = 0.0,
        occlusion_hidden_dim: int = 128,
        occlusion_aux_weight: float = 1.0,
        gate_hidden_dim: int = 64,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
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
        if mlp_layers is None:
            mlp_layers = (256, 128, 64)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.aux_occlusion_gt_key = str(aux_occlusion_gt_key)
        self.aux_tactile_contact_gt_key = str(aux_tactile_contact_gt_key)
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        self.vector_keys = tuple(key for key in vector_keys if self._has_obs_key(observation_space, key))
        self.tactile_time_window = max(1, int(tactile_time_window))
        self.tactile_gru_hidden_dim = int(tactile_gru_hidden_dim)
        self.tactile_gru_layers = int(tactile_gru_layers)
        self.tactile_latent_dim = int(tactile_latent_dim)
        self.fused_dim = int(fused_dim)
        self.occlusion_aux_weight = float(occlusion_aux_weight)

        required_keys = (
            self.vision_key,
            *self.tactile_keys,
            *self.vector_keys,
            self.proprio_key,
            self.aux_occlusion_gt_key,
            self.aux_tactile_contact_gt_key,
        )
        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if len(self.tactile_keys) != 4:
            raise ValueError(f"Expected 4 tactile observation keys, got {self.tactile_keys}")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")
        if not self._has_obs_key(observation_space, self.aux_occlusion_gt_key):
            raise ValueError(f"Missing aux observation key '{self.aux_occlusion_gt_key}'")
        if not self._has_obs_key(observation_space, self.aux_tactile_contact_gt_key):
            raise ValueError(f"Missing aux observation key '{self.aux_tactile_contact_gt_key}'")

        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in required_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }
        if self._obs_dims[self.aux_occlusion_gt_key] != 1:
            raise ValueError(f"Expected '{self.aux_occlusion_gt_key}' dim=1")
        if self._obs_dims[self.aux_tactile_contact_gt_key] != 4:
            raise ValueError(f"Expected '{self.aux_tactile_contact_gt_key}' dim=4")

        self._sensor_feat_dims: dict[str, int] = {}
        for key in self.tactile_keys:
            total_dim = int(self._obs_dims[key])
            if total_dim % self.tactile_time_window == 0:
                self._sensor_feat_dims[key] = total_dim // self.tactile_time_window
            else:
                self._sensor_feat_dims[key] = total_dim

        act_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        vision_dim = self._obs_dims[self.vision_key]
        proprio_dim = self._obs_dims[self.proprio_key]
        vector_dim = sum(self._obs_dims[key] for key in self.vector_keys)

        self.vision_proj = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, self.fused_dim), act_cls())
        self.occlusion_head = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, int(occlusion_hidden_dim)),
            act_cls(),
            nn.Linear(int(occlusion_hidden_dim), 1),
            nn.Sigmoid(),
        )

        self.tactile_grus = nn.ModuleDict()
        self.tactile_heads = nn.ModuleDict()
        for key in self.tactile_keys:
            self.tactile_grus[key] = nn.GRU(
                input_size=self._sensor_feat_dims[key],
                hidden_size=self.tactile_gru_hidden_dim,
                num_layers=self.tactile_gru_layers,
                batch_first=True,
            )
            self.tactile_heads[key] = nn.Sequential(
                nn.LayerNorm(self.tactile_gru_hidden_dim),
                nn.Linear(self.tactile_gru_hidden_dim, self.tactile_latent_dim),
                act_cls(),
            )

        tactile_total_latent = self.tactile_latent_dim * len(self.tactile_keys)
        self.tactile_base_proj = nn.Sequential(
            nn.LayerNorm(tactile_total_latent),
            nn.Linear(tactile_total_latent, self.fused_dim),
            act_cls(),
        )
        self.tactile_token_proj = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.LayerNorm(self.tactile_latent_dim),
                    nn.Linear(self.tactile_latent_dim, self.fused_dim),
                    act_cls(),
                )
                for key in self.tactile_keys
            }
        )
        self.tactile_sensor_embed = nn.Embedding(len(self.tactile_keys), self.fused_dim)
        self.tactile_cross_proj = nn.Sequential(
            nn.LayerNorm(len(self.tactile_keys) * self.fused_dim),
            nn.Linear(len(self.tactile_keys) * self.fused_dim, self.fused_dim),
            act_cls(),
        )

        self.query_norm = nn.LayerNorm(self.fused_dim)
        self.context_norm = nn.LayerNorm(self.fused_dim)
        self.tactile_visual_attn = nn.MultiheadAttention(
            embed_dim=self.fused_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_ff_norm = nn.LayerNorm(self.fused_dim)
        self.attn_ff = nn.Sequential(
            nn.Linear(self.fused_dim, int(attention_mlp_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(attention_mlp_dim), self.fused_dim),
            nn.Dropout(float(dropout)),
        )
        self.alpha_gate = nn.Sequential(
            nn.Linear(1 + 4 + proprio_dim, int(gate_hidden_dim)),
            act_cls(),
            nn.Linear(int(gate_hidden_dim), 1),
            nn.Sigmoid(),
        )

        mlp: list[nn.Module] = []
        last_dim = self.fused_dim + vector_dim
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
            return torch.zeros((batch_size, self._obs_dims[key]), device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(device=self.device, dtype=torch.float32)
        return value.reshape(batch_size, -1)

    def _encode_tactile_tokens(self, obs: Dict[str, Any], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        latent_chunks = []
        token_chunks = []
        for key in self.tactile_keys:
            feature_dim = self._sensor_feat_dims[key]
            flat = self._flatten_obs(obs, key, batch_size)
            if flat.shape[-1] == feature_dim:
                sequence = flat.unsqueeze(1).expand(-1, self.tactile_time_window, -1)
            else:
                sequence = flat.reshape(batch_size, self.tactile_time_window, feature_dim)
            _, h_n = self.tactile_grus[key](sequence)
            latent = self.tactile_heads[key](h_n[-1])
            latent_chunks.append(latent)
            token_chunks.append(self.tactile_token_proj[key](latent))
        tactile_flat = torch.cat(latent_chunks, dim=-1)
        tactile_tokens = torch.stack(token_chunks, dim=1)
        return tactile_flat, tactile_tokens

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
        vision = self.vision_proj(vision_raw).unsqueeze(1)
        tactile_flat, tactile_tokens = self._encode_tactile_tokens(obs, batch_size)
        tactile = self.tactile_base_proj(tactile_flat)

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


def quick_tactile_cross_alpha_aux_gru_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
    """Minimal forward/backward check without Isaac Sim."""
    import gymnasium as gym

    window = 10
    tactile_dim = 256 * window
    obs_space = gym.spaces.Dict(
        {
            "third_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(tactile_dim,), dtype=float),
            "tactile_right_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(tactile_dim,), dtype=float),
            "tactile_left_down_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(tactile_dim,), dtype=float),
            "tactile_right_down_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(tactile_dim,), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
            "aux_occlusion_gt": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=float),
            "aux_tactile_contact_gt": gym.spaces.Box(0.0, 1.0, shape=(4,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)
    model = OccludedGraspingVTTactileCrossAlphaAuxGRUPolicy(obs_space, action_space, device=device)
    states = {
        "third_resnet": torch.rand(batch_size, 256),
        "tactile_left_depth_resnet": torch.rand(batch_size, tactile_dim),
        "tactile_right_depth_resnet": torch.rand(batch_size, tactile_dim),
        "tactile_left_down_depth_resnet": torch.rand(batch_size, tactile_dim),
        "tactile_right_down_depth_resnet": torch.rand(batch_size, tactile_dim),
        "proprio_obs": torch.rand(batch_size, 18),
        "aux_occlusion_gt": torch.rand(batch_size, 1),
        "aux_tactile_contact_gt": torch.rand(batch_size, 4),
    }
    mean, log_std, outputs = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean() + outputs["aux_loss"]
    loss.backward()
    grad_norm = model.tactile_grus["tactile_left_depth_resnet"].weight_ih_l0.grad.norm().item()
    print(
        f"[tactile_cross_alpha_aux_gru_smoke_test] mean shape={tuple(mean.shape)} "
        f"aux_loss={outputs['aux_loss'].item():.6f} gru_grad_norm={grad_norm:.6f}"
    )
