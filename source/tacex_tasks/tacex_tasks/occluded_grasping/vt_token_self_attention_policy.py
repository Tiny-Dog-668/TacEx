"""Vision/tactile token self-attention policy."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


class OccludedGraspingVTTokenSelfAttentionPolicy(GaussianMixin, Model):
    """Run self-attention over visual/tactile tokens, compress tactile tokens, then concatenate."""

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
        token_dim: int = 256,
        num_heads: int = 4,
        attention_mlp_dim: int = 512,
        dropout: float = 0.0,
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

        self.vision_key = str(vision_key)
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        self.vector_keys = tuple(key for key in vector_keys if self._has_obs_key(observation_space, key))
        self.token_dim = int(token_dim)

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self.tactile_keys:
            raise ValueError("No tactile observation keys found for token self-attention policy")

        self._obs_shapes = {
            key: self._get_obs_shape(observation_space, key)
            for key in (self.vision_key, *self.tactile_keys, *self.vector_keys)
        }
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }

        vision_dim = self._obs_dims[self.vision_key]
        vector_dim = sum(self._obs_dims[key] for key in self.vector_keys)
        self.num_tokens = 1 + len(self.tactile_keys)

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, self.token_dim),
            nn.ELU(),
        )
        self.tactile_proj = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.LayerNorm(self._obs_dims[key]),
                    nn.Linear(self._obs_dims[key], self.token_dim),
                    nn.ELU(),
                )
                for key in self.tactile_keys
            }
        )
        self.modality_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.token_dim))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

        self.token_norm = nn.LayerNorm(self.token_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(self.token_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.token_dim, int(attention_mlp_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(attention_mlp_dim), self.token_dim),
            nn.Dropout(float(dropout)),
        )
        self.tactile_compress = nn.Sequential(
            nn.LayerNorm(len(self.tactile_keys) * self.token_dim),
            nn.Linear(len(self.tactile_keys) * self.token_dim, self.token_dim),
            nn.ELU(),
        )

        if mlp_layers is None:
            mlp_layers = (512, 256, 128, 64)
        activation_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        mlp: list[nn.Module] = []
        last_dim = 2 * self.token_dim + vector_dim
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(activation_cls())
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

        vision_token = self.vision_proj(self._flatten_obs(obs, self.vision_key, batch_size)).unsqueeze(1)
        tactile_tokens = torch.stack(
            [self.tactile_proj[key](self._flatten_obs(obs, key, batch_size)) for key in self.tactile_keys],
            dim=1,
        )
        tokens = torch.cat([vision_token, tactile_tokens], dim=1) + self.modality_embed

        attn_input = self.token_norm(tokens)
        attn_tokens, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_tokens
        tokens = tokens + self.ff(self.ff_norm(tokens))

        vision_feature = tokens[:, 0]
        tactile_feature = self.tactile_compress(tokens[:, 1:].reshape(batch_size, -1))
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision_feature, tactile_feature, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}
