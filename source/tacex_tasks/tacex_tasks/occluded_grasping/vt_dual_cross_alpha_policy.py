"""Bidirectional vision/tactile cross-attention policy with alpha mixing."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha


class OccludedGraspingVTDualCrossAlphaPolicy(GaussianMixin, Model):
    """Compute V' from tactile context and T' from visual context, then alpha-mix both."""

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
        fused_dim: int = 256,
        num_heads: int = 4,
        attention_mlp_dim: int = 512,
        dropout: float = 0.0,
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

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        self.vector_keys = tuple(key for key in vector_keys if self._has_obs_key(observation_space, key))
        self.fused_dim = int(fused_dim)

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self.tactile_keys:
            raise ValueError("No tactile observation keys found for dual-cross-alpha policy")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")

        self._obs_shapes = {
            key: self._get_obs_shape(observation_space, key)
            for key in (self.vision_key, *self.tactile_keys, *self.vector_keys)
        }
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }

        vision_dim = self._obs_dims[self.vision_key]
        proprio_dim = self._obs_dims[self.proprio_key]
        vector_dim = sum(self._obs_dims[key] for key in self.vector_keys)
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, self.fused_dim),
            nn.ELU(),
        )
        self.tactile_token_proj = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.LayerNorm(self._obs_dims[key]),
                    nn.Linear(self._obs_dims[key], self.fused_dim),
                    nn.ELU(),
                )
                for key in self.tactile_keys
            }
        )
        self.tactile_sensor_embed = nn.Embedding(len(self.tactile_keys), self.fused_dim)
        self.tactile_base_proj = nn.Sequential(
            nn.LayerNorm(len(self.tactile_keys) * self.fused_dim),
            nn.Linear(len(self.tactile_keys) * self.fused_dim, self.fused_dim),
            nn.ELU(),
        )

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
            nn.ELU(),
        )

        self.alpha_gate = nn.Sequential(
            nn.Linear(proprio_dim, int(gate_hidden_dim)),
            nn.ELU(),
            nn.Linear(int(gate_hidden_dim), 1),
            nn.Sigmoid(),
        )

        if mlp_layers is None:
            mlp_layers = (512, 256, 128, 64)
        activation_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        mlp: list[nn.Module] = []
        last_dim = (2 * self.fused_dim) + vector_dim
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

        vision = self.vision_proj(self._flatten_obs(obs, self.vision_key, batch_size))
        tactile_tokens = torch.stack(
            [self.tactile_token_proj[key](self._flatten_obs(obs, key, batch_size)) for key in self.tactile_keys],
            dim=1,
        )
        sensor_ids = torch.arange(len(self.tactile_keys), device=self.device)
        tactile_tokens = tactile_tokens + self.tactile_sensor_embed(sensor_ids).unsqueeze(0)
        tactile = self.tactile_base_proj(tactile_tokens.reshape(batch_size, -1))

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

        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        alpha = self.alpha_gate(proprio)
        set_latest_alpha(alpha)
        vision_mixed = (1.0 - alpha) * vision + alpha * vision_cross
        tactile_mixed = alpha * tactile + (1.0 - alpha) * tactile_cross

        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision_mixed, tactile_mixed, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}


def quick_dual_cross_alpha_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
    """Minimal forward/backward check without Isaac Sim."""
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "third_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_right_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_down_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_right_down_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)

    model = OccludedGraspingVTDualCrossAlphaPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device=device,
    )
    states = {
        "third_resnet": torch.rand(batch_size, 256),
        "tactile_left_depth_resnet": torch.rand(batch_size, 256),
        "tactile_right_depth_resnet": torch.rand(batch_size, 256),
        "tactile_left_down_depth_resnet": torch.rand(batch_size, 256),
        "tactile_right_down_depth_resnet": torch.rand(batch_size, 256),
        "proprio_obs": torch.rand(batch_size, 18),
    }
    mean, log_std, _ = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean()
    loss.backward()
    print(f"[dual_cross_alpha_smoke_test] mean shape={tuple(mean.shape)}")
