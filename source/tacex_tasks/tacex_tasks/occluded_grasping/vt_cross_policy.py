"""ReTac-ACT style token-level vision-tactile cross-attention policy."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha


class CrossAttentionBlock(nn.Module):
    """Cross-attention block that keeps the token sequence structure intact."""

    def __init__(self, embed_dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), embed_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(query)
        ctx = self.context_norm(context)
        attn_out, _ = self.attn(q, ctx, ctx, need_weights=False)
        x = q + attn_out
        x = x + self.ff(self.ff_norm(x))
        return x


class BidirectionalCrossAttentionLayer(nn.Module):
    """Bidirectional tactile<->vision cross-attention with optional scalar gating."""

    def __init__(self, embed_dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.visual_from_tactile = CrossAttentionBlock(embed_dim, heads, ff_dim, dropout)
        self.tactile_from_visual = CrossAttentionBlock(embed_dim, heads, ff_dim, dropout)
        mixer_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(mixer_layer, num_layers=1)

    def forward(
        self,
        visual: torch.Tensor,
        tactile: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_tilde = self.visual_from_tactile(visual, tactile)
        tactile_tilde = self.tactile_from_visual(tactile, visual)

        if alpha is not None:
            visual = (1.0 - alpha) * visual + alpha * visual_tilde
            tactile = alpha * tactile + (1.0 - alpha) * tactile_tilde
        else:
            visual = visual_tilde
            tactile = tactile_tilde

        mixed = self.token_mixer(torch.cat([visual, tactile], dim=1))
        visual_len = visual.shape[1]
        return mixed[:, :visual_len], mixed[:, visual_len:]


class OccludedGraspingVisionTactileCrossAttentionPolicy(GaussianMixin, Model):
    """Token-level actor for occluded grasping with bidirectional cross-attention."""

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
        vision_keys: Iterable[str] | None = None,
        tactile_keys: Iterable[str] | None = None,
        vector_keys: Iterable[str] | None = None,
        proprio_key: str = "proprio_obs",
        vision_token_hw: Iterable[int] = (4, 5),
        tactile_token_hw: Iterable[int] = (3, 4),
        embed_dim: int = 512,
        depth: int = 2,
        heads: int = 8,
        attention_mlp_dim: int = 1024,
        dropout: float = 0.0,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        use_proprio_gate: bool = True,
        gate_hidden_dim: int = 128,
        alpha_print_interval: int = 0,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        if vision_keys is None:
            vision_keys = ("third_resnet_tokens",)
        if tactile_keys is None:
            tactile_keys = (
                "tactile_left_tokens",
                "tactile_right_tokens",
                "tactile_left_down_tokens",
                "tactile_right_down_tokens",
            )
        if vector_keys is None:
            vector_keys = (proprio_key,)

        self.embed_dim = int(embed_dim)
        self.proprio_key = str(proprio_key)
        self.vision_keys = tuple(key for key in vision_keys if self._has_obs_key(observation_space, key))
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        self.vector_keys = tuple(key for key in vector_keys if self._has_obs_key(observation_space, key))
        self.use_proprio_gate = bool(use_proprio_gate) and self._has_obs_key(observation_space, self.proprio_key)
        self.vision_token_hw = self._normalize_hw(vision_token_hw, "vision_token_hw")
        self.tactile_token_hw = self._normalize_hw(tactile_token_hw, "tactile_token_hw")

        if not self.vision_keys:
            raise ValueError("No visual token observation keys found for cross-attention policy")
        if not self.tactile_keys:
            raise ValueError("No tactile token observation keys found for cross-attention policy")

        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in (*self.vision_keys, *self.tactile_keys, *self.vector_keys)}
        self._obs_dims = {key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None}

        self.vision_projs = nn.ModuleDict(
            {key: nn.Linear(self._obs_dims[key], self.embed_dim) for key in self.vision_keys}
        )
        self.tactile_projs = nn.ModuleDict(
            {key: nn.Linear(self._obs_dims[key], self.embed_dim) for key in self.tactile_keys}
        )

        self.vision_type_embeddings = nn.ParameterDict(
            {key: nn.Parameter(torch.zeros(1, 1, self.embed_dim)) for key in self.vision_keys}
        )
        self.tactile_type_embeddings = nn.ParameterDict(
            {key: nn.Parameter(torch.zeros(1, 1, self.embed_dim)) for key in self.tactile_keys}
        )
        self.token_dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0.0 else nn.Identity()
        self._vision_pos_buffer_names = {}
        self._tactile_pos_buffer_names = {}
        self._register_position_buffers()

        self.layers = nn.ModuleList(
            [
                BidirectionalCrossAttentionLayer(
                    embed_dim=self.embed_dim,
                    heads=int(heads),
                    ff_dim=int(attention_mlp_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )

        proprio_dim = self._obs_dims.get(self.proprio_key, 0)
        if self.use_proprio_gate:
            self.alpha_gate = nn.Sequential(
                nn.Linear(proprio_dim, int(gate_hidden_dim)),
                nn.ELU(),
                nn.Linear(int(gate_hidden_dim), 1),
                nn.Sigmoid(),
            )
        else:
            self.alpha_gate = None

        vector_dim = sum(self._obs_dims[key] for key in self.vector_keys)
        mlp_input_dim = (2 * self.embed_dim) + vector_dim

        if mlp_layers is None:
            mlp_layers = (512, 256, 128)
        activation_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        mlp = []
        last_dim = mlp_input_dim
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

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.vision_type_embeddings.values():
            nn.init.normal_(parameter, std=0.02)
        for parameter in self.tactile_type_embeddings.values():
            nn.init.normal_(parameter, std=0.02)

    def _normalize_hw(self, hw: Iterable[int], name: str) -> tuple[int, int]:
        values = tuple(int(v) for v in hw)
        if len(values) != 2 or any(v <= 0 for v in values):
            raise ValueError(f"{name} must contain exactly two positive integers, got {hw}")
        return values

    def _make_buffer_name(self, prefix: str, key: str) -> str:
        suffix = re.sub(r"[^0-9a-zA-Z_]+", "_", str(key)).strip("_")
        return f"_{prefix}_{suffix}"

    def _build_2d_sincos_position_encoding(self, height: int, width: int) -> torch.Tensor:
        if self.embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4 for 2D sine-cosine encoding, got {self.embed_dim}")
        dtype = torch.float32
        pos_dim = self.embed_dim // 4
        y = torch.arange(height, dtype=dtype)
        x = torch.arange(width, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        omega = torch.arange(pos_dim, dtype=dtype)
        omega = omega / max(pos_dim - 1, 1)
        omega = 1.0 / (10000.0 ** omega)
        out_y = grid_y.reshape(-1, 1) * omega.reshape(1, -1)
        out_x = grid_x.reshape(-1, 1) * omega.reshape(1, -1)
        pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
        return pos.unsqueeze(0)

    def _register_position_buffer_for_key(self, key: str, hw: tuple[int, int], prefix: str) -> str:
        shape = self._obs_shapes.get(key)
        if shape is None or len(shape) != 2:
            raise ValueError(f"Observation '{key}' must have sequence shape [N, C], got {shape}")
        token_count = int(shape[0])
        if token_count != int(hw[0] * hw[1]):
            raise ValueError(
                f"Token grid mismatch for '{key}': observation has {token_count} tokens, but {prefix}_token_hw={hw}"
            )
        buffer_name = self._make_buffer_name(f"{prefix}_pos", key)
        self.register_buffer(buffer_name, self._build_2d_sincos_position_encoding(hw[0], hw[1]), persistent=False)
        return buffer_name

    def _register_position_buffers(self) -> None:
        for key in self.vision_keys:
            self._vision_pos_buffer_names[key] = self._register_position_buffer_for_key(
                key, self.vision_token_hw, prefix="vision"
            )
        for key in self.tactile_keys:
            self._tactile_pos_buffer_names[key] = self._register_position_buffer_for_key(
                key, self.tactile_token_hw, prefix="tactile"
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

    def _tokenize_sequence(
        self,
        obs: Dict[str, Any],
        key: str,
        projection: nn.Linear,
        type_embedding: nn.Parameter,
        position_encoding: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        value = obs.get(key)
        if value is None:
            shape = self._obs_shapes[key]
            if shape is None or len(shape) != 2:
                raise ValueError(f"Missing sequence observation '{key}' and shape metadata is unavailable")
            value = torch.zeros((batch_size, shape[0], shape[1]), device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(device=self.device, dtype=torch.float32)
        if value.dim() != 3:
            raise ValueError(f"Expected token sequence [B, N, C] for '{key}', got shape {tuple(value.shape)}")
        tokens = projection(value)
        tokens = self.token_dropout(tokens + position_encoding + type_embedding)
        return tokens

    def _collect_tokens(
        self,
        obs: Dict[str, Any],
        keys: tuple[str, ...],
        projections: nn.ModuleDict,
        type_embeddings: nn.ParameterDict,
        position_buffer_names: dict[str, str],
        batch_size: int,
    ) -> torch.Tensor:
        tokens = [
            self._tokenize_sequence(
                obs,
                key,
                projections[key],
                type_embeddings[key],
                getattr(self, position_buffer_names[key]),
                batch_size,
            )
            for key in keys
        ]
        return torch.cat(tokens, dim=1)

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

        visual = self._collect_tokens(
            obs,
            self.vision_keys,
            self.vision_projs,
            self.vision_type_embeddings,
            self._vision_pos_buffer_names,
            batch_size,
        )
        tactile = self._collect_tokens(
            obs,
            self.tactile_keys,
            self.tactile_projs,
            self.tactile_type_embeddings,
            self._tactile_pos_buffer_names,
            batch_size,
        )

        alpha = None
        if self.alpha_gate is not None:
            proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
            alpha = self.alpha_gate(proprio).view(batch_size, 1, 1)
            set_latest_alpha(alpha.view(batch_size))
        else:
            set_latest_alpha(None)

        for layer in self.layers:
            visual, tactile = layer(visual, tactile, alpha=alpha)

        visual_summary = visual.mean(dim=1)
        tactile_summary = tactile.mean(dim=1)
        fused = torch.cat([visual_summary, tactile_summary], dim=-1)

        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        if vector_features:
            fused = torch.cat([fused, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}


def quick_cross_attention_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
    """Minimal forward/backward check without Isaac Sim."""
    import gymnasium as gym

    tactile_token_dim = 128  # Default VT-Cross env uses the CNN tactile encoder.
    obs_space = gym.spaces.Dict(
        {
            "third_resnet_tokens": gym.spaces.Box(-1.0, 1.0, shape=(20, 256), dtype=float),
            "tactile_left_tokens": gym.spaces.Box(-1.0, 1.0, shape=(12, tactile_token_dim), dtype=float),
            "tactile_right_tokens": gym.spaces.Box(-1.0, 1.0, shape=(12, tactile_token_dim), dtype=float),
            "tactile_left_down_tokens": gym.spaces.Box(-1.0, 1.0, shape=(12, tactile_token_dim), dtype=float),
            "tactile_right_down_tokens": gym.spaces.Box(-1.0, 1.0, shape=(12, tactile_token_dim), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)

    model = OccludedGraspingVisionTactileCrossAttentionPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device=device,
    )
    states = {
        "third_resnet_tokens": torch.rand(batch_size, 20, 256),
        "tactile_left_tokens": torch.rand(batch_size, 12, tactile_token_dim),
        "tactile_right_tokens": torch.rand(batch_size, 12, tactile_token_dim),
        "tactile_left_down_tokens": torch.rand(batch_size, 12, tactile_token_dim),
        "tactile_right_down_tokens": torch.rand(batch_size, 12, tactile_token_dim),
        "proprio_obs": torch.rand(batch_size, 18),
    }
    mean, log_std, _ = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean()
    loss.backward()
    grad_norm = model.vision_projs["third_resnet_tokens"].weight.grad.norm().item()
    print(f"[smoke_test] mean shape={tuple(mean.shape)} grad_norm={grad_norm:.6f}")
