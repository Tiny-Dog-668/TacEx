"""Policy-token Transformer fusion for env-side vision and tactile tokens."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


class OccludedGraspingVTPolicyTokenTransformerPolicy(GaussianMixin, Model):
    """Fuse pre-extracted visual/tactile tokens through a learnable policy token."""

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
        vision_key: str = "vision_tokens",
        tactile_key: str = "tactile_tokens",
        proprio_key: str = "proprio_obs",
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        action_chunk_size: int = 1,
        action_dim: int | None = None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.vision_key = str(vision_key)
        self.tactile_key = str(tactile_key)
        self.proprio_key = str(proprio_key)
        self.d_model = int(d_model)
        self.action_chunk_size = max(1, int(action_chunk_size))
        self.action_dim = int(action_dim) if action_dim is not None else int(self.num_actions)

        self._vision_shape = self._get_obs_shape(observation_space, self.vision_key)
        self._tactile_shape = self._get_obs_shape(observation_space, self.tactile_key)
        self._proprio_shape = self._get_obs_shape(observation_space, self.proprio_key)
        if self._vision_shape is None:
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if self._tactile_shape is None:
            raise ValueError(f"Missing tactile observation key '{self.tactile_key}'")
        if self._proprio_shape is None:
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")
        if len(self._vision_shape) != 2 or int(self._vision_shape[0]) != 49:
            raise ValueError(f"Expected vision token shape [49,D], got {self._vision_shape}")
        if len(self._tactile_shape) != 2 or int(self._tactile_shape[0]) != 16:
            raise ValueError(f"Expected tactile token shape [16,D], got {self._tactile_shape}")

        self.proprio_dim = int(torch.tensor(self._proprio_shape).prod().item())
        vision_token_dim = int(self._vision_shape[-1])
        tactile_token_dim = int(self._tactile_shape[-1])

        self.vision_proj = nn.Linear(vision_token_dim, self.d_model)
        self.vision_pos_embed = nn.Parameter(torch.zeros(1, 49, self.d_model))
        self.vision_modality_embed = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.tactile_proj = nn.Linear(tactile_token_dim, self.d_model)
        self.tactile_sensor_embed = nn.Embedding(4, self.d_model)
        self.tactile_local_embed = nn.Embedding(4, self.d_model)
        self.tactile_type_embed = nn.Embedding(2, self.d_model)
        self.tactile_modality_embed = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.policy_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        ff_dim = int(dim_feedforward) if dim_feedforward is not None else 4 * self.d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

        output_dim = self.action_dim * self.action_chunk_size
        if output_dim != self.num_actions:
            raise ValueError(
                "Policy output dimension must match the environment action space for skrl. "
                f"Got action_dim * action_chunk_size = {output_dim}, num_actions = {self.num_actions}."
            )

        if mlp_layers is None:
            mlp_layers = (512, 256, 128)
        activation_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        mlp: list[nn.Module] = []
        last_dim = self.d_model + self.proprio_dim
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(activation_cls())
            last_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, output_dim)
        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(initial_log_std)),
            requires_grad=not bool(fixed_log_std),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.policy_token, std=0.02)
        nn.init.trunc_normal_(self.vision_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.vision_modality_embed, std=0.02)
        nn.init.trunc_normal_(self.tactile_modality_embed, std=0.02)

    def _has_obs_key(self, obs_space, key: str) -> bool:
        if hasattr(obs_space, "spaces"):
            return key in obs_space.spaces
        return isinstance(obs_space, dict) and key in obs_space

    def _get_obs_shape(self, obs_space, key: str) -> Tuple[int, ...] | None:
        if hasattr(obs_space, "spaces") and key in obs_space.spaces:
            shape = getattr(obs_space.spaces[key], "shape", None)
            return tuple(int(s) for s in shape) if shape is not None else None
        if isinstance(obs_space, dict) and key in obs_space:
            spec = obs_space[key]
            if isinstance(spec, int):
                return (int(spec),)
            if isinstance(spec, (tuple, list)):
                return tuple(int(s) for s in spec)
            shape = getattr(spec, "shape", None)
            if shape is not None:
                return tuple(int(s) for s in shape)
        return None

    def _obs_tensor(self, obs: Dict[str, Any], key: str, shape: Tuple[int, ...], batch_size: int) -> torch.Tensor:
        value = obs.get(key)
        if value is None:
            return torch.zeros((batch_size, *shape), device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(device=self.device, dtype=torch.float32)
        return value.reshape(batch_size, *shape)

    def _encode_vision(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.vision_proj(vision_tokens)
        return tokens + self.vision_pos_embed + self.vision_modality_embed

    def _encode_tactile(self, tactile_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tactile_tokens.shape[0]
        local_tokens = self.tactile_proj(tactile_tokens).reshape(batch_size, 4, 4, self.d_model)

        sensor_ids = torch.arange(4, device=self.device)
        local_ids = torch.arange(4, device=self.device)
        # tactile order is bottom-left, bottom-right, inner-left, inner-right.
        type_ids = torch.tensor([0, 0, 1, 1], device=self.device, dtype=torch.long)

        tokens = local_tokens
        tokens = tokens + self.tactile_sensor_embed(sensor_ids).view(1, 4, 1, self.d_model)
        tokens = tokens + self.tactile_local_embed(local_ids).view(1, 1, 4, self.d_model)
        tokens = tokens + self.tactile_type_embed(type_ids).view(1, 4, 1, self.d_model)
        tokens = tokens.reshape(batch_size, 16, self.d_model)
        return tokens + self.tactile_modality_embed

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

        vision_tokens = self._obs_tensor(obs, self.vision_key, self._vision_shape, batch_size)
        tactile_tokens = self._obs_tensor(obs, self.tactile_key, self._tactile_shape, batch_size)
        proprio = self._obs_tensor(obs, self.proprio_key, self._proprio_shape, batch_size).reshape(batch_size, -1)

        vision_tokens = self._encode_vision(vision_tokens)
        tactile_tokens = self._encode_tactile(tactile_tokens)
        policy_token = self.policy_token.expand(batch_size, -1, -1)

        tokens = torch.cat([policy_token, vision_tokens, tactile_tokens], dim=1)
        if tokens.shape[1] != 66:
            raise RuntimeError(f"Expected 66 tokens, got {tokens.shape[1]}")
        tokens = self.transformer(tokens)
        c_out = tokens[:, 0]

        hidden = self.mlp(torch.cat([c_out, proprio], dim=-1))
        mu = self.mu(hidden)
        if self.action_chunk_size > 1:
            mu = mu.view(batch_size, self.action_chunk_size, self.action_dim)
        return mu, self.log_std_parameter, {}


def quick_policy_token_transformer_smoke_test(batch_size: int = 2, device: str = "cpu") -> None:
    """Minimal forward/backward check without Isaac Sim."""
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "vision_tokens": gym.spaces.Box(-10.0, 10.0, shape=(49, 512), dtype=float),
            "tactile_tokens": gym.spaces.Box(-10.0, 10.0, shape=(16, 128), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)
    model = OccludedGraspingVTPolicyTokenTransformerPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device=device,
    ).to(device)
    states = {
        "vision_tokens": torch.rand(batch_size, 49, 512, device=device),
        "tactile_tokens": torch.rand(batch_size, 16, 128, device=device),
        "proprio_obs": torch.rand(batch_size, 18, device=device),
    }
    mean, log_std, _ = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean()
    loss.backward()
    print(f"[policy_token_transformer_smoke_test] mean shape={tuple(mean.shape)}")
