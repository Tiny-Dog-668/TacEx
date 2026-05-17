"""Alpha-gated VT policy with temporal GRU encoders for vision and tactile features."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha


class OccludedGraspingVTAlphaDualGRUPolicy(GaussianMixin, Model):
    """Late-fusion alpha policy using 2-layer GRUs for vision and tactile branches."""

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
        vision_time_window: int = 5,
        vision_gru_hidden_dim: int = 64,
        vision_gru_layers: int = 2,
        vision_latent_dim: int = 256,
        tactile_time_window: int = 10,
        tactile_gru_hidden_dim: int = 64,
        tactile_gru_layers: int = 2,
        tactile_latent_dim: int = 128,
        fused_dim: int = 256,
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
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        self.vector_keys = tuple(key for key in vector_keys if self._has_obs_key(observation_space, key))
        self.vision_time_window = max(1, int(vision_time_window))
        self.vision_gru_hidden_dim = int(vision_gru_hidden_dim)
        self.vision_gru_layers = int(vision_gru_layers)
        self.vision_latent_dim = int(vision_latent_dim)
        self.tactile_time_window = max(1, int(tactile_time_window))
        self.tactile_gru_hidden_dim = int(tactile_gru_hidden_dim)
        self.tactile_gru_layers = int(tactile_gru_layers)
        self.tactile_latent_dim = int(tactile_latent_dim)
        self.fused_dim = int(fused_dim)

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self.tactile_keys:
            raise ValueError("No tactile observation keys found for alpha dual-GRU policy")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")

        self._obs_shapes = {
            key: self._get_obs_shape(observation_space, key)
            for key in (self.vision_key, *self.tactile_keys, *self.vector_keys)
        }
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }

        vision_total_dim = int(self._obs_dims[self.vision_key])
        if vision_total_dim % self.vision_time_window != 0:
            raise ValueError(
                f"Vision observation '{self.vision_key}' dim={vision_total_dim} "
                f"is not divisible by window={self.vision_time_window}"
            )
        self._vision_feat_dim = vision_total_dim // self.vision_time_window

        self._sensor_feat_dims: dict[str, int] = {}
        for key in self.tactile_keys:
            total_dim = int(self._obs_dims[key])
            if total_dim % self.tactile_time_window != 0:
                raise ValueError(
                    f"Tactile observation '{key}' dim={total_dim} is not divisible by window={self.tactile_time_window}"
                )
            self._sensor_feat_dims[key] = total_dim // self.tactile_time_window

        proprio_dim = int(self._obs_dims[self.proprio_key])
        vector_dim = sum(int(self._obs_dims[key]) for key in self.vector_keys)

        activation_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU

        self.vision_gru = nn.GRU(
            input_size=self._vision_feat_dim,
            hidden_size=self.vision_gru_hidden_dim,
            num_layers=self.vision_gru_layers,
            batch_first=True,
        )
        self.vision_head = nn.Sequential(
            nn.LayerNorm(self.vision_gru_hidden_dim),
            nn.Linear(self.vision_gru_hidden_dim, self.vision_latent_dim),
            nn.ELU(),
            nn.Linear(self.vision_latent_dim, self.fused_dim),
            nn.ELU(),
        )

        self.tactile_grus = nn.ModuleDict()
        self.tactile_heads = nn.ModuleDict()
        for key in self.tactile_keys:
            input_dim = self._sensor_feat_dims[key]
            self.tactile_grus[key] = nn.GRU(
                input_size=input_dim,
                hidden_size=self.tactile_gru_hidden_dim,
                num_layers=self.tactile_gru_layers,
                batch_first=True,
            )
            self.tactile_heads[key] = nn.Sequential(
                nn.LayerNorm(self.tactile_gru_hidden_dim),
                nn.Linear(self.tactile_gru_hidden_dim, self.tactile_latent_dim),
                nn.ELU(),
            )

        tactile_total_latent = self.tactile_latent_dim * len(self.tactile_keys)
        self.tactile_proj = nn.Sequential(
            nn.LayerNorm(tactile_total_latent),
            nn.Linear(tactile_total_latent, self.fused_dim),
            nn.ELU(),
        )

        self.alpha_gate = nn.Sequential(
            nn.Linear(proprio_dim, int(gate_hidden_dim)),
            nn.ELU(),
            nn.Linear(int(gate_hidden_dim), 1),
            nn.Sigmoid(),
        )

        mlp_input_dim = (2 * self.fused_dim) + vector_dim
        mlp: list[nn.Module] = []
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

    def _encode_vision(self, obs: Dict[str, Any], batch_size: int) -> torch.Tensor:
        flat = self._flatten_obs(obs, self.vision_key, batch_size)
        sequence = flat.reshape(batch_size, self.vision_time_window, self._vision_feat_dim)
        _, h_n = self.vision_gru(sequence)
        last_hidden = h_n[-1]
        return self.vision_head(last_hidden)

    def _encode_tactile(self, obs: Dict[str, Any], batch_size: int) -> torch.Tensor:
        latent_chunks = []
        for key in self.tactile_keys:
            feature_dim = self._sensor_feat_dims[key]
            flat = self._flatten_obs(obs, key, batch_size)
            sequence = flat.reshape(batch_size, self.tactile_time_window, feature_dim)
            _, h_n = self.tactile_grus[key](sequence)
            last_hidden = h_n[-1]
            latent = self.tactile_heads[key](last_hidden)
            latent_chunks.append(latent)
        return torch.cat(latent_chunks, dim=-1)

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

        vision = self._encode_vision(obs, batch_size)
        tactile = self.tactile_proj(self._encode_tactile(obs, batch_size))
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        alpha = self.alpha_gate(proprio)
        set_latest_alpha(alpha)

        gated = torch.cat([(1.0 - alpha) * vision, alpha * tactile], dim=-1)
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([gated, *vector_features], dim=-1) if vector_features else gated

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}
