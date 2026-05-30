"""GRU-style VT policy with Sparsh inner tactile and down-depth GRU tactile."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha


class OccludedGraspingVTGRUSparshDownDepthPolicy(GaussianMixin, Model):
    """Fuse vision, non-recurrent inner Sparsh tactile, and recurrent down-depth tactile.

    Expected policy observations:
        third_resnet: 256
        tactile_left_rgb / tactile_right_rgb: Sparsh features, 256 each
        tactile_left_down_depth / tactile_right_down_depth: depth-CNN feature windows, 10 * 256 each
        proprio_obs: 18

    Fusion dimensions:
        vision -> 256
        inner RGB Sparsh pair -> 128
        down depth GRU pair -> 128
        final MLP input = 256 + 128 + 128 + 18 = 530
    """

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
        inner_tactile_keys: Iterable[str] | None = None,
        down_tactile_keys: Iterable[str] | None = None,
        vector_keys: Iterable[str] | None = None,
        proprio_key: str = "proprio_obs",
        tactile_time_window: int = 10,
        tactile_feature_dim: int = 256,
        down_gru_hidden_dim: int = 128,
        down_gru_layers: int = 2,
        vision_latent_dim: int = 256,
        inner_latent_dim: int = 128,
        down_latent_dim: int = 128,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        if inner_tactile_keys is None:
            inner_tactile_keys = ("tactile_left_rgb", "tactile_right_rgb")
        if down_tactile_keys is None:
            down_tactile_keys = ("tactile_left_down_depth", "tactile_right_down_depth")
        if vector_keys is None:
            vector_keys = (proprio_key,)
        if mlp_layers is None:
            mlp_layers = (256, 128, 64)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.inner_tactile_keys = tuple(key for key in inner_tactile_keys if self._has_obs_key(observation_space, key))
        self.down_tactile_keys = tuple(key for key in down_tactile_keys if self._has_obs_key(observation_space, key))
        self.vector_keys = tuple(key for key in vector_keys if self._has_obs_key(observation_space, key))
        self.tactile_time_window = max(1, int(tactile_time_window))
        self.tactile_feature_dim = int(tactile_feature_dim)
        self.down_gru_hidden_dim = int(down_gru_hidden_dim)
        self.down_gru_layers = int(down_gru_layers)
        self.vision_latent_dim = int(vision_latent_dim)
        self.inner_latent_dim = int(inner_latent_dim)
        self.down_latent_dim = int(down_latent_dim)

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self.inner_tactile_keys:
            raise ValueError("No inner Sparsh tactile observation keys found")
        if not self.down_tactile_keys:
            raise ValueError("No down-depth tactile observation keys found")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")

        used_keys = (self.vision_key, *self.inner_tactile_keys, *self.down_tactile_keys, *self.vector_keys)
        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in used_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1])
            for key in self._obs_shapes
            if self._obs_shapes[key] is not None
        }

        self._inner_feat_dims = {key: self._infer_single_feature_dim(key, allow_window=False) for key in self.inner_tactile_keys}
        self._down_feat_dims = {key: self._infer_single_feature_dim(key, allow_window=True) for key in self.down_tactile_keys}

        activation_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        vision_dim = int(self._obs_dims[self.vision_key])
        vector_dim = sum(int(self._obs_dims[key]) for key in self.vector_keys)
        inner_total_dim = sum(int(self._inner_feat_dims[key]) for key in self.inner_tactile_keys)

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, self.vision_latent_dim),
            nn.ELU(),
        )

        self.inner_proj = nn.Sequential(
            nn.LayerNorm(inner_total_dim),
            nn.Linear(inner_total_dim, self.inner_latent_dim),
            nn.ELU(),
        )

        self.down_grus = nn.ModuleDict()
        for key in self.down_tactile_keys:
            self.down_grus[key] = nn.GRU(
                input_size=int(self._down_feat_dims[key]),
                hidden_size=self.down_gru_hidden_dim,
                num_layers=self.down_gru_layers,
                batch_first=True,
            )

        down_total_dim = self.down_gru_hidden_dim * len(self.down_tactile_keys)
        self.down_proj = nn.Sequential(
            nn.LayerNorm(down_total_dim),
            nn.Linear(down_total_dim, self.down_latent_dim),
            nn.ELU(),
        )

        mlp_input_dim = self.vision_latent_dim + self.inner_latent_dim + self.down_latent_dim + vector_dim
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

    def _infer_single_feature_dim(self, key: str, allow_window: bool) -> int:
        total_dim = int(self._obs_dims[key])
        if allow_window and total_dim % self.tactile_time_window == 0:
            return total_dim // self.tactile_time_window
        if allow_window and total_dim == self.tactile_feature_dim:
            return self.tactile_feature_dim
        return total_dim

    def _flatten_obs(self, obs: Dict[str, Any], key: str, batch_size: int) -> torch.Tensor:
        value = obs.get(key)
        if value is None:
            return torch.zeros((batch_size, self._obs_dims[key]), device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(device=self.device, dtype=torch.float32)
        return value.reshape(batch_size, -1)

    def _inner_feature(self, obs: Dict[str, Any], key: str, batch_size: int) -> torch.Tensor:
        feature_dim = int(self._inner_feat_dims[key])
        flat = self._flatten_obs(obs, key, batch_size)
        if flat.shape[-1] == feature_dim:
            return flat
        if flat.shape[-1] % feature_dim == 0:
            return flat.reshape(batch_size, -1, feature_dim)[:, -1, :]
        raise ValueError(f"Inner tactile observation '{key}' dim={flat.shape[-1]} is incompatible with {feature_dim}")

    def _down_sequence(self, obs: Dict[str, Any], key: str, batch_size: int) -> torch.Tensor:
        feature_dim = int(self._down_feat_dims[key])
        flat = self._flatten_obs(obs, key, batch_size)
        if flat.shape[-1] == feature_dim:
            return flat.unsqueeze(1).expand(-1, self.tactile_time_window, -1)
        if flat.shape[-1] % feature_dim != 0:
            raise ValueError(f"Down tactile observation '{key}' dim={flat.shape[-1]} is incompatible with {feature_dim}")
        return flat.reshape(batch_size, -1, feature_dim)

    def _encode_inner_tactile(self, obs: Dict[str, Any], batch_size: int) -> torch.Tensor:
        return torch.cat([self._inner_feature(obs, key, batch_size) for key in self.inner_tactile_keys], dim=-1)

    def _encode_down_tactile(self, obs: Dict[str, Any], batch_size: int) -> torch.Tensor:
        chunks = []
        for key in self.down_tactile_keys:
            sequence = self._down_sequence(obs, key, batch_size)
            _, h_n = self.down_grus[key](sequence)
            chunks.append(h_n[-1])
        return torch.cat(chunks, dim=-1)

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
        inner_tactile = self.inner_proj(self._encode_inner_tactile(obs, batch_size))
        down_tactile = self.down_proj(self._encode_down_tactile(obs, batch_size))
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        set_latest_alpha(None)

        fused = torch.cat([vision, inner_tactile, down_tactile, *vector_features], dim=-1)
        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}


def quick_gru_sparsh_down_depth_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
    """Minimal forward/backward check without Isaac Sim."""
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "third_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_rgb": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_right_rgb": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_down_depth": gym.spaces.Box(-1.0, 1.0, shape=(2560,), dtype=float),
            "tactile_right_down_depth": gym.spaces.Box(-1.0, 1.0, shape=(2560,), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)
    model = OccludedGraspingVTGRUSparshDownDepthPolicy(obs_space, action_space, device=device)
    states = {
        "third_resnet": torch.rand(batch_size, 256),
        "tactile_left_rgb": torch.rand(batch_size, 256),
        "tactile_right_rgb": torch.rand(batch_size, 256),
        "tactile_left_down_depth": torch.rand(batch_size, 2560),
        "tactile_right_down_depth": torch.rand(batch_size, 2560),
        "proprio_obs": torch.rand(batch_size, 18),
    }
    mean, log_std, _ = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean()
    loss.backward()
    grad_norm = model.down_grus["tactile_left_down_depth"].weight_ih_l0.grad.norm().item()
    print(f"[gru_sparsh_down_depth_smoke_test] mean shape={tuple(mean.shape)} grad_norm={grad_norm:.6f}")
