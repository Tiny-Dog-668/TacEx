"""Alpha-gated VT policy using visual, tactile, and proprio inputs plus tactile validity."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha


class OccludedGraspingVTGateAlphaPolicy(GaussianMixin, Model):
    """Late-fusion policy whose alpha gate sees compressed visual/tactile features and proprio."""

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
        tactile_valid_key: str = "tactile_valid",
        fused_dim: int = 128,
        vision_fused_dim: int | None = None,
        tactile_fused_dim: int | None = None,
        gate_hidden_dim: int = 128,
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
        self.tactile_valid_key = str(tactile_valid_key)
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        self.vector_keys = tuple(key for key in vector_keys if self._has_obs_key(observation_space, key))

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self.tactile_keys:
            raise ValueError("No tactile observation keys found for gated alpha policy")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")

        self._has_tactile_valid = self._has_obs_key(observation_space, self.tactile_valid_key)
        tracked_keys = [self.vision_key, *self.tactile_keys, *self.vector_keys]
        if self._has_tactile_valid and self.tactile_valid_key not in tracked_keys:
            tracked_keys.append(self.tactile_valid_key)

        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in tracked_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }

        vision_dim = self._obs_dims[self.vision_key]
        tactile_dim = sum(self._obs_dims[key] for key in self.tactile_keys)
        proprio_dim = self._obs_dims[self.proprio_key]
        vector_dim = sum(self._obs_dims[key] for key in self.vector_keys)
        tactile_valid_dim = self._obs_dims.get(self.tactile_valid_key, 0)
        self.vision_fused_dim = int(fused_dim if vision_fused_dim is None else vision_fused_dim)
        self.tactile_fused_dim = int(fused_dim if tactile_fused_dim is None else tactile_fused_dim)

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, self.vision_fused_dim),
            nn.ELU(),
        )
        self.tactile_proj = nn.Sequential(
            nn.LayerNorm(tactile_dim),
            nn.Linear(tactile_dim, self.tactile_fused_dim),
            nn.ELU(),
        )
        fused_pair_dim = self.vision_fused_dim + self.tactile_fused_dim
        self.alpha_gate = nn.Sequential(
            nn.Linear(fused_pair_dim + proprio_dim + tactile_valid_dim, int(gate_hidden_dim)),
            nn.ELU(),
            nn.Linear(int(gate_hidden_dim), 1),
        )

        mlp_input_dim = fused_pair_dim + vector_dim
        if mlp_layers is None:
            mlp_layers = (256, 128, 64)
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
        tactile = self.tactile_proj(
            torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.tactile_keys], dim=-1)
        )
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        tactile_valid = (
            self._flatten_obs(obs, self.tactile_valid_key, batch_size).clamp(0.0, 1.0)
            if self._has_tactile_valid
            else torch.ones((batch_size, 1), device=self.device, dtype=torch.float32)
        )

        alpha_parts = [vision, tactile, proprio]
        if self._has_tactile_valid:
            alpha_parts.append(tactile_valid)
        alpha_input = torch.cat(alpha_parts, dim=-1)
        alpha = torch.sigmoid(self.alpha_gate(alpha_input))
        alpha = alpha * tactile_valid[:, :1]
        set_latest_alpha(alpha)

        gated = torch.cat([(1.0 - alpha) * vision, alpha * tactile], dim=-1)
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([gated, *vector_features], dim=-1) if vector_features else gated

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}


def quick_gate_alpha_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
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
            "tactile_valid": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)

    model = OccludedGraspingVTGateAlphaPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        vision_fused_dim=128,
        tactile_fused_dim=256,
    )
    states = {
        "third_resnet": torch.rand(batch_size, 256),
        "tactile_left_depth_resnet": torch.rand(batch_size, 256),
        "tactile_right_depth_resnet": torch.rand(batch_size, 256),
        "tactile_left_down_depth_resnet": torch.rand(batch_size, 256),
        "tactile_right_down_depth_resnet": torch.rand(batch_size, 256),
        "proprio_obs": torch.rand(batch_size, 18),
        "tactile_valid": torch.randint(0, 2, (batch_size, 1), dtype=torch.float32),
    }
    mean, log_std, _ = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean()
    loss.backward()
    grad_norm = model.vision_proj[1].weight.grad.norm().item()
    print(f"[gate_alpha_smoke_test] mean shape={tuple(mean.shape)} grad_norm={grad_norm:.6f}")
