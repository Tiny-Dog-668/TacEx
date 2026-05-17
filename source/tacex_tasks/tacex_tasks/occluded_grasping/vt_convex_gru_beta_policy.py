"""Convex alpha/beta VT policy with per-tactile GRU temporal encoders."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha_beta
from .vt_alpha_gru_policy import OccludedGraspingVTAlphaGRUPolicy


class OccludedGraspingVTConvexGRUBetaPolicy(OccludedGraspingVTAlphaGRUPolicy):
    """Convex-GRU policy with separate down/inner tactile gates."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        tactile_inner_keys: Iterable[str] | None = None,
        tactile_down_keys: Iterable[str] | None = None,
        fused_dim: int = 512,
        gate_hidden_dim: int = 64,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        **kwargs: Any,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            fused_dim=fused_dim,
            gate_hidden_dim=gate_hidden_dim,
            mlp_layers=mlp_layers,
            mlp_activation=mlp_activation,
            **kwargs,
        )

        if tactile_inner_keys is None:
            tactile_inner_keys = tuple(key for key in self.tactile_keys if "down" not in str(key))
        if tactile_down_keys is None:
            tactile_down_keys = tuple(key for key in self.tactile_keys if "down" in str(key))

        self.tactile_inner_keys = tuple(key for key in tactile_inner_keys if key in self.tactile_keys)
        self.tactile_down_keys = tuple(key for key in tactile_down_keys if key in self.tactile_keys)
        if not self.tactile_inner_keys:
            raise ValueError("No inner tactile observation keys found for convex-GRU-beta policy")
        if not self.tactile_down_keys:
            raise ValueError("No down tactile observation keys found for convex-GRU-beta policy")

        if mlp_layers is None:
            mlp_layers = (256, 128, 64)
        activation_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        proprio_dim = int(self._obs_dims[self.proprio_key])
        vector_dim = sum(int(self._obs_dims[key]) for key in self.vector_keys)

        self.tactile_inner_proj = nn.Sequential(
            nn.LayerNorm(self.tactile_latent_dim * len(self.tactile_inner_keys)),
            nn.Linear(self.tactile_latent_dim * len(self.tactile_inner_keys), self.fused_dim),
            activation_cls(),
        )
        self.tactile_down_proj = nn.Sequential(
            nn.LayerNorm(self.tactile_latent_dim * len(self.tactile_down_keys)),
            nn.Linear(self.tactile_latent_dim * len(self.tactile_down_keys), self.fused_dim),
            activation_cls(),
        )
        self.alpha_beta_gate = nn.Sequential(
            nn.Linear(proprio_dim, int(gate_hidden_dim)),
            activation_cls(),
            nn.Linear(int(gate_hidden_dim), 3),
        )

        mlp_input_dim = self.fused_dim + vector_dim
        mlp: list[nn.Module] = []
        last_dim = mlp_input_dim
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(activation_cls())
            last_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)

    def _encode_tactile_group(self, obs: Dict[str, Any], batch_size: int, keys: tuple[str, ...]) -> torch.Tensor:
        latent_chunks = []
        for key in keys:
            feature_dim = self._sensor_feat_dims[key]
            flat = self._flatten_obs(obs, key, batch_size)
            if flat.shape[-1] == feature_dim:
                sequence = flat.unsqueeze(1).expand(-1, self.tactile_time_window, -1)
            else:
                sequence = flat.reshape(batch_size, self.tactile_time_window, feature_dim)
            _, h_n = self.tactile_grus[key](sequence)
            latent_chunks.append(self.tactile_heads[key](h_n[-1]))
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

        vision = self.vision_proj(self._flatten_obs(obs, self.vision_key, batch_size))
        tactile_down = self.tactile_down_proj(self._encode_tactile_group(obs, batch_size, self.tactile_down_keys))
        tactile_inner = self.tactile_inner_proj(self._encode_tactile_group(obs, batch_size, self.tactile_inner_keys))
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)

        weights = torch.softmax(self.alpha_beta_gate(proprio), dim=-1)
        alpha = weights[:, 1:2]
        beta = weights[:, 2:3]
        visual_weight = 1.0 - alpha - beta
        set_latest_alpha_beta(alpha, beta)

        fused = (visual_weight * vision) + (alpha * tactile_down) + (beta * tactile_inner)
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([fused, *vector_features], dim=-1) if vector_features else fused

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}


def quick_convex_gru_beta_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
    """Minimal forward/backward check without Isaac Sim."""
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "third_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(2560,), dtype=float),
            "tactile_right_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(2560,), dtype=float),
            "tactile_left_down_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(2560,), dtype=float),
            "tactile_right_down_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(2560,), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)

    model = OccludedGraspingVTConvexGRUBetaPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device=device,
    )
    states = {
        "third_resnet": torch.rand(batch_size, 256),
        "tactile_left_depth_resnet": torch.rand(batch_size, 2560),
        "tactile_right_depth_resnet": torch.rand(batch_size, 2560),
        "tactile_left_down_depth_resnet": torch.rand(batch_size, 2560),
        "tactile_right_down_depth_resnet": torch.rand(batch_size, 2560),
        "proprio_obs": torch.rand(batch_size, 18),
    }
    mean, log_std, _ = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean()
    loss.backward()
    grad_norm = model.vision_proj[1].weight.grad.norm().item()
    print(f"[convex_gru_beta_smoke_test] mean shape={tuple(mean.shape)} grad_norm={grad_norm:.6f}")
