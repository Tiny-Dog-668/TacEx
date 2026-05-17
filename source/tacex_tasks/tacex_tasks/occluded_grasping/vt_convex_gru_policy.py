"""Convex-combination VT policy with per-tactile GRU temporal encoders."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_alpha_gru_policy import OccludedGraspingVTAlphaGRUPolicy


class OccludedGraspingVTConvexGRUPolicy(OccludedGraspingVTAlphaGRUPolicy):
    """Alpha-GRU tactile encoder with 512-d convex visual/tactile fusion."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
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

        if mlp_layers is None:
            mlp_layers = (256, 128, 64)
        activation_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        vector_dim = sum(int(self._obs_dims[key]) for key in self.vector_keys)
        mlp_input_dim = self.fused_dim + vector_dim

        mlp: list[nn.Module] = []
        last_dim = mlp_input_dim
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(activation_cls())
            last_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)

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
        tactile = self.tactile_proj(self._encode_tactile(obs, batch_size))
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        alpha = self.alpha_gate(proprio)
        set_latest_alpha(alpha)

        fused = (1.0 - alpha) * vision + alpha * tactile
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([fused, *vector_features], dim=-1) if vector_features else fused

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}


def quick_convex_gru_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
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

    model = OccludedGraspingVTConvexGRUPolicy(
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
    print(f"[convex_gru_smoke_test] mean shape={tuple(mean.shape)} grad_norm={grad_norm:.6f}")
