"""VT policy with per-tactile GRU temporal encoders and no alpha gate."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_alpha_gru_policy import OccludedGraspingVTAlphaGRUPolicy


class OccludedGraspingVTGRUPolicy(OccludedGraspingVTAlphaGRUPolicy):
    """GRU tactile encoder with direct VT concatenation fusion."""

    def __init__(self, observation_space, action_space, device, **kwargs: Any):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            **kwargs,
        )
        self.alpha_gate = nn.Identity()

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
        set_latest_alpha(None)

        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision, tactile, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}


def quick_gru_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
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

    model = OccludedGraspingVTGRUPolicy(
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
    print(f"[gru_smoke_test] mean shape={tuple(mean.shape)} grad_norm={grad_norm:.6f}")
