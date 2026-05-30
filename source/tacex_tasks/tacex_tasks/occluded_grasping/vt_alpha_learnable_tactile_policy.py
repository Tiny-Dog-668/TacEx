"""Alpha-gated tactile replacement policy for occluded grasping."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_alpha_policy import OccludedGraspingVTAlphaPolicy


class OccludedGraspingVTAlphaLearnableTactilePolicy(OccludedGraspingVTAlphaPolicy):
    """Keep vision unweighted and gate tactile against a learnable fallback vector.

    The fusion is:
        tactile_mixed = alpha * tactile + (1 - alpha) * learned_tactile
        fused = concat(vision, tactile_mixed, proprio)

    Alpha is still predicted from proprioception only.
    """

    def __init__(
        self,
        *args,
        fallback_init: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.learned_tactile_fallback = nn.Parameter(
            torch.full((1, self.fused_dim), float(fallback_init), device=self.device, dtype=torch.float32)
        )

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
        alpha = self.alpha_gate(proprio)
        set_latest_alpha(alpha)

        fallback = self.learned_tactile_fallback.expand(batch_size, -1)
        tactile_mixed = alpha * tactile + (1.0 - alpha) * fallback
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        fused = torch.cat([vision, tactile_mixed, *vector_features], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)
        return mu, self.log_std_parameter, {}
