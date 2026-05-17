"""Vision-base policy with tactile gated residual correction for occluded grasping."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha

_LEGACY_TACTILE_KEY_ALIASES = {
    "tactile_left_depth_resnet": "tactile_left_rgb",
    "tactile_right_depth_resnet": "tactile_right_rgb",
    "tactile_left_down_depth_resnet": "tactile_left_down_depth",
    "tactile_right_down_depth_resnet": "tactile_right_down_depth",
}


class OccludedGraspingVTResidualPolicy(GaussianMixin, Model):
    """Visual base action plus tactile gated residual correction."""

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
        proprio_key: str = "proprio_obs",
        tactile_contact_bits_key: str = "tactile_contact_bits",
        tactile_valid_key: str = "tactile_valid",
        visual_hidden_layers: Iterable[int] | None = None,
        tactile_token_dim: int = 128,
        residual_hidden_dim: int = 256,
        gate_per_dim: bool = True,
        residual_action_scales: Iterable[float] | None = None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        if tactile_keys is None:
            tactile_keys = (
                "tactile_left_rgb",
                "tactile_right_rgb",
                "tactile_left_down_depth",
                "tactile_right_down_depth",
            )
        if visual_hidden_layers is None:
            visual_hidden_layers = (512, 256, 128, 64)
        if residual_action_scales is None:
            residual_action_scales = (0.04, 0.04, 0.04, 0.20, 0.10)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.tactile_contact_bits_key = str(tactile_contact_bits_key)
        self.tactile_valid_key = str(tactile_valid_key)
        resolved_tactile_keys = []
        for key in tactile_keys:
            resolved_key = self._resolve_obs_key(observation_space, key)
            if resolved_key is not None and resolved_key not in resolved_tactile_keys:
                resolved_tactile_keys.append(resolved_key)
        self.tactile_keys = tuple(resolved_tactile_keys)

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")
        if not self.tactile_keys:
            raise ValueError("No tactile observation keys found for residual policy")
        if not self._has_obs_key(observation_space, self.tactile_contact_bits_key):
            raise ValueError(f"Missing tactile contact bits key '{self.tactile_contact_bits_key}'")
        self._has_tactile_valid = self._has_obs_key(observation_space, self.tactile_valid_key)

        tracked_keys = [self.vision_key, self.proprio_key, self.tactile_contact_bits_key, *self.tactile_keys]
        if self._has_tactile_valid:
            tracked_keys.append(self.tactile_valid_key)
        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in tracked_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }

        self.vision_dim = self._obs_dims[self.vision_key]
        self.proprio_dim = self._obs_dims[self.proprio_key]
        self.contact_dim = self._obs_dims[self.tactile_contact_bits_key]
        self.tactile_dim = self._obs_dims[self.tactile_keys[0]]
        self.num_tactile = len(self.tactile_keys)
        self.gate_per_dim = bool(gate_per_dim)

        visual_layers = []
        last_dim = self.vision_dim + self.proprio_dim
        for hidden_dim in visual_hidden_layers:
            visual_layers.append(nn.Linear(last_dim, int(hidden_dim)))
            visual_layers.append(nn.ELU())
            last_dim = int(hidden_dim)
        self.visual_base_mlp = nn.Sequential(*visual_layers) if visual_layers else nn.Identity()
        self.visual_action_head = nn.Linear(last_dim, self.num_actions)

        self.tactile_token_proj = nn.Sequential(
            nn.LayerNorm(self.tactile_dim),
            nn.Linear(self.tactile_dim, int(tactile_token_dim)),
            nn.ELU(),
        )
        self.sensor_id_embedding = nn.Embedding(self.num_tactile, int(tactile_token_dim))

        token_plus_contact_dim = int(tactile_token_dim) + 1
        tactile_agg_dim = self.num_tactile * token_plus_contact_dim
        residual_input_dim = tactile_agg_dim + self.contact_dim + self.proprio_dim + self.num_actions

        self.residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, int(residual_hidden_dim)),
            nn.ELU(),
            nn.Linear(int(residual_hidden_dim), int(residual_hidden_dim // 2)),
            nn.ELU(),
            nn.Linear(int(residual_hidden_dim // 2), self.num_actions),
        )

        residual_scales = torch.tensor(list(residual_action_scales), dtype=torch.float32)
        if residual_scales.numel() != self.num_actions:
            raise ValueError(
                f"Expected {self.num_actions} residual scales, got {residual_scales.numel()}"
            )
        self.register_buffer("residual_action_scales", residual_scales.view(1, -1), persistent=False)

        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(initial_log_std)),
            requires_grad=not bool(fixed_log_std),
        )

    def _has_obs_key(self, obs_space, key: str) -> bool:
        if hasattr(obs_space, "spaces"):
            return key in obs_space.spaces
        return isinstance(obs_space, dict) and key in obs_space

    def _resolve_obs_key(self, obs_space, key: str) -> str | None:
        if self._has_obs_key(obs_space, key):
            return key
        alias_key = _LEGACY_TACTILE_KEY_ALIASES.get(key)
        if alias_key is not None and self._has_obs_key(obs_space, alias_key):
            return alias_key
        return None

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

    def _collect_batch_size(self, obs: Dict[str, Any]) -> int:
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        raise ValueError("No tensor observations found in inputs['states']")

    def compute(self, inputs: Dict[str, Any], role: str = ""):
        states = inputs.get("states")
        if isinstance(states, dict):
            obs = states
        else:
            obs = unflatten_tensorized_space(self.observation_space, states)

        batch_size = self._collect_batch_size(obs)
        vision = self._flatten_obs(obs, self.vision_key, batch_size)
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        contact_bits = self._flatten_obs(obs, self.tactile_contact_bits_key, batch_size).clamp(0.0, 1.0)
        tactile_valid = (
            self._flatten_obs(obs, self.tactile_valid_key, batch_size).clamp(0.0, 1.0)
            if self._has_tactile_valid
            else contact_bits.max(dim=-1, keepdim=True).values
        )

        visual_input = torch.cat([vision, proprio], dim=-1)
        visual_hidden = self.visual_base_mlp(visual_input)
        a_vis = self.visual_action_head(visual_hidden)

        tactile_tokens = []
        sensor_ids = torch.arange(self.num_tactile, device=self.device, dtype=torch.long)
        for sensor_idx, key in enumerate(self.tactile_keys):
            tactile_feat = self._flatten_obs(obs, key, batch_size)
            tactile_token = self.tactile_token_proj(tactile_feat)
            tactile_token = tactile_token + self.sensor_id_embedding(sensor_ids[sensor_idx]).unsqueeze(0)
            tactile_token = torch.cat([tactile_token, contact_bits[:, sensor_idx : sensor_idx + 1]], dim=-1)
            tactile_tokens.append(tactile_token)
        tactile_agg = torch.cat(tactile_tokens, dim=-1)

        residual_input = torch.cat([tactile_agg, contact_bits, proprio, a_vis], dim=-1)
        delta_a_tac = torch.tanh(self.residual_head(residual_input)) * self.residual_action_scales

        gate = tactile_valid
        if not self.gate_per_dim:
            gate = gate.expand(-1, self.num_actions)
        set_latest_alpha(gate.mean(dim=-1, keepdim=True))

        mu = a_vis + gate * delta_a_tac
        return mu, self.log_std_parameter, {
            "a_vis": a_vis,
            "delta_a_tac": delta_a_tac,
            "gate": gate,
        }


def quick_residual_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
    """Minimal forward/backward check without Isaac Sim."""
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "third_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_rgb": gym.spaces.Box(-1.0, 1.0, shape=(128,), dtype=float),
            "tactile_right_rgb": gym.spaces.Box(-1.0, 1.0, shape=(128,), dtype=float),
            "tactile_left_down_depth": gym.spaces.Box(-1.0, 1.0, shape=(128,), dtype=float),
            "tactile_right_down_depth": gym.spaces.Box(-1.0, 1.0, shape=(128,), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
            "tactile_contact_bits": gym.spaces.Box(0.0, 1.0, shape=(4,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)

    model = OccludedGraspingVTResidualPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device=device,
    )
    states = {
        "third_resnet": torch.rand(batch_size, 256),
        "tactile_left_rgb": torch.rand(batch_size, 128),
        "tactile_right_rgb": torch.rand(batch_size, 128),
        "tactile_left_down_depth": torch.rand(batch_size, 128),
        "tactile_right_down_depth": torch.rand(batch_size, 128),
        "proprio_obs": torch.rand(batch_size, 18),
        "tactile_contact_bits": torch.randint(0, 2, (batch_size, 4), dtype=torch.float32),
    }
    mean, log_std, _ = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean()
    loss.backward()
    grad_norm = model.visual_action_head.weight.grad.norm().item()
    print(f"[residual_smoke_test] mean shape={tuple(mean.shape)} grad_norm={grad_norm:.6f}")
