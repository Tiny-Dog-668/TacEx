"""VT downsample policy with a router-gated MLP expert block."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import math

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


class OccludedGraspingVTRouterMAEPolicy(GaussianMixin, Model):
    """Visual-tactile policy with four MLP experts and router auxiliary losses."""

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
        expert_dim: int = 256,
        vision_downsample_dim: int | None = None,
        expert_hidden_dim: int = 256,
        num_experts: int = 4,
        router_hard: bool = True,
        load_loss_weight: float = 0.01,
        entropy_loss_weight: float = 0.001,
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
        if mlp_layers is None:
            mlp_layers = (512, 256, 128, 64)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        self.expert_dim = int(expert_dim)
        self.vision_downsample_dim = None if vision_downsample_dim is None else int(vision_downsample_dim)
        self.num_experts = int(num_experts)
        self.router_hard = bool(router_hard)
        self.load_loss_weight = float(load_loss_weight)
        self.entropy_loss_weight = float(entropy_loss_weight)

        if self.num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")
        if len(self.tactile_keys) != 4:
            raise ValueError(f"Expected 4 tactile keys, got {self.tactile_keys}")

        used_keys = (self.vision_key, self.proprio_key, *self.tactile_keys)
        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in used_keys}
        self._obs_dims = {key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes}

        act_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        self.vision_dim = int(self._obs_dims[self.vision_key])
        self.proprio_dim = int(self._obs_dims[self.proprio_key])
        tactile_total_dim = sum(int(self._obs_dims[key]) for key in self.tactile_keys)

        if self.vision_downsample_dim is not None and self.vision_downsample_dim > 0:
            self.vision_proj = nn.Sequential(
                nn.LayerNorm(self.vision_dim),
                nn.Linear(self.vision_dim, self.vision_downsample_dim),
                act_cls(),
                nn.Linear(self.vision_downsample_dim, self.expert_dim),
                act_cls(),
            )
        else:
            self.vision_proj = nn.Sequential(
                nn.LayerNorm(self.vision_dim),
                nn.Linear(self.vision_dim, self.expert_dim),
                act_cls(),
            )
        self.tactile_proj = nn.Sequential(
            nn.LayerNorm(tactile_total_dim),
            nn.Linear(tactile_total_dim, self.expert_dim),
            act_cls(),
        )
        self.router = nn.Linear(self.expert_dim * 2, self.num_experts)
        self.expert_input = nn.Sequential(
            nn.Linear(self.expert_dim * 2, self.expert_dim),
            act_cls(),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.expert_dim),
                    nn.Linear(self.expert_dim, int(expert_hidden_dim)),
                    act_cls(),
                    nn.Linear(int(expert_hidden_dim), self.expert_dim),
                    act_cls(),
                )
                for _ in range(self.num_experts)
            ]
        )

        policy_input_dim = self.expert_dim + self.proprio_dim
        mlp: list[nn.Module] = []
        last_dim = policy_input_dim
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(act_cls())
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

    def _get_obs_shape(self, obs_space, key: str) -> Tuple[int, ...]:
        if hasattr(obs_space, "spaces") and key in obs_space.spaces:
            shape = getattr(obs_space.spaces[key], "shape", None)
        else:
            spec = obs_space[key]
            shape = getattr(spec, "shape", spec)
        if shape is None:
            raise ValueError(f"Missing shape for observation key '{key}'")
        if isinstance(shape, int):
            return (int(shape),)
        return tuple(int(s) for s in shape)

    def _flatten_obs(self, obs: Dict[str, Any], key: str, batch_size: int) -> torch.Tensor:
        value = obs.get(key)
        if value is None:
            return torch.zeros((batch_size, self._obs_dims[key]), device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(device=self.device, dtype=torch.float32)
        return value.reshape(batch_size, -1)

    def _router_weights(self, probs: torch.Tensor) -> torch.Tensor:
        if not self.router_hard:
            return probs
        selected = probs.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(probs).scatter_(dim=-1, index=selected, value=1.0)
        return hard - probs.detach() + probs

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

        vision = self.vision_proj(self._flatten_obs(obs, self.vision_key, batch_size))
        tactile = self.tactile_proj(
            torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.tactile_keys], dim=-1)
        )
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)

        router_input = torch.cat([vision, tactile], dim=-1)
        router_logits = self.router(router_input)
        router_probs = torch.softmax(router_logits, dim=-1)
        router_weights = self._router_weights(router_probs)

        expert_base = self.expert_input(router_input)
        expert_outputs = torch.stack([expert(expert_base) for expert in self.experts], dim=1)
        moe_feature = torch.sum(router_weights.unsqueeze(-1) * expert_outputs, dim=1)

        mu = self.mu(self.mlp(torch.cat([moe_feature, proprio], dim=-1)))

        mean_prob = router_probs.mean(dim=0)
        load_loss = self.num_experts * torch.sum(mean_prob * mean_prob) - 1.0
        entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1)
        entropy_loss = entropy.mean() / math.log(self.num_experts)
        aux_loss = self.load_loss_weight * load_loss + self.entropy_loss_weight * entropy_loss

        selected = router_probs.argmax(dim=-1).to(torch.float32)
        return mu, self.log_std_parameter, {
            "aux_loss": aux_loss,
            "aux_reliability_loss": load_loss.detach(),
            "aux_probe_loss": entropy_loss.detach(),
            "router_load_loss": load_loss.detach(),
            "router_entropy_loss": entropy_loss.detach(),
            "router_probs": router_probs.detach(),
            "router_selected": selected.detach(),
            "alpha": router_probs.max(dim=-1).values.detach(),
            "g_probe": selected.detach(),
        }


def quick_router_mae_policy_smoke_test(batch_size: int = 8, device: str = "cpu") -> None:
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
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)
    model = OccludedGraspingVTRouterMAEPolicy(obs_space, action_space, device=device)
    states = {
        "third_resnet": torch.rand(batch_size, 256),
        "tactile_left_depth_resnet": torch.rand(batch_size, 256),
        "tactile_right_depth_resnet": torch.rand(batch_size, 256),
        "tactile_left_down_depth_resnet": torch.rand(batch_size, 256),
        "tactile_right_down_depth_resnet": torch.rand(batch_size, 256),
        "proprio_obs": torch.rand(batch_size, 18),
    }
    mean, log_std, outputs = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean() + outputs["aux_loss"]
    loss.backward()
    grad_norm = model.router.weight.grad.norm().item()
    print(
        f"[router_mae_smoke_test] mean shape={tuple(mean.shape)} "
        f"load={float(outputs['router_load_loss']):.6f} "
        f"entropy={float(outputs['router_entropy_loss']):.6f} grad_norm={grad_norm:.6f}"
    )
