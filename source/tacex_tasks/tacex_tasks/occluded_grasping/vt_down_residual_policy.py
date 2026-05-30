"""VT policy with hard-gated down-tactile residual corrections for dx/dy."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .policy_aux_heads import set_latest_aux_predictions


class OccludedGraspingVTDownResidualPolicy(GaussianMixin, Model):
    """VT main action plus bottom tactile residual for dx/dy only."""

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
        down_tactile_keys: Iterable[str] | None = None,
        proprio_key: str = "proprio_obs",
        down_contact_ratio_key: str = "down_tactile_contact_ratio",
        down_contact_threshold: float = 0.01,
        tactile_latent_dim: int = 256,
        down_latent_dim: int = 128,
        residual_hidden_dim: int = 128,
        residual_action_scale: float = 0.2,
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
        if down_tactile_keys is None:
            down_tactile_keys = ("tactile_left_down_depth_resnet", "tactile_right_down_depth_resnet")
        if mlp_layers is None:
            mlp_layers = (512, 256, 128, 64)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.down_contact_ratio_key = str(down_contact_ratio_key)
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        self.down_tactile_keys = tuple(key for key in down_tactile_keys if self._has_obs_key(observation_space, key))
        self.down_contact_threshold = float(down_contact_threshold)
        self.tactile_latent_dim = int(tactile_latent_dim)
        self.down_latent_dim = int(down_latent_dim)
        self.residual_action_scale = float(residual_action_scale)

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")
        if not self._has_obs_key(observation_space, self.down_contact_ratio_key):
            raise ValueError(f"Missing down contact ratio key '{self.down_contact_ratio_key}'")
        if len(self.tactile_keys) != 4:
            raise ValueError(f"Expected 4 tactile keys, got {self.tactile_keys}")
        if len(self.down_tactile_keys) != 2:
            raise ValueError(f"Expected 2 down tactile keys, got {self.down_tactile_keys}")

        used_keys = (self.vision_key, self.proprio_key, self.down_contact_ratio_key, *self.tactile_keys)
        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in used_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1])
            for key in self._obs_shapes
            if self._obs_shapes[key] is not None
        }

        act_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        self.vision_dim = int(self._obs_dims[self.vision_key])
        self.proprio_dim = int(self._obs_dims[self.proprio_key])
        tactile_total_dim = sum(int(self._obs_dims[key]) for key in self.tactile_keys)
        down_total_dim = sum(int(self._obs_dims[key]) for key in self.down_tactile_keys)

        self.tactile_proj = nn.Sequential(
            nn.LayerNorm(tactile_total_dim),
            nn.Linear(tactile_total_dim, self.tactile_latent_dim),
            act_cls(),
        )

        mlp_input_dim = self.vision_dim + self.tactile_latent_dim + self.proprio_dim
        mlp: list[nn.Module] = []
        last_dim = mlp_input_dim
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(act_cls())
            last_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)

        self.down_proj = nn.Sequential(
            nn.LayerNorm(down_total_dim),
            nn.Linear(down_total_dim, self.down_latent_dim),
            act_cls(),
        )
        residual_input_dim = self.down_latent_dim + 2 + self.proprio_dim
        self.down_residual = nn.Sequential(
            nn.Linear(residual_input_dim, int(residual_hidden_dim)),
            act_cls(),
            nn.Linear(int(residual_hidden_dim), max(1, int(residual_hidden_dim // 2))),
            act_cls(),
            nn.Linear(max(1, int(residual_hidden_dim // 2)), 2),
            nn.Tanh(),
        )

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
                batch_size = int(value.shape[0])
                break
        if batch_size is None:
            raise ValueError("No tensor observations found in inputs['states']")

        vision = self._flatten_obs(obs, self.vision_key, batch_size)
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        tactile = self.tactile_proj(
            torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.tactile_keys], dim=-1)
        )
        main_input = torch.cat([vision, tactile, proprio], dim=-1)
        mu = self.mu(self.mlp(main_input))

        down_contact_ratio = self._flatten_obs(obs, self.down_contact_ratio_key, batch_size)[:, :2].clamp(0.0, 1.0)
        hard_gate = (down_contact_ratio.max(dim=-1, keepdim=True).values > self.down_contact_threshold).to(torch.float32)
        down_feat = self.down_proj(
            torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.down_tactile_keys], dim=-1)
        )
        residual_input = torch.cat([down_feat, down_contact_ratio, proprio], dim=-1)
        residual_xy = self.down_residual(residual_input) * self.residual_action_scale

        mu = mu.clone()
        mu[:, 0:2] = mu[:, 0:2] + hard_gate * residual_xy

        set_latest_alpha(hard_gate)
        dummy_stage = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        set_latest_aux_predictions(
            reliability_pred=hard_gate,
            stage_logits=dummy_stage,
            w_probe=hard_gate,
            m_probe=hard_gate,
            m_grasp=torch.zeros_like(hard_gate),
        )
        return mu, self.log_std_parameter, {
            "down_residual_gate": hard_gate,
            "down_residual_xy": residual_xy,
            "down_contact_ratio": down_contact_ratio,
        }


def quick_down_residual_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
    """Minimal forward/backward check without Isaac Sim."""
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "third_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_right_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_down_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_right_down_depth_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "down_tactile_contact_ratio": gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)
    model = OccludedGraspingVTDownResidualPolicy(obs_space, action_space, device=device)
    states = {
        "third_resnet": torch.rand(batch_size, 256),
        "tactile_left_depth_resnet": torch.rand(batch_size, 256),
        "tactile_right_depth_resnet": torch.rand(batch_size, 256),
        "tactile_left_down_depth_resnet": torch.rand(batch_size, 256),
        "tactile_right_down_depth_resnet": torch.rand(batch_size, 256),
        "down_tactile_contact_ratio": torch.tensor([[0.0, 0.0], [0.02, 0.0], [0.0, 0.03], [0.04, 0.05]]),
        "proprio_obs": torch.rand(batch_size, 18),
    }
    mean, log_std, outputs = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean()
    loss.backward()
    grad_norm = model.down_residual[0].weight.grad.norm().item()
    print(
        f"[down_residual_smoke_test] mean shape={tuple(mean.shape)} "
        f"gate={outputs['down_residual_gate'].view(-1).tolist()} grad_norm={grad_norm:.6f}"
    )
