"""VT policy with trainable tactile CNN encoder and image reconstruction decoder."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha


class OccludedGraspingVTCNNReconPolicy(GaussianMixin, Model):
    """Policy-side tactile CNN with decoder reconstruction supervision."""

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
        tactile_raw_keys: Iterable[str] | None = None,
        proprio_key: str = "proprio_obs",
        vector_keys: Iterable[str] | None = None,
        tactile_img_hw: Iterable[int] = (32, 32),
        tactile_feature_dim: int = 256,
        fused_dim: int = 256,
        recon_loss_weight: float = 1.0,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        if tactile_raw_keys is None:
            tactile_raw_keys = (
                "tactile_left_rgb_raw",
                "tactile_right_rgb_raw",
                "tactile_left_down_rgb_raw",
                "tactile_right_down_rgb_raw",
            )
        if vector_keys is None:
            vector_keys = (proprio_key,)
        if mlp_layers is None:
            mlp_layers = (512, 256, 128, 64)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.tactile_raw_keys = tuple(key for key in tactile_raw_keys if self._has_obs_key(observation_space, key))
        self.vector_keys = tuple(key for key in vector_keys if self._has_obs_key(observation_space, key))
        self.tactile_img_h, self.tactile_img_w = tuple(int(v) for v in tactile_img_hw)
        self.tactile_feature_dim = int(tactile_feature_dim)
        self.fused_dim = int(fused_dim)
        self.recon_loss_weight = float(recon_loss_weight)

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")
        if len(self.tactile_raw_keys) != 4:
            raise ValueError(f"Expected 4 raw tactile keys, got {self.tactile_raw_keys}")

        used_keys = (self.vision_key, *self.tactile_raw_keys, *self.vector_keys)
        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in used_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1])
            for key in self._obs_shapes
            if self._obs_shapes[key] is not None
        }

        raw_dim = 3 * self.tactile_img_h * self.tactile_img_w
        for key in self.tactile_raw_keys:
            if int(self._obs_dims[key]) != raw_dim:
                raise ValueError(
                    f"Raw tactile observation '{key}' dim={self._obs_dims[key]} does not match "
                    f"3*{self.tactile_img_h}*{self.tactile_img_w}={raw_dim}"
                )

        act_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        vision_dim = int(self._obs_dims[self.vision_key])
        vector_dim = sum(int(self._obs_dims[key]) for key in self.vector_keys)

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, self.fused_dim),
            act_cls(),
        )

        self.tactile_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.tactile_feature_dim),
            nn.ELU(),
        )
        self.tactile_decoder = nn.Sequential(
            nn.Linear(self.tactile_feature_dim, 128 * 4 * 4),
            nn.ELU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.tactile_proj = nn.Sequential(
            nn.LayerNorm(self.tactile_feature_dim * len(self.tactile_raw_keys)),
            nn.Linear(self.tactile_feature_dim * len(self.tactile_raw_keys), self.fused_dim),
            act_cls(),
        )

        mlp_input_dim = (2 * self.fused_dim) + vector_dim
        mlp: list[nn.Module] = []
        last_dim = mlp_input_dim
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

    def _collect_tactile_images(self, obs: Dict[str, Any], batch_size: int) -> torch.Tensor:
        images = []
        for key in self.tactile_raw_keys:
            flat = self._flatten_obs(obs, key, batch_size)
            images.append(flat.reshape(batch_size, 3, self.tactile_img_h, self.tactile_img_w))
        return torch.stack(images, dim=1)

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
        tactile_images = self._collect_tactile_images(obs, batch_size)
        tactile_batch = tactile_images.reshape(batch_size * len(self.tactile_raw_keys), 3, self.tactile_img_h, self.tactile_img_w)
        tactile_features = self.tactile_encoder(tactile_batch)
        recon_images = self.tactile_decoder(tactile_features)
        if recon_images.shape[-2:] != (self.tactile_img_h, self.tactile_img_w):
            recon_images = F.interpolate(
                recon_images,
                size=(self.tactile_img_h, self.tactile_img_w),
                mode="bilinear",
                align_corners=False,
            )

        tactile_features = tactile_features.reshape(batch_size, len(self.tactile_raw_keys), self.tactile_feature_dim)
        tactile = self.tactile_proj(tactile_features.reshape(batch_size, -1))
        vector_features = [self._flatten_obs(obs, key, batch_size) for key in self.vector_keys]
        set_latest_alpha(None)

        fused = torch.cat([vision, tactile, *vector_features], dim=-1)
        hidden = self.mlp(fused)
        mu = self.mu(hidden)

        recon_target = tactile_images.reshape(batch_size * len(self.tactile_raw_keys), 3, self.tactile_img_h, self.tactile_img_w)
        recon_loss = F.mse_loss(recon_images, recon_target)
        aux_loss = self.recon_loss_weight * recon_loss
        outputs = {
            "aux_loss": aux_loss,
            "aux_recon_loss": recon_loss.detach(),
        }
        return mu, self.log_std_parameter, outputs


def quick_cnn_recon_policy_smoke_test(batch_size: int = 4, device: str = "cpu") -> None:
    """Minimal forward/backward check without Isaac Sim."""
    import gymnasium as gym

    raw_dim = 3 * 32 * 32
    obs_space = gym.spaces.Dict(
        {
            "third_resnet": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=float),
            "tactile_left_rgb_raw": gym.spaces.Box(0.0, 1.0, shape=(raw_dim,), dtype=float),
            "tactile_right_rgb_raw": gym.spaces.Box(0.0, 1.0, shape=(raw_dim,), dtype=float),
            "tactile_left_down_rgb_raw": gym.spaces.Box(0.0, 1.0, shape=(raw_dim,), dtype=float),
            "tactile_right_down_rgb_raw": gym.spaces.Box(0.0, 1.0, shape=(raw_dim,), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float)
    model = OccludedGraspingVTCNNReconPolicy(obs_space, action_space, device=device)
    states = {
        "third_resnet": torch.rand(batch_size, 256),
        "tactile_left_rgb_raw": torch.rand(batch_size, raw_dim),
        "tactile_right_rgb_raw": torch.rand(batch_size, raw_dim),
        "tactile_left_down_rgb_raw": torch.rand(batch_size, raw_dim),
        "tactile_right_down_rgb_raw": torch.rand(batch_size, raw_dim),
        "proprio_obs": torch.rand(batch_size, 18),
    }
    mean, log_std, outputs = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean() + outputs["aux_loss"]
    loss.backward()
    grad_norm = model.tactile_encoder[0].weight.grad.norm().item()
    print(
        f"[cnn_recon_smoke_test] mean shape={tuple(mean.shape)} "
        f"aux_loss={outputs['aux_loss'].item():.6f} encoder_grad_norm={grad_norm:.6f}"
    )
