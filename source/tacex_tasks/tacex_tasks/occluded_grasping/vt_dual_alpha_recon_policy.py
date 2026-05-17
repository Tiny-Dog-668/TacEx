"""Dual-alpha VT policy with supervised task heads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha


class OccludedGraspingVTDualAlphaReconPolicy(GaussianMixin, Model):
    """Vision + dual tactile branches with separate alpha gates.

    Fusion layout:
    [vision_256, alpha_down * down_128, alpha_inner * inner_128]
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
        tactile_inner_keys: Iterable[str] | None = None,
        tactile_down_keys: Iterable[str] | None = None,
        proprio_key: str = "proprio_obs",
        aux_occlusion_gt_key: str = "aux_occlusion_gt",
        aux_down_contact_gt_key: str = "aux_down_contact_gt",
        aux_object_side_gt_key: str = "aux_object_side_gt",
        aux_inner_contact_gt_key: str = "aux_inner_contact_gt",
        vision_latent_dim: int = 256,
        tactile_latent_dim: int = 128,
        task_hidden_dim: int = 256,
        alpha_hidden_dim: int = 128,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        aux_occlusion_weight: float = 1.0,
        aux_down_contact_weight: float = 1.0,
        aux_object_side_weight: float = 1.0,
        aux_inner_contact_weight: float = 1.0,
        side_loss_contact_threshold: float = 0.05,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        if tactile_inner_keys is None:
            tactile_inner_keys = ("tactile_left_rgb_resnet", "tactile_right_rgb_resnet")
        if tactile_down_keys is None:
            tactile_down_keys = ("tactile_left_down_rgb_resnet", "tactile_right_down_rgb_resnet")
        if mlp_layers is None:
            mlp_layers = (256, 128, 64)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.aux_occlusion_gt_key = str(aux_occlusion_gt_key)
        self.aux_down_contact_gt_key = str(aux_down_contact_gt_key)
        self.aux_object_side_gt_key = str(aux_object_side_gt_key)
        self.aux_inner_contact_gt_key = str(aux_inner_contact_gt_key)

        self.tactile_inner_keys = tuple(key for key in tactile_inner_keys if self._has_obs_key(observation_space, key))
        self.tactile_down_keys = tuple(key for key in tactile_down_keys if self._has_obs_key(observation_space, key))
        self.tactile_all_keys = self.tactile_inner_keys + self.tactile_down_keys

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")
        if len(self.tactile_inner_keys) != 2:
            raise ValueError(f"Expected 2 inner tactile keys, got {self.tactile_inner_keys}")
        if len(self.tactile_down_keys) != 2:
            raise ValueError(f"Expected 2 down tactile keys, got {self.tactile_down_keys}")

        tracked_keys = [self.vision_key, self.proprio_key, *self.tactile_all_keys]
        for key in (
            self.aux_occlusion_gt_key,
            self.aux_down_contact_gt_key,
            self.aux_object_side_gt_key,
            self.aux_inner_contact_gt_key,
        ):
            if self._has_obs_key(observation_space, key):
                tracked_keys.append(key)

        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in tracked_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }

        self.vision_dim = int(self._obs_dims[self.vision_key])
        self.proprio_dim = int(self._obs_dims[self.proprio_key])
        self.inner_dim = sum(int(self._obs_dims[key]) for key in self.tactile_inner_keys)
        self.down_dim = sum(int(self._obs_dims[key]) for key in self.tactile_down_keys)

        self.vision_latent_dim = int(vision_latent_dim)
        self.tactile_latent_dim = int(tactile_latent_dim)

        act_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, self.vision_latent_dim),
            act_cls(),
        )
        self.down_proj = nn.Sequential(
            nn.LayerNorm(self.down_dim),
            nn.Linear(self.down_dim, self.tactile_latent_dim),
            act_cls(),
        )
        self.inner_proj = nn.Sequential(
            nn.LayerNorm(self.inner_dim),
            nn.Linear(self.inner_dim, self.tactile_latent_dim),
            act_cls(),
        )

        task_hidden_dim = int(task_hidden_dim)
        self.occlusion_head = nn.Sequential(
            nn.Linear(self.vision_dim, task_hidden_dim),
            act_cls(),
            nn.Linear(task_hidden_dim, 1),
        )
        self.down_contact_head = nn.Sequential(
            nn.Linear(self.down_dim, task_hidden_dim),
            act_cls(),
            nn.Linear(task_hidden_dim, 2),
        )
        self.object_side_head = nn.Sequential(
            nn.Linear(self.down_dim, task_hidden_dim),
            act_cls(),
            nn.Linear(task_hidden_dim, 3),
        )
        self.inner_contact_head = nn.Sequential(
            nn.Linear(self.inner_dim, task_hidden_dim),
            act_cls(),
            nn.Linear(task_hidden_dim, 3),
        )

        alpha_down_in_dim = 1 + 2 + 3 + self.proprio_dim
        alpha_inner_in_dim = 1 + 3 + self.proprio_dim
        self.alpha_down_gate = nn.Sequential(
            nn.Linear(alpha_down_in_dim, int(alpha_hidden_dim)),
            act_cls(),
            nn.Linear(int(alpha_hidden_dim), 1),
            nn.Sigmoid(),
        )
        self.alpha_inner_gate = nn.Sequential(
            nn.Linear(alpha_inner_in_dim, int(alpha_hidden_dim)),
            act_cls(),
            nn.Linear(int(alpha_hidden_dim), 1),
            nn.Sigmoid(),
        )

        mlp_input_dim = self.vision_latent_dim + self.tactile_latent_dim + self.tactile_latent_dim
        mlp: list[nn.Module] = []
        last_dim = mlp_input_dim
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            mlp.append(act_cls())
            last_dim = int(hidden_dim)
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)

        self.aux_occlusion_weight = float(aux_occlusion_weight)
        self.aux_down_contact_weight = float(aux_down_contact_weight)
        self.aux_object_side_weight = float(aux_object_side_weight)
        self.aux_inner_contact_weight = float(aux_inner_contact_weight)
        self.side_loss_contact_threshold = float(side_loss_contact_threshold)

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
            return torch.zeros((batch_size, int(self._obs_dims.get(key, 0))), device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(device=self.device, dtype=torch.float32)
        return value.reshape(batch_size, -1)

    def _build_aux_losses(
        self,
        obs: Dict[str, Any],
        batch_size: int,
        occlusion_pred: torch.Tensor,
        down_contact_pred: torch.Tensor,
        object_side_logits: torch.Tensor,
        inner_contact_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = occlusion_pred.new_zeros(())

        occlusion_loss = zero
        if self._has_obs_key(self.observation_space, self.aux_occlusion_gt_key):
            gt = self._flatten_obs(obs, self.aux_occlusion_gt_key, batch_size)[:, :1].clamp(0.0, 1.0)
            occlusion_loss = F.mse_loss(occlusion_pred, gt)

        down_loss = zero
        side_loss = zero
        down_gt = None
        if self._has_obs_key(self.observation_space, self.aux_down_contact_gt_key):
            down_gt = self._flatten_obs(obs, self.aux_down_contact_gt_key, batch_size)[:, :2].clamp(0.0, 1.0)
            down_loss = F.mse_loss(down_contact_pred, down_gt)
        if self._has_obs_key(self.observation_space, self.aux_object_side_gt_key):
            gt = self._flatten_obs(obs, self.aux_object_side_gt_key, batch_size)
            if gt.shape[1] >= 3:
                side_index = torch.argmax(gt[:, :3], dim=-1).to(dtype=torch.long)
                if down_gt is not None:
                    contact_valid = down_gt.max(dim=-1).values > self.side_loss_contact_threshold
                    if contact_valid.any():
                        side_loss = F.cross_entropy(object_side_logits[contact_valid], side_index[contact_valid])
                    else:
                        side_loss = zero
                else:
                    side_loss = F.cross_entropy(object_side_logits, side_index)
        probe_loss = down_loss + side_loss

        inner_loss = zero
        if self._has_obs_key(self.observation_space, self.aux_inner_contact_gt_key):
            gt = self._flatten_obs(obs, self.aux_inner_contact_gt_key, batch_size)[:, :3].clamp(0.0, 1.0)
            inner_loss = F.mse_loss(inner_contact_pred, gt)

        aux_loss = (
            self.aux_occlusion_weight * occlusion_loss
            + self.aux_down_contact_weight * down_loss
            + self.aux_object_side_weight * side_loss
            + self.aux_inner_contact_weight * inner_loss
        )
        return aux_loss, occlusion_loss, probe_loss, inner_loss

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

        vision_raw = self._flatten_obs(obs, self.vision_key, batch_size)
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        inner_raw = torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.tactile_inner_keys], dim=-1)
        down_raw = torch.cat([self._flatten_obs(obs, key, batch_size) for key in self.tactile_down_keys], dim=-1)

        occlusion_pred = torch.sigmoid(self.occlusion_head(vision_raw))
        down_contact_pred = torch.sigmoid(self.down_contact_head(down_raw))
        object_side_logits = self.object_side_head(down_raw)
        object_side_prob = torch.softmax(object_side_logits, dim=-1)
        inner_contact_pred = torch.sigmoid(self.inner_contact_head(inner_raw))

        alpha_down_in = torch.cat([occlusion_pred, down_contact_pred, object_side_prob, proprio], dim=-1)
        alpha_inner_in = torch.cat([occlusion_pred, inner_contact_pred, proprio], dim=-1)
        alpha_down = self.alpha_down_gate(alpha_down_in)
        alpha_inner = self.alpha_inner_gate(alpha_inner_in)
        set_latest_alpha(0.5 * (alpha_down + alpha_inner))

        vision_feat = self.vision_proj(vision_raw)
        down_feat = self.down_proj(down_raw)
        inner_feat = self.inner_proj(inner_raw)
        fused = torch.cat([vision_feat, alpha_down * down_feat, alpha_inner * inner_feat], dim=-1)

        hidden = self.mlp(fused)
        mu = self.mu(hidden)

        aux_loss, occlusion_loss, probe_loss, inner_loss = self._build_aux_losses(
            obs=obs,
            batch_size=batch_size,
            occlusion_pred=occlusion_pred,
            down_contact_pred=down_contact_pred,
            object_side_logits=object_side_logits,
            inner_contact_pred=inner_contact_pred,
        )

        outputs = {
            "aux_loss": aux_loss,
            "aux_reliability_loss": occlusion_loss,
            "aux_probe_loss": probe_loss,
            "aux_grasp_loss": inner_loss,
            "g_probe": alpha_down.mean(),
            "g_grasp": alpha_inner.mean(),
            "alpha": (0.5 * (alpha_down + alpha_inner)).mean(),
        }
        return mu, self.log_std_parameter, outputs
