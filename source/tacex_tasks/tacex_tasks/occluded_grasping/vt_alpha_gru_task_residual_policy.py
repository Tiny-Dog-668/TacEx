"""Alpha-GRU VT policy with visual task heads and gated tactile residual corrections."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .policy_aux_heads import set_latest_aux_predictions


class OccludedGraspingVTAlphaGRUTaskResidualPolicy(GaussianMixin, Model):
    """Alpha-GRU policy with supervised task heads and tactile residual action branches."""

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
        tactile_bottom_keys: Iterable[str] | None = None,
        proprio_key: str = "proprio_obs",
        aux_visibility_gt_key: str = "aux_visibility_gt",
        aux_occlusion_gt_key: str = "aux_occlusion_gt",
        aux_bottom_contact_gt_key: str = "aux_bottom_contact_gt",
        aux_bottom_rel_pos_gt_key: str = "aux_bottom_rel_pos_gt",
        aux_inner_contact_gt_key: str = "aux_inner_contact_gt",
        aux_inner_contact_diff_gt_key: str = "aux_inner_contact_diff_gt",
        bottom_tactile_mask_key: str = "bottom_tactile_current_mask",
        inner_tactile_mask_key: str = "inner_tactile_current_mask",
        tactile_time_window: int = 10,
        tactile_gru_hidden_dim: int = 64,
        tactile_gru_layers: int = 2,
        tactile_latent_dim: int = 128,
        fused_dim: int = 256,
        task_hidden_dim: int = 256,
        pair_hidden_layers: Iterable[int] | None = None,
        alpha_hidden_dim: int = 64,
        gate_hidden_dim: int = 64,
        base_mlp_layers: Iterable[int] | None = None,
        residual_hidden_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        bottom_dx_scale: float = 0.25,
        bottom_dy_scale: float = 0.12,
        inner_gripper_scale: float = 0.25,
        aux_visibility_weight: float = 1.0,
        aux_occlusion_weight: float = 1.0,
        aux_bottom_contact_weight: float = 1.0,
        aux_bottom_rel_pos_weight: float = 1.0,
        aux_inner_contact_weight: float = 1.0,
        aux_inner_contact_diff_weight: float = 1.0,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        if tactile_inner_keys is None:
            tactile_inner_keys = ("tactile_left_depth_resnet", "tactile_right_depth_resnet")
        if tactile_bottom_keys is None:
            tactile_bottom_keys = ("tactile_left_down_depth_resnet", "tactile_right_down_depth_resnet")
        if pair_hidden_layers is None:
            pair_hidden_layers = (128,)
        if base_mlp_layers is None:
            base_mlp_layers = (256, 128, 64)
        if residual_hidden_layers is None:
            residual_hidden_layers = (128, 64)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.aux_visibility_gt_key = str(aux_visibility_gt_key)
        self.aux_occlusion_gt_key = str(aux_occlusion_gt_key)
        self.aux_bottom_contact_gt_key = str(aux_bottom_contact_gt_key)
        self.aux_bottom_rel_pos_gt_key = str(aux_bottom_rel_pos_gt_key)
        self.aux_inner_contact_gt_key = str(aux_inner_contact_gt_key)
        self.aux_inner_contact_diff_gt_key = str(aux_inner_contact_diff_gt_key)
        self.bottom_tactile_mask_key = str(bottom_tactile_mask_key)
        self.inner_tactile_mask_key = str(inner_tactile_mask_key)

        self.tactile_inner_keys = tuple(key for key in tactile_inner_keys if self._has_obs_key(observation_space, key))
        self.tactile_bottom_keys = tuple(key for key in tactile_bottom_keys if self._has_obs_key(observation_space, key))
        self.tactile_keys = self.tactile_inner_keys + self.tactile_bottom_keys
        self.tactile_time_window = max(1, int(tactile_time_window))
        self.tactile_gru_hidden_dim = int(tactile_gru_hidden_dim)
        self.tactile_gru_layers = int(tactile_gru_layers)
        self.tactile_latent_dim = int(tactile_latent_dim)
        self.fused_dim = int(fused_dim)

        if not self._has_obs_key(observation_space, self.vision_key):
            raise ValueError(f"Missing vision observation key '{self.vision_key}'")
        if not self._has_obs_key(observation_space, self.proprio_key):
            raise ValueError(f"Missing proprio observation key '{self.proprio_key}'")
        if len(self.tactile_inner_keys) != 2:
            raise ValueError(f"Expected 2 inner tactile keys, got {self.tactile_inner_keys}")
        if len(self.tactile_bottom_keys) != 2:
            raise ValueError(f"Expected 2 bottom tactile keys, got {self.tactile_bottom_keys}")

        tracked_keys = [
            self.vision_key,
            self.proprio_key,
            *self.tactile_keys,
            self.bottom_tactile_mask_key,
            self.inner_tactile_mask_key,
        ]
        for key in (
            self.aux_visibility_gt_key,
            self.aux_occlusion_gt_key,
            self.aux_bottom_contact_gt_key,
            self.aux_bottom_rel_pos_gt_key,
            self.aux_inner_contact_gt_key,
            self.aux_inner_contact_diff_gt_key,
        ):
            if self._has_obs_key(observation_space, key):
                tracked_keys.append(key)

        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in tracked_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }

        self._sensor_feat_dims: dict[str, int] = {}
        for key in self.tactile_keys:
            total_dim = int(self._obs_dims[key])
            if total_dim % self.tactile_time_window != 0:
                raise ValueError(
                    f"Tactile observation '{key}' dim={total_dim} is not divisible by window={self.tactile_time_window}"
                )
            self._sensor_feat_dims[key] = total_dim // self.tactile_time_window

        self.vision_dim = int(self._obs_dims[self.vision_key])
        self.proprio_dim = int(self._obs_dims[self.proprio_key])

        act_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, self.fused_dim),
            act_cls(),
        )

        self.tactile_grus = nn.ModuleDict()
        self.tactile_heads = nn.ModuleDict()
        for key in self.tactile_keys:
            input_dim = self._sensor_feat_dims[key]
            self.tactile_grus[key] = nn.GRU(
                input_size=input_dim,
                hidden_size=self.tactile_gru_hidden_dim,
                num_layers=self.tactile_gru_layers,
                batch_first=True,
            )
            self.tactile_heads[key] = nn.Sequential(
                nn.LayerNorm(self.tactile_gru_hidden_dim),
                nn.Linear(self.tactile_gru_hidden_dim, self.tactile_latent_dim),
                act_cls(),
            )

        tactile_total_latent = self.tactile_latent_dim * len(self.tactile_keys)
        self.tactile_proj = nn.Sequential(
            nn.LayerNorm(tactile_total_latent),
            nn.Linear(tactile_total_latent, self.fused_dim),
            act_cls(),
        )

        self.vision_task_encoder = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, int(task_hidden_dim)),
            act_cls(),
        )
        self.visibility_head = nn.Linear(int(task_hidden_dim), 1)
        self.occlusion_head = nn.Linear(int(task_hidden_dim), 1)

        self.inner_pair_mlp = self._build_mlp(
            input_dim=2 * self.tactile_latent_dim,
            hidden_layers=pair_hidden_layers,
            output_dim=self.tactile_latent_dim,
            act_cls=act_cls,
        )
        self.bottom_pair_mlp = self._build_mlp(
            input_dim=2 * self.tactile_latent_dim,
            hidden_layers=pair_hidden_layers,
            output_dim=self.tactile_latent_dim,
            act_cls=act_cls,
        )

        self.bottom_contact_head = self._build_mlp(
            input_dim=self.tactile_latent_dim,
            hidden_layers=(int(task_hidden_dim) // 2,),
            output_dim=2,
            act_cls=act_cls,
        )
        self.bottom_rel_pos_head = self._build_mlp(
            input_dim=self.tactile_latent_dim,
            hidden_layers=(int(task_hidden_dim) // 2,),
            output_dim=2,
            act_cls=act_cls,
        )
        self.inner_contact_head = self._build_mlp(
            input_dim=self.tactile_latent_dim,
            hidden_layers=(int(task_hidden_dim) // 2,),
            output_dim=1,
            act_cls=act_cls,
        )
        self.inner_diff_head = self._build_mlp(
            input_dim=self.tactile_latent_dim,
            hidden_layers=(int(task_hidden_dim) // 2,),
            output_dim=1,
            act_cls=act_cls,
        )

        self.alpha_gate = nn.Sequential(
            nn.Linear(self.proprio_dim + 2, int(alpha_hidden_dim)),
            act_cls(),
            nn.Linear(int(alpha_hidden_dim), 1),
            nn.Sigmoid(),
        )

        base_input_dim = (2 * self.fused_dim) + self.proprio_dim
        self.base_mlp, base_last_dim = self._build_stack(base_input_dim, base_mlp_layers, act_cls)
        self.base_mu = nn.Linear(base_last_dim, self.num_actions)

        bottom_input_dim = self.tactile_latent_dim + 4 + self.proprio_dim
        inner_input_dim = self.tactile_latent_dim + 2 + self.proprio_dim
        self.bottom_soft_gate = nn.Sequential(
            nn.Linear(bottom_input_dim, int(gate_hidden_dim)),
            act_cls(),
            nn.Linear(int(gate_hidden_dim), 1),
            nn.Sigmoid(),
        )
        self.inner_soft_gate = nn.Sequential(
            nn.Linear(inner_input_dim, int(gate_hidden_dim)),
            act_cls(),
            nn.Linear(int(gate_hidden_dim), 1),
            nn.Sigmoid(),
        )
        self.bottom_residual = self._build_mlp(bottom_input_dim, residual_hidden_layers, 2, act_cls)
        self.inner_residual = self._build_mlp(inner_input_dim, residual_hidden_layers, 1, act_cls)

        self.bottom_dx_scale = float(bottom_dx_scale)
        self.bottom_dy_scale = float(bottom_dy_scale)
        self.inner_gripper_scale = float(inner_gripper_scale)

        self.aux_visibility_weight = float(aux_visibility_weight)
        self.aux_occlusion_weight = float(aux_occlusion_weight)
        self.aux_bottom_contact_weight = float(aux_bottom_contact_weight)
        self.aux_bottom_rel_pos_weight = float(aux_bottom_rel_pos_weight)
        self.aux_inner_contact_weight = float(aux_inner_contact_weight)
        self.aux_inner_contact_diff_weight = float(aux_inner_contact_diff_weight)

        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(initial_log_std)),
            requires_grad=not bool(fixed_log_std),
        )

    @staticmethod
    def _build_mlp(input_dim: int, hidden_layers: Iterable[int], output_dim: int, act_cls):
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                continue
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_cls())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, int(output_dim)))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_stack(input_dim: int, hidden_layers: Iterable[int], act_cls):
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                continue
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_cls())
            last_dim = hidden_dim
        return (nn.Sequential(*layers) if layers else nn.Identity(), last_dim)

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
            dim = int(self._obs_dims.get(key, 0))
            return torch.zeros((batch_size, dim), device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(device=self.device, dtype=torch.float32)
        return value.reshape(batch_size, -1)

    def _collect_batch_size(self, obs: Dict[str, Any]) -> int:
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        raise ValueError("No tensor observations found in inputs['states']")

    def _encode_tactile(self, obs: Dict[str, Any], batch_size: int) -> Dict[str, torch.Tensor]:
        latent_by_key: Dict[str, torch.Tensor] = {}
        for key in self.tactile_keys:
            feature_dim = self._sensor_feat_dims[key]
            flat = self._flatten_obs(obs, key, batch_size)
            sequence = flat.reshape(batch_size, self.tactile_time_window, feature_dim)
            _, h_n = self.tactile_grus[key](sequence)
            latent_by_key[key] = self.tactile_heads[key](h_n[-1])
        return latent_by_key

    def _resolve_mask(self, obs: Dict[str, Any], key: str, batch_size: int, fallback: torch.Tensor) -> torch.Tensor:
        if self._has_obs_key(self.observation_space, key):
            mask = self._flatten_obs(obs, key, batch_size)[:, :1].clamp(0.0, 1.0)
            return mask
        return fallback

    def _build_aux_losses(
        self,
        obs: Dict[str, Any],
        batch_size: int,
        visibility_pred: torch.Tensor,
        occlusion_pred: torch.Tensor,
        bottom_contact_pred: torch.Tensor,
        bottom_rel_pos_pred: torch.Tensor,
        inner_contact_pred: torch.Tensor,
        inner_diff_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = visibility_pred.new_zeros(())

        visibility_loss = zero
        occlusion_loss = zero
        bottom_contact_loss = zero
        bottom_rel_pos_loss = zero
        inner_contact_loss = zero
        inner_diff_loss = zero

        if self._has_obs_key(self.observation_space, self.aux_visibility_gt_key):
            gt = self._flatten_obs(obs, self.aux_visibility_gt_key, batch_size)[:, :1].clamp(0.0, 1.0)
            visibility_loss = F.smooth_l1_loss(visibility_pred, gt)
        if self._has_obs_key(self.observation_space, self.aux_occlusion_gt_key):
            gt = self._flatten_obs(obs, self.aux_occlusion_gt_key, batch_size)[:, :1].clamp(0.0, 1.0)
            occlusion_loss = F.smooth_l1_loss(occlusion_pred, gt)
        if self._has_obs_key(self.observation_space, self.aux_bottom_contact_gt_key):
            gt = self._flatten_obs(obs, self.aux_bottom_contact_gt_key, batch_size)[:, :2].clamp(0.0, 1.0)
            bottom_contact_loss = F.smooth_l1_loss(bottom_contact_pred, gt)
        if self._has_obs_key(self.observation_space, self.aux_bottom_rel_pos_gt_key):
            gt = self._flatten_obs(obs, self.aux_bottom_rel_pos_gt_key, batch_size)[:, :2].clamp(-1.0, 1.0)
            bottom_rel_pos_loss = F.smooth_l1_loss(bottom_rel_pos_pred, gt)
        if self._has_obs_key(self.observation_space, self.aux_inner_contact_gt_key):
            gt = self._flatten_obs(obs, self.aux_inner_contact_gt_key, batch_size)[:, :1].clamp(0.0, 1.0)
            inner_contact_loss = F.smooth_l1_loss(inner_contact_pred, gt)
        if self._has_obs_key(self.observation_space, self.aux_inner_contact_diff_gt_key):
            gt = self._flatten_obs(obs, self.aux_inner_contact_diff_gt_key, batch_size)[:, :1].clamp(0.0, 1.0)
            inner_diff_loss = F.smooth_l1_loss(inner_diff_pred, gt)

        reliability_loss = (
            self.aux_visibility_weight * visibility_loss + self.aux_occlusion_weight * occlusion_loss
        )
        probe_loss = (
            self.aux_bottom_contact_weight * bottom_contact_loss + self.aux_bottom_rel_pos_weight * bottom_rel_pos_loss
        )
        grasp_loss = (
            self.aux_inner_contact_weight * inner_contact_loss
            + self.aux_inner_contact_diff_weight * inner_diff_loss
        )
        aux_loss = reliability_loss + probe_loss + grasp_loss
        return aux_loss, reliability_loss, probe_loss, grasp_loss

    def compute(self, inputs: Dict[str, Any], role: str = ""):
        states = inputs.get("states")
        if isinstance(states, dict):
            obs = states
        else:
            obs = unflatten_tensorized_space(self.observation_space, states)

        batch_size = self._collect_batch_size(obs)

        vision_raw = self._flatten_obs(obs, self.vision_key, batch_size)
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        tactile_latents = self._encode_tactile(obs, batch_size)

        vision_task_hidden = self.vision_task_encoder(vision_raw)
        visibility_pred = torch.sigmoid(self.visibility_head(vision_task_hidden))
        occlusion_pred = torch.sigmoid(self.occlusion_head(vision_task_hidden))

        inner_pair = self.inner_pair_mlp(
            torch.cat(
                [
                    tactile_latents[self.tactile_inner_keys[0]],
                    tactile_latents[self.tactile_inner_keys[1]],
                ],
                dim=-1,
            )
        )
        bottom_pair = self.bottom_pair_mlp(
            torch.cat(
                [
                    tactile_latents[self.tactile_bottom_keys[0]],
                    tactile_latents[self.tactile_bottom_keys[1]],
                ],
                dim=-1,
            )
        )

        bottom_contact_pred = torch.sigmoid(self.bottom_contact_head(bottom_pair))
        bottom_rel_pos_pred = torch.tanh(self.bottom_rel_pos_head(bottom_pair))
        inner_contact_pred = torch.sigmoid(self.inner_contact_head(inner_pair))
        inner_diff_pred = torch.sigmoid(self.inner_diff_head(inner_pair))

        alpha = self.alpha_gate(torch.cat([proprio, visibility_pred, occlusion_pred], dim=-1))
        set_latest_alpha(alpha)

        vision_feat = self.vision_proj(vision_raw)
        tactile_global = self.tactile_proj(torch.cat([tactile_latents[key] for key in self.tactile_keys], dim=-1))
        base_input = torch.cat([(1.0 - alpha) * vision_feat, alpha * tactile_global, proprio], dim=-1)
        mu = self.base_mu(self.base_mlp(base_input))

        bottom_summary = torch.cat([bottom_contact_pred, bottom_rel_pos_pred], dim=-1)
        inner_summary = torch.cat([inner_contact_pred, inner_diff_pred], dim=-1)
        bottom_input = torch.cat([bottom_pair, bottom_summary, proprio], dim=-1)
        inner_input = torch.cat([inner_pair, inner_summary, proprio], dim=-1)

        bottom_mask_fallback = (bottom_contact_pred.max(dim=-1, keepdim=True).values > 0.1).to(torch.float32)
        inner_mask_fallback = (inner_contact_pred > 0.1).to(torch.float32)
        m_probe = self._resolve_mask(obs, self.bottom_tactile_mask_key, batch_size, bottom_mask_fallback)
        m_grasp = self._resolve_mask(obs, self.inner_tactile_mask_key, batch_size, inner_mask_fallback)
        g_probe = m_probe * self.bottom_soft_gate(bottom_input)
        g_grasp = m_grasp * self.inner_soft_gate(inner_input)

        bottom_delta = torch.tanh(self.bottom_residual(bottom_input))
        inner_delta = torch.tanh(self.inner_residual(inner_input))

        mu = mu.clone()
        mu[:, 0:1] = mu[:, 0:1] + g_probe * self.bottom_dx_scale * bottom_delta[:, 0:1]
        mu[:, 1:2] = mu[:, 1:2] + g_probe * self.bottom_dy_scale * bottom_delta[:, 1:2]
        mu[:, 4:5] = mu[:, 4:5] + g_grasp * self.inner_gripper_scale * inner_delta

        aux_loss, reliability_loss, probe_loss, grasp_loss = self._build_aux_losses(
            obs=obs,
            batch_size=batch_size,
            visibility_pred=visibility_pred,
            occlusion_pred=occlusion_pred,
            bottom_contact_pred=bottom_contact_pred,
            bottom_rel_pos_pred=bottom_rel_pos_pred,
            inner_contact_pred=inner_contact_pred,
            inner_diff_pred=inner_diff_pred,
        )

        dummy_stage_logits = torch.zeros((batch_size, 4), device=self.device, dtype=torch.float32)
        set_latest_aux_predictions(
            visibility_pred,
            dummy_stage_logits,
            w_probe=g_probe,
            w_grasp=g_grasp,
            alpha=alpha,
            m_probe=m_probe,
            m_grasp=m_grasp,
        )

        outputs = {
            "aux_loss": aux_loss,
            "aux_reliability_loss": reliability_loss,
            "aux_probe_loss": probe_loss,
            "aux_grasp_loss": grasp_loss,
            "aux_visibility_pred": visibility_pred,
            "aux_occlusion_pred": occlusion_pred,
            "aux_bottom_contact_pred": bottom_contact_pred,
            "aux_bottom_rel_pos_pred": bottom_rel_pos_pred,
            "aux_inner_contact_pred": inner_contact_pred,
            "aux_inner_contact_diff_pred": inner_diff_pred,
            "g_probe": g_probe.mean(),
            "g_grasp": g_grasp.mean(),
            "alpha": alpha.mean(),
            "m_probe": m_probe.mean(),
            "m_grasp": m_grasp.mean(),
        }
        return mu, self.log_std_parameter, outputs
