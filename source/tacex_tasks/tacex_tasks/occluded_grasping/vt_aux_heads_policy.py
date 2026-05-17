"""VT policy with constrained modality gating and rule-based stage priors."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_aux_heads import set_latest_aux_predictions


class OccludedGraspingVTAuxHeadsPolicy(GaussianMixin, Model):
    """Action policy + supervised aux heads with rule-stage constrained gating."""

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
        prev_action_key: str = "prev_action",
        aux_input_key: str = "aux_head_input",
        aux_reliability_gt_key: str = "aux_reliability_gt",
        aux_stage_gt_key: str = "aux_stage_gt",
        aux_stage_onehot_gt_key: str = "aux_stage_onehot_gt",
        aux_probe_gt_key: str = "aux_probe_gt",
        aux_grasp_gt_key: str = "aux_grasp_gt",
        bottom_valid_hist_key: str = "bottom_valid_hist",
        inner_valid_hist_key: str = "inner_valid_hist",
        mlp_layers: Iterable[int] | None = None,
        aux_hidden_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        aux_reliability_weight: float = 1.0,
        aux_stage_weight: float = 0.0,
        aux_probe_weight: float = 1.0,
        aux_grasp_weight: float = 1.0,
        tactile_pos_enable: bool = True,
        tactile_pos_hidden_dim: int = 64,
        tactile_pos_scale: float = 1.0,
        tactile_pair_fusion_enable: bool = True,
        tactile_pair_out_dim: int = 256,
        tactile_pair_hidden_layers: Iterable[int] | None = None,
        tactile_probe_keys: Iterable[str] | None = None,
        tactile_grasp_keys: Iterable[str] | None = None,
        tactile_gate_hidden_layers: Iterable[int] | None = None,
        gate_use_vision: bool | None = None,
        use_alpha_correction_head: bool | None = None,
        alpha_correction_eps: float = 0.0,
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
        if aux_hidden_layers is None:
            aux_hidden_layers = (256, 128)
        if tactile_pair_hidden_layers is None:
            tactile_pair_hidden_layers = (256, 128)
        if tactile_gate_hidden_layers is None:
            tactile_gate_hidden_layers = (256, 128)

        self.vision_key = str(vision_key)
        self.proprio_key = str(proprio_key)
        self.prev_action_key = str(prev_action_key)
        self.aux_input_key = str(aux_input_key)
        self.aux_reliability_gt_key = str(aux_reliability_gt_key)
        self.aux_stage_gt_key = str(aux_stage_gt_key)
        self.aux_stage_onehot_gt_key = str(aux_stage_onehot_gt_key)
        self.aux_probe_gt_key = str(aux_probe_gt_key)
        self.aux_grasp_gt_key = str(aux_grasp_gt_key)
        self.bottom_valid_hist_key = str(bottom_valid_hist_key)
        self.inner_valid_hist_key = str(inner_valid_hist_key)
        self.tactile_keys = tuple(key for key in tactile_keys if self._has_obs_key(observation_space, key))
        if not self.tactile_keys:
            raise ValueError("No tactile observation keys found for auxiliary-head policy")

        required_keys = [self.vision_key, self.proprio_key, self.aux_input_key]
        for key in required_keys:
            if not self._has_obs_key(observation_space, key):
                raise ValueError(f"Missing required observation key '{key}'")

        tracked_keys = [*required_keys, *self.tactile_keys]
        if self._has_obs_key(observation_space, self.prev_action_key):
            tracked_keys.append(self.prev_action_key)
        if self._has_obs_key(observation_space, self.aux_reliability_gt_key):
            tracked_keys.append(self.aux_reliability_gt_key)
        if self._has_obs_key(observation_space, self.aux_stage_gt_key):
            tracked_keys.append(self.aux_stage_gt_key)
        if self._has_obs_key(observation_space, self.aux_stage_onehot_gt_key):
            tracked_keys.append(self.aux_stage_onehot_gt_key)
        if self._has_obs_key(observation_space, self.aux_probe_gt_key):
            tracked_keys.append(self.aux_probe_gt_key)
        if self._has_obs_key(observation_space, self.aux_grasp_gt_key):
            tracked_keys.append(self.aux_grasp_gt_key)
        if self._has_obs_key(observation_space, self.bottom_valid_hist_key):
            tracked_keys.append(self.bottom_valid_hist_key)
        if self._has_obs_key(observation_space, self.inner_valid_hist_key):
            tracked_keys.append(self.inner_valid_hist_key)

        self._obs_shapes = {key: self._get_obs_shape(observation_space, key) for key in tracked_keys}
        self._obs_dims = {
            key: int(self._obs_shapes[key][-1]) for key in self._obs_shapes if self._obs_shapes[key] is not None
        }
        if gate_use_vision is None:
            gate_use_vision = float(aux_stage_weight) > 0.0
        self._gate_use_vision = bool(gate_use_vision)

        act_cls = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU

        self.tactile_pos_enable = bool(tactile_pos_enable)
        self.tactile_pos_scale = float(tactile_pos_scale)
        self._num_tactile_keys = len(self.tactile_keys)
        self._tactile_pos_mlps = nn.ModuleDict()
        if self.tactile_pos_enable:
            hidden_dim = int(tactile_pos_hidden_dim)
            for key in self.tactile_keys:
                tactile_dim = int(self._obs_dims[key])
                if hidden_dim > 0:
                    self._tactile_pos_mlps[key] = nn.Sequential(
                        nn.Linear(self._num_tactile_keys, hidden_dim),
                        act_cls(),
                        nn.Linear(hidden_dim, tactile_dim),
                    )
                else:
                    self._tactile_pos_mlps[key] = nn.Linear(self._num_tactile_keys, tactile_dim)

        self.tactile_pair_fusion_enable = bool(tactile_pair_fusion_enable)
        self.tactile_pair_out_dim = int(tactile_pair_out_dim)
        if self.tactile_pair_out_dim <= 0:
            raise ValueError(f"tactile_pair_out_dim must be positive, got {self.tactile_pair_out_dim}")

        self._tactile_probe_keys, self._tactile_grasp_keys = self._resolve_tactile_pair_keys(
            tactile_probe_keys=tactile_probe_keys,
            tactile_grasp_keys=tactile_grasp_keys,
        )
        if self.tactile_pair_fusion_enable:
            proprio_dim = int(self._obs_dims[self.proprio_key])
            probe_in_dim = (
                int(self._obs_dims[self._tactile_probe_keys[0]])
                + int(self._obs_dims[self._tactile_probe_keys[1]])
                + proprio_dim
            )
            grasp_in_dim = (
                int(self._obs_dims[self._tactile_grasp_keys[0]])
                + int(self._obs_dims[self._tactile_grasp_keys[1]])
                + proprio_dim
            )
            self._probe_pair_mlp = self._build_mlp(
                input_dim=probe_in_dim,
                hidden_layers=tactile_pair_hidden_layers,
                output_dim=self.tactile_pair_out_dim,
                act_cls=act_cls,
            )
            self._grasp_pair_mlp = self._build_mlp(
                input_dim=grasp_in_dim,
                hidden_layers=tactile_pair_hidden_layers,
                output_dim=self.tactile_pair_out_dim,
                act_cls=act_cls,
            )
            gate_h_dim = self.tactile_pair_out_dim
        else:
            gate_h_dim = 1

        # Gate context:
        # new mode -> [r_v, s(4), q_probe(4), q_grasp(3)]
        # legacy mode -> [vision, r_v, s(4), q_probe(4), q_grasp(3)]
        gate_in_dim = 1 + 4 + 4 + 3
        if self._gate_use_vision:
            gate_in_dim += int(self._obs_dims[self.vision_key])
        self._tactile_gate_mlp = self._build_mlp(
            input_dim=gate_in_dim,
            hidden_layers=tactile_gate_hidden_layers,
            output_dim=2,
            act_cls=act_cls,
        )
        if use_alpha_correction_head is None:
            use_alpha_correction_head = abs(float(alpha_correction_eps)) > 0.0
        self._use_alpha_correction_head = bool(use_alpha_correction_head)
        self._alpha_correction_head = nn.Linear(gate_in_dim, 1) if self._use_alpha_correction_head else None
        self._alpha_correction_eps = float(alpha_correction_eps)

        # Keep action-network input shape unchanged:
        # [fused_vision, tactile_slot_1, tactile_slot_2, proprio, prev_action]
        prev_action_dim = int(self._obs_dims.get(self.prev_action_key, 0))
        action_in_dim = self._obs_dims[self.vision_key] + gate_h_dim + gate_h_dim + self._obs_dims[self.proprio_key] + prev_action_dim
        aux_in_dim = self._obs_dims[self.aux_input_key]

        action_mlp: list[nn.Module] = []
        last_dim = action_in_dim
        for hidden_dim in mlp_layers:
            action_mlp.append(nn.Linear(last_dim, int(hidden_dim)))
            action_mlp.append(act_cls())
            last_dim = int(hidden_dim)
        self.action_mlp = nn.Sequential(*action_mlp) if action_mlp else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)

        aux_mlp: list[nn.Module] = []
        aux_last_dim = aux_in_dim
        for hidden_dim in aux_hidden_layers:
            aux_mlp.append(nn.Linear(aux_last_dim, int(hidden_dim)))
            aux_mlp.append(act_cls())
            aux_last_dim = int(hidden_dim)
        self.aux_mlp = nn.Sequential(*aux_mlp) if aux_mlp else nn.Identity()
        self.reliability_head = nn.Linear(aux_last_dim, 1)
        # Keep optional trainable stage head for backwards-compatible checkpoint loading.
        self.stage_head = nn.Linear(aux_last_dim, 4) if float(aux_stage_weight) > 0.0 else None
        pair_head_in_dim = self.tactile_pair_out_dim if self.tactile_pair_fusion_enable else 1
        self.probe_head = self._build_mlp(
            input_dim=pair_head_in_dim,
            hidden_layers=(128,),
            output_dim=4,
            act_cls=act_cls,
        )
        self.grasp_head = self._build_mlp(
            input_dim=pair_head_in_dim,
            hidden_layers=(128,),
            output_dim=3,
            act_cls=act_cls,
        )

        self.aux_reliability_weight = float(aux_reliability_weight)
        self.aux_stage_weight = float(aux_stage_weight)
        self.aux_probe_weight = float(aux_probe_weight)
        self.aux_grasp_weight = float(aux_grasp_weight)

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

    def _resolve_tactile_pair_keys(
        self,
        tactile_probe_keys: Iterable[str] | None,
        tactile_grasp_keys: Iterable[str] | None,
    ) -> tuple[tuple[str, str], tuple[str, str]]:
        valid_keys = set(self.tactile_keys)

        def _clean(keys: Iterable[str] | None) -> list[str]:
            if keys is None:
                return []
            return [str(key) for key in keys if str(key) in valid_keys]

        probe_keys = _clean(tactile_probe_keys)
        grasp_keys = _clean(tactile_grasp_keys)

        if len(probe_keys) != 2:
            probe_keys = [key for key in self.tactile_keys if "down" in key.lower()]
        if len(grasp_keys) != 2:
            grasp_keys = [key for key in self.tactile_keys if key not in probe_keys]

        if len(probe_keys) != 2 or len(grasp_keys) != 2:
            raise ValueError(
                "Failed to resolve tactile probe/grasp key pairs from tactile_keys="
                f"{self.tactile_keys}, probe={probe_keys}, grasp={grasp_keys}"
            )
        return (probe_keys[0], probe_keys[1]), (grasp_keys[0], grasp_keys[1])

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

    def _encode_tactile_with_position(self, obs: Dict[str, Any], batch_size: int) -> Dict[str, torch.Tensor]:
        tactile_features: Dict[str, torch.Tensor] = {}
        for index, key in enumerate(self.tactile_keys):
            feat = self._flatten_obs(obs, key, batch_size)
            if self.tactile_pos_enable:
                one_hot = F.one_hot(
                    torch.full((batch_size,), index, device=self.device, dtype=torch.long),
                    num_classes=self._num_tactile_keys,
                ).to(dtype=feat.dtype)
                pos_feat = self._tactile_pos_mlps[key](one_hot)
                feat = feat + self.tactile_pos_scale * pos_feat
            tactile_features[key] = feat
        return tactile_features

    def _fuse_tactile_pairs(
        self, tactile_feats: Dict[str, torch.Tensor], proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.tactile_pair_fusion_enable:
            probe_input = torch.cat(
                [
                    tactile_feats[self._tactile_probe_keys[0]],
                    tactile_feats[self._tactile_probe_keys[1]],
                    proprio,
                ],
                dim=-1,
            )
            grasp_input = torch.cat(
                [
                    tactile_feats[self._tactile_grasp_keys[0]],
                    tactile_feats[self._tactile_grasp_keys[1]],
                    proprio,
                ],
                dim=-1,
            )
            h_probe = self._probe_pair_mlp(probe_input)
            h_grasp = self._grasp_pair_mlp(grasp_input)
            tactile_action_feat = torch.cat([h_grasp, h_probe], dim=-1)
            return tactile_action_feat, h_probe, h_grasp

        tactile_action_feat = torch.cat([tactile_feats[key] for key in self.tactile_keys], dim=-1)
        zero = tactile_action_feat.new_zeros((tactile_action_feat.shape[0], 1))
        return tactile_action_feat, zero, zero

    def _build_aux_losses(
        self,
        obs: Dict[str, Any],
        batch_size: int,
        reliability_pred: torch.Tensor,
        stage_logits_pred: torch.Tensor | None,
        probe_pred: torch.Tensor,
        grasp_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = reliability_pred.new_zeros(())
        reliability_loss = zero
        stage_loss = zero
        probe_loss = zero
        grasp_loss = zero

        if self._has_obs_key(self.observation_space, self.aux_reliability_gt_key):
            reliability_gt = self._flatten_obs(obs, self.aux_reliability_gt_key, batch_size)[:, :1].clamp(0.0, 1.0)
            reliability_loss = F.smooth_l1_loss(reliability_pred, reliability_gt)

        if self.aux_stage_weight > 0.0 and stage_logits_pred is not None:
            if self._has_obs_key(self.observation_space, self.aux_stage_onehot_gt_key):
                stage_onehot_gt = self._flatten_obs(obs, self.aux_stage_onehot_gt_key, batch_size)
                if stage_onehot_gt.shape[1] >= 4:
                    target = stage_onehot_gt[:, :4].clamp_min(0.0)
                    target_sum = target.sum(dim=-1, keepdim=True)
                    uniform = torch.full_like(target, 0.25)
                    target = torch.where(target_sum > 0.0, target / target_sum.clamp_min(1e-6), uniform)
                    stage_loss = -(target * F.log_softmax(stage_logits_pred, dim=-1)).sum(dim=-1).mean()
            elif self._has_obs_key(self.observation_space, self.aux_stage_gt_key):
                stage_raw = self._flatten_obs(obs, self.aux_stage_gt_key, batch_size)[:, :1]
                stage_idx = torch.round(stage_raw).clamp(0.0, 3.0).to(dtype=torch.long).squeeze(-1)
                stage_loss = F.cross_entropy(stage_logits_pred, stage_idx)

        if self._has_obs_key(self.observation_space, self.aux_probe_gt_key):
            probe_gt = self._flatten_obs(obs, self.aux_probe_gt_key, batch_size)
            if probe_gt.shape[1] >= 4:
                probe_loss = F.smooth_l1_loss(probe_pred, probe_gt[:, :4].clamp(0.0, 1.0))

        if self._has_obs_key(self.observation_space, self.aux_grasp_gt_key):
            grasp_gt = self._flatten_obs(obs, self.aux_grasp_gt_key, batch_size)
            if grasp_gt.shape[1] >= 3:
                grasp_loss = F.smooth_l1_loss(grasp_pred, grasp_gt[:, :3].clamp(0.0, 1.0))

        aux_loss = (
            self.aux_reliability_weight * reliability_loss
            + self.aux_stage_weight * stage_loss
            + self.aux_probe_weight * probe_loss
            + self.aux_grasp_weight * grasp_loss
        )
        return aux_loss, reliability_loss, stage_loss, probe_loss, grasp_loss

    def _get_rule_stage(self, obs: Dict[str, Any], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        stage_prob = None
        if self._has_obs_key(self.observation_space, self.aux_stage_onehot_gt_key):
            stage_onehot = self._flatten_obs(obs, self.aux_stage_onehot_gt_key, batch_size)
            if stage_onehot.shape[1] >= 4:
                stage_prob = stage_onehot[:, :4]
        if stage_prob is None and self._has_obs_key(self.observation_space, self.aux_stage_gt_key):
            stage_raw = self._flatten_obs(obs, self.aux_stage_gt_key, batch_size)[:, :1]
            stage_idx = torch.round(stage_raw).clamp(0.0, 3.0).to(dtype=torch.long).squeeze(-1)
            stage_prob = F.one_hot(stage_idx, num_classes=4).to(dtype=torch.float32)
        if stage_prob is None:
            stage_prob = torch.full((batch_size, 4), 0.25, dtype=torch.float32, device=self.device)
        else:
            stage_prob = stage_prob.to(device=self.device, dtype=torch.float32)

        stage_prob = stage_prob.clamp_min(0.0)
        stage_prob_sum = stage_prob.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(stage_prob, 0.25)
        stage_prob = torch.where(stage_prob_sum > 0.0, stage_prob / stage_prob_sum.clamp_min(1e-6), uniform)
        stage_logits = torch.log(stage_prob.clamp_min(1e-6))
        return stage_logits, stage_prob

    def _get_recent_tactile_masks(self, obs: Dict[str, Any], batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _mask_from_hist(key: str) -> torch.Tensor:
            hist_flat = self._flatten_obs(obs, key, batch_size)
            if hist_flat.numel() == 0 or hist_flat.shape[1] == 0:
                return torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
            if hist_flat.shape[1] >= 2 and hist_flat.shape[1] % 2 == 0:
                hist = hist_flat.reshape(batch_size, -1, 2)
                return (hist > 0.5).any(dim=(1, 2)).to(dtype=torch.float32).unsqueeze(-1)
            return (hist_flat > 0.5).any(dim=1).to(dtype=torch.float32).unsqueeze(-1)

        m_probe = _mask_from_hist(self.bottom_valid_hist_key)
        m_grasp = _mask_from_hist(self.inner_valid_hist_key)
        m_tac = torch.maximum(m_probe, m_grasp)
        return m_probe, m_grasp, m_tac

    def _get_stage_priors(self, stage_prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [alpha_prior, b_probe, b_grasp] for [approach, probe, grasp, lift]
        prior_table = stage_prob.new_tensor(
            [
                [0.0, 0.0, 0.0],
                [0.4, 1.0, 0.2],
                [0.7, 0.2, 1.0],
                [0.5, 0.1, 0.8],
            ]
        )
        priors = stage_prob @ prior_table
        return priors[:, 0:1], priors[:, 1:2], priors[:, 2:3]

    def _inject_tactile_into_vision(self, vision: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
        """Inject tactile residual into vision channels without changing dimensionality."""
        vision_dim = vision.shape[1]
        tactile_dim = tactile.shape[1]
        if tactile_dim == vision_dim:
            tactile_v = tactile
        elif tactile_dim < vision_dim:
            pad = tactile.new_zeros((tactile.shape[0], vision_dim - tactile_dim))
            tactile_v = torch.cat([tactile, pad], dim=-1)
        else:
            tactile_v = tactile[:, :vision_dim]
        return vision + tactile_v

    def compute(self, inputs: Dict[str, Any], role: str = ""):
        states = inputs.get("states")
        if isinstance(states, dict):
            obs = states
        else:
            obs = unflatten_tensorized_space(self.observation_space, states)

        batch_size = self._collect_batch_size(obs)
        vision = self._flatten_obs(obs, self.vision_key, batch_size)
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        tactile_feats = self._encode_tactile_with_position(obs, batch_size)
        _, h_probe, h_grasp = self._fuse_tactile_pairs(tactile_feats, proprio)

        aux_input = self._flatten_obs(obs, self.aux_input_key, batch_size)
        aux_hidden = self.aux_mlp(aux_input)
        reliability_pred = torch.sigmoid(self.reliability_head(aux_hidden))
        stage_logits_rule, stage_prob = self._get_rule_stage(obs, batch_size)
        stage_logits_pred = self.stage_head(aux_hidden) if self.stage_head is not None else None
        stage_logits = stage_logits_pred if stage_logits_pred is not None else stage_logits_rule
        if self._gate_use_vision and stage_logits_pred is not None:
            stage_prob_gate = F.softmax(stage_logits_pred, dim=-1)
        else:
            stage_prob_gate = stage_prob
        probe_pred = torch.sigmoid(self.probe_head(h_probe))
        grasp_pred = torch.sigmoid(self.grasp_head(h_grasp))
        m_probe, m_grasp, m_tac = self._get_recent_tactile_masks(obs, batch_size)
        alpha_prior, b_probe, b_grasp = self._get_stage_priors(stage_prob_gate)

        gate_components = [reliability_pred, stage_prob_gate, probe_pred, grasp_pred]
        if self._gate_use_vision:
            gate_components.insert(0, vision)
        gate_input = torch.cat(gate_components, dim=-1)
        branch_raw = self._tactile_gate_mlp(gate_input)
        u_probe = branch_raw[:, 0:1]
        u_grasp = branch_raw[:, 1:2]
        eps0 = 1e-6
        logit_probe = u_probe + torch.log(b_probe + eps0) + torch.log(m_probe + eps0)
        logit_grasp = u_grasp + torch.log(b_grasp + eps0) + torch.log(m_grasp + eps0)
        # Avoid hard 0/1 saturation: use bounded sigmoid weighting instead of softmax.
        probe_share = torch.sigmoid(logit_probe - logit_grasp)
        g_probe = torch.clamp(probe_share, 0.1, 0.9)
        g_grasp = 1.0 - g_probe

        if self._alpha_correction_head is not None and self._alpha_correction_eps != 0.0:
            delta_alpha = self._alpha_correction_eps * torch.tanh(self._alpha_correction_head(gate_input))
        else:
            delta_alpha = torch.zeros_like(alpha_prior)
        alpha = torch.clamp(alpha_prior + delta_alpha, 0.0, 1.0)
        alpha = alpha * m_tac

        z_probe = h_probe
        z_grasp = h_grasp
        z_tac = g_probe * z_probe + g_grasp * z_grasp
        weighted_vision = self._inject_tactile_into_vision(vision, alpha * z_tac)

        # Preserve action-network input dimensionality; tactile influence enters through fused vision.
        direct_probe = torch.zeros_like(h_probe)
        direct_grasp = torch.zeros_like(h_grasp)
        prev_action = self._flatten_obs(obs, self.prev_action_key, batch_size)
        action_input = torch.cat([weighted_vision, direct_probe, direct_grasp, proprio, prev_action], dim=-1)
        mu = self.mu(self.action_mlp(action_input))

        set_latest_aux_predictions(
            reliability_pred,
            stage_logits,
            w_probe=g_probe,
            w_grasp=g_grasp,
            alpha=alpha,
            u_probe=u_probe,
            u_grasp=u_grasp,
            logit_probe=logit_probe,
            logit_grasp=logit_grasp,
            b_probe=b_probe,
            b_grasp=b_grasp,
            m_probe=m_probe,
            m_grasp=m_grasp,
        )

        aux_loss, reliability_loss, stage_loss, probe_loss, grasp_loss = self._build_aux_losses(
            obs=obs,
            batch_size=batch_size,
            reliability_pred=reliability_pred,
            stage_logits_pred=stage_logits_pred,
            probe_pred=probe_pred,
            grasp_pred=grasp_pred,
        )

        outputs = {
            "aux_loss": aux_loss,
            "aux_reliability_loss": reliability_loss,
            "aux_stage_loss": stage_loss,
            "aux_probe_loss": probe_loss,
            "aux_grasp_loss": grasp_loss,
            "aux_reliability_pred": reliability_pred,
            "aux_stage_logits": stage_logits,
            "aux_probe_pred": probe_pred,
            "aux_grasp_pred": grasp_pred,
            "r_v": reliability_pred,
            "s": stage_prob_gate,
            "q_probe": probe_pred,
            "q_grasp": grasp_pred,
            "g_probe": g_probe,
            "g_grasp": g_grasp,
            "alpha": alpha,
            "m_probe": m_probe,
            "m_grasp": m_grasp,
            "m_tac": m_tac,
            "h_probe": h_probe,
            "h_grasp": h_grasp,
        }
        return mu, self.log_std_parameter, outputs
