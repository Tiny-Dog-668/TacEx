"""Occluded grasping variant with tactile validity gating for alpha fusion."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from isaaclab.utils import configclass

from .vt_box import (
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
)

_TACTILE_SENSOR_FEATURE_KEYS: Dict[str, str] = {
    "left": "tactile_left_depth_resnet",
    "right": "tactile_right_depth_resnet",
    "left_down": "tactile_left_down_depth_resnet",
    "right_down": "tactile_right_down_depth_resnet",
}

_TACTILE_SENSOR_ATTRS: Dict[str, str] = {
    "left": "gsmini_left",
    "right": "gsmini_right",
    "left_down": "gsmini_left_down",
    "right_down": "gsmini_right_down",
}


def _maybe_call_parent_post_init(instance) -> None:
    parent_post_init = getattr(super(instance.__class__, instance), "__post_init__", None)
    if callable(parent_post_init):
        parent_post_init()


@configclass
class OccludedGraspingVTGateAlphaBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for occluded grasping with tactile validity gating."""

    tactile_gate_enable = True
    tactile_gate_metric = "l1"
    tactile_gate_diff_threshold = 0.02
    tactile_gate_baseline_delay_steps = 4

    def __post_init__(self):
        _maybe_call_parent_post_init(self)
        self.observation_space = dict(self.observation_space)
        self.observation_space["tactile_valid"] = 1


class OccludedGraspingVTGateAlphaBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Occluded grasping env with per-sensor tactile baselines and a fused tactile-valid flag."""

    cfg: OccludedGraspingVTGateAlphaBoxCfg

    def __init__(self, cfg: OccludedGraspingVTGateAlphaBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        baseline_delay = max(int(getattr(self.cfg, "tactile_gate_baseline_delay_steps", 0)), 0)

        self._tactile_gate_refs = {
            key: torch.zeros((self.num_envs, 3, target_h, target_w), device=self.device, dtype=torch.float32)
            for key in _TACTILE_SENSOR_ATTRS
        }
        self._tactile_gate_pending = {
            key: torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
            for key in _TACTILE_SENSOR_ATTRS
        }
        self._tactile_gate_delay = {
            key: torch.full((self.num_envs,), baseline_delay, device=self.device, dtype=torch.long)
            for key in _TACTILE_SENSOR_ATTRS
        }
        self._tactile_gate_any = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._tactile_gate_scores = {
            key: torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            for key in _TACTILE_SENSOR_ATTRS
        }

    def _prep_tactile_rgb_01(self, rgb_tensor: torch.Tensor | None) -> torch.Tensor:
        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        if rgb_tensor is None:
            return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
        x = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = x.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x

    def _compute_tactile_gate_score(self, current: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        metric = str(getattr(self.cfg, "tactile_gate_metric", "l1")).strip().lower()
        diff = current - baseline
        if metric in {"l1", "mae"}:
            return diff.abs().mean(dim=(1, 2, 3))
        if metric in {"l2", "rmse"}:
            return torch.sqrt(diff.square().mean(dim=(1, 2, 3)) + 1.0e-12)
        if metric in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss(current, baseline, reduction="none").mean(dim=(1, 2, 3))
        raise ValueError(f"Unsupported tactile_gate_metric='{metric}'. Use 'l1', 'l2', or 'smooth_l1'.")

    def _get_tactile_gate_masks(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        gate_masks: Dict[str, torch.Tensor] = {}
        gate_scores: Dict[str, torch.Tensor] = {}
        threshold = float(getattr(self.cfg, "tactile_gate_diff_threshold", 0.02))
        gate_enabled = bool(getattr(self.cfg, "tactile_gate_enable", True))
        cached_frames = getattr(self, "_latest_tactile_rgb_01", {})

        for key, attr_name in _TACTILE_SENSOR_ATTRS.items():
            frame = cached_frames.get(key)
            has_sensor = frame is not None
            if not has_sensor:
                sensor = getattr(self, attr_name, None)
                raw_rgb = sensor.data.output.get("tactile_rgb") if sensor is not None else None
                has_sensor = raw_rgb is not None
            gate = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
            score = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

            if has_sensor:
                if not gate_enabled:
                    gate_masks[key] = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
                    gate_scores[key] = score
                    continue

                if frame is None:
                    frame = self._prep_tactile_rgb_01(raw_rgb)
                pending = self._tactile_gate_pending[key]
                delay = self._tactile_gate_delay[key]

                countdown_mask = pending & (delay > 0)
                if torch.any(countdown_mask):
                    delay[countdown_mask] -= 1

                capture_mask = pending & (delay <= 0)
                if torch.any(capture_mask):
                    self._tactile_gate_refs[key][capture_mask] = frame[capture_mask].detach()
                    pending[capture_mask] = False

                valid_mask = ~pending
                if torch.any(valid_mask):
                    current_score = self._compute_tactile_gate_score(frame, self._tactile_gate_refs[key]).detach()
                    score = torch.where(valid_mask, current_score, score)
                    gate = valid_mask & (score >= threshold)

            gate_masks[key] = gate
            gate_scores[key] = score

        return gate_masks, gate_scores

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs_dict = super()._get_observations()
        obs = obs_dict["policy"]

        gate_masks, gate_scores = self._get_tactile_gate_masks()
        tactile_valid = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        for key, feature_key in _TACTILE_SENSOR_FEATURE_KEYS.items():
            gate = gate_masks[key]
            tactile_valid |= gate
            if feature_key in obs:
                obs[feature_key] = obs[feature_key] * gate.to(dtype=obs[feature_key].dtype).unsqueeze(-1)

        self._tactile_gate_any = tactile_valid
        self._tactile_gate_scores = gate_scores
        obs["tactile_valid"] = tactile_valid.to(dtype=torch.float32).unsqueeze(-1)

        log = self.extras.setdefault("log", {})
        log["info/tactile_valid_ratio"] = tactile_valid.to(dtype=torch.float32).mean().detach()
        log["info/tactile_gate_threshold"] = torch.tensor(
            float(getattr(self.cfg, "tactile_gate_diff_threshold", 0.02)), device=self.device
        )
        for key, score in gate_scores.items():
            log[f"info/tactile_score_{key}"] = score.mean().detach()

        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        baseline_delay = max(int(getattr(self.cfg, "tactile_gate_baseline_delay_steps", 0)), 0)
        self._tactile_gate_any[env_ids] = False
        for key in _TACTILE_SENSOR_ATTRS:
            self._tactile_gate_refs[key][env_ids] = 0.0
            self._tactile_gate_pending[key][env_ids] = True
            self._tactile_gate_delay[key][env_ids] = baseline_delay
            self._tactile_gate_scores[key][env_ids] = 0.0
