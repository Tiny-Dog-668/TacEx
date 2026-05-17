"""Occluded grasping variant with per-sensor hard tactile gating from depth contact cues."""

from __future__ import annotations

from typing import Dict

import torch

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
class OccludedGraspingVTHardGateBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """VT box config that hard-gates each tactile feature with baseline-depth contact thresholds."""

    tactile_hard_gate_enable = True
    tactile_baseline_diff_threshold = 0.05
    tactile_contact_area_threshold = 0.01

    def __post_init__(self):
        _maybe_call_parent_post_init(self)
        for attr_name in _TACTILE_SENSOR_ATTRS.values():
            sensor_cfg = getattr(self, attr_name).replace()
            sensor_cfg.data_types = ["tactile_rgb", "camera_depth"]
            setattr(self, attr_name, sensor_cfg)


class OccludedGraspingVTHardGateBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Hard-gate each tactile branch by per-sensor depth-based contact detection."""

    cfg: OccludedGraspingVTHardGateBoxCfg

    def __init__(self, cfg: OccludedGraspingVTHardGateBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._tactile_baseline_depth: dict[str, torch.Tensor | None] = {
            key: None for key in _TACTILE_SENSOR_ATTRS
        }
        self._pending_tactile_baseline_refresh = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._tactile_gate_any = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._latest_tactile_contact_ratios = {
            key: torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            for key in _TACTILE_SENSOR_ATTRS
        }

    def _get_tactile_sensor_depths(self) -> dict[str, torch.Tensor | None]:
        sensor_depths = {}
        for key, attr_name in _TACTILE_SENSOR_ATTRS.items():
            sensor = getattr(self, attr_name, None)
            sensor_depths[key] = sensor.data.output.get("camera_depth") if sensor is not None else None
        return sensor_depths

    def _normalize_depth_01(self, depth_tensor: torch.Tensor | None) -> torch.Tensor | None:
        if depth_tensor is None:
            return None
        depth = depth_tensor.to(device=self.device, dtype=torch.float32)
        max_val = depth.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            depth = depth / 255.0
        depth = depth.clamp(0.0, 1.0)
        if depth.ndim == 3:
            depth = depth.unsqueeze(-1)
        return depth

    def _maybe_refresh_tactile_baselines(self, sensor_depths: dict[str, torch.Tensor | None]) -> None:
        pending = self._pending_tactile_baseline_refresh
        if not torch.any(pending):
            return

        pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
        if pending_ids.numel() == 0:
            return

        captured_any = False
        for sensor_name, depth_tensor in sensor_depths.items():
            depth_01 = self._normalize_depth_01(depth_tensor)
            if depth_01 is None:
                continue

            baseline = self._tactile_baseline_depth.get(sensor_name)
            if baseline is None or baseline.shape != depth_01.shape:
                baseline = torch.zeros_like(depth_01)
                self._tactile_baseline_depth[sensor_name] = baseline
            baseline[pending_ids] = depth_01[pending_ids]
            captured_any = True

        if captured_any:
            pending[pending_ids] = False

    def _contact_ratio_from_depth(self, sensor_name: str, depth_tensor: torch.Tensor | None) -> torch.Tensor:
        depth = self._normalize_depth_01(depth_tensor)
        baseline = self._tactile_baseline_depth.get(sensor_name)
        if depth is None or baseline is None:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        diff = torch.abs(depth - baseline)
        diff_threshold = float(getattr(self.cfg, "tactile_baseline_diff_threshold", 0.05))
        return (diff >= diff_threshold).to(torch.float32).mean(dim=(1, 2, 3), keepdim=False)

    def _contact_bits_from_ratios(self, contact_ratios: dict[str, torch.Tensor]) -> torch.Tensor:
        area_threshold = float(getattr(self.cfg, "tactile_contact_area_threshold", 0.01))
        return torch.cat(
            [
                (contact_ratios[key] > area_threshold).to(torch.float32).unsqueeze(-1)
                for key in _TACTILE_SENSOR_ATTRS
            ],
            dim=-1,
        )

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        gate_enabled = bool(getattr(self.cfg, "tactile_hard_gate_enable", True))
        if gate_enabled:
            sensor_depths = self._get_tactile_sensor_depths()
            self._maybe_refresh_tactile_baselines(sensor_depths)
            contact_ratios = {
                key: self._contact_ratio_from_depth(key, sensor_depths[key]) for key in _TACTILE_SENSOR_ATTRS
            }
            contact_bits = self._contact_bits_from_ratios(contact_ratios)
        else:
            contact_ratios = {
                key: torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
                for key in _TACTILE_SENSOR_ATTRS
            }
            contact_bits = torch.ones((self.num_envs, len(_TACTILE_SENSOR_ATTRS)), dtype=torch.float32, device=self.device)

        for sensor_idx, (sensor_name, feature_key) in enumerate(_TACTILE_SENSOR_FEATURE_KEYS.items()):
            if feature_key not in obs:
                continue
            gate = contact_bits[:, sensor_idx : sensor_idx + 1].to(dtype=obs[feature_key].dtype)
            obs[feature_key] = obs[feature_key] * gate

        tactile_valid = contact_bits.max(dim=-1).values > 0.5
        self._tactile_gate_any = tactile_valid
        self._latest_tactile_contact_ratios = contact_ratios

        log = self.extras.setdefault("log", {})
        log["info/tactile_valid_ratio"] = tactile_valid.to(dtype=torch.float32).mean().detach()
        log["info/tactile_baseline_diff_threshold"] = torch.tensor(
            float(getattr(self.cfg, "tactile_baseline_diff_threshold", 0.05)), device=self.device
        )
        log["info/tactile_contact_area_threshold"] = torch.tensor(
            float(getattr(self.cfg, "tactile_contact_area_threshold", 0.01)), device=self.device
        )
        for key, ratio in contact_ratios.items():
            log[f"info/tactile_ratio_{key}"] = ratio.mean().detach()

        return observations

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        self._pending_tactile_baseline_refresh[env_ids] = True
        self._tactile_gate_any[env_ids] = False
        for key in _TACTILE_SENSOR_ATTRS:
            baseline = self._tactile_baseline_depth[key]
            if baseline is not None:
                baseline[env_ids] = 0.0
            self._latest_tactile_contact_ratios[key][env_ids] = 0.0
