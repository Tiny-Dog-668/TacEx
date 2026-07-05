"""GelFusion-style occluded grasping task variants.

This module keeps the base grasping dynamics unchanged and only adds the
low-dimensional tactile dynamics used by GelFusion: per-sensor binary frame
difference mean and variance.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vt_box import (
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
    OccludedGraspingVisionFourTactileDownsampleBoxCfg,
)


def _enable_gelfusion_observations(cfg) -> None:
    cfg.observation_space = dict(cfg.observation_space)
    cfg.observation_space["tactile_dynamic_stats"] = 8


@configclass
class OccludedGraspingVTGelFusionBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """VT task exposing GelFusion-style tactile dynamic statistics."""

    tactile_dynamic_diff_threshold = 0.05
    tactile_encoder_type = "resnet"

    def __post_init__(self):
        super().__post_init__()
        _enable_gelfusion_observations(self)

    def _post_configure_scene_object(self):
        _enable_gelfusion_observations(self)


@configclass
class OccludedGraspingVTGelFusionDownsampleBoxCfg(OccludedGraspingVisionFourTactileDownsampleBoxCfg):
    """Visual-downsample VT task with GelFusion-style tactile dynamics."""

    tactile_dynamic_diff_threshold = 0.05
    tactile_encoder_type = "resnet"

    def __post_init__(self):
        super().__post_init__()
        _enable_gelfusion_observations(self)

    def _post_configure_scene_object(self):
        _enable_gelfusion_observations(self)


class OccludedGraspingVTGelFusionBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Append per-sensor tactile frame-difference mean/variance observations.

    Output shape:
      tactile_dynamic_stats: [N, 8], ordered as
      left_mean, left_var, right_mean, right_var,
      left_down_mean, left_down_var, right_down_mean, right_down_var.
    """

    cfg: OccludedGraspingVTGelFusionBoxCfg

    def __init__(self, cfg: OccludedGraspingVTGelFusionBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._gelfusion_prev_tactile_rgb = {
            key: torch.zeros_like(value) for key, value in self._latest_tactile_rgb_01.items()
        }
        self._gelfusion_pending_prev_refresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._latest_tactile_dynamic_stats = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)

    def _compute_tactile_dynamic_stats(self) -> torch.Tensor:
        pending = self._gelfusion_pending_prev_refresh
        threshold = float(getattr(self.cfg, "tactile_dynamic_diff_threshold", 0.05))
        stats = []

        for key in self._tactile_sensor_keys:
            current = self._latest_tactile_rgb_01[key].to(device=self.device, dtype=torch.float32)
            previous = self._gelfusion_prev_tactile_rgb[key]

            if torch.any(pending):
                pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
                previous[pending_ids] = current[pending_ids]

            diff = torch.abs(current - previous).mean(dim=1)
            binary_diff = (diff >= threshold).to(torch.float32)
            mean = binary_diff.mean(dim=(1, 2), keepdim=False)
            var = binary_diff.var(dim=(1, 2), unbiased=False, keepdim=False)
            stats.extend([mean, var])

        if torch.any(pending):
            pending[:] = False
        for key in self._tactile_sensor_keys:
            self._gelfusion_prev_tactile_rgb[key] = self._latest_tactile_rgb_01[key].detach().clone()

        return torch.stack(stats, dim=-1)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]
        dynamic_stats = self._compute_tactile_dynamic_stats()
        obs["tactile_dynamic_stats"] = dynamic_stats
        self._latest_tactile_dynamic_stats = dynamic_stats

        log = self.extras.setdefault("log", {})
        log["info/tactile_dynamic_mean"] = dynamic_stats[:, 0::2].mean().detach()
        log["info/tactile_dynamic_var"] = dynamic_stats[:, 1::2].mean().detach()
        return observations

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        self._gelfusion_pending_prev_refresh[env_ids] = True
        for key in self._tactile_sensor_keys:
            self._gelfusion_prev_tactile_rgb[key][env_ids] = 0.0
        self._latest_tactile_dynamic_stats[env_ids] = 0.0
