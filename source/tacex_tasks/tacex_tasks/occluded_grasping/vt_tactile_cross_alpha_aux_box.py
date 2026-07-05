"""Tactile-cross-alpha aux task using XY-predicted occlusion instead of tiled-camera bbox."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vt_tactile_cross_alpha_visual_predict_visible_box import (
    OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleBoxEnv,
    OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleDownsampleBoxCfg,
    _enable_predict_visible_observations,
)


def _enable_predict_occlusion_aux_observations(cfg) -> None:
    _enable_predict_visible_observations(cfg)
    cfg.observation_space = dict(cfg.observation_space)
    cfg.observation_space["aux_occlusion_gt"] = 1
    cfg.observation_space["aux_tactile_contact_gt"] = 4


@configclass
class OccludedGraspingVTTactileCrossAlphaAuxDownsampleBoxCfg(
    OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleDownsampleBoxCfg
):
    """Downsample VT cfg exposing predictor occlusion and tactile contact ratios for alpha."""

    def __post_init__(self):
        super().__post_init__()
        _enable_predict_occlusion_aux_observations(self)

    def _post_configure_scene_object(self):
        _enable_predict_occlusion_aux_observations(self)


class OccludedGraspingVTTactileCrossAlphaAuxBoxEnv(
    OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleBoxEnv
):
    """Append predictor-derived occlusion and tactile contact ratios for alpha gating."""

    cfg: OccludedGraspingVTTactileCrossAlphaAuxDownsampleBoxCfg

    def __init__(
        self,
        cfg: OccludedGraspingVTTactileCrossAlphaAuxDownsampleBoxCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        fallback_occlusion = 1.0 - float(getattr(self.cfg, "pseudo_visible_fallback_ratio", 1.0))
        self._latest_aux_occlusion_gt = torch.full(
            (self.num_envs, 1),
            fallback_occlusion,
            dtype=torch.float32,
            device=self.device,
        )
        self._latest_aux_tactile_contact_gt = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        pseudo_visible_ratio = obs["pseudo_visible_ratio"].clamp(0.0, 1.0)
        tactile_contact_ratio = obs["tactile_contact_ratio"].clamp(0.0, 1.0)
        occlusion = (1.0 - pseudo_visible_ratio).clamp(0.0, 1.0)

        obs["aux_occlusion_gt"] = occlusion
        obs["aux_tactile_contact_gt"] = tactile_contact_ratio
        self._latest_aux_occlusion_gt = occlusion
        self._latest_aux_tactile_contact_gt = tactile_contact_ratio

        log = self.extras.setdefault("log", {})
        log["aux/occlusion_predictor_mean"] = occlusion.mean().detach()
        log["aux/tactile_contact_gt_mean"] = tactile_contact_ratio.mean().detach()
        return observations

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        fallback_occlusion = 1.0 - float(getattr(self.cfg, "pseudo_visible_fallback_ratio", 1.0))
        self._latest_aux_occlusion_gt[env_ids] = fallback_occlusion
        self._latest_aux_tactile_contact_gt[env_ids] = 0.0
