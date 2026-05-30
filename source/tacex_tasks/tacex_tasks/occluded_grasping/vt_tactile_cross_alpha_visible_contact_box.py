"""Tactile-cross-alpha task gated by visual visibility and tactile contact ratios."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vt_box import OccludedGraspingVTAlphaBoxCfg, OccludedGraspingVTAlphaBoxEnv, OccludedGraspingVTAlphaDownsampleBoxCfg
from .vt_dual_cross_alpha_aux_box import OccludedGraspingVTDualCrossAlphaAuxBoxEnv


def _enable_visible_contact_observations(cfg) -> None:
    cfg.observation_space = dict(cfg.observation_space)
    cfg.observation_space["visual_visible_ratio"] = 1
    cfg.observation_space["tactile_contact_ratio"] = 4

    try:
        cfg.can = cfg.can.replace(spawn=cfg.can.spawn.replace(semantic_tags=[("class", "can")]))
    except Exception:
        try:
            cfg.can.spawn.semantic_tags = [("class", "can")]
        except Exception:
            pass


@configclass
class OccludedGraspingVTTactileCrossAlphaVisibleContactBoxCfg(OccludedGraspingVTAlphaBoxCfg):
    """VT cfg exposing visible ratio and four tactile contact ratios for alpha gating."""

    aux_alpha_use_bbox3d = True
    aux_alpha_fallback_occlusion = 1.0
    aux_alpha_bbox_lazy_init = True

    def __post_init__(self):
        super().__post_init__()
        _enable_visible_contact_observations(self)

    def _post_configure_scene_object(self):
        _enable_visible_contact_observations(self)


@configclass
class OccludedGraspingVTTactileCrossAlphaVisibleContactDownsampleBoxCfg(OccludedGraspingVTAlphaDownsampleBoxCfg):
    """Downsample VT cfg exposing visible ratio and four tactile contact ratios for alpha gating."""

    aux_alpha_use_bbox3d = True
    aux_alpha_fallback_occlusion = 1.0
    aux_alpha_bbox_lazy_init = True

    def __post_init__(self):
        super().__post_init__()
        _enable_visible_contact_observations(self)

    def _post_configure_scene_object(self):
        _enable_visible_contact_observations(self)


class OccludedGraspingVTTactileCrossAlphaVisibleContactBoxEnv(OccludedGraspingVTDualCrossAlphaAuxBoxEnv):
    """Tactile-cross-alpha env exposing direct perceptual gate inputs."""

    cfg: OccludedGraspingVTTactileCrossAlphaVisibleContactBoxCfg

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = OccludedGraspingVTAlphaBoxEnv._get_observations(self)
        obs = observations["policy"]

        visual_visible_ratio = (1.0 - self._compute_occlusion_gt()).clamp(0.0, 1.0).unsqueeze(-1)
        tactile_contact_ratio = self._compute_tactile_contact_gt()
        obs["visual_visible_ratio"] = visual_visible_ratio
        obs["tactile_contact_ratio"] = tactile_contact_ratio

        log = self.extras.setdefault("log", {})
        log["gate/visual_visible_ratio_mean"] = visual_visible_ratio.mean().detach()
        log["gate/tactile_contact_ratio_mean"] = tactile_contact_ratio.mean().detach()
        return {"policy": obs}
