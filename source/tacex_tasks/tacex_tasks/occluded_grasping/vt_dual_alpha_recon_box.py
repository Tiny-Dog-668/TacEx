"""Dual-alpha reconstruction box variant.

This env reuses alpha-reconstruction supervision targets and pairs with a
dual-alpha policy that separately gates down/inner tactile branches.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .vt_alpha_recon_box import (
    OccludedGraspingVTAlphaReconBoxCfg,
    OccludedGraspingVTAlphaReconBoxEnv,
)


@configclass
class OccludedGraspingVTDualAlphaReconBoxCfg(OccludedGraspingVTAlphaReconBoxCfg):
    """Configuration alias for dual-alpha reconstruction task."""


class OccludedGraspingVTDualAlphaReconBoxEnv(OccludedGraspingVTAlphaReconBoxEnv):
    """Environment alias for dual-alpha reconstruction task."""

    cfg: OccludedGraspingVTDualAlphaReconBoxCfg

