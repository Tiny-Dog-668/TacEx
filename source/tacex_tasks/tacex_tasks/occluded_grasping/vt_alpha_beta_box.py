"""Alpha-beta gated VT box task aliases."""

from __future__ import annotations

from .vt_box import OccludedGraspingVTAlphaBoxCfg, OccludedGraspingVTAlphaBoxEnv


class OccludedGraspingVTAlphaBetaBoxCfg(OccludedGraspingVTAlphaBoxCfg):
    """Configuration alias for the alpha-beta gated VT task."""


class OccludedGraspingVTAlphaBetaBoxEnv(OccludedGraspingVTAlphaBoxEnv):
    """Environment alias for the alpha-beta gated VT task."""

    cfg: OccludedGraspingVTAlphaBetaBoxCfg
