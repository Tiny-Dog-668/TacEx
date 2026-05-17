"""Alpha-GRU-beta gated VT box task aliases."""

from __future__ import annotations

from .vt_alpha_gru_box import OccludedGraspingVTAlphaGRUBoxCfg, OccludedGraspingVTAlphaGRUBoxEnv


class OccludedGraspingVTAlphaGRUBetaBoxCfg(OccludedGraspingVTAlphaGRUBoxCfg):
    """Configuration alias for the alpha-GRU-beta gated VT task."""


class OccludedGraspingVTAlphaGRUBetaBoxEnv(OccludedGraspingVTAlphaGRUBoxEnv):
    """Environment alias for the alpha-GRU-beta gated VT task."""

    cfg: OccludedGraspingVTAlphaGRUBetaBoxCfg
