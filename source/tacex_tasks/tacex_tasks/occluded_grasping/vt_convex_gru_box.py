"""Convex-GRU gated VT box task aliases."""

from __future__ import annotations

from .vt_alpha_gru_box import OccludedGraspingVTAlphaGRUBoxCfg, OccludedGraspingVTAlphaGRUBoxEnv


class OccludedGraspingVTConvexGRUBoxCfg(OccludedGraspingVTAlphaGRUBoxCfg):
    """Configuration alias for the Convex-GRU VT task."""


class OccludedGraspingVTConvexGRUBoxEnv(OccludedGraspingVTAlphaGRUBoxEnv):
    """Environment alias for the Convex-GRU VT task."""

    cfg: OccludedGraspingVTConvexGRUBoxCfg
