"""Convex-GRU-beta gated VT box task aliases."""

from __future__ import annotations

from .vt_convex_gru_box import OccludedGraspingVTConvexGRUBoxCfg, OccludedGraspingVTConvexGRUBoxEnv


class OccludedGraspingVTConvexGRUBetaBoxCfg(OccludedGraspingVTConvexGRUBoxCfg):
    """Configuration alias for the Convex-GRU-beta VT task."""


class OccludedGraspingVTConvexGRUBetaBoxEnv(OccludedGraspingVTConvexGRUBoxEnv):
    """Environment alias for the Convex-GRU-beta VT task."""

    cfg: OccludedGraspingVTConvexGRUBetaBoxCfg
