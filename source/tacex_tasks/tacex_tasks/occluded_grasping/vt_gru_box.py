"""GRU VT box task aliases without alpha gating."""

from __future__ import annotations

from .vt_alpha_gru_box import OccludedGraspingVTAlphaGRUBoxCfg, OccludedGraspingVTAlphaGRUBoxEnv


class OccludedGraspingVTGRUBoxCfg(OccludedGraspingVTAlphaGRUBoxCfg):
    """Configuration alias for the no-alpha GRU VT task."""


class OccludedGraspingVTGRUBoxEnv(OccludedGraspingVTAlphaGRUBoxEnv):
    """Environment alias for the no-alpha GRU VT task."""

    cfg: OccludedGraspingVTGRUBoxCfg
