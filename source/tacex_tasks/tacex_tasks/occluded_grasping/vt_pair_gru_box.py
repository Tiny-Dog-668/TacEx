"""VT-Pair-GRU box task aliases."""

from __future__ import annotations

from .vt_alpha_gru_box import OccludedGraspingVTAlphaGRUBoxCfg, OccludedGraspingVTAlphaGRUBoxEnv


class OccludedGraspingVTPairGRUBoxCfg(OccludedGraspingVTAlphaGRUBoxCfg):
    """Configuration alias for VT-Pair with tactile GRU temporal windows."""


class OccludedGraspingVTPairGRUBoxEnv(OccludedGraspingVTAlphaGRUBoxEnv):
    """Environment alias for VT-Pair with tactile GRU temporal windows."""

    cfg: OccludedGraspingVTPairGRUBoxCfg
