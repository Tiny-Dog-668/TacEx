"""GRU VT box task aliases with an extra current-tactile feature branch."""

from __future__ import annotations

from .vt_gru_box import OccludedGraspingVTGRUBoxCfg, OccludedGraspingVTGRUBoxEnv


class OccludedGraspingVTGRUExtraTactileBoxCfg(OccludedGraspingVTGRUBoxCfg):
    """Configuration alias for GRU plus extra current tactile fusion."""


class OccludedGraspingVTGRUExtraTactileBoxEnv(OccludedGraspingVTGRUBoxEnv):
    """Environment alias for GRU plus extra current tactile fusion."""

    cfg: OccludedGraspingVTGRUExtraTactileBoxCfg
