"""VT pairwise tactile projection box task aliases."""

from __future__ import annotations

from .vt_box import OccludedGraspingVisionFourTactileBoxCfg, OccludedGraspingVisionFourTactileBoxEnv


class OccludedGraspingVTPairBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration alias for VT with pairwise tactile projections."""


class OccludedGraspingVTPairBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Environment alias for VT with pairwise tactile projections."""

    cfg: OccludedGraspingVTPairBoxCfg
