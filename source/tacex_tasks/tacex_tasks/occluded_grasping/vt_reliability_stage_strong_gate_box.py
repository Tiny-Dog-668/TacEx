"""Reliability/stage box variant with 5-pair tactile DINO observations for strong gating."""

from __future__ import annotations

from isaaclab.utils import configclass

from .vt_reliability_stage_box import (
    OccludedGraspingVTReliabilityStageBoxCfg,
    OccludedGraspingVTReliabilityStageBoxEnv,
)


@configclass
class OccludedGraspingVTReliabilityStageStrongGateBoxCfg(OccludedGraspingVTReliabilityStageBoxCfg):
    """Keep auxiliary labels while exposing 5 temporal tactile pairs per sensor."""

    def __post_init__(self):
        super().__post_init__()

        self.tactile_encoder_type = "dino"
        self.tactile_dino_total_frames = 2

        sync_tactile_obs = getattr(self, "_sync_tactile_observation_space", None)
        if callable(sync_tactile_obs):
            sync_tactile_obs()

        tactile_obs_dim = int(getattr(self, "_compute_tactile_obs_dim")())
        self.observation_space = dict(self.observation_space)
        for old_key, new_key in getattr(self, "tactile_obs_rename_pairs", ()):
            if old_key in self.observation_space:
                self.observation_space[old_key] = tactile_obs_dim
                self.observation_space[new_key] = tactile_obs_dim


class OccludedGraspingVTReliabilityStageStrongGateBoxEnv(OccludedGraspingVTReliabilityStageBoxEnv):
    """Aux-supervised box task with temporal tactile observations for strong-gate policies."""

    cfg: OccludedGraspingVTReliabilityStageStrongGateBoxCfg

