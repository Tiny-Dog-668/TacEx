"""Third-view + proprio policy variant with concatenated four-tactile CNN features."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vt_box import (
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
)


@configclass
class OccludedGraspingThirdViewTactileConcatBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for third-view + concatenated four-tactile policy observations."""

    tactile_encoder_type = "cnn"

    observation_space = {
        "proprio_obs": 18,
        "third_resnet": 512,
        "tactile_concat": 1024,
        "critic_can_pos": 3,
        "critic_can_quat": 4,
        "critic_can_lin_vel": 3,
        "critic_can_ang_vel": 3,
        "critic_gripper_pos": 3,
        "critic_gripper_quat": 4,
        "critic_gripper_lin_vel": 3,
        "critic_gripper_ang_vel": 3,
        "critic_target_pos": 3,
        "critic_target_distance": 1,
    }

    def __post_init__(self):
        # Force the four tactile streams to use the lightweight shared CNN encoder.
        self.tactile_encoder_type = "cnn"
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        tactile_obs_dim = int(self._compute_tactile_obs_dim())
        self.observation_space = dict(self.observation_space)
        self.observation_space["tactile_concat"] = 4 * tactile_obs_dim


class OccludedGraspingThirdViewTactileConcatBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Occluded grasping env exposing third-view and four-tactile CNN features."""

    cfg: OccludedGraspingThirdViewTactileConcatBoxCfg

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        tactile_concat = torch.cat(
            [
                obs["tactile_left_depth_resnet"],
                obs["tactile_right_depth_resnet"],
                obs["tactile_left_down_depth_resnet"],
                obs["tactile_right_down_depth_resnet"],
            ],
            dim=-1,
        )

        obs["tactile_concat"] = tactile_concat
        obs.pop("tactile_left_depth_resnet", None)
        obs.pop("tactile_right_depth_resnet", None)
        obs.pop("tactile_left_down_depth_resnet", None)
        obs.pop("tactile_right_down_depth_resnet", None)
        return observations
