"""Third-view + four-tactile environment with fused vision-tactile policy feature."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vt_box import (
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
)


@configclass
class OccludedGraspingVTConcatBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for policy observations using fused third-view and tactile features."""

    observation_space = {
        "proprio_obs": 18,
        "vision_tactile_concat": 1536,
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
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        tactile_obs_dim = int(self._compute_tactile_obs_dim())
        # third(512) + four tactile streams
        self.observation_space = dict(self.observation_space)
        self.observation_space["vision_tactile_concat"] = 512 + 4 * tactile_obs_dim


class OccludedGraspingVTConcatBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Occluded grasping environment with policy feature fused inside the environment."""

    cfg: OccludedGraspingVTConcatBoxCfg

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        fused = torch.cat(
            [
                obs["third_resnet"],
                obs["tactile_left_depth_resnet"],
                obs["tactile_right_depth_resnet"],
                obs["tactile_left_down_depth_resnet"],
                obs["tactile_right_down_depth_resnet"],
            ],
            dim=-1,
        )

        obs["vision_tactile_concat"] = fused
        obs.pop("third_resnet", None)
        obs.pop("tactile_left_depth_resnet", None)
        obs.pop("tactile_right_depth_resnet", None)
        obs.pop("tactile_left_down_depth_resnet", None)
        obs.pop("tactile_right_down_depth_resnet", None)
        return observations
