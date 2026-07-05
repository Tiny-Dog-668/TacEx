"""Occluded grasping baseline with four tactile feature streams and proprioception only."""

from __future__ import annotations

from isaaclab.utils import configclass

from .vt_box import OccludedGraspingVisionFourTactileBoxCfg, OccludedGraspingVisionFourTactileBoxEnv


@configclass
class OccludedGraspingTactileProprioBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for a tactile-proprioception actor baseline.

    The simulator scene and reward logic remain identical to the full VT task, but
    the policy observation removes the third-person visual feature.
    """

    observation_space = {
        "proprio_obs": 18,
        "tactile_left_depth_resnet": 256,
        "tactile_right_depth_resnet": 256,
        "tactile_left_down_depth_resnet": 256,
        "tactile_right_down_depth_resnet": 256,
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


class OccludedGraspingTactileProprioBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Occluded grasping environment exposing only tactile features and proprioception to the actor."""

    cfg: OccludedGraspingTactileProprioBoxCfg

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        observations["policy"].pop("third_resnet", None)
        return observations
