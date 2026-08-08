"""PandaHand-only RMA environments using cube XY and force-derived grasp state."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import configclass

from .sim2real_cube_real_alignment_rma_env import (
    _CRITIC_OBSERVATION_SPACE,
    _RMATerminalMixin,
    _make_rma_cube_cfg,
    _make_rma_cube_contact_sensor_cfg,
    _make_rma_robot_cfg,
    Sim2RealCubeRealAlignmentRMAStudentDREnv,
    Sim2RealCubeRealAlignmentRMAStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMAStudentEnv,
    Sim2RealCubeRealAlignmentRMAStudentEnvCfg,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnv,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnvCfg,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapEnv,
    Sim2RealCubeRealAlignmentRMAStudentHeatmapEnvCfg,
    Sim2RealCubeRealAlignmentRMATeacherEnv,
    Sim2RealCubeRealAlignmentRMATeacherEnvCfg,
)


class _RMAXYContractMixin:
    """Replace legacy RMA XYZ/contact labels with PandaHand XY/force inputs."""

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]
        cube_position = obs.pop("rma_cube_pos")
        obs.pop("rma_contact_state", None)
        obs["rma_cube_xy"] = cube_position[..., :2]
        # [N,2] is max force norm over both 60 Hz substeps in one 30 Hz policy step.
        obs["rma_contact_force"] = self._compute_rma_contact_forces()
        return observations

    def _compute_additional_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Skip the legacy single-finger RMA reward but preserve inherited table/DR terms.
        reward, log = super(_RMATerminalMixin, self)._compute_additional_reward()
        forces = self._compute_rma_contact_forces()
        contact = (forces >= float(self.cfg.rma_contact_force_threshold_n)).to(torch.float32)
        grasped = contact.prod(dim=-1)
        weighted_reward = float(self.cfg.rma_contact_reward_weight) * grasped
        action_rate_penalty, action_rate_log = self._compute_rma_action_rate_penalty()
        self._last_rma_contact_forces = forces.detach().clone()
        self._last_rma_contact_state = contact.detach().clone()
        self._last_rma_contact_reward = weighted_reward.detach().clone()
        self._last_rma_action_rate_penalty = action_rate_penalty.detach().clone()
        log.update({
            "reward/rma_contact": weighted_reward.mean().detach(),
            "info/rma_left_contact_fraction": contact[:, 0].mean().detach(),
            "info/rma_right_contact_fraction": contact[:, 1].mean().detach(),
            "info/rma_bilateral_contact_fraction": grasped.mean().detach(),
            "info/rma_grasped_fraction": grasped.mean().detach(),
            "info/rma_left_contact_force_n": forces[:, 0].mean().detach(),
            "info/rma_right_contact_force_n": forces[:, 1].mean().detach(),
        })
        log.update(action_rate_log)
        return reward + weighted_reward + action_rate_penalty, log


@configclass
class Sim2RealCubeRealAlignmentRMAXYTeacherEnvCfg(Sim2RealCubeRealAlignmentRMATeacherEnvCfg):
    """PandaHand privileged Teacher: cube XY plus two physical contact forces."""

    robot = _make_rma_robot_cfg()
    cube = _make_rma_cube_cfg()
    rma_cube_contact_sensor = _make_rma_cube_contact_sensor_cfg()
    rma_contact_force_threshold_n = 1.0
    rma_actor_feature_dim = 26
    rma_object_pose_components = "position_xy_only"
    rma_contact_components = "bilateral_cube_finger_force_n_gte_1N_binary"
    observation_space = {"proprio_obs": 15, "action_history": 4, "rma_cube_xy": 2, "rma_contact_force": 2, **_CRITIC_OBSERVATION_SPACE}


class Sim2RealCubeRealAlignmentRMAXYTeacherEnv(_RMAXYContractMixin, Sim2RealCubeRealAlignmentRMATeacherEnv):
    cfg: Sim2RealCubeRealAlignmentRMAXYTeacherEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAXYStudentEnvCfg(Sim2RealCubeRealAlignmentRMAStudentEnvCfg):
    """PandaHand Student: RGB predicts XY only; contact force is a runtime input."""

    robot = _make_rma_robot_cfg()
    cube = _make_rma_cube_cfg()
    rma_cube_contact_sensor = _make_rma_cube_contact_sensor_cfg()
    rma_contact_force_threshold_n = 1.0
    rma_actor_feature_dim = 26
    rma_object_pose_components = "position_xy_only"
    rma_contact_components = "bilateral_cube_finger_force_n_gte_1N_binary"
    observation_space = {
        "proprio_obs": 15, "action_history": 4,
        "wrist_rgb": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        "rma_cube_xy": 2, "rma_contact_force": 2, **_CRITIC_OBSERVATION_SPACE,
    }


class Sim2RealCubeRealAlignmentRMAXYStudentEnv(_RMAXYContractMixin, Sim2RealCubeRealAlignmentRMAStudentEnv):
    cfg: Sim2RealCubeRealAlignmentRMAXYStudentEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAXYStudentHeatmapEnvCfg(Sim2RealCubeRealAlignmentRMAXYStudentEnvCfg, Sim2RealCubeRealAlignmentRMAStudentHeatmapEnvCfg):
    rma_heatmap_supervision_enabled = True
    rma_heatmap_loss_weight = 1.0


class Sim2RealCubeRealAlignmentRMAXYStudentHeatmapEnv(Sim2RealCubeRealAlignmentRMAXYStudentEnv):
    cfg: Sim2RealCubeRealAlignmentRMAXYStudentHeatmapEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAXYStudentDREnvCfg(Sim2RealCubeRealAlignmentRMAStudentDREnvCfg):
    """Full-strength visual DR with the same PandaHand XY/force Actor contract."""

    robot = _make_rma_robot_cfg()
    cube = _make_rma_cube_cfg()
    rma_cube_contact_sensor = _make_rma_cube_contact_sensor_cfg()
    rma_contact_force_threshold_n = 1.0
    rma_actor_feature_dim = 26
    rma_object_pose_components = "position_xy_only"
    rma_contact_components = "bilateral_cube_finger_force_n_gte_1N_binary"
    observation_space = Sim2RealCubeRealAlignmentRMAXYStudentEnvCfg().observation_space


class Sim2RealCubeRealAlignmentRMAXYStudentDREnv(_RMAXYContractMixin, Sim2RealCubeRealAlignmentRMAStudentDREnv):
    cfg: Sim2RealCubeRealAlignmentRMAXYStudentDREnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAXYStudentHeatmapDREnvCfg(Sim2RealCubeRealAlignmentRMAXYStudentDREnvCfg, Sim2RealCubeRealAlignmentRMAStudentHeatmapDREnvCfg):
    rma_heatmap_supervision_enabled = True
    rma_heatmap_loss_weight = 1.0


class Sim2RealCubeRealAlignmentRMAXYStudentHeatmapDREnv(Sim2RealCubeRealAlignmentRMAXYStudentDREnv):
    cfg: Sim2RealCubeRealAlignmentRMAXYStudentHeatmapDREnvCfg
