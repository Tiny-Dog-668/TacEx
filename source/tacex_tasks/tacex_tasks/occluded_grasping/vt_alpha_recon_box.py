"""Alpha-reconstruction box variant with multi-task supervision targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from .vt_reliability_stage_box import (
    OccludedGraspingVTReliabilityStageBoxCfg,
    OccludedGraspingVTReliabilityStageBoxEnv,
)


@configclass
class OccludedGraspingVTAlphaReconBoxCfg(OccludedGraspingVTReliabilityStageBoxCfg):
    """Configuration for alpha task-head reconstruction supervision."""

    object_side_center_threshold = 0.015

    def __post_init__(self):
        super().__post_init__()

        self.observation_space = dict(self.observation_space)
        self.observation_space["aux_occlusion_gt"] = 1
        self.observation_space["aux_down_contact_gt"] = 2
        self.observation_space["aux_object_side_gt"] = 3
        self.observation_space["aux_inner_contact_gt"] = 3


class OccludedGraspingVTAlphaReconBoxEnv(OccludedGraspingVTReliabilityStageBoxEnv):
    """Reliability-stage env extended with alpha task-head GT signals."""

    cfg: OccludedGraspingVTAlphaReconBoxCfg

    def _compute_object_side_onehot_gt(self) -> torch.Tensor:
        hand_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos_w, ee_quat_w = math_utils.combine_frame_transforms(
            hand_pos_w, hand_quat_w, self._offset_pos, self._offset_rot
        )
        can_pos_w = self._can.data.root_pos_w

        rel_pos_w = can_pos_w - ee_pos_w
        rot_w_from_ee = math_utils.matrix_from_quat(ee_quat_w)
        rot_ee_from_w = rot_w_from_ee.transpose(1, 2)
        rel_pos_ee = torch.bmm(rot_ee_from_w, rel_pos_w.unsqueeze(-1)).squeeze(-1)
        rel_lateral = rel_pos_ee[:, 1]

        center_threshold = float(getattr(self.cfg, "object_side_center_threshold", 0.015))
        side_index = torch.full((self.num_envs,), 1, dtype=torch.long, device=self.device)
        side_index = torch.where(rel_lateral > center_threshold, torch.full_like(side_index, 0), side_index)
        side_index = torch.where(rel_lateral < -center_threshold, torch.full_like(side_index, 2), side_index)
        return F.one_hot(side_index, num_classes=3).to(dtype=torch.float32)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        visibility = obs.get("aux_reliability_gt")
        if visibility is None:
            visibility = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        else:
            visibility = visibility.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, 1)
        occlusion_gt = (1.0 - visibility).clamp(0.0, 1.0)

        tactile_ratio = self._latest_tactile_over_thresh_ratio_by_sensor.to(device=self.device, dtype=torch.float32)
        if tactile_ratio.ndim == 1:
            tactile_ratio = tactile_ratio.view(self.num_envs, 4)

        down_contact_gt = tactile_ratio[:, 2:4].clamp(0.0, 1.0)
        inner_left = tactile_ratio[:, 0:1].clamp(0.0, 1.0)
        inner_right = tactile_ratio[:, 1:2].clamp(0.0, 1.0)
        inner_diff = torch.abs(inner_left - inner_right)
        inner_contact_gt = torch.cat([inner_left, inner_right, inner_diff], dim=-1)
        object_side_gt = self._compute_object_side_onehot_gt()

        obs["aux_occlusion_gt"] = occlusion_gt
        obs["aux_down_contact_gt"] = down_contact_gt
        obs["aux_object_side_gt"] = object_side_gt
        obs["aux_inner_contact_gt"] = inner_contact_gt

        log = self.extras.setdefault("log", {})
        log["aux/occlusion_gt_mean"] = occlusion_gt.mean().detach()
        log["aux/down_contact_left_gt_mean"] = down_contact_gt[:, 0].mean().detach()
        log["aux/down_contact_right_gt_mean"] = down_contact_gt[:, 1].mean().detach()
        log["aux/inner_contact_left_gt_mean"] = inner_contact_gt[:, 0].mean().detach()
        log["aux/inner_contact_right_gt_mean"] = inner_contact_gt[:, 1].mean().detach()
        log["aux/inner_contact_diff_gt_mean"] = inner_contact_gt[:, 2].mean().detach()
        log["aux/object_side_left_gt_mean"] = object_side_gt[:, 0].mean().detach()
        log["aux/object_side_center_gt_mean"] = object_side_gt[:, 1].mean().detach()
        log["aux/object_side_right_gt_mean"] = object_side_gt[:, 2].mean().detach()

        return {"policy": obs}
