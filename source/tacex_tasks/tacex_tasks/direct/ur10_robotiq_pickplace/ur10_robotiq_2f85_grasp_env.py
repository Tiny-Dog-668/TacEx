# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_apply

from .ur10_robotiq_2f85_grasp_env_cfg import UR10Robotiq2F85GraspEnvCfg
from .ur10_robotiq_pick_place_env import UR10RobotiqPickPlaceEnv


class UR10Robotiq2F85GraspEnv(UR10RobotiqPickPlaceEnv):
    cfg: UR10Robotiq2F85GraspEnvCfg

    def __init__(self, cfg: UR10Robotiq2F85GraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._success_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._object_local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self._lift_upright_cos_threshold = math.cos(math.radians(float(self.cfg.lift_upright_tilt_threshold_deg)))

    def _compute_object_upright_cos(self, object_quat: torch.Tensor) -> torch.Tensor:
        local_z_axis = self._object_local_z_axis[: object_quat.shape[0]]
        object_up_axis = quat_apply(object_quat, local_z_axis)
        return torch.clamp(object_up_axis[:, 2], min=-1.0, max=1.0)

    def _compute_object_lowest_height(self, object_pos: torch.Tensor, object_quat: torch.Tensor) -> torch.Tensor:
        axis_z = torch.abs(self._compute_object_upright_cos(object_quat))
        radial_component = torch.sqrt(torch.clamp(1.0 - axis_z * axis_z, min=0.0))
        return (
            object_pos[:, 2]
            - 0.5 * float(self.cfg.bottle_height) * axis_z
            - float(self.cfg.bottle_radius) * radial_component
        )

    def _compute_task_terms(self):
        task_terms = super()._compute_task_terms()
        object_quat = self._object.data.root_quat_w
        object_upright_cos = self._compute_object_upright_cos(object_quat)
        object_lowest_height = self._compute_object_lowest_height(task_terms["object_pos"], object_quat)
        task_terms["object_quat"] = object_quat
        task_terms["object_upright_cos"] = object_upright_cos
        task_terms["object_lowest_height"] = object_lowest_height
        return task_terms

    def _get_rewards(self) -> torch.Tensor:
        task_terms = self._compute_task_terms()

        sigma = max(float(self.cfg.reach_sigma), 1.0e-6)
        reach_reward = 1.0 - torch.tanh(task_terms["grip_to_object_dist"] / sigma)

        upright_cos = task_terms["object_upright_cos"]
        upright_tilt_deg = torch.rad2deg(torch.acos(upright_cos))
        upright_mask = (upright_cos >= self._lift_upright_cos_threshold).float()

        current_center_height = task_terms["object_pos"][:, 2]
        current_lowest_height = task_terms["object_lowest_height"]
        lift_start_height = float(self.cfg.lift_reward_start_center_height)
        success_height = float(self.cfg.success_center_height)
        lift_span = max(success_height - lift_start_height, 1.0e-6)
        base_reward = (current_center_height > lift_start_height).float()
        linear_reward = torch.clamp((current_center_height - lift_start_height) / lift_span, min=0.0, max=1.0)
        lift_reward = (base_reward + linear_reward) * upright_mask

        success_reward = (current_center_height > success_height).float() * upright_mask
        above_success_height = (current_center_height > success_height) & (
            upright_cos >= self._lift_upright_cos_threshold
        )
        success_hold_counter_preview = torch.where(
            above_success_height,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        object_center = task_terms["object_pos"]
        grip_center = task_terms["grip_pos"]
        object_center_mean = object_center.mean(dim=0)
        grip_center_mean = grip_center.mean(dim=0)
        object_center_env0 = object_center[0]
        grip_center_env0 = grip_center[0]

        rewards = (
            float(self.cfg.reward_reach_weight) * reach_reward
            + float(self.cfg.reward_lift_weight) * lift_reward
            + float(self.cfg.reward_success_weight) * success_reward
        )

        if self.reward_print_interval > 0 and (self.step_count + 1) % self.reward_print_interval == 0:
            print(
                f"[奖励] step {self.step_count + 1}: "
                f"reach={reach_reward.mean().item():.3f} (w={self.cfg.reward_reach_weight}), "
                f"lift={lift_reward.mean().item():.3f} (w={self.cfg.reward_lift_weight}), "
                f"success={success_reward.mean().item():.3f} (w={self.cfg.reward_success_weight}), "
                f"tilt={upright_tilt_deg.mean().item():.2f} deg, "
                f"hold={success_hold_counter_preview.float().mean().item():.2f}/{self.cfg.success_hold_steps}, "
                f"center_z={current_center_height.mean().item():.4f} m, "
                f"lowest_z={current_lowest_height.mean().item():.4f} m, "
                f"obj_center_mean=({object_center_mean[0].item():.4f}, {object_center_mean[1].item():.4f}, "
                f"{object_center_mean[2].item():.4f}), "
                f"grip_center_mean=({grip_center_mean[0].item():.4f}, {grip_center_mean[1].item():.4f}, "
                f"{grip_center_mean[2].item():.4f}), "
                f"obj_center_env0=({object_center_env0[0].item():.4f}, {object_center_env0[1].item():.4f}, "
                f"{object_center_env0[2].item():.4f}), "
                f"grip_center_env0=({grip_center_env0[0].item():.4f}, {grip_center_env0[1].item():.4f}, "
                f"{grip_center_env0[2].item():.4f}), "
                f"total={rewards.mean().item():.3f}",
                flush=True,
            )

        self.extras["log"] = {
            "reward/total": rewards.mean().detach(),
            "reward/reach": reach_reward.mean().detach(),
            "reward/lift": lift_reward.mean().detach(),
            "reward/success": success_reward.mean().detach(),
            "metric/grip_to_object_dist": task_terms["grip_to_object_dist"].mean().detach(),
            "metric/object_height": task_terms["object_pos"][:, 2].mean().detach(),
            "metric/object_center_height": current_center_height.mean().detach(),
            "metric/object_lowest_height": current_lowest_height.mean().detach(),
            "metric/object_center_x": object_center_mean[0].detach(),
            "metric/object_center_y": object_center_mean[1].detach(),
            "metric/object_center_z": object_center_mean[2].detach(),
            "metric/grip_center_x": grip_center_mean[0].detach(),
            "metric/grip_center_y": grip_center_mean[1].detach(),
            "metric/grip_center_z": grip_center_mean[2].detach(),
            "metric/object_upright_cos": upright_cos.mean().detach(),
            "metric/object_tilt_deg": upright_tilt_deg.mean().detach(),
            "metric/success_hold_counter": success_hold_counter_preview.to(torch.float32).mean().detach(),
            "metric/gripper_target_deg": self._gripper_targets[:, 0].mean().detach(),
            "metric/gripper_closed_ratio": self._gripper_closed.float().mean().detach(),
        }
        self.step_count += 1
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_terms = self._compute_task_terms()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        invalid_state = torch.isnan(self._robot.data.joint_pos[:, self._finger_joint_id]) | torch.isnan(
            self._object.data.root_pos_w[:, 2]
        )
        current_center_height = task_terms["object_pos"][:, 2]
        upright_cos = task_terms["object_upright_cos"]
        above_success_height = (current_center_height > float(self.cfg.success_center_height)) & (
            upright_cos >= self._lift_upright_cos_threshold
        )
        self._success_hold_counter = torch.where(
            above_success_height,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        success = self._success_hold_counter >= int(self.cfg.success_hold_steps)
        object_fell = task_terms["object_lowest_height"] < float(self.cfg.fall_height)
        terminated = invalid_state | object_fell | success
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._success_hold_counter[env_ids] = 0
