"""Three-frame visual Student for the X040-Wide Size-Buckets task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import configclass

from .sim2real_cube_real_alignment_rma_env import _CRITIC_OBSERVATION_SPACE
from .sim2real_cube_real_alignment_rma_x040_wide_size_buckets_env import (
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnv,
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnvCfg,
)


@configclass
class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnvCfg(
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnvCfg
):
    """Size-Buckets Student exposing the latest three 30 Hz wrist RGB frames."""

    wrist_rgb_history_length = 3
    wrist_rgb_history_stride_policy_steps = 1
    wrist_rgb_history_order = "oldest_to_newest"
    wrist_rgb_history_reset_fill = "repeat_first_post_reset_frame"
    observation_space = {
        "proprio_obs": 15,
        "action_history": 4,
        "wrist_rgb_history": gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 224, 224, 3),
            dtype=np.uint8,
        ),
        "rma_cube_pos": 3,
        **_CRITIC_OBSERVATION_SPACE,
    }


class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnv(
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnv
):
    """Maintain a reset-safe ``[oldest, middle, newest]`` uint8 frame history."""

    cfg: Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnvCfg

    def _ensure_wrist_rgb_history(self) -> None:
        expected_shape = (self.num_envs, 3, 224, 224, 3)
        if (
            not hasattr(self, "_wrist_rgb_history")
            or self._wrist_rgb_history.shape != expected_shape
        ):
            self._wrist_rgb_history = torch.zeros(
                expected_shape,
                device=self.device,
                dtype=torch.uint8,
            )
            self._wrist_rgb_history_reset_pending = torch.ones(
                (self.num_envs,),
                device=self.device,
                dtype=torch.bool,
            )

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self._ensure_wrist_rgb_history()
        self._wrist_rgb_history_reset_pending[
            env_ids.to(device=self.device, dtype=torch.long)
        ] = True

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        policy = observations["policy"]
        current_rgb = policy.pop("wrist_rgb")
        self._ensure_wrist_rgb_history()

        # Input current_rgb: [N,224,224,3]. Output history:
        # [N,3,224,224,3] ordered oldest -> newest at 30 Hz policy cadence.
        self._wrist_rgb_history[:, 0].copy_(self._wrist_rgb_history[:, 1])
        self._wrist_rgb_history[:, 1].copy_(self._wrist_rgb_history[:, 2])
        self._wrist_rgb_history[:, 2].copy_(current_rgb)

        reset_ids = torch.nonzero(
            self._wrist_rgb_history_reset_pending,
            as_tuple=False,
        ).squeeze(-1)
        if reset_ids.numel() > 0:
            repeated = current_rgb[reset_ids].unsqueeze(1).expand(-1, 3, -1, -1, -1)
            self._wrist_rgb_history[reset_ids] = repeated
            self._wrist_rgb_history_reset_pending[reset_ids] = False

        policy["wrist_rgb_history"] = self._wrist_rgb_history
        return observations


__all__ = (
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnv",
)
