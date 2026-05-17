from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from isaaclab.utils import configclass

from ..can_grasping.can_grasping_vision_only_resnet18 import (
    CanGraspingVisionOnlyEnv as CanGraspingVisionOnlyResNet18Env,
)
from ..can_grasping.can_grasping_vision_two_tactile import (
    CanGraspingVisionTwoTactileCfg,
    CanGraspingVisionTwoTactileEnv,
)
from .sparsh_encoder import SparshFrozenEncoder, project_sparsh_features


@configclass
class SparchGraspVisionTwoTactileIJEPACfg(CanGraspingVisionTwoTactileCfg):
    """Can grasping config using a frozen Sparsh IJEPA tactile encoder."""

    sparsh_repo_path = "/home/tinydog/Projects/sparsh"
    sparsh_checkpoint_path = "/home/tinydog/桌面/ijepa_vitsmall.ckpt"
    sparsh_model_size = "small"
    sparsh_encoder_type = "ijepa"
    sparsh_num_register_tokens = 0
    sparsh_encoder_embed_dim = 384
    sparsh_feature_dim = 256
    sparsh_frame_stride = 5
    sparsh_baseline_warmup_steps = 4
    sparsh_baseline_retry_steps = 2
    sparsh_baseline_diff_threshold = 0.02
    resnet18_frozen = True

    observation_space = {
        "proprio_obs": 18,
        "gripper_state": 1,
        "action_history": 5,
        "wrist_resnet": 512,
        "tactile_left_resnet": sparsh_feature_dim,
        "tactile_right_resnet": sparsh_feature_dim,
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


class SparchGraspVisionTwoTactileIJEPAEnv(CanGraspingVisionTwoTactileEnv):
    """Two-tactile can grasping environment backed by Sparsh IJEPA features."""

    cfg: SparchGraspVisionTwoTactileIJEPACfg

    def __init__(self, cfg: SparchGraspVisionTwoTactileIJEPACfg, render_mode: str | None = None, **kwargs):
        # Skip the legacy tactile ResNet initialization and install only the Sparsh path.
        CanGraspingVisionOnlyResNet18Env.__init__(self, cfg, render_mode, **kwargs)

        self._tactile_encoder = SparshFrozenEncoder(
            repo_path=self.cfg.sparsh_repo_path,
            checkpoint_path=self.cfg.sparsh_checkpoint_path,
            img_size_hw=tuple(getattr(self.cfg, "tactile_img_res_hw", (96, 128))),
            model_size=self.cfg.sparsh_model_size,
            encoder_type=self.cfg.sparsh_encoder_type,
            num_register_tokens=int(getattr(self.cfg, "sparsh_num_register_tokens", 0)),
        ).to(self.device)
        self._tactile_encoder.eval()
        self.cfg.sparsh_encoder_embed_dim = int(self._tactile_encoder.output_dim)
        self._tactile_feature_dim = int(self.cfg.sparsh_feature_dim)
        self.cfg.observation_space["tactile_left_resnet"] = self._tactile_feature_dim
        self.cfg.observation_space["tactile_right_resnet"] = self._tactile_feature_dim
        self.single_observation_space["policy"]["tactile_left_resnet"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._tactile_feature_dim,), dtype=np.float32
        )
        self.single_observation_space["policy"]["tactile_right_resnet"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._tactile_feature_dim,), dtype=np.float32
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)

        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        warmup_steps = int(getattr(self.cfg, "sparsh_baseline_warmup_steps", 4))
        self._tactile_ref_left = torch.zeros(
            (self.num_envs, 3, target_h, target_w), device=self.device, dtype=torch.float32
        )
        self._tactile_ref_right = torch.zeros(
            (self.num_envs, 3, target_h, target_w), device=self.device, dtype=torch.float32
        )
        self._tactile_ref_pending = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self._tactile_ref_delay = torch.full(
            (self.num_envs,), warmup_steps, device=self.device, dtype=torch.long
        )

        self._sparsh_frame_stride = int(getattr(self.cfg, "sparsh_frame_stride", 5))
        self._sparsh_history_len = self._sparsh_frame_stride + 1
        self._sparsh_hist_left = torch.zeros(
            (self._sparsh_history_len, self.num_envs, 3, target_h, target_w),
            device=self.device,
            dtype=torch.float32,
        )
        self._sparsh_hist_right = torch.zeros_like(self._sparsh_hist_left)
        self._sparsh_hist_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

    def _prep_rgb_01(self, rgb_tensor: torch.Tensor | None) -> torch.Tensor:
        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        if rgb_tensor is None:
            return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
        x = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = x.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _sparsh_diff(curr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return torch.clamp(curr - ref + 0.5, 0.0, 1.0)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs_dict = CanGraspingVisionOnlyResNet18Env._get_observations(self)
        obs = obs_dict["policy"]

        tact_l_raw = getattr(self, "gsmini_left", None)
        tact_l_raw = tact_l_raw.data.output.get("tactile_rgb") if tact_l_raw is not None else None
        tact_r_raw = getattr(self, "gsmini_right", None)
        tact_r_raw = tact_r_raw.data.output.get("tactile_rgb") if tact_r_raw is not None else None

        has_tact_l = tact_l_raw is not None
        has_tact_r = tact_r_raw is not None

        tact_l_01 = self._prep_rgb_01(tact_l_raw)
        tact_r_01 = self._prep_rgb_01(tact_r_raw)

        diff_thresh = float(getattr(self.cfg, "sparsh_baseline_diff_threshold", 0.02))
        score_l = (tact_l_01 - self._tactile_ref_left).abs().mean(dim=(1, 2, 3))
        score_r = (tact_r_01 - self._tactile_ref_right).abs().mean(dim=(1, 2, 3))

        pending = self._tactile_ref_pending
        if torch.any(pending):
            dec_mask = pending & (self._tactile_ref_delay > 0)
            if torch.any(dec_mask):
                self._tactile_ref_delay[dec_mask] -= 1
            ready = pending & (self._tactile_ref_delay <= 0)
            if torch.any(ready):
                baseline_empty = (
                    (self._tactile_ref_left.abs().sum(dim=(1, 2, 3)) < 1e-6)
                    & (self._tactile_ref_right.abs().sum(dim=(1, 2, 3)) < 1e-6)
                )
                unsafe = (score_l > diff_thresh) | (score_r > diff_thresh)
                safe = ready & (baseline_empty | ~unsafe)
                if not (has_tact_l and has_tact_r):
                    safe = torch.zeros_like(safe, dtype=torch.bool, device=self.device)
                if torch.any(safe):
                    self._tactile_ref_left[safe] = tact_l_01[safe]
                    self._tactile_ref_right[safe] = tact_r_01[safe]
                    self._tactile_ref_pending[safe] = False
                    self._tactile_ref_delay[safe] = 0
                retry = ready & ~safe
                if torch.any(retry):
                    retry_steps = int(getattr(self.cfg, "sparsh_baseline_retry_steps", 2))
                    retry_steps = max(retry_steps, 1)
                    if retry_steps == 1:
                        self._tactile_ref_delay[retry] = 1
                    else:
                        self._tactile_ref_delay[retry] = torch.randint(
                            1, retry_steps + 1, (int(retry.sum().item()),), device=self.device
                        )

        ready_mask = (~self._tactile_ref_pending).clone()
        if not (has_tact_l and has_tact_r):
            ready_mask = torch.zeros_like(ready_mask, dtype=torch.bool, device=self.device)

        if torch.any(~ready_mask):
            self._sparsh_hist_left[:, ~ready_mask] = 0
            self._sparsh_hist_right[:, ~ready_mask] = 0
            self._sparsh_hist_count[~ready_mask] = 0

        diff_l = torch.zeros_like(tact_l_01)
        diff_r = torch.zeros_like(tact_r_01)
        if torch.any(ready_mask):
            diff_l[ready_mask] = self._sparsh_diff(tact_l_01[ready_mask], self._tactile_ref_left[ready_mask])
            diff_r[ready_mask] = self._sparsh_diff(tact_r_01[ready_mask], self._tactile_ref_right[ready_mask])

        self._sparsh_hist_left = torch.roll(self._sparsh_hist_left, shifts=-1, dims=0)
        self._sparsh_hist_right = torch.roll(self._sparsh_hist_right, shifts=-1, dims=0)
        if torch.any(ready_mask):
            self._sparsh_hist_left[-1, ready_mask] = diff_l[ready_mask]
            self._sparsh_hist_right[-1, ready_mask] = diff_r[ready_mask]
            self._sparsh_hist_count[ready_mask] = torch.clamp(
                self._sparsh_hist_count[ready_mask] + 1,
                max=self._sparsh_history_len,
            )

        past_l = diff_l.clone()
        past_r = diff_r.clone()
        history_ready = ready_mask & (self._sparsh_hist_count >= self._sparsh_history_len)
        if torch.any(history_ready):
            past_l[history_ready] = self._sparsh_hist_left[0, history_ready]
            past_r[history_ready] = self._sparsh_hist_right[0, history_ready]

        zeros = torch.zeros((self.num_envs, self._tactile_feature_dim), device=self.device, dtype=torch.float32)
        fl = zeros
        fr = zeros
        if torch.any(ready_mask):
            sparsh_l = torch.cat([diff_l, past_l], dim=1)
            sparsh_r = torch.cat([diff_r, past_r], dim=1)

            dev = self.device
            dev_type = getattr(dev, "type", None)
            if dev_type is None:
                dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
            use_amp = dev_type == "cuda"
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                fl = self._tactile_encoder(sparsh_l).mean(dim=1).float()
                fr = self._tactile_encoder(sparsh_r).mean(dim=1).float()
            fl = project_sparsh_features(fl, self._tactile_feature_dim)
            fr = project_sparsh_features(fr, self._tactile_feature_dim)

        obs.update(
            {
                "tactile_left_resnet": fl,
                "tactile_right_resnet": fr,
            }
        )
        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        warmup_steps = int(getattr(self.cfg, "sparsh_baseline_warmup_steps", 4))
        self._tactile_ref_left[env_ids] = 0
        self._tactile_ref_right[env_ids] = 0
        self._tactile_ref_pending[env_ids] = True
        self._tactile_ref_delay[env_ids] = warmup_steps
        self._sparsh_hist_left[:, env_ids] = 0
        self._sparsh_hist_right[:, env_ids] = 0
        self._sparsh_hist_count[env_ids] = 0
