# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cube Grasping Environment with Vision Degradation + Two Tactile Sensors."""

from __future__ import annotations

import os
import torch

from isaaclab.utils import configclass

try:
    import torchvision
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

from .cube_grasping_vision_tactile_fusion import (
    CubeGraspingVisionTwoTactileCfg,
    CubeGraspingVisionTwoTactileEnv,
)


@configclass
class CubeGraspingVisionDegradationTactileFusionCfg(CubeGraspingVisionTwoTactileCfg):
    """Configuration for cube grasping with degraded wrist vision and 2 tactile RGB sensors."""

    vision_degradation_enable = True
    vision_occlusion_ratio_range = (0.1, 0.4)
    vision_occlusion_prob = 1.0
    vision_occlusion_value = 0.0
    vision_occlusion_save_images_enabled = True
    vision_occlusion_save_images_dir = "auto"
    vision_occlusion_save_images_max = 6
    vision_occlusion_save_images_interval = 1
    vision_occlusion_save_images_env_id = 0


class CubeGraspingVisionDegradationTactileFusionEnv(CubeGraspingVisionTwoTactileEnv):
    """Cube grasping environment with degraded wrist vision and 2 tactile RGB sensors."""

    cfg: CubeGraspingVisionDegradationTactileFusionCfg

    def __init__(self, cfg: CubeGraspingVisionDegradationTactileFusionCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._occlusion_save_enabled = bool(getattr(self.cfg, "vision_occlusion_save_images_enabled", False))
        if self._occlusion_save_enabled and not _HAS_TORCHVISION:
            print("[WARN] vision_occlusion_save_images_enabled requires torchvision; disabling image dump.")
            self._occlusion_save_enabled = False
        self._occlusion_save_dir = self._resolve_occlusion_save_dir(
            getattr(self.cfg, "vision_occlusion_save_images_dir", "auto")
        )
        self._occlusion_save_max = int(getattr(self.cfg, "vision_occlusion_save_images_max", 6))
        self._occlusion_save_interval = max(int(getattr(self.cfg, "vision_occlusion_save_images_interval", 1)), 1)
        self._occlusion_save_env_id = int(getattr(self.cfg, "vision_occlusion_save_images_env_id", 0))
        self._occlusion_save_count = 0

    def _apply_random_occlusion(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb is None or not bool(getattr(self.cfg, "vision_degradation_enable", True)):
            return rgb
        if rgb.numel() == 0:
            return rgb

        h = rgb.shape[1]
        w = rgb.shape[2]
        min_r, max_r = getattr(self.cfg, "vision_occlusion_ratio_range", (0.0, 0.0))
        min_r = float(min_r)
        max_r = float(max_r)
        if max_r <= 0.0:
            return rgb
        if min_r < 0.0:
            min_r = 0.0
        if max_r < min_r:
            min_r, max_r = max_r, min_r

        prob = float(getattr(self.cfg, "vision_occlusion_prob", 1.0))
        prob = max(0.0, min(1.0, prob))

        out = rgb.clone()
        num_envs = out.shape[0]
        device = out.device
        if prob < 1.0:
            apply_mask = torch.rand((num_envs,), device=device) < prob
        else:
            apply_mask = torch.ones((num_envs,), device=device, dtype=torch.bool)
        if not apply_mask.any():
            return out

        ratios = torch.empty((num_envs,), device=device).uniform_(min_r, max_r)

        fill_value = getattr(self.cfg, "vision_occlusion_value", 0.0)
        if out.dtype.is_floating_point:
            fill_value = float(fill_value)
        else:
            fill_value = int(round(float(fill_value)))

        for i in range(num_envs):
            if not apply_mask[i]:
                continue
            r = float(ratios[i].item())
            if r <= 0.0:
                continue
            r = min(max(r, 0.0), 0.95)
            pixel_mask = torch.rand((h, w), device=device) < r
            if pixel_mask.any():
                out[i][pixel_mask] = fill_value

        return out

    def _resolve_occlusion_save_dir(self, save_dir: str | None) -> str:
        if save_dir is None:
            normalized = ""
        else:
            normalized = str(save_dir).strip()
        if not normalized or normalized.lower() == "auto":
            return os.path.join(self._find_latest_skrl_run_dir(), "wrist_occlusion")
        if "{run_dir}" in normalized:
            return normalized.format(run_dir=self._find_latest_skrl_run_dir())
        return normalized

    def _maybe_save_occlusion_images(self, rgb: torch.Tensor):
        if not self._occlusion_save_enabled:
            return
        if self._occlusion_save_count >= self._occlusion_save_max:
            return
        if self._occlusion_save_env_id < 0 or self._occlusion_save_env_id >= self.num_envs:
            print(f"[WARN] vision_occlusion_save_images_env_id out of range: {self._occlusion_save_env_id}")
            self._occlusion_save_enabled = False
            return
        step_id = int(getattr(self, "common_step_counter", 0))
        if step_id % self._occlusion_save_interval != 0:
            return
        os.makedirs(self._occlusion_save_dir, exist_ok=True)
        env_id = self._occlusion_save_env_id
        img = rgb[env_id]
        if not img.dtype.is_floating_point:
            img = img.to(torch.float32) / 255.0
        img = img.clamp(0.0, 1.0).permute(2, 0, 1).contiguous()
        idx = self._occlusion_save_count
        out_path = os.path.join(
            self._occlusion_save_dir, f"wrist_occluded_env{env_id}_step{step_id:06d}_{idx:03d}.png"
        )
        torchvision.utils.save_image(img, out_path)
        self._occlusion_save_count += 1

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        wrist_rgb = None
        output = getattr(self.wrist_camera, "data", None)
        output = getattr(output, "output", None) if output is not None else None
        if isinstance(output, dict):
            wrist_rgb = output.get("rgb")

        if wrist_rgb is None or not bool(getattr(self.cfg, "vision_degradation_enable", True)):
            return super()._get_observations()

        degraded = self._apply_random_occlusion(wrist_rgb)
        self._maybe_save_occlusion_images(degraded)

        if isinstance(output, dict):
            output["rgb"] = degraded
            try:
                obs = super()._get_observations()
            finally:
                output["rgb"] = wrist_rgb
        else:
            obs = super()._get_observations()

        return obs
