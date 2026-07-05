"""Tactile-cross-alpha visual task with predictor pseudo visible-ratio labels."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from isaaclab.utils import configclass

from .vt_box import OccludedGraspingVTAlphaBoxEnv, OccludedGraspingVTAlphaDownsampleBoxCfg


def _enable_predict_visible_observations(cfg) -> None:
    cfg.observation_space = dict(cfg.observation_space)
    cfg.observation_space["pseudo_visible_ratio"] = 1
    cfg.observation_space["tactile_contact_ratio"] = 4


@configclass
class OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleDownsampleBoxCfg(OccludedGraspingVTAlphaDownsampleBoxCfg):
    """Downsample VT cfg exposing predictor pseudo visibility and tactile contact ratios."""

    pseudo_visible_predictor_path = ""
    pseudo_visible_fallback_ratio = 1.0
    print_pseudo_visible_ratio = False
    pseudo_visible_ratio_print_interval = 200

    def __post_init__(self):
        super().__post_init__()
        _enable_predict_visible_observations(self)

    def _post_configure_scene_object(self):
        _enable_predict_visible_observations(self)


class _XYOcclusionPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ELU())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleBoxEnv(OccludedGraspingVTAlphaBoxEnv):
    """Append pseudo visible ratio from a frozen XY predictor and current tactile contact ratios."""

    cfg: OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleDownsampleBoxCfg

    def __init__(
        self,
        cfg: OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleDownsampleBoxCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._pseudo_visible_predictor: nn.Module | None = None
        self._pseudo_visible_feature_names: list[str] = []
        self._pseudo_visible_feature_mean: torch.Tensor | None = None
        self._pseudo_visible_feature_std: torch.Tensor | None = None
        self._latest_pseudo_visible_ratio = torch.full(
            (self.num_envs, 1),
            float(getattr(self.cfg, "pseudo_visible_fallback_ratio", 1.0)),
            dtype=torch.float32,
            device=self.device,
        )
        self._load_pseudo_visible_predictor()

    def _load_pseudo_visible_predictor(self) -> None:
        path = str(getattr(self.cfg, "pseudo_visible_predictor_path", "") or "")
        if not path:
            if hasattr(self.cfg, "cube_side"):
                path = "logs/occlusion_predictor/cube_xy_predictor.pt"
            elif hasattr(self.cfg, "cuboid_size"):
                path = "logs/occlusion_predictor/cuboid_xy_predictor.pt"
            else:
                path = "logs/occlusion_predictor/cylinder_xy_predictor.pt"
        if not path:
            return
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            print(f"[WARN] pseudo visible predictor not found: {path}; using fallback visible ratio.", flush=True)
            return

        checkpoint: dict[str, Any] = torch.load(path, map_location=self.device)
        metadata = checkpoint.get("metadata", {})
        feature_names = list(metadata.get("feature_names", []))
        if feature_names != ["x", "y"]:
            raise ValueError(
                "Pseudo visible env currently supports predictors with feature_names ['x', 'y']; "
                f"got {feature_names}"
            )
        hidden_dims = [int(v) for v in metadata.get("hidden_dims", [64, 64])]
        model = _XYOcclusionPredictor(input_dim=2, hidden_dims=hidden_dims).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self._pseudo_visible_predictor = model
        self._pseudo_visible_feature_names = feature_names
        self._pseudo_visible_feature_mean = torch.tensor(
            metadata.get("feature_mean", [0.0, 0.0]), device=self.device, dtype=torch.float32
        )
        self._pseudo_visible_feature_std = torch.tensor(
            metadata.get("feature_std", [1.0, 1.0]), device=self.device, dtype=torch.float32
        ).clamp_min(1.0e-6)
        print(f"[INFO] Loaded pseudo visible predictor: {path}", flush=True)

    def _compute_pseudo_visible_ratio(self) -> torch.Tensor:
        if (
            self._pseudo_visible_predictor is None
            or self._pseudo_visible_feature_mean is None
            or self._pseudo_visible_feature_std is None
        ):
            return torch.full(
                (self.num_envs, 1),
                float(getattr(self.cfg, "pseudo_visible_fallback_ratio", 1.0)),
                dtype=torch.float32,
                device=self.device,
            )

        can_pos_local = self._can.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        features = (can_pos_local.to(torch.float32) - self._pseudo_visible_feature_mean) / self._pseudo_visible_feature_std
        with torch.no_grad():
            occlusion = self._pseudo_visible_predictor(features).clamp(0.0, 1.0)
        return (1.0 - occlusion).clamp(0.0, 1.0).unsqueeze(-1)

    def _compute_current_tactile_contact_ratio(self) -> torch.Tensor:
        raw_rgbs = self._get_tactile_contact_raw_rgbs()
        sensor_rgbs = {name: self._prep_contact_rgb_01(raw) for name, raw in raw_rgbs.items()}
        self._maybe_refresh_tactile_contact_baseline(sensor_rgbs)

        ratios = []
        diff_threshold = float(getattr(self.cfg, "tactile_contact_diff_threshold", 0.05))
        for sensor_name in self._tactile_sensor_keys:
            rgb = sensor_rgbs.get(sensor_name)
            if rgb is None:
                ratios.append(torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device))
                continue
            baseline = self._tactile_contact_rgb_baseline[sensor_name]
            diff = torch.abs(rgb - baseline).mean(dim=1)
            ratios.append((diff >= diff_threshold).to(torch.float32).mean(dim=(1, 2)))
        return torch.stack(ratios, dim=-1).clamp(0.0, 1.0)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        pseudo_visible_ratio = self._compute_pseudo_visible_ratio()
        tactile_contact_ratio = self._compute_current_tactile_contact_ratio()
        self._latest_pseudo_visible_ratio = pseudo_visible_ratio

        obs["pseudo_visible_ratio"] = pseudo_visible_ratio
        obs["tactile_contact_ratio"] = tactile_contact_ratio

        log = self.extras.setdefault("log", {})
        log["gate/pseudo_visible_ratio_mean"] = pseudo_visible_ratio.mean().detach()
        log["gate/tactile_contact_ratio_mean"] = tactile_contact_ratio.mean().detach()

        if bool(getattr(self.cfg, "print_pseudo_visible_ratio", False)):
            interval = int(getattr(self.cfg, "pseudo_visible_ratio_print_interval", 200))
            step_count = int(getattr(self, "step_count", 0))
            if interval <= 1 or step_count % interval == 0:
                mean_value = float(pseudo_visible_ratio.mean().detach().cpu().item())
                print(f"[predict_visible] step={step_count} pseudo_visible_ratio_mean={mean_value:.4f}", flush=True)

        return observations

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        self._latest_pseudo_visible_ratio[env_ids] = float(getattr(self.cfg, "pseudo_visible_fallback_ratio", 1.0))
