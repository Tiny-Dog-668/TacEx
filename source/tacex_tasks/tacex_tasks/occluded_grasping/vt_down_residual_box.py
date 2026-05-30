"""VT task with down-tactile contact ratios for hard-gated residual actions."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vt_box import OccludedGraspingVisionFourTactileBoxCfg, OccludedGraspingVisionFourTactileBoxEnv


@configclass
class OccludedGraspingVTDownResidualBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """VT config exposing bottom tactile contact degree for residual-action gates."""

    down_residual_contact_threshold = 0.01

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = {
            **dict(self.observation_space),
            "down_tactile_contact_ratio": 2,
        }


class OccludedGraspingVTDownResidualBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """VT env that adds current bottom tactile contact ratios to policy observations."""

    cfg: OccludedGraspingVTDownResidualBoxCfg

    def __init__(self, cfg: OccludedGraspingVTDownResidualBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._latest_down_residual_contact_ratio = torch.zeros((self.num_envs, 2), device=self.device)
        self._latest_down_residual_gate = torch.zeros((self.num_envs, 1), device=self.device)

    def _compute_current_down_contact_ratio(self) -> torch.Tensor:
        raw_rgbs = self._get_tactile_contact_raw_rgbs()
        sensor_rgbs = {name: self._prep_contact_rgb_01(raw) for name, raw in raw_rgbs.items()}
        self._maybe_refresh_tactile_contact_baseline(sensor_rgbs)

        ratios = []
        diff_threshold = float(getattr(self.cfg, "tactile_contact_diff_threshold", 0.05))
        for sensor_name in ("left_down", "right_down"):
            rgb = sensor_rgbs.get(sensor_name)
            if rgb is None:
                ratios.append(torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device))
                continue
            baseline = self._tactile_contact_rgb_baseline[sensor_name]
            diff = torch.abs(rgb - baseline).mean(dim=1)
            ratios.append((diff >= diff_threshold).to(torch.float32).mean(dim=(1, 2)))
        return torch.stack(ratios, dim=-1)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        down_ratio = self._compute_current_down_contact_ratio().clamp(0.0, 1.0)
        threshold = float(getattr(self.cfg, "down_residual_contact_threshold", 0.01))
        down_gate = (down_ratio.max(dim=-1, keepdim=True).values > threshold).to(torch.float32)

        self._latest_down_residual_contact_ratio = down_ratio
        self._latest_down_residual_gate = down_gate
        obs["down_tactile_contact_ratio"] = down_ratio

        log = self.extras.setdefault("log", {})
        log["info/down_residual_contact_left"] = down_ratio[:, 0].mean().detach()
        log["info/down_residual_contact_right"] = down_ratio[:, 1].mean().detach()
        log["info/down_residual_gate"] = down_gate.mean().detach()
        return observations

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        self._latest_down_residual_contact_ratio[env_ids] = 0.0
        self._latest_down_residual_gate[env_ids] = 0.0
