"""Third-view + concatenated four-tactile env with mixed-env style tactile gating."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .vt_box import OccludedGraspingVisionFourTactileBoxEnv
from .third_view_tactile_concat_box import (
    OccludedGraspingThirdViewTactileConcatBoxCfg,
    OccludedGraspingThirdViewTactileConcatBoxEnv,
)

_TACTILE_SENSOR_ATTRS: dict[str, str] = {
    "left": "gsmini_left",
    "right": "gsmini_right",
    "left_down": "gsmini_left_down",
    "right_down": "gsmini_right_down",
}


@configclass
class OccludedGraspingThirdViewTactileConcatGateBoxCfg(OccludedGraspingThirdViewTactileConcatBoxCfg):
    """Configuration for third-view tactile gating with residual-policy friendly observations."""

    tactile_gate_enable = True
    tactile_baseline_diff_threshold = 0.05
    tactile_contact_area_threshold = 0.01
    tactile_gate_history_len = 1

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()

        self.gsmini_left = self.gsmini_left.replace()
        self.gsmini_left.data_types = ["tactile_rgb", "camera_depth"]

        self.gsmini_right = self.gsmini_right.replace()
        self.gsmini_right.data_types = ["tactile_rgb", "camera_depth"]

        self.gsmini_left_down = self.gsmini_left_down.replace()
        self.gsmini_left_down.data_types = ["tactile_rgb", "camera_depth"]

        self.gsmini_right_down = self.gsmini_right_down.replace()
        self.gsmini_right_down.data_types = ["tactile_rgb", "camera_depth"]

        tactile_obs_dim = int(self._compute_tactile_obs_dim())
        critic_keys = (
            "critic_can_pos",
            "critic_can_quat",
            "critic_can_lin_vel",
            "critic_can_ang_vel",
            "critic_gripper_pos",
            "critic_gripper_quat",
            "critic_gripper_lin_vel",
            "critic_gripper_ang_vel",
            "critic_target_pos",
            "critic_target_distance",
        )
        self.observation_space = {
            "proprio_obs": int(self.observation_space["proprio_obs"]),
            "third_resnet": int(self.observation_space["third_resnet"]),
            "tactile_left_depth_resnet": tactile_obs_dim,
            "tactile_right_depth_resnet": tactile_obs_dim,
            "tactile_left_down_depth_resnet": tactile_obs_dim,
            "tactile_right_down_depth_resnet": tactile_obs_dim,
            "tactile_concat": 4 * tactile_obs_dim,
            "tactile_contact_bits": len(_TACTILE_SENSOR_ATTRS),
            "tactile_valid": 1,
            **{key: int(self.observation_space[key]) for key in critic_keys},
        }


class OccludedGraspingThirdViewTactileConcatGateBoxEnv(OccludedGraspingThirdViewTactileConcatBoxEnv):
    """Expose tactile contact gates and keep a compatibility tactile concat tensor."""

    cfg: OccludedGraspingThirdViewTactileConcatGateBoxCfg

    def __init__(self, cfg: OccludedGraspingThirdViewTactileConcatGateBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._tactile_baseline_depth: dict[str, torch.Tensor | None] = {
            key: None for key in _TACTILE_SENSOR_ATTRS
        }
        self._pending_tactile_baseline_refresh = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        history_len = max(1, int(getattr(self.cfg, "tactile_gate_history_len", 1)))
        self._tactile_contact_history = torch.zeros(
            (self.num_envs, history_len, len(_TACTILE_SENSOR_ATTRS)),
            dtype=torch.float32,
            device=self.device,
        )
        self._tactile_gate_any = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._latest_tactile_contact_ratios = {
            key: torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            for key in _TACTILE_SENSOR_ATTRS
        }

    def _get_tactile_sensor_depths(self) -> dict[str, torch.Tensor | None]:
        """Collect per-sensor depth frames used by tactile contact gating."""
        sensor_depths = {}
        for key, attr_name in _TACTILE_SENSOR_ATTRS.items():
            sensor = getattr(self, attr_name, None)
            sensor_depths[key] = sensor.data.output.get("camera_depth") if sensor is not None else None
        return sensor_depths

    def _normalize_depth_01(self, depth_tensor: torch.Tensor | None) -> torch.Tensor | None:
        """Convert camera-depth output into float32 [0, 1] NHWC1 tensor."""
        if depth_tensor is None:
            return None
        depth = depth_tensor.to(device=self.device, dtype=torch.float32)
        max_val = depth.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            depth = depth / 255.0
        depth = depth.clamp(0.0, 1.0)
        if depth.ndim == 3:
            depth = depth.unsqueeze(-1)
        return depth

    def _maybe_refresh_tactile_baselines(self, sensor_depths: dict[str, torch.Tensor | None]) -> None:
        """Capture per-env baseline depth images right after reset."""
        pending = self._pending_tactile_baseline_refresh
        if not torch.any(pending):
            return
        pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
        if pending_ids.numel() == 0:
            return

        captured_any = False
        for sensor_name, depth_tensor in sensor_depths.items():
            depth_01 = self._normalize_depth_01(depth_tensor)
            if depth_01 is None:
                continue
            baseline = self._tactile_baseline_depth.get(sensor_name)
            if baseline is None or baseline.shape != depth_01.shape:
                baseline = torch.zeros_like(depth_01)
                self._tactile_baseline_depth[sensor_name] = baseline
            baseline[pending_ids] = depth_01[pending_ids]
            captured_any = True

        if captured_any:
            pending[pending_ids] = False

    def _contact_ratio_from_depth(self, sensor_name: str, depth_tensor: torch.Tensor | None) -> torch.Tensor:
        """Match vt_mixed_box: ratio of pixels whose baseline-depth diff exceeds threshold."""
        depth = self._normalize_depth_01(depth_tensor)
        baseline = self._tactile_baseline_depth.get(sensor_name)
        if depth is None or baseline is None:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        diff = torch.abs(depth - baseline)
        diff_threshold = float(getattr(self.cfg, "tactile_baseline_diff_threshold", 0.05))
        return (diff >= diff_threshold).to(torch.float32).mean(dim=(1, 2, 3), keepdim=False)

    def _contact_bits_from_ratio(self, contact_ratio: torch.Tensor) -> torch.Tensor:
        """Match vt_mixed_box: binarize changed-pixel ratio into a per-sensor contact bit."""
        area_threshold = float(getattr(self.cfg, "tactile_contact_area_threshold", 0.01))
        return (contact_ratio > area_threshold).to(torch.float32).unsqueeze(-1)

    def _update_tactile_global_gate(
        self, sensor_depths: dict[str, torch.Tensor | None]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Update global tactile gate state from per-sensor contact estimates."""
        gate_enabled = bool(getattr(self.cfg, "tactile_gate_enable", True))
        if gate_enabled:
            self._maybe_refresh_tactile_baselines(sensor_depths)
            contact_ratios = {
                key: self._contact_ratio_from_depth(key, sensor_depths[key]) for key in _TACTILE_SENSOR_ATTRS
            }
            contact_bits = torch.cat(
                [self._contact_bits_from_ratio(contact_ratios[key]) for key in _TACTILE_SENSOR_ATTRS],
                dim=-1,
            )
        else:
            contact_ratios = {
                key: torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
                for key in _TACTILE_SENSOR_ATTRS
            }
            contact_bits = torch.ones((self.num_envs, len(_TACTILE_SENSOR_ATTRS)), dtype=torch.float32, device=self.device)

        self._latest_tactile_contact_ratios = contact_ratios
        self._tactile_contact_history[:, :-1, :] = self._tactile_contact_history[:, 1:, :].clone()
        self._tactile_contact_history[:, -1, :] = contact_bits
        global_gate = self._tactile_contact_history.max(dim=1).values.max(dim=-1, keepdim=True).values
        self._tactile_gate_any = global_gate.squeeze(-1) > 0.5
        return contact_ratios, contact_bits, global_gate

    def _gate_tactile_branch(self, tactile_feature: torch.Tensor, global_gate: torch.Tensor) -> torch.Tensor:
        """Disable tactile branch when gate is off and keep the off-branch out of autograd."""
        gate_mask = global_gate > 0.5
        tactile_off = torch.zeros_like(tactile_feature.detach())
        return torch.where(gate_mask, tactile_feature, tactile_off)

    def _log_tactile_gate_stats(
        self, contact_ratios: dict[str, torch.Tensor], global_gate: torch.Tensor
    ) -> None:
        """Record tactile gate diagnostics."""
        log = self.extras.setdefault("log", {})
        log["info/tactile_global_gate_mean"] = global_gate.mean().detach()
        log["info/tactile_baseline_diff_threshold"] = torch.tensor(
            float(getattr(self.cfg, "tactile_baseline_diff_threshold", 0.05)), device=self.device
        )
        log["info/tactile_contact_area_threshold"] = torch.tensor(
            float(getattr(self.cfg, "tactile_contact_area_threshold", 0.01)), device=self.device
        )
        for key, ratio in contact_ratios.items():
            log[f"info/tactile_ratio_{key}"] = ratio.mean().detach()

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = OccludedGraspingVisionFourTactileBoxEnv._get_observations(self)
        obs = observations["policy"]

        sensor_depths = self._get_tactile_sensor_depths()
        contact_ratios, contact_bits, global_gate = self._update_tactile_global_gate(sensor_depths)

        tactile_concat = torch.cat(
            [
                obs["tactile_left_depth_resnet"],
                obs["tactile_right_depth_resnet"],
                obs["tactile_left_down_depth_resnet"],
                obs["tactile_right_down_depth_resnet"],
            ],
            dim=-1,
        )
        obs["tactile_concat"] = self._gate_tactile_branch(tactile_concat, global_gate)
        obs["tactile_contact_bits"] = contact_bits
        obs["tactile_valid"] = global_gate.to(dtype=torch.float32)

        self._log_tactile_gate_stats(contact_ratios, global_gate)

        return observations

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        self._tactile_gate_any[env_ids] = False
        for key in _TACTILE_SENSOR_ATTRS:
            baseline = self._tactile_baseline_depth[key]
            if baseline is not None:
                baseline[env_ids] = 0.0
            self._latest_tactile_contact_ratios[key][env_ids] = 0.0
        self._pending_tactile_baseline_refresh[env_ids] = True
        self._tactile_contact_history[env_ids] = 0.0
