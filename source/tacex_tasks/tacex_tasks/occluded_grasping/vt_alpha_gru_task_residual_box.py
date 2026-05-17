"""Alpha-GRU box variant with auxiliary task heads and tactile residual-control targets."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from .vt_alpha_gru_box import OccludedGraspingVTAlphaGRUBoxCfg, OccludedGraspingVTAlphaGRUBoxEnv


@configclass
class OccludedGraspingVTAlphaGRUTaskResidualBoxCfg(OccludedGraspingVTAlphaGRUBoxCfg):
    """Configuration for Alpha-GRU with task-head supervision and tactile residual gates."""

    reliability_use_bbox3d = True
    reliability_fallback_visible_ratio = 0.0
    reliability_bbox_lazy_init = True

    stage_contact_diff_threshold = 0.05
    stage_contact_area_threshold = 0.01
    bottom_rel_pos_scale_x = 0.12
    bottom_rel_pos_scale_y = 0.08

    def __post_init__(self):
        super().__post_init__()

        self.observation_space = dict(self.observation_space)
        self.observation_space["aux_visibility_gt"] = 1
        self.observation_space["aux_occlusion_gt"] = 1
        self.observation_space["aux_bottom_contact_gt"] = 2
        self.observation_space["aux_bottom_rel_pos_gt"] = 2
        self.observation_space["aux_inner_contact_gt"] = 1
        self.observation_space["aux_inner_contact_diff_gt"] = 1
        self.observation_space["bottom_tactile_current_mask"] = 1
        self.observation_space["inner_tactile_current_mask"] = 1

        try:
            self.can = self.can.replace(
                spawn=self.can.spawn.replace(
                    semantic_tags=[("class", "can")],
                )
            )
        except Exception:
            try:
                self.can.spawn.semantic_tags = [("class", "can")]
            except Exception:
                pass


class OccludedGraspingVTAlphaGRUTaskResidualBoxEnv(OccludedGraspingVTAlphaGRUBoxEnv):
    """Alpha-GRU env exposing task-head targets and current tactile-valid masks."""

    cfg: OccludedGraspingVTAlphaGRUTaskResidualBoxCfg

    _CONTACT_SENSOR_ORDER = ("left", "right", "left_down", "right_down")
    _CONTACT_SENSOR_ATTRS = {
        "left": "gsmini_left",
        "right": "gsmini_right",
        "left_down": "gsmini_left_down",
        "right_down": "gsmini_right_down",
    }

    def __init__(self, cfg: OccludedGraspingVTAlphaGRUTaskResidualBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._stage_contact_rgb_baseline: dict[str, torch.Tensor | None] = {
            key: None for key in self._CONTACT_SENSOR_ORDER
        }
        self._pending_stage_contact_rgb_baseline_refresh = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._latest_tactile_over_thresh_ratio_by_sensor = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._latest_visibility_gt = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._latest_occlusion_gt = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._latest_bottom_contact_gt = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._latest_bottom_rel_pos_gt = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._latest_inner_contact_gt = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._latest_inner_contact_diff_gt = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._latest_bottom_mask = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._latest_inner_mask = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        self._bbox_annotator = None
        self._bbox_render_products: list[str] = []
        self._bbox_enabled = False
        self._bbox_init_attempted = False
        if not bool(getattr(self.cfg, "reliability_bbox_lazy_init", True)):
            self._init_bbox_annotator()

    def _resolve_render_product_paths(self) -> list[str]:
        candidates = (
            "render_product_paths",
            "render_product_path",
            "_render_product_paths",
            "_render_product_path",
        )
        for attr in candidates:
            value = getattr(self.third_person_camera, attr, None)
            if value is None:
                continue
            if isinstance(value, str):
                return [value]
            if isinstance(value, (tuple, list)):
                return [str(v) for v in value if v is not None and str(v)]
        return []

    def _init_bbox_annotator(self) -> None:
        self._bbox_init_attempted = True
        if not bool(getattr(self.cfg, "reliability_use_bbox3d", True)):
            return

        try:
            import omni.replicator.core as rep  # type: ignore
        except Exception:
            return

        try:
            render_products = self._resolve_render_product_paths()
            if not render_products:
                return

            annotator = rep.AnnotatorRegistry.get_annotator(
                "bounding_box_3d",
                init_params={"semanticTypes": ["class"]},
            )
            for render_product in render_products:
                try:
                    annotator.attach(render_product)
                except Exception:
                    annotator.attach([render_product])

            self._bbox_annotator = annotator
            self._bbox_render_products = render_products
            self._bbox_enabled = True
        except Exception:
            self._bbox_annotator = None
            self._bbox_render_products = []
            self._bbox_enabled = False

    def _detach_bbox_annotator(self) -> None:
        if self._bbox_annotator is None:
            return
        try:
            if self._bbox_render_products:
                self._bbox_annotator.detach(self._bbox_render_products)
        except Exception:
            pass
        self._bbox_annotator = None
        self._bbox_render_products = []
        self._bbox_enabled = False

    def close(self):
        self._detach_bbox_annotator()
        super().close()

    @staticmethod
    def _unpack_bbox_output(output):
        if isinstance(output, dict):
            data = output.get("data")
            info = output.get("info", {})
            if not info:
                info = {key: value for key, value in output.items() if key != "data"}
            return data, info
        return output, {}

    @staticmethod
    def _extract_env_id_from_path(path: str) -> int | None:
        prefix = "/World/envs/env_"
        if not path.startswith(prefix):
            return None
        suffix = path[len(prefix):]
        digits: list[str] = []
        for ch in suffix:
            if ch.isdigit():
                digits.append(ch)
                continue
            break
        if not digits:
            return None
        try:
            return int("".join(digits))
        except ValueError:
            return None

    @staticmethod
    def _extract_occlusion_ratio(row) -> float | None:
        value = None
        if isinstance(row, dict):
            value = row.get("occlusionRatio")
        else:
            try:
                value = row["occlusionRatio"]
            except Exception:
                value = getattr(row, "occlusionRatio", None)

        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(ratio) or ratio < 0.0:
            return None
        return ratio

    def _compute_bbox_visible_ratio_gt(self) -> torch.Tensor:
        fallback = float(getattr(self.cfg, "reliability_fallback_visible_ratio", 0.0))
        visible_ratio = torch.full((self.num_envs,), fallback, dtype=torch.float32, device=self.device)

        if (
            bool(getattr(self.cfg, "reliability_use_bbox3d", True))
            and not self._bbox_enabled
            and not self._bbox_init_attempted
        ):
            self._init_bbox_annotator()

        if not self._bbox_enabled or self._bbox_annotator is None:
            return visible_ratio

        try:
            bbox_output = self._bbox_annotator.get_data()
        except Exception:
            return visible_ratio

        bbox_data, bbox_info = self._unpack_bbox_output(bbox_output)
        prim_paths = bbox_info.get("primPaths", [])
        if bbox_data is None or not isinstance(prim_paths, Iterable):
            return visible_ratio

        max_occlusion_by_env: list[float | None] = [None] * self.num_envs
        for row_idx, prim_path in enumerate(prim_paths):
            path_str = str(prim_path)
            env_id = self._extract_env_id_from_path(path_str)
            if env_id is None or env_id < 0 or env_id >= self.num_envs:
                continue

            can_prefix = f"/World/envs/env_{env_id}/can"
            if path_str != can_prefix and not path_str.startswith(f"{can_prefix}/"):
                continue

            try:
                row = bbox_data[row_idx]
            except Exception:
                continue
            occlusion_ratio = self._extract_occlusion_ratio(row)
            if occlusion_ratio is None:
                continue

            previous = max_occlusion_by_env[env_id]
            max_occlusion_by_env[env_id] = occlusion_ratio if previous is None else max(previous, occlusion_ratio)

        for env_id, occlusion_ratio in enumerate(max_occlusion_by_env):
            if occlusion_ratio is None:
                continue
            visible_ratio[env_id] = float(max(0.0, min(1.0, 1.0 - occlusion_ratio)))

        return visible_ratio

    def _prep_rgb_01(self, rgb_tensor: torch.Tensor | None) -> torch.Tensor | None:
        if rgb_tensor is None:
            return None
        rgb = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = rgb.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            rgb = rgb / 255.0
        rgb = rgb.clamp(0.0, 1.0)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(-1)
        if rgb.ndim != 4:
            return None
        return rgb.permute(0, 3, 1, 2).contiguous()

    def _maybe_refresh_stage_contact_baselines(self, sensor_rgbs: dict[str, torch.Tensor | None]) -> None:
        pending = self._pending_stage_contact_rgb_baseline_refresh
        if not torch.any(pending):
            return

        pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
        if pending_ids.numel() == 0:
            return

        captured_any = False
        for sensor_name, rgb_tensor in sensor_rgbs.items():
            rgb = self._prep_rgb_01(rgb_tensor)
            if rgb is None:
                continue

            baseline = self._stage_contact_rgb_baseline.get(sensor_name)
            if baseline is None or baseline.shape != rgb.shape:
                baseline = torch.zeros_like(rgb, dtype=torch.float32, device=self.device)
                self._stage_contact_rgb_baseline[sensor_name] = baseline

            baseline[pending_ids] = rgb[pending_ids]
            captured_any = True

        if captured_any:
            pending[pending_ids] = False

    def _contact_ratio_from_rgb(self, sensor_name: str, rgb_tensor: torch.Tensor | None) -> torch.Tensor:
        rgb = self._prep_rgb_01(rgb_tensor)
        baseline = self._stage_contact_rgb_baseline.get(sensor_name)
        if rgb is None or baseline is None:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        diff = torch.abs(rgb - baseline)
        diff_threshold = float(getattr(self.cfg, "stage_contact_diff_threshold", 0.05))
        return (diff >= diff_threshold).to(torch.float32).mean(dim=(1, 2, 3), keepdim=False)

    def _compute_bottom_relative_pos_gt(self) -> torch.Tensor:
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

        scale_x = max(float(getattr(self.cfg, "bottom_rel_pos_scale_x", 0.12)), 1e-6)
        scale_y = max(float(getattr(self.cfg, "bottom_rel_pos_scale_y", 0.08)), 1e-6)
        rel_x = (rel_pos_ee[:, 0:1] / scale_x).clamp(-1.0, 1.0)
        rel_y = (rel_pos_ee[:, 1:2] / scale_y).clamp(-1.0, 1.0)
        return torch.cat([rel_x, rel_y], dim=-1)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        sensor_rgbs = {}
        for sensor_name, sensor_attr in self._CONTACT_SENSOR_ATTRS.items():
            sensor = getattr(self, sensor_attr, None)
            rgb = sensor.data.output.get("tactile_rgb") if sensor is not None else None
            sensor_rgbs[sensor_name] = rgb
        self._maybe_refresh_stage_contact_baselines(sensor_rgbs)

        contact_ratios = {
            sensor_name: self._contact_ratio_from_rgb(sensor_name, sensor_rgbs[sensor_name])
            for sensor_name in self._CONTACT_SENSOR_ORDER
        }
        self._latest_tactile_over_thresh_ratio_by_sensor = torch.stack(
            [contact_ratios[sensor_name] for sensor_name in self._CONTACT_SENSOR_ORDER], dim=-1
        )

        visibility_gt = self._compute_bbox_visible_ratio_gt().unsqueeze(-1)
        occlusion_gt = (1.0 - visibility_gt).clamp(0.0, 1.0)
        bottom_contact_gt = self._latest_tactile_over_thresh_ratio_by_sensor[:, 2:4].clamp(0.0, 1.0)
        bottom_rel_pos_gt = self._compute_bottom_relative_pos_gt()
        inner_left = self._latest_tactile_over_thresh_ratio_by_sensor[:, 0:1].clamp(0.0, 1.0)
        inner_right = self._latest_tactile_over_thresh_ratio_by_sensor[:, 1:2].clamp(0.0, 1.0)
        inner_contact_gt = (0.5 * (inner_left + inner_right)).clamp(0.0, 1.0)
        inner_contact_diff_gt = torch.abs(inner_left - inner_right).clamp(0.0, 1.0)

        area_threshold = float(getattr(self.cfg, "stage_contact_area_threshold", 0.01))
        bottom_mask = (bottom_contact_gt.max(dim=-1, keepdim=True).values > area_threshold).to(torch.float32)
        inner_mask = (torch.maximum(inner_left, inner_right) > area_threshold).to(torch.float32)

        self._latest_visibility_gt = visibility_gt
        self._latest_occlusion_gt = occlusion_gt
        self._latest_bottom_contact_gt = bottom_contact_gt
        self._latest_bottom_rel_pos_gt = bottom_rel_pos_gt
        self._latest_inner_contact_gt = inner_contact_gt
        self._latest_inner_contact_diff_gt = inner_contact_diff_gt
        self._latest_bottom_mask = bottom_mask
        self._latest_inner_mask = inner_mask

        obs["aux_visibility_gt"] = visibility_gt
        obs["aux_occlusion_gt"] = occlusion_gt
        obs["aux_bottom_contact_gt"] = bottom_contact_gt
        obs["aux_bottom_rel_pos_gt"] = bottom_rel_pos_gt
        obs["aux_inner_contact_gt"] = inner_contact_gt
        obs["aux_inner_contact_diff_gt"] = inner_contact_diff_gt
        obs["bottom_tactile_current_mask"] = bottom_mask
        obs["inner_tactile_current_mask"] = inner_mask

        log = self.extras.setdefault("log", {})
        log["aux/visibility_gt_mean"] = visibility_gt.mean().detach()
        log["aux/occlusion_gt_mean"] = occlusion_gt.mean().detach()
        log["aux/bottom_contact_left_gt_mean"] = bottom_contact_gt[:, 0].mean().detach()
        log["aux/bottom_contact_right_gt_mean"] = bottom_contact_gt[:, 1].mean().detach()
        log["aux/bottom_rel_x_gt_mean"] = bottom_rel_pos_gt[:, 0].mean().detach()
        log["aux/bottom_rel_y_gt_mean"] = bottom_rel_pos_gt[:, 1].mean().detach()
        log["aux/inner_contact_gt_mean"] = inner_contact_gt.mean().detach()
        log["aux/inner_contact_diff_gt_mean"] = inner_contact_diff_gt.mean().detach()
        log["aux/bottom_mask_mean"] = bottom_mask.mean().detach()
        log["aux/inner_mask_mean"] = inner_mask.mean().detach()

        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return

        self._pending_stage_contact_rgb_baseline_refresh[env_ids] = True
        for baseline in self._stage_contact_rgb_baseline.values():
            if baseline is not None:
                baseline[env_ids] = 0.0

        self._latest_tactile_over_thresh_ratio_by_sensor[env_ids] = 0.0
        self._latest_visibility_gt[env_ids] = float(getattr(self.cfg, "reliability_fallback_visible_ratio", 0.0))
        self._latest_occlusion_gt[env_ids] = 1.0 - float(getattr(self.cfg, "reliability_fallback_visible_ratio", 0.0))
        self._latest_bottom_contact_gt[env_ids] = 0.0
        self._latest_bottom_rel_pos_gt[env_ids] = 0.0
        self._latest_inner_contact_gt[env_ids] = 0.0
        self._latest_inner_contact_diff_gt[env_ids] = 0.0
        self._latest_bottom_mask[env_ids] = 0.0
        self._latest_inner_mask[env_ids] = 0.0
