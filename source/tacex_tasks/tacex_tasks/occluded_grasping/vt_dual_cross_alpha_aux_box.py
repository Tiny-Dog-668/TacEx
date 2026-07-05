"""Dual-cross-alpha downsample task with occlusion/contact supervised alpha inputs."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from isaaclab.utils import configclass

from .vt_box import OccludedGraspingVTAlphaBoxEnv, OccludedGraspingVTAlphaDownsampleBoxCfg


def _enable_aux_alpha_observations(cfg) -> None:
    cfg.observation_space = dict(cfg.observation_space)
    cfg.observation_space["aux_occlusion_gt"] = 1
    cfg.observation_space["aux_tactile_contact_gt"] = 4

    try:
        cfg.can = cfg.can.replace(spawn=cfg.can.spawn.replace(semantic_tags=[("class", "can")]))
    except Exception:
        try:
            cfg.can.spawn.semantic_tags = [("class", "can")]
        except Exception:
            pass


@configclass
class OccludedGraspingVTDualCrossAlphaAuxDownsampleBoxCfg(OccludedGraspingVTAlphaDownsampleBoxCfg):
    """Downsample VT cfg exposing bbox occlusion and tactile contact GT labels."""

    aux_alpha_use_bbox3d = True
    aux_alpha_fallback_occlusion = 0.0
    aux_alpha_bbox_lazy_init = True

    def __post_init__(self):
        super().__post_init__()
        _enable_aux_alpha_observations(self)

    def _post_configure_scene_object(self):
        _enable_aux_alpha_observations(self)


class OccludedGraspingVTDualCrossAlphaAuxBoxEnv(OccludedGraspingVTAlphaBoxEnv):
    """VT env that appends bbox-derived occlusion and tactile-contact GT labels."""

    cfg: OccludedGraspingVTDualCrossAlphaAuxDownsampleBoxCfg

    def __init__(
        self,
        cfg: OccludedGraspingVTDualCrossAlphaAuxDownsampleBoxCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._bbox_annotator = None
        self._bbox_render_products: list[str] = []
        self._bbox_enabled = False
        self._bbox_init_attempted = False
        self._latest_aux_occlusion_gt = torch.full(
            (self.num_envs, 1),
            float(getattr(self.cfg, "aux_alpha_fallback_occlusion", 0.0)),
            dtype=torch.float32,
            device=self.device,
        )
        self._latest_aux_tactile_contact_gt = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        if not bool(getattr(self.cfg, "aux_alpha_bbox_lazy_init", True)):
            self._init_bbox_annotator()

    def _resolve_render_product_paths(self) -> list[str]:
        candidates = ("render_product_paths", "render_product_path", "_render_product_paths", "_render_product_path")
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
        if not bool(getattr(self.cfg, "aux_alpha_use_bbox3d", True)):
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
        return max(0.0, min(1.0, ratio))

    def _compute_occlusion_gt(self) -> torch.Tensor:
        fallback = float(getattr(self.cfg, "aux_alpha_fallback_occlusion", 0.0))
        occlusion = torch.full((self.num_envs,), fallback, dtype=torch.float32, device=self.device)
        debug_bbox = bool(getattr(self.cfg, "print_bbox_debug", False))
        debug_interval = int(
            getattr(
                self.cfg,
                "bbox_debug_print_interval",
                getattr(
                    self.cfg,
                    "visual_visible_ratio_print_interval",
                    getattr(self.cfg, "reward_print_interval", 200),
                ),
            )
        )
        if debug_interval <= 0:
            debug_interval = int(getattr(self.cfg, "reward_print_interval", 200))
        step_count = int(getattr(self, "step_count", 0))
        should_debug_bbox = debug_bbox and (debug_interval <= 1 or step_count % debug_interval == 0)

        if (
            bool(getattr(self.cfg, "aux_alpha_use_bbox3d", True))
            and not self._bbox_enabled
            and not self._bbox_init_attempted
        ):
            self._init_bbox_annotator()

        if not self._bbox_enabled or self._bbox_annotator is None:
            if should_debug_bbox:
                print(
                    f"[bbox_debug] step={step_count} annotator_enabled={self._bbox_enabled} "
                    f"annotator_present={self._bbox_annotator is not None} "
                    f"render_products={self._bbox_render_products}",
                    flush=True,
                )
            return occlusion

        try:
            bbox_output = self._bbox_annotator.get_data()
        except Exception as exc:
            if should_debug_bbox:
                print(f"[bbox_debug] step={step_count} get_data_failed={exc!r}", flush=True)
            return occlusion

        bbox_data, bbox_info = self._unpack_bbox_output(bbox_output)
        prim_paths = bbox_info.get("primPaths", [])
        if bbox_data is None or not isinstance(prim_paths, Iterable):
            if should_debug_bbox:
                print(
                    f"[bbox_debug] step={step_count} invalid_output "
                    f"data_type={type(bbox_data).__name__} primPaths_type={type(prim_paths).__name__} "
                    f"info_keys={list(bbox_info.keys())}",
                    flush=True,
                )
            return occlusion
        prim_paths = list(prim_paths)

        max_occlusion_by_env: list[float | None] = [None] * self.num_envs
        debug_rows: list[str] = []
        for row_idx, prim_path in enumerate(prim_paths):
            path_str = str(prim_path)
            env_id = self._extract_env_id_from_path(path_str)
            debug_occlusion_ratio: float | None = None
            if env_id is None or env_id < 0 or env_id >= self.num_envs:
                if should_debug_bbox and len(debug_rows) < 80:
                    debug_rows.append(f"{row_idx}:env=None occ=None path={path_str}")
                continue

            can_prefix = f"/World/envs/env_{env_id}/can"
            if path_str != can_prefix and not path_str.startswith(f"{can_prefix}/"):
                if should_debug_bbox and len(debug_rows) < 80:
                    debug_rows.append(f"{row_idx}:env={env_id} can=False occ=None path={path_str}")
                continue

            try:
                row = bbox_data[row_idx]
            except Exception:
                if should_debug_bbox and len(debug_rows) < 80:
                    debug_rows.append(f"{row_idx}:env={env_id} can=True row_error path={path_str}")
                continue
            occlusion_ratio = self._extract_occlusion_ratio(row)
            debug_occlusion_ratio = occlusion_ratio
            if should_debug_bbox and len(debug_rows) < 80:
                debug_rows.append(
                    f"{row_idx}:env={env_id} can=True occ={debug_occlusion_ratio} path={path_str}"
                )
            if occlusion_ratio is None:
                continue
            previous = max_occlusion_by_env[env_id]
            max_occlusion_by_env[env_id] = occlusion_ratio if previous is None else max(previous, occlusion_ratio)

        for env_id, occlusion_ratio in enumerate(max_occlusion_by_env):
            if occlusion_ratio is not None:
                occlusion[env_id] = float(occlusion_ratio)
        if should_debug_bbox:
            matched_text = ", ".join(
                f"env_{env_id}={value if value is not None else 'None'}"
                for env_id, value in enumerate(max_occlusion_by_env)
            )
            row_text = " | ".join(debug_rows) if debug_rows else "<no rows>"
            print(
                f"[bbox_debug] step={step_count} rows={len(prim_paths)} "
                f"matched_occlusion={matched_text}",
                flush=True,
            )
            print(f"[bbox_debug] rows: {row_text}", flush=True)
        return occlusion

    def _compute_tactile_contact_gt(self) -> torch.Tensor:
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
        ratio_by_sensor = torch.stack(ratios, dim=-1)
        self._latest_tactile_contact_ratio_by_sensor = ratio_by_sensor
        self._latest_tactile_over_thresh_ratio_by_sensor = ratio_by_sensor
        self._latest_tactile_over_thresh_ratio_mean = ratio_by_sensor.mean(dim=-1)
        return ratio_by_sensor

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        occlusion_gt = self._compute_occlusion_gt().unsqueeze(-1)
        tactile_contact_gt = self._compute_tactile_contact_gt()
        self._latest_aux_occlusion_gt = occlusion_gt
        self._latest_aux_tactile_contact_gt = tactile_contact_gt
        obs["aux_occlusion_gt"] = occlusion_gt
        obs["aux_tactile_contact_gt"] = tactile_contact_gt

        log = self.extras.setdefault("log", {})
        log["aux/occlusion_gt_mean"] = occlusion_gt.mean().detach()
        log["aux/tactile_contact_gt_mean"] = tactile_contact_gt.mean().detach()
        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        self._latest_aux_occlusion_gt[env_ids] = float(getattr(self.cfg, "aux_alpha_fallback_occlusion", 0.0))
        self._latest_aux_tactile_contact_gt[env_ids] = 0.0
