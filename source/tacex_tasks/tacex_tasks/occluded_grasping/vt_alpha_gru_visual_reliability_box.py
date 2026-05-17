"""Self-occlusion Alpha-GRU box task with visual reliability supervision."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from isaaclab.utils import configclass

from .vt_alpha_gru_box import (
    OccludedGraspingVTAlphaGRUBoxEnv,
    OccludedGraspingVTAlphaGRUDrawerOcclusionBoxCfg,
    OccludedGraspingVTAlphaGRUDrawerOcclusionCubeCfg,
    OccludedGraspingVTAlphaGRUSelfOcclusionBoxCfg,
    OccludedGraspingVTAlphaGRUSelfOcclusionCubeCfg,
)


def _enable_visual_reliability_observation(cfg) -> None:
    cfg.observation_space = dict(cfg.observation_space)
    cfg.observation_space["aux_visual_reliability_gt"] = 1

    try:
        cfg.can = cfg.can.replace(
            spawn=cfg.can.spawn.replace(
                semantic_tags=[("class", "can")],
            )
        )
    except Exception:
        try:
            cfg.can.spawn.semantic_tags = [("class", "can")]
        except Exception:
            pass


class _VisualReliabilityCfg:
    reliability_use_bbox3d = True
    reliability_fallback_visible_ratio = 0.0
    reliability_bbox_lazy_init = True


@configclass
class OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionBoxCfg(
    _VisualReliabilityCfg,
    OccludedGraspingVTAlphaGRUDrawerOcclusionBoxCfg,
):
    """Drawer-occlusion Alpha-GRU cylinder task exposing a visual reliability target."""

    def __post_init__(self):
        super().__post_init__()
        _enable_visual_reliability_observation(self)


@configclass
class OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionCubeCfg(
    _VisualReliabilityCfg,
    OccludedGraspingVTAlphaGRUDrawerOcclusionCubeCfg,
):
    """Drawer-occlusion Alpha-GRU cube task exposing a visual reliability target."""

    def __post_init__(self):
        super().__post_init__()
        _enable_visual_reliability_observation(self)


@configclass
class OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxCfg(
    _VisualReliabilityCfg,
    OccludedGraspingVTAlphaGRUSelfOcclusionBoxCfg
):
    """Self-occlusion Alpha-GRU task exposing a visual reliability target."""

    def __post_init__(self):
        super().__post_init__()
        _enable_visual_reliability_observation(self)


@configclass
class OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionCubeCfg(
    _VisualReliabilityCfg,
    OccludedGraspingVTAlphaGRUSelfOcclusionCubeCfg
):
    """Self-occlusion Alpha-GRU cube task exposing a visual reliability target."""

    def __post_init__(self):
        super().__post_init__()
        _enable_visual_reliability_observation(self)


class OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxEnv(OccludedGraspingVTAlphaGRUBoxEnv):
    """Alpha-GRU self-occlusion env with bbox-derived visual reliability labels."""

    cfg: (
        OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionBoxCfg
        | OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionCubeCfg
        |
        OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxCfg
        | OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionCubeCfg
    )

    def __init__(
        self,
        cfg: (
            OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionBoxCfg
            | OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionCubeCfg
            |
            OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxCfg
            | OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionCubeCfg
        ),
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._bbox_annotator = None
        self._bbox_render_products: list[str] = []
        self._bbox_enabled = False
        self._bbox_init_attempted = False
        self._latest_visual_reliability_gt = torch.full(
            (self.num_envs, 1),
            float(getattr(self.cfg, "reliability_fallback_visible_ratio", 0.0)),
            dtype=torch.float32,
            device=self.device,
        )
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

    def _compute_visual_reliability_gt(self) -> torch.Tensor:
        fallback = float(getattr(self.cfg, "reliability_fallback_visible_ratio", 0.0))
        reliability = torch.full((self.num_envs,), fallback, dtype=torch.float32, device=self.device)

        if (
            bool(getattr(self.cfg, "reliability_use_bbox3d", True))
            and not self._bbox_enabled
            and not self._bbox_init_attempted
        ):
            self._init_bbox_annotator()

        if not self._bbox_enabled or self._bbox_annotator is None:
            return reliability

        try:
            bbox_output = self._bbox_annotator.get_data()
        except Exception:
            return reliability

        bbox_data, bbox_info = self._unpack_bbox_output(bbox_output)
        prim_paths = bbox_info.get("primPaths", [])
        if bbox_data is None or not isinstance(prim_paths, Iterable):
            return reliability

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
            reliability[env_id] = float(max(0.0, min(1.0, 1.0 - occlusion_ratio)))

        return reliability

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        visual_reliability_gt = self._compute_visual_reliability_gt().unsqueeze(-1)
        self._latest_visual_reliability_gt = visual_reliability_gt
        obs["aux_visual_reliability_gt"] = visual_reliability_gt

        log = self.extras.setdefault("log", {})
        log["aux/visual_reliability_gt_mean"] = visual_reliability_gt.mean().detach()
        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        self._latest_visual_reliability_gt[env_ids] = float(
            getattr(self.cfg, "reliability_fallback_visible_ratio", 0.0)
        )
