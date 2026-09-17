"""Replicator bbox helpers shared by Large Drawer scan and evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def enable_target_object_semantics(cfg) -> None:
    try:
        cfg.cube = cfg.cube.replace(
            spawn=cfg.cube.spawn.replace(semantic_tags=[("class", "target_object")])
        )
    except Exception:
        cfg.cube.spawn.semantic_tags = [("class", "target_object")]


def _env_id(path: str) -> int | None:
    prefix = "/World/envs/env_"
    if not path.startswith(prefix):
        return None
    digits = []
    for char in path[len(prefix) :]:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else None


def _unpack(value):
    if isinstance(value, dict):
        return value.get("data"), value.get("info", {})
    return value, {}


class LargeDrawerBBoxReader:
    """Read target-cylinder occlusion ratio for every replicated wrist camera."""

    def __init__(self, base_env) -> None:
        self.base_env = base_env
        self.records = []

    def initialize(self) -> None:
        import omni.replicator.core as rep  # type: ignore

        camera = self.base_env.wrist_camera
        paths = getattr(getattr(camera, "_view", None), "prim_paths", None)
        if not isinstance(paths, (tuple, list)) or not paths:
            raise RuntimeError("Could not resolve per-environment wrist camera prims")
        for camera_path in paths:
            env_id = _env_id(str(camera_path))
            if env_id is None:
                continue
            product = rep.create.render_product(
                str(camera_path), resolution=(int(camera.cfg.width), int(camera.cfg.height))
            )
            product_path = product if isinstance(product, str) else product.path
            annotator = rep.AnnotatorRegistry.get_annotator(
                "bounding_box_3d", init_params={"semanticTypes": ["class"]}
            )
            try:
                annotator.attach(product_path)
            except Exception:
                annotator.attach([product_path])
            self.records.append((env_id, annotator, str(product_path)))
        if not self.records:
            raise RuntimeError("No bbox annotators were attached")

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        occlusion = torch.full(
            (self.base_env.num_envs,), float("nan"), device=self.base_env.device
        )
        found = torch.zeros(
            (self.base_env.num_envs,), dtype=torch.bool, device=self.base_env.device
        )
        for record_env, annotator, _ in self.records:
            try:
                data, info = _unpack(annotator.get_data())
            except Exception:
                continue
            paths = info.get("primPaths", [])
            if data is None or not isinstance(paths, Iterable):
                continue
            for row_index, prim_path_value in enumerate(paths):
                prim_path = str(prim_path_value)
                env_id = _env_id(prim_path)
                target = f"/World/envs/env_{record_env}/cube"
                if env_id != record_env or not (
                    prim_path == target or prim_path.startswith(f"{target}/")
                ):
                    continue
                row = data[row_index]
                value = row.get("occlusionRatio") if isinstance(row, dict) else row["occlusionRatio"]
                try:
                    ratio = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(ratio) and ratio >= 0.0:
                    ratio = min(1.0, ratio)
                    current = occlusion[env_id]
                    occlusion[env_id] = ratio if torch.isnan(current) else max(current.item(), ratio)
                    found[env_id] = True
        return occlusion, found

    def close(self) -> None:
        for _, annotator, product in self.records:
            try:
                annotator.detach([product])
            except Exception:
                try:
                    annotator.detach(product)
                except Exception:
                    pass


__all__ = ("LargeDrawerBBoxReader", "enable_target_object_semantics")
