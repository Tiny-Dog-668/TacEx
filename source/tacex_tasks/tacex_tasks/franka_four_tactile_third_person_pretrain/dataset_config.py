from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_OBJECT_IDS = (
    "sphere_xs_25",
    "sphere_s_35",
    "sphere_m_50",
    "sphere_l_65",
    "box_cube_25",
    "box_cube_40",
    "box_cube_55",
    "box_rect_60_40_30",
    "box_rect_80_30_25",
    "box_flat_50_50_20",
    "box_tall_20_20_60",
    "box_ridge_80_12_30",
    "box_slim_16_16_35",
    "cylinder_slim_18_60",
    "cylinder_short_30_60",
    "cylinder_mid_25_70",
    "cylinder_tall_25_100",
    "cylinder_thick_40_50",
    "plate_small_60_60_8",
    "plate_80_80_10",
    "plate_rect_100_60_8",
)

SUPPORTED_PRIMITIVES = ("touch", "press", "slide", "pinch_grasp")


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    family: str
    shape: str
    size_mm: tuple[float, ...]
    target_primitives: tuple[str, ...]
    split: str

    @property
    def size_m(self) -> tuple[float, ...]:
        return tuple(v / 1000.0 for v in self.size_mm)

    @property
    def half_height_m(self) -> float:
        if self.shape == "sphere":
            return self.size_m[0] * 0.5
        if self.shape == "cylinder":
            return self.size_m[1] * 0.5
        if self.shape in {"box", "plate"}:
            return self.size_m[2] * 0.5
        raise ValueError(f"Unsupported shape for height inference: {self.shape}")

    @property
    def top_clearance_hover_m(self) -> float:
        return max(0.05, self.half_height_m + 0.05)

    @property
    def side_grasp_height_offset_m(self) -> float:
        return min(max(self.half_height_m * 0.5, 0.012), 0.04)

    @property
    def nominal_grasp_width_m(self) -> float:
        if self.shape == "sphere":
            return min(max(self.size_m[0] * 1.15, 0.045), 0.08)
        if self.shape == "cylinder":
            radius = self.size_m[0]
            diameter = radius * 2.0
            return min(max(diameter * 1.20, 0.045), 0.08)
        if self.shape == "box":
            lateral = self.size_m[1]
            return min(max(lateral * 1.20, 0.045), 0.08)
        return 0.08

    def supports(self, primitive: str) -> bool:
        return primitive in self.target_primitives and primitive in SUPPORTED_PRIMITIVES


def resolve_manifest_paths(base_dir: Path | None = None) -> tuple[Path, Path]:
    root = base_dir or Path(__file__).resolve().parent
    return root / "object_pool.yaml", root / "collection_protocol.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_collection_protocol(path: Path | None = None) -> dict[str, Any]:
    _, protocol_path = resolve_manifest_paths(path.parent if path is not None else None)
    if path is None:
        path = protocol_path
    return _load_yaml(path)


def load_supported_object_specs(path: Path | None = None, splits: tuple[str, ...] | None = None) -> dict[str, ObjectSpec]:
    object_pool_path, _ = resolve_manifest_paths(path.parent if path is not None else None)
    if path is None:
        path = object_pool_path
    data = _load_yaml(path)
    allowed_splits = set(splits) if splits else None

    specs: dict[str, ObjectSpec] = {}
    for family, payload in data["families"].items():
        for obj in payload["objects"]:
            object_id = obj["id"]
            if object_id not in SUPPORTED_OBJECT_IDS:
                continue
            split = obj["split"]
            if allowed_splits is not None and split not in allowed_splits:
                continue
            spec = ObjectSpec(
                id=object_id,
                family=family,
                shape=obj["shape"],
                size_mm=tuple(float(v) for v in obj["size_mm"]),
                target_primitives=tuple(str(v) for v in obj["target_primitives"]),
                split=split,
            )
            specs[object_id] = spec
    return specs


def supported_object_ids() -> tuple[str, ...]:
    return SUPPORTED_OBJECT_IDS
