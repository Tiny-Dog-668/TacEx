"""Offline utilities for Large Drawer XY-to-initial-occlusion artifacts."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


OCCLUSION_PREDICTOR_KIND = "tacex_large_drawer_xy_occlusion_predictor"
OCCLUSION_PREDICTOR_VERSION = 1
OCCLUSION_PROFILE = "large_pulled_drawer_cylinder_initial_bbox_occlusion_v1"
DEFAULT_OCCLUSION_BINS = tuple(round(index * 0.1, 1) for index in range(8)) + (1.0,)


def validate_occlusion_predictor_metadata(metadata: Mapping[str, Any]) -> None:
    """Fail closed on the scan protocol and held-out validation provenance."""
    metrics = metadata.get("metrics")
    if (
        metadata.get("profile") != OCCLUSION_PROFILE
        or metadata.get("feature_names") != ["x", "y"]
        or metadata.get("feature_frame") != "robot_root"
        or metadata.get("hidden_dims") != [64, 64]
        or metadata.get("target") != "initial_geometric_bbox_occlusion_ratio"
        or metadata.get("target_source")
        != "replicator_bounding_box_3d_before_rgb_downsample"
        or metadata.get("split") != "grouped_by_geometry_seed"
        or not isinstance(metadata.get("source_csv_sha256"), Mapping)
        or not isinstance(metrics, Mapping)
        or not all(isinstance(metrics.get(name), Mapping) for name in ("train", "val", "test"))
    ):
        raise RuntimeError("Unsupported Large Drawer occlusion predictor metadata")


class LargeDrawerXYOcclusionPredictor(nn.Module):
    """Small frozen-runtime MLP: robot-root object XY -> bbox occlusion [0, 1]."""

    def __init__(self, hidden_dims: Iterable[int] = (64, 64)) -> None:
        super().__init__()
        dims = [2, *(int(value) for value in hidden_dims), 1]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)
        self.register_buffer("feature_mean", torch.zeros(2))
        self.register_buffer("feature_std", torch.ones(2))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean.to(self.feature_mean).reshape(2))
        self.feature_std.copy_(std.to(self.feature_std).reshape(2).clamp_min(1.0e-6))

    def forward(self, object_xy_root_m: torch.Tensor) -> torch.Tensor:
        normalized = (object_xy_root_m.to(torch.float32) - self.feature_mean) / self.feature_std
        return torch.sigmoid(self.network(normalized))


def load_xy_occlusion_predictor(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[LargeDrawerXYOcclusionPredictor, dict[str, Any]]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Occlusion predictor not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Invalid Large Drawer occlusion predictor payload")
    metadata = payload.get("metadata")
    if (
        payload.get("kind") != OCCLUSION_PREDICTOR_KIND
        or payload.get("version") != OCCLUSION_PREDICTOR_VERSION
        or not isinstance(metadata, Mapping)
    ):
        raise RuntimeError("Unsupported Large Drawer occlusion predictor contract")
    validate_occlusion_predictor_metadata(metadata)
    hidden_dims = tuple(int(value) for value in metadata.get("hidden_dims", (64, 64)))
    model = LargeDrawerXYOcclusionPredictor(hidden_dims).to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Occlusion predictor state_dict is missing")
    model.load_state_dict(dict(state), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dict(metadata)


def load_occlusion_scan_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {"geometry_seed", "x", "y", "occlusion_ratio", "bbox_found"}
    for value in paths:
        path = Path(value).expanduser().resolve()
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                missing = sorted(required.difference(reader.fieldnames or ()))
                raise ValueError(f"Occlusion CSV {path} is missing columns: {missing}")
            for row in reader:
                if int(float(row["bbox_found"])) != 1:
                    continue
                parsed = {
                    "geometry_seed": int(float(row["geometry_seed"])),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "occlusion_ratio": float(row["occlusion_ratio"]),
                    "source": str(path),
                }
                if all(math.isfinite(float(parsed[key])) for key in ("x", "y", "occlusion_ratio")):
                    parsed["occlusion_ratio"] = min(1.0, max(0.0, parsed["occlusion_ratio"]))
                    rows.append(parsed)
    if not rows:
        raise RuntimeError("No valid bbox occlusion rows were loaded")
    return rows


def split_rows_by_geometry_seed(
    rows: list[dict[str, Any]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["geometry_seed"])].append(row)
    group_ids = sorted(groups)
    if len(group_ids) < 3:
        raise ValueError("At least three geometry_seed groups are required")
    random.Random(seed).shuffle(group_ids)
    test_count = max(1, int(round(len(group_ids) * test_fraction)))
    val_count = max(1, int(round(len(group_ids) * val_fraction)))
    if test_count + val_count >= len(group_ids):
        raise ValueError("Geometry split leaves no training groups")
    ids = {
        "test": set(group_ids[:test_count]),
        "val": set(group_ids[test_count : test_count + val_count]),
        "train": set(group_ids[test_count + val_count :]),
    }
    return {
        split: [row for group_id in selected for row in groups[group_id]]
        for split, selected in ids.items()
    }


def rows_to_tensors(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor([[row["x"], row["y"]] for row in rows], dtype=torch.float32)
    targets = torch.tensor([[row["occlusion_ratio"]] for row in rows], dtype=torch.float32)
    return features, targets


def occlusion_bin_index(value: float, edges: tuple[float, ...] = DEFAULT_OCCLUSION_BINS) -> int:
    clipped = min(1.0, max(0.0, float(value)))
    for index in range(len(edges) - 1):
        upper = edges[index + 1]
        if clipped < upper or index == len(edges) - 2:
            return index
    return len(edges) - 2


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


__all__ = (
    "DEFAULT_OCCLUSION_BINS",
    "LargeDrawerXYOcclusionPredictor",
    "OCCLUSION_PREDICTOR_KIND",
    "OCCLUSION_PREDICTOR_VERSION",
    "OCCLUSION_PROFILE",
    "load_occlusion_scan_rows",
    "load_xy_occlusion_predictor",
    "occlusion_bin_index",
    "rows_to_tensors",
    "split_rows_by_geometry_seed",
    "validate_occlusion_predictor_metadata",
    "wilson_interval",
)
