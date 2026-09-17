"""Offline tests for Large Drawer occlusion artifacts and statistics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tacex_tasks/sim2real_gelsight_rma/large_drawer_occlusion.py"
    )
    spec = importlib.util.spec_from_file_location("large_drawer_occlusion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


occlusion = _module()


def test_geometry_group_split_has_no_seed_leakage() -> None:
    rows = [
        {
            "geometry_seed": seed,
            "x": float(index),
            "y": 0.0,
            "occlusion_ratio": 0.1 * index,
        }
        for seed in range(10)
        for index in range(3)
    ]
    split = occlusion.split_rows_by_geometry_seed(
        rows, val_fraction=0.2, test_fraction=0.2, seed=42
    )
    seed_sets = {
        name: {row["geometry_seed"] for row in values}
        for name, values in split.items()
    }
    assert seed_sets["train"].isdisjoint(seed_sets["val"])
    assert seed_sets["train"].isdisjoint(seed_sets["test"])
    assert seed_sets["val"].isdisjoint(seed_sets["test"])
    assert sum(len(values) for values in split.values()) == len(rows)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, 0), (0.0, 0), (0.099, 0), (0.1, 1), (0.699, 6), (0.7, 7), (1.0, 7)],
)
def test_occlusion_bin_boundaries(value: float, expected: int) -> None:
    assert occlusion.occlusion_bin_index(value) == expected


def test_predictor_artifact_round_trip_and_wilson_interval(tmp_path: Path) -> None:
    model = occlusion.LargeDrawerXYOcclusionPredictor((64, 64))
    model.set_normalization(torch.tensor([0.5, 0.0]), torch.tensor([0.1, 0.1]))
    metadata = {
        "profile": occlusion.OCCLUSION_PROFILE,
        "feature_names": ["x", "y"],
        "feature_frame": "robot_root",
        "hidden_dims": [64, 64],
        "target": "initial_geometric_bbox_occlusion_ratio",
        "target_source": "replicator_bounding_box_3d_before_rgb_downsample",
        "split": "grouped_by_geometry_seed",
        "source_csv_sha256": {"scan.csv": "abc"},
        "metrics": {"train": {}, "val": {}, "test": {}},
    }
    path = tmp_path / "predictor.pt"
    torch.save(
        {
            "kind": occlusion.OCCLUSION_PREDICTOR_KIND,
            "version": occlusion.OCCLUSION_PREDICTOR_VERSION,
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    loaded, loaded_metadata = occlusion.load_xy_occlusion_predictor(path)
    probe = torch.tensor([[0.5, 0.0]])
    assert torch.equal(model(probe), loaded(probe))
    assert loaded_metadata == metadata
    low, high = occlusion.wilson_interval(50, 100)
    assert low < 0.5 < high
