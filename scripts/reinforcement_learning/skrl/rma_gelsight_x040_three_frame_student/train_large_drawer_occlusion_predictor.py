"""Train the grouped-split XY occlusion predictor for Large Drawer Aux Students."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _extend_repo_pythonpath() -> None:
    root = Path(__file__).resolve().parents[4]
    for relative in ("source/tacex_tasks", "source/tacex", "source/tacex_assets"):
        value = str(root / relative)
        if value not in sys.path:
            sys.path.insert(0, value)


_extend_repo_pythonpath()

from tacex_tasks.sim2real_gelsight_rma.large_drawer_occlusion import (  # noqa: E402
    LargeDrawerXYOcclusionPredictor,
    OCCLUSION_PREDICTOR_KIND,
    OCCLUSION_PREDICTOR_VERSION,
    OCCLUSION_PROFILE,
    load_occlusion_scan_rows,
    rows_to_tensors,
    split_rows_by_geometry_seed,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _metrics(model, features, targets) -> dict[str, float]:
    prediction = model(features)
    error = prediction - targets
    target_centered = targets - targets.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = torch.sqrt(
        target_centered.square().sum() * prediction_centered.square().sum()
    )
    correlation = (
        float((target_centered * prediction_centered).sum().div(denominator).item())
        if denominator.item() > 0.0
        else float("nan")
    )
    return {
        "mae": float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "correlation": correlation,
        "target_mean": float(targets.mean().item()),
        "prediction_mean": float(prediction.mean().item()),
    }


def _atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch_size must be positive")
    if args.val_fraction <= 0.0 or args.test_fraction <= 0.0:
        raise ValueError("val_fraction and test_fraction must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = [Path(value).expanduser().resolve() for value in args.csv]
    rows = load_occlusion_scan_rows(paths)
    split_rows = split_rows_by_geometry_seed(
        rows,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    train_x, train_y = rows_to_tensors(split_rows["train"])
    val_x, val_y = rows_to_tensors(split_rows["val"])
    test_x, test_y = rows_to_tensors(split_rows["test"])
    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0).clamp_min(1.0e-6)

    model = LargeDrawerXYOcclusionPredictor((64, 64))
    model.set_normalization(feature_mean, feature_std)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True
    )
    best_state = None
    best_mae = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, target in loader:
            loss = F.mse_loss(model(features), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        val_metrics = _metrics(model, val_x, val_y)
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 100 == 0:
            print(f"[occlusion predictor] epoch={epoch} val_mae={val_metrics['mae']:.5f}")
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Occlusion predictor training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()

    metrics = {
        split: _metrics(model, *rows_to_tensors(split_rows[split]))
        for split in ("train", "val", "test")
    }
    metadata = {
        "profile": OCCLUSION_PROFILE,
        "feature_names": ["x", "y"],
        "feature_frame": "robot_root",
        "hidden_dims": [64, 64],
        "target": "initial_geometric_bbox_occlusion_ratio",
        "target_source": "replicator_bounding_box_3d_before_rgb_downsample",
        "split": "grouped_by_geometry_seed",
        "seed": args.seed,
        "row_counts": {name: len(values) for name, values in split_rows.items()},
        "geometry_seed_counts": {
            name: len({int(row["geometry_seed"]) for row in values})
            for name, values in split_rows.items()
        },
        "source_csv": [str(path) for path in paths],
        "source_csv_sha256": {str(path): _sha256(path) for path in paths},
        "metrics": metrics,
    }
    output = Path(args.output).expanduser().resolve()
    _atomic_save(
        {
            "kind": OCCLUSION_PREDICTOR_KIND,
            "version": OCCLUSION_PREDICTOR_VERSION,
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
        },
        output,
    )
    _atomic_json(metadata, output.with_suffix(".json"))
    print(json.dumps(metrics, indent=2))
    print(f"[INFO] Saved predictor: {output}")


if __name__ == "__main__":
    main()
