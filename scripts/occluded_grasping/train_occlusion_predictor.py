from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PredictorMetadata:
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    hidden_dims: list[int]
    target_name: str
    train_rows: int
    val_rows: int
    test_rows: int
    metrics: dict[str, float]


class OcclusionPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ELU())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def _as_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _target_column(fieldnames: list[str]) -> str:
    for name in ("occlusion_ratio", "bbox_occlusion_ratio"):
        if name in fieldnames:
            return name
    raise ValueError("CSV must contain either 'occlusion_ratio' or 'bbox_occlusion_ratio'")


def _available_features(fieldnames: list[str], requested: list[str] | None) -> list[str]:
    if requested:
        missing = [name for name in requested if name not in fieldnames]
        if missing:
            raise ValueError(f"Requested feature columns missing from CSV: {missing}")
        return requested
    features = ["x", "y"]
    for optional in ("geom_occlusion_ratio", "geom_visible_ratio"):
        if optional in fieldnames:
            features.append(optional)
    return features


def load_rows(paths: list[str], requested_features: list[str] | None) -> tuple[torch.Tensor, torch.Tensor, list[str], str]:
    xs: list[list[float]] = []
    ys: list[float] = []
    feature_names: list[str] | None = None
    target_name: str | None = None

    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            fieldnames = list(reader.fieldnames)
            this_target = _target_column(fieldnames)
            this_features = _available_features(fieldnames, requested_features)
            if target_name is None:
                target_name = this_target
            if feature_names is None:
                feature_names = this_features
            if this_features != feature_names:
                raise ValueError(
                    f"Feature columns differ in {path}: expected {feature_names}, got {this_features}"
                )

            for row in reader:
                if "bbox_found" in row and int(float(row.get("bbox_found") or 0)) == 0:
                    continue
                target = _as_float(row.get(this_target))
                features = [_as_float(row.get(name)) for name in this_features]
                if not math.isfinite(target) or any(not math.isfinite(value) for value in features):
                    continue
                ys.append(max(0.0, min(1.0, target)))
                xs.append(features)

    if not xs or feature_names is None or target_name is None:
        raise RuntimeError("No valid training rows were loaded")
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
        feature_names,
        target_name,
    )


def split_indices(n: int, val_fraction: float, test_fraction: float, seed: int):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    test_n = int(round(n * test_fraction))
    val_n = int(round(n * val_fraction))
    test_idx = indices[:test_n]
    val_idx = indices[test_n : test_n + val_n]
    train_idx = indices[test_n + val_n :]
    if not train_idx or not val_idx:
        raise ValueError("Split produced empty train or val set; reduce val/test fractions")
    return train_idx, val_idx, test_idx


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict[str, float]:
    model.eval()
    preds = []
    for start in range(0, x.shape[0], batch_size):
        preds.append(model(x[start : start + batch_size]))
    pred = torch.cat(preds, dim=0)
    err = pred - y
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err * err).mean()).item()
    y_mean = y.mean()
    pred_mean = pred.mean()
    cov = ((y - y_mean) * (pred - pred_mean)).sum()
    den = torch.sqrt(((y - y_mean) ** 2).sum() * ((pred - pred_mean) ** 2).sum())
    corr = (cov / den).item() if den.item() > 0 else float("nan")
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "corr": float(corr),
        "pred_mean": float(pred.mean().item()),
        "target_mean": float(y.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small MLP predictor for bbox occlusion ratio.")
    parser.add_argument("--csv", nargs="+", required=True, help="Training CSV(s) from scan/compare occlusion scripts.")
    parser.add_argument("--output", default=None, help="Output checkpoint path.")
    parser.add_argument("--features", nargs="*", default=None, help="Feature columns. Default: x y plus geometry columns if present.")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x_raw, y, feature_names, target_name = load_rows(args.csv, args.features)
    train_idx, val_idx, test_idx = split_indices(
        x_raw.shape[0], args.val_fraction, args.test_fraction, args.seed
    )
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long)

    mean = x_raw[train_idx_t].mean(dim=0)
    std = x_raw[train_idx_t].std(dim=0).clamp_min(1.0e-6)
    x = (x_raw - mean) / std

    train_loader = DataLoader(
        TensorDataset(x[train_idx_t], y[train_idx_t]),
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    model = OcclusionPredictor(input_dim=x.shape[1], hidden_dims=list(args.hidden_dims))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, x[val_idx_t], y[val_idx_t], args.batch_size)
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"[epoch {epoch:04d}] val_mae={val_metrics['mae']:.5f} "
                f"val_rmse={val_metrics['rmse']:.5f} val_corr={val_metrics['corr']:.4f}"
            )
        if stale_epochs >= int(args.patience):
            print(f"[INFO] Early stopping at epoch {epoch}; best val_mae={best_val:.5f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, x[train_idx_t], y[train_idx_t], args.batch_size)
    val_metrics = evaluate(model, x[val_idx_t], y[val_idx_t], args.batch_size)
    test_metrics = evaluate(model, x[test_idx_t], y[test_idx_t], args.batch_size) if len(test_idx) else {}
    metrics = {
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "train_corr": train_metrics["corr"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_corr": val_metrics["corr"],
    }
    for key, value in test_metrics.items():
        metrics[f"test_{key}"] = value

    metadata = PredictorMetadata(
        feature_names=feature_names,
        feature_mean=[float(v) for v in mean.tolist()],
        feature_std=[float(v) for v in std.tolist()],
        hidden_dims=list(args.hidden_dims),
        target_name=target_name,
        train_rows=len(train_idx),
        val_rows=len(val_idx),
        test_rows=len(test_idx),
        metrics=metrics,
    )

    out_path = args.output
    if out_path is None:
        out_dir = os.path.join("logs", "occlusion_predictor")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"occlusion_predictor_{time.strftime('%Y%m%d_%H%M%S')}.pt")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": asdict(metadata),
        },
        out_path,
    )
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)

    print("[INFO] Features:", ", ".join(feature_names))
    print("[INFO] Metrics:", json.dumps(metrics, indent=2))
    print(f"[INFO] Saved predictor: {out_path}")


if __name__ == "__main__":
    main()
