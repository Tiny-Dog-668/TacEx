#!/usr/bin/env python3
"""Plot success_rate and avg curves from play_metrics CSV."""

from __future__ import annotations

import argparse
import csv
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot play metrics CSV into a PNG figure.")
    parser.add_argument("--csv", required=True, help="Path to play_metrics_*.csv")
    parser.add_argument("--out", default=None, help="Output PNG path (default: same stem as CSV)")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    steps: list[float] = []
    success_rate: list[float] = []
    avg: list[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(float(row.get("step", "nan")))
                success_rate.append(float(row.get("success_rate", "nan")))
                avg_text = row.get("avg", "")
                avg.append(float(avg_text) if avg_text not in {"", None} else float("nan"))
            except ValueError:
                continue

    if not steps:
        raise RuntimeError(f"No valid rows found in CSV: {csv_path}")

    out_path = os.path.abspath(args.out) if args.out else os.path.splitext(csv_path)[0] + ".png"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, success_rate, label="success_rate", linewidth=1.6)
    if any(v == v for v in avg):
        ax.plot(steps, avg, label="avg", linewidth=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title("Play Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved figure: {out_path}")


if __name__ == "__main__":
    main()
