from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict


parser = argparse.ArgumentParser(description="Analyze bucket success rate grouped by precomputed occlusion ratio.")
parser.add_argument("--play_csv", required=True, help="play_bucket.py detail CSV.")
parser.add_argument("--occlusion_csv", required=True, help="scan_bucket_occlusion.py CSV for the same object/grid.")
parser.add_argument(
    "--out_prefix",
    default=None,
    help=(
        "Output prefix. Defaults to logs/bucket_analysis/<run_dir>/<play_csv_stem>_by_occlusion, "
        "where <run_dir> is inferred from --play_csv."
    ),
)
parser.add_argument("--bin_size", type=float, default=0.1, help="Occlusion bin size, e.g. 0.1 gives 10 bins.")
parser.add_argument(
    "--include_missing",
    action="store_true",
    default=False,
    help="Include rows whose occlusion map has bbox_found=0 or non-finite occlusion as a missing bin.",
)
parser.add_argument(
    "--title",
    default=None,
    help="Optional plot title.",
)
args = parser.parse_args()


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _as_float(value, default: float = float("nan")) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _read_occlusion_map(path: str) -> dict[int, tuple[float, bool]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    mapping: dict[int, tuple[float, bool]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"grid_index", "occlusion_ratio"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Occlusion CSV missing columns: {sorted(missing)}")
        for row in reader:
            grid_index = _as_int(row.get("grid_index"), -1)
            if grid_index < 0:
                continue
            ratio = _as_float(row.get("occlusion_ratio"))
            bbox_found = _as_int(row.get("bbox_found", "1"), 1) != 0
            mapping[grid_index] = (ratio, bbox_found)
    if not mapping:
        raise RuntimeError(f"No occlusion rows loaded from {path}")
    return mapping


def _bin_label(value: float, bin_size: float) -> tuple[str, float, float]:
    if not math.isfinite(value):
        return "missing", float("nan"), float("nan")
    value = min(max(value, 0.0), 1.0)
    if value >= 1.0:
        lo = max(0.0, 1.0 - bin_size)
        hi = 1.0
    else:
        idx = int(math.floor(value / bin_size))
        lo = idx * bin_size
        hi = min(1.0, lo + bin_size)
    right = "]" if hi >= 1.0 else ")"
    return f"[{lo:.2f},{hi:.2f}{right}", lo, hi


def _read_and_group(play_csv: str, occlusion_map: dict[int, tuple[float, bool]], bin_size: float):
    if not os.path.isfile(play_csv):
        raise FileNotFoundError(play_csv)
    groups = defaultdict(lambda: {"trials": 0, "successes": 0, "occlusion_sum": 0.0, "occlusion_count": 0})
    per_grid = defaultdict(lambda: {"trials": 0, "successes": 0})
    total_rows = 0
    skipped_missing = 0
    skipped_no_map = 0

    with open(play_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"grid_index", "success"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Play CSV missing columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            grid_index = _as_int(row.get("grid_index"), -1)
            success = 1 if _as_int(row.get("success"), 0) != 0 else 0
            if grid_index not in occlusion_map:
                skipped_no_map += 1
                continue
            occlusion_ratio, bbox_found = occlusion_map[grid_index]
            if (not bbox_found or not math.isfinite(occlusion_ratio)) and not args.include_missing:
                skipped_missing += 1
                continue

            label, lo, hi = _bin_label(occlusion_ratio, bin_size)
            group = groups[label]
            group["trials"] += 1
            group["successes"] += success
            if math.isfinite(occlusion_ratio):
                group["occlusion_sum"] += occlusion_ratio
                group["occlusion_count"] += 1
            per_grid[grid_index]["trials"] += 1
            per_grid[grid_index]["successes"] += success

    return groups, per_grid, total_rows, skipped_missing, skipped_no_map


def _sorted_group_rows(groups: dict, bin_size: float) -> list[dict[str, object]]:
    rows = []
    for label, data in groups.items():
        if label == "missing":
            sort_key = 999.0
            lo = hi = float("nan")
        else:
            inner = label.strip("[]()")
            lo_s, hi_s = inner.split(",", 1)
            lo = float(lo_s)
            hi = float(hi_s)
            sort_key = lo
        trials = int(data["trials"])
        successes = int(data["successes"])
        success_rate = float("nan") if trials <= 0 else successes / trials
        occ_count = int(data["occlusion_count"])
        occ_mean = float("nan") if occ_count <= 0 else float(data["occlusion_sum"]) / occ_count
        rows.append(
            {
                "occlusion_bin": label,
                "bin_min": lo,
                "bin_max": hi,
                "trials": trials,
                "successes": successes,
                "success_rate": success_rate,
                "occlusion_mean": occ_mean,
                "_sort": sort_key,
            }
        )
    rows.sort(key=lambda row: row["_sort"])
    for row in rows:
        row.pop("_sort", None)
    return rows


def _write_summary_csv(path: str, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["occlusion_bin", "bin_min", "bin_max", "trials", "successes", "success_rate", "occlusion_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: str, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("| Occlusion bin | Trials | Successes | Success rate | Mean occlusion |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['occlusion_bin']} | {row['trials']} | {row['successes']} | "
                f"{float(row['success_rate']):.4f} | {float(row['occlusion_mean']):.4f} |\n"
            )


def _plot_outputs(prefix: str, rows: list[dict[str, object]], title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = [str(row["occlusion_bin"]) for row in rows]
    rates = [float(row["success_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.bar(labels, rates, width=0.45, color="#3b82f6")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Occlusion Ratio Bin")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", alpha=0.25)
    for idx, rate in enumerate(rates):
        ax.text(idx, min(rate + 0.025, 0.98), f"{rate * 100:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(prefix + "_bar.png", dpi=180)
    plt.close(fig)

    table_rows = [
        [
            row["occlusion_bin"],
            str(row["trials"]),
            str(row["successes"]),
            f"{float(row['success_rate']):.3f}",
            f"{float(row['occlusion_mean']):.3f}",
        ]
        for row in rows
    ]
    fig_h = max(3.0, 0.45 * (len(table_rows) + 1))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")
    ax.set_title(title, pad=12)
    table = ax.table(
        cellText=table_rows,
        colLabels=["Occlusion", "Trials", "Successes", "Success Rate", "Mean Occ."],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    fig.tight_layout()
    fig.savefig(prefix + "_table.png", dpi=180)
    plt.close(fig)


def _infer_run_name(play_csv: str) -> str:
    parts = os.path.abspath(play_csv).split(os.sep)
    for part in reversed(parts):
        if "_ppo_" in part or "_sac_" in part or "_amp_" in part:
            return part
    parent = os.path.basename(os.path.dirname(play_csv))
    return parent or "bucket_run"


def main() -> None:
    if args.bin_size <= 0.0 or args.bin_size > 1.0:
        raise ValueError("--bin_size must be in (0, 1]")
    play_csv = os.path.abspath(args.play_csv)
    occlusion_csv = os.path.abspath(args.occlusion_csv)
    out_prefix = args.out_prefix
    if out_prefix is None:
        run_name = _infer_run_name(play_csv)
        out_dir = os.path.join("logs", "bucket_analysis", run_name)
        play_stem = os.path.splitext(os.path.basename(play_csv))[0]
        out_prefix = os.path.join(out_dir, f"{play_stem}_by_occlusion")
    out_prefix = os.path.abspath(out_prefix)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    occlusion_map = _read_occlusion_map(occlusion_csv)
    groups, _per_grid, total_rows, skipped_missing, skipped_no_map = _read_and_group(
        play_csv, occlusion_map, args.bin_size
    )
    rows = _sorted_group_rows(groups, args.bin_size)
    if not rows:
        raise RuntimeError("No rows left after joining play CSV with occlusion map")

    summary_csv = out_prefix + "_summary.csv"
    markdown = out_prefix + "_summary.md"
    _write_summary_csv(summary_csv, rows)
    _write_markdown(markdown, rows)
    title = args.title or "Success Rate by Initial Occlusion"
    _plot_outputs(out_prefix, rows, title)

    print(f"[INFO] Loaded play rows: {total_rows}")
    print(f"[INFO] Loaded occlusion grid entries: {len(occlusion_map)}")
    if skipped_no_map:
        print(f"[WARN] Skipped rows without occlusion map entry: {skipped_no_map}")
    if skipped_missing:
        print(f"[WARN] Skipped rows with missing bbox occlusion: {skipped_missing}")
    print(f"[INFO] Saved summary CSV: {summary_csv}")
    print(f"[INFO] Saved markdown table: {markdown}")
    print(f"[INFO] Saved bar plot: {out_prefix + '_bar.png'}")
    print(f"[INFO] Saved table image: {out_prefix + '_table.png'}")


if __name__ == "__main__":
    main()
