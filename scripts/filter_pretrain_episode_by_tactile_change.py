#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


TACTILE_RGB_KEYS = {
    "inner_left": "tactile_inner_left_rgb",
    "inner_right": "tactile_inner_right_rgb",
    "down_left": "tactile_down_left_rgb",
    "down_right": "tactile_down_right_rgb",
}

TACTILE_DEPTH_KEYS = {
    "inner_left": "tactile_inner_left_depth",
    "inner_right": "tactile_inner_right_depth",
    "down_left": "tactile_down_left_depth",
    "down_right": "tactile_down_right_depth",
}

CONTACT_KEYS = {
    "inner_left": "contact_inner_left",
    "inner_right": "contact_inner_right",
    "down_left": "contact_down_left",
    "down_right": "contact_down_right",
}

IMAGE_KEYS = (
    "third_person_rgb",
    "tactile_inner_left_rgb",
    "tactile_inner_right_rgb",
    "tactile_down_left_rgb",
    "tactile_down_right_rgb",
    "tactile_inner_left_depth",
    "tactile_inner_right_depth",
    "tactile_down_left_depth",
    "tactile_down_right_depth",
)


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def to_uint8_depth(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def frame_change_score(frame0: np.ndarray, frame1: np.ndarray) -> float:
    rgb0 = to_uint8_rgb(frame0).astype(np.float32)
    rgb1 = to_uint8_rgb(frame1).astype(np.float32)
    return float(np.mean(np.abs(rgb1 - rgb0)))


def compute_sensor_scores(frames_rgb: np.ndarray) -> np.ndarray:
    frames_u8 = to_uint8_rgb(frames_rgb).astype(np.int16)
    baseline = frames_u8[0:1]
    return np.mean(np.abs(frames_u8 - baseline), axis=(1, 2, 3), dtype=np.float32)


def export_frame_bundle(arrays: dict[str, np.ndarray], frame_idx: int, export_dir: Path):
    export_dir.mkdir(parents=True, exist_ok=True)

    for key in IMAGE_KEYS:
        if key not in arrays:
            continue
        frame = arrays[key][frame_idx]
        if key.endswith("_rgb") or key == "third_person_rgb":
            imageio.imwrite(export_dir / f"{key}.png", to_uint8_rgb(frame))
        else:
            imageio.imwrite(export_dir / f"{key}.png", to_uint8_depth(frame))

    frame_payload = {}
    num_frames = arrays["third_person_rgb"].shape[0]
    for key, value in arrays.items():
        if value.shape[0] == num_frames:
            frame_payload[key] = value[frame_idx]
        else:
            frame_payload[key] = value
    # Per-frame bundles are small enough that skipping compression is much faster.
    np.savez(export_dir / "frame_data.npz", **frame_payload)


def main():
    parser = argparse.ArgumentParser(
        description="Keep only frames whose tactile RGB changes enough from the initial frame."
    )
    parser.add_argument("npz", help="Episode npz file.")
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["down_right"],
        choices=tuple(TACTILE_RGB_KEYS.keys()),
        help="Tactile sensors to monitor.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=12.0,
        help="Mean absolute RGB difference threshold against the initial frame, in uint8 scale.",
    )
    parser.add_argument(
        "--min_frame",
        type=int,
        default=1,
        help="Start checking from this frame index. Default skips frame 0.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional export directory. Default: <episode>/filtered_tactile_change/",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path)
    arrays = {key: np.asarray(data[key]) for key in data.files}
    num_frames = int(arrays["third_person_rgb"].shape[0])
    export_root = Path(args.output_dir) if args.output_dir else npz_path.with_suffix("") / "filtered_tactile_change"
    export_root.mkdir(parents=True, exist_ok=True)

    score_arrays = {}
    score_progress = tqdm(args.sensors, desc="Scoring sensors", unit="sensor")
    for sensor in score_progress:
        score_arrays[sensor] = compute_sensor_scores(arrays[TACTILE_RGB_KEYS[sensor]])

    start_frame = max(args.min_frame, 1)
    kept = []
    kept_indices: list[int] = []
    scan_progress = tqdm(
        range(start_frame, num_frames),
        total=max(0, num_frames - start_frame),
        desc="Selecting frames",
        unit="frame",
    )
    for frame_idx in scan_progress:
        triggered = []
        scores = {}
        for sensor in args.sensors:
            score = float(score_arrays[sensor][frame_idx])
            scores[sensor] = score
            if score >= args.threshold:
                triggered.append(sensor)
        if not triggered:
            continue
        kept.append(
            {
                "frame_index": frame_idx,
                "triggered_sensors": triggered,
                "scores": scores,
            }
        )
        kept_indices.append(frame_idx)
        scan_progress.set_postfix(kept=len(kept_indices))

    export_progress = tqdm(kept, desc="Exporting frames", unit="frame")
    for item in export_progress:
        frame_idx = item["frame_index"]
        frame_dir = export_root / f"frame_{frame_idx:06d}"
        export_frame_bundle(arrays, frame_idx, frame_dir)

        meta = {
            "frame_index": frame_idx,
            "triggered_sensors": item["triggered_sensors"],
            "scores": item["scores"],
            "threshold": args.threshold,
            "object_pose": arrays["object_pose"][frame_idx].tolist() if "object_pose" in arrays else None,
            "ee_pose": arrays["ee_pose"][frame_idx].tolist() if "ee_pose" in arrays else None,
            "contacts": {
                sensor: float(arrays[CONTACT_KEYS[sensor]].reshape(-1)[frame_idx])
                for sensor in args.sensors
                if CONTACT_KEYS[sensor] in arrays
            },
        }
        (frame_dir / "frame_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        item.update(meta)
        export_progress.set_postfix(kept=len(kept))

    summary = {
        "npz": str(npz_path),
        "num_frames": num_frames,
        "sensors": args.sensors,
        "threshold": args.threshold,
        "num_kept_frames": len(kept_indices),
        "kept_indices": kept_indices,
    }
    (export_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (export_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"processed {npz_path}")
    print(f"kept {len(kept)} / {num_frames} frames")
    print(f"output -> {export_root}")


if __name__ == "__main__":
    main()
