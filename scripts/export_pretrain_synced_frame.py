#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


RGB_KEYS = {
    "third_person": "third_person_rgb",
    "inner_left": "tactile_inner_left_rgb",
    "inner_right": "tactile_inner_right_rgb",
    "down_left": "tactile_down_left_rgb",
    "down_right": "tactile_down_right_rgb",
}

DEPTH_KEYS = {
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


def choose_frame_index(
    data: np.lib.npyio.NpzFile,
    sensor: str,
    mode: str,
    threshold: float,
    frame_index: int | None,
) -> int:
    num_frames = int(data["third_person_rgb"].shape[0])
    if frame_index is not None:
        if frame_index < 0 or frame_index >= num_frames:
            raise ValueError(f"frame_index={frame_index} is out of range [0, {num_frames - 1}]")
        return frame_index

    contact = np.asarray(data[CONTACT_KEYS[sensor]]).reshape(-1)
    if contact.shape[0] != num_frames:
        raise ValueError(f"Contact length mismatch for {sensor}: {contact.shape[0]} vs {num_frames}")

    if mode == "max_contact":
        return int(np.argmax(contact))

    if mode == "first_contact":
        hit = np.flatnonzero(contact >= threshold)
        if hit.size > 0:
            return int(hit[0])
        return int(np.argmax(contact))

    raise ValueError(f"Unsupported mode: {mode}")


def export_synced_frame(
    npz_path: Path,
    sensor: str,
    mode: str,
    threshold: float,
    frame_index: int | None,
    output_dir: Path | None,
) -> Path:
    data = np.load(npz_path)
    idx = choose_frame_index(data, sensor=sensor, mode=mode, threshold=threshold, frame_index=frame_index)

    export_dir = output_dir or npz_path.with_suffix("") / f"synced_frame_{idx:04d}_{sensor}"
    export_dir.mkdir(parents=True, exist_ok=True)

    third_rgb = to_uint8_rgb(data[RGB_KEYS["third_person"]][idx])
    tactile_rgb = to_uint8_rgb(data[RGB_KEYS[sensor]][idx])
    tactile_depth = to_uint8_depth(data[DEPTH_KEYS[sensor]][idx])

    imageio.imwrite(export_dir / "third_person_rgb.png", third_rgb)
    imageio.imwrite(export_dir / f"{RGB_KEYS[sensor]}.png", tactile_rgb)
    imageio.imwrite(export_dir / f"{DEPTH_KEYS[sensor]}.png", tactile_depth)

    meta = {
        "npz": str(npz_path),
        "frame_index": idx,
        "sensor": sensor,
        "selection_mode": mode if frame_index is None else "manual",
        "contact_key": CONTACT_KEYS[sensor],
        "contact_value": float(np.asarray(data[CONTACT_KEYS[sensor]]).reshape(-1)[idx]),
        "rgb_key_third_person": RGB_KEYS["third_person"],
        "rgb_key_tactile": RGB_KEYS[sensor],
        "depth_key_tactile": DEPTH_KEYS[sensor],
    }
    if "object_pose" in data:
        meta["object_pose"] = np.asarray(data["object_pose"][idx]).tolist()
    if "ee_pose" in data:
        meta["ee_pose"] = np.asarray(data["ee_pose"][idx]).tolist()
    if CONTACT_KEYS[sensor] in data:
        meta["sensor_contact_series_max"] = float(np.max(np.asarray(data[CONTACT_KEYS[sensor]]).reshape(-1)))

    (export_dir / "frame_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return export_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export one synchronized visual+tactile frame from a pretrain episode npz."
    )
    parser.add_argument("npz", help="Episode npz file.")
    parser.add_argument(
        "--sensor",
        default="down_right",
        choices=tuple(CONTACT_KEYS.keys()),
        help="Which tactile sensor to pair with the third-person image.",
    )
    parser.add_argument(
        "--mode",
        default="max_contact",
        choices=("max_contact", "first_contact"),
        help="How to choose the frame when --frame_index is not provided.",
    )
    parser.add_argument(
        "--contact_threshold",
        type=float,
        default=0.01,
        help="Threshold used by --mode first_contact.",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=None,
        help="Use a specific frame index instead of automatic selection.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Default: <episode>/synced_frame_<idx>_<sensor>/",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    export_dir = export_synced_frame(
        npz_path=npz_path,
        sensor=args.sensor,
        mode=args.mode,
        threshold=args.contact_threshold,
        frame_index=args.frame_index,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"exported {npz_path} -> {export_dir}")


if __name__ == "__main__":
    main()
