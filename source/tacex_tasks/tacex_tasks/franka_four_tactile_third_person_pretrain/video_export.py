from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


TACTILE_RGB_KEYS = [
    "tactile_inner_left_rgb",
    "tactile_inner_right_rgb",
    "tactile_down_left_rgb",
    "tactile_down_right_rgb",
]

TACTILE_DEPTH_KEYS = [
    "tactile_inner_left_depth",
    "tactile_inner_right_depth",
    "tactile_down_left_depth",
    "tactile_down_right_depth",
]


def to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        return arr
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def depth_to_rgb(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = []
    for frame in arr:
        colored = cv2.applyColorMap(frame, cv2.COLORMAP_VIRIDIS)
        out.append(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    return np.stack(out, axis=0)


def write_mp4(frames_rgb: np.ndarray, output_path: Path, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb.shape[1], frames_rgb.shape[2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _build_grid(frames_map: dict[str, np.ndarray], labels: list[str]) -> np.ndarray:
    rendered = []
    for label in labels:
        frame = frames_map[label]
        h, w = frame.shape[:2]
        canvas = np.zeros((h + 24, w, 3), dtype=np.uint8)
        canvas[24:] = frame
        cv2.putText(
            canvas,
            label,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered.append(canvas)
    top = np.concatenate(rendered[:2], axis=1)
    bottom = np.concatenate(rendered[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def export_episode_videos(npz_path: str | Path, fps: int = 20, include_per_sensor: bool = False) -> Path:
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    export_dir = npz_path.with_suffix("")
    export_dir.mkdir(parents=True, exist_ok=True)

    visual = to_uint8_rgb(data["third_person_rgb"])
    write_mp4(visual, export_dir / "third_person_rgb.mp4", fps)

    tactile_rgb_frames = {key: to_uint8_rgb(data[key]) for key in TACTILE_RGB_KEYS}
    tactile_depth_frames = {key: depth_to_rgb(data[key]) for key in TACTILE_DEPTH_KEYS}

    if include_per_sensor:
        for key, frames in tactile_rgb_frames.items():
            write_mp4(frames, export_dir / f"{key}.mp4", fps)
        for key, frames in tactile_depth_frames.items():
            write_mp4(frames, export_dir / f"{key}.mp4", fps)

    rgb_grid = []
    depth_grid = []
    num_frames = visual.shape[0]
    for idx in range(num_frames):
        rgb_grid.append(_build_grid({key: tactile_rgb_frames[key][idx] for key in TACTILE_RGB_KEYS}, TACTILE_RGB_KEYS))
        depth_grid.append(
            _build_grid({key: tactile_depth_frames[key][idx] for key in TACTILE_DEPTH_KEYS}, TACTILE_DEPTH_KEYS)
        )

    write_mp4(np.stack(rgb_grid, axis=0), export_dir / "tactile_rgb_grid.mp4", fps)
    write_mp4(np.stack(depth_grid, axis=0), export_dir / "tactile_depth_grid.mp4", fps)
    return export_dir
