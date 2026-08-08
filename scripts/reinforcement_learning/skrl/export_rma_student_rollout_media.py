"""Export one synchronized RMA rollout diagnostic video from an episode NPZ.

The MP4 contains third view plus available GelSight views, followed by plots of
post-action contact force and pre-action true/Student-predicted cube XY. It
does not write individual camera videos or static images.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


STREAM_LABELS = {
    "wrist_rgb": "Third view",
    "gsmini_left_rgb": "GelSight left",
    "gsmini_right_rgb": "GelSight right",
}
_POSITION_COLORS = {
    "X true": (0, 102, 204),
    "X Student": (128, 178, 255),
    "Y true": (0, 153, 0),
    "Y Student": (128, 220, 128),
}
_CONTACT_STATE_COLORS = {
    "left >=1N": (204, 102, 0),
    "right >=1N": (0, 153, 0),
}


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _as_rgb_frames(name: str, value: np.ndarray) -> np.ndarray:
    """Validate an RGB rollout stream as [frames, height, width, 3] uint8."""
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [T,H,W,3], got {value.shape}")
    if value.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 RGB, got {value.dtype}")
    if value.shape[0] == 0:
        raise ValueError(f"{name} has no frames")
    return value


def _as_series(name: str, value: np.ndarray, trailing_shape: tuple[int, ...]) -> np.ndarray:
    if value.ndim != len(trailing_shape) + 1 or value.shape[1:] != trailing_shape:
        raise ValueError(f"{name} must have shape [T,{','.join(map(str, trailing_shape))}], got {value.shape}")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got {value.dtype}")
    return value.astype(np.float32, copy=False)


def load_rollout_diagnostics(
    episode_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Load visual streams plus the diagnostic signal contract from one rollout."""
    with np.load(episode_path, allow_pickle=False) as rollout:
        required = (
            "frame_step_indices",
            "contact_force_n",
            "rma_cube_xy",
            "student_predicted_cube_xy",
            "rma_contact_force",
        )
        missing = [key for key in required if key not in rollout]
        if missing:
            raise KeyError(f"Rollout is missing diagnostic key(s): {', '.join(missing)}")
        frame_step_indices = np.asarray(rollout["frame_step_indices"], dtype=np.int64)
        if frame_step_indices.ndim != 1 or frame_step_indices.size == 0:
            raise ValueError("frame_step_indices must be a non-empty one-dimensional array")
        if np.any(np.diff(frame_step_indices) <= 0):
            raise ValueError("frame_step_indices must be strictly increasing")
        streams = {
            key: _as_rgb_frames(key, np.asarray(rollout[key]))
            for key in STREAM_LABELS
            if key in rollout
        }
        signals = {
            "contact_force_n": _as_series("contact_force_n", np.asarray(rollout["contact_force_n"]), (2,)),
            "rma_cube_xy": _as_series("rma_cube_xy", np.asarray(rollout["rma_cube_xy"]), (2,)),
            "student_predicted_cube_xy": _as_series(
                "student_predicted_cube_xy",
                np.asarray(rollout["student_predicted_cube_xy"]),
                (2,),
            ),
            "rma_contact_force": _as_series("rma_contact_force", np.asarray(rollout["rma_contact_force"]), (2,)),
        }
    if "wrist_rgb" not in streams:
        raise KeyError("Rollout is missing required third-view wrist_rgb frames")
    for key, frames in streams.items():
        if frames.shape[0] != frame_step_indices.size:
            raise ValueError(
                f"{key} has {frames.shape[0]} frames but frame_step_indices has {frame_step_indices.size}"
            )
    signal_steps = signals["contact_force_n"].shape[0]
    if any(series.shape[0] != signal_steps for series in signals.values()):
        raise ValueError("contact force, true cube position, and predicted cube position must share T")
    if int(frame_step_indices[-1]) >= signal_steps:
        raise ValueError("frame_step_indices refers to a policy step missing from diagnostic signals")
    return streams, frame_step_indices, signals


def _rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _camera_mosaic(frame_by_key: Mapping[str, np.ndarray]) -> np.ndarray:
    """Place available RGB streams side-by-side at a common display height."""
    display_height = max(frame.shape[0] for frame in frame_by_key.values())
    panels: list[np.ndarray] = []
    for key in STREAM_LABELS:
        if key not in frame_by_key:
            continue
        frame = frame_by_key[key]
        height, width = frame.shape[:2]
        display_width = round(width * display_height / height)
        panel = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
        panel = _rgb_to_bgr(panel)
        cv2.putText(panel, STREAM_LABELS[key], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(panel, STREAM_LABELS[key], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def _plot_rect(
    canvas: np.ndarray,
    top: int,
    height: int,
    values_by_label: Mapping[str, np.ndarray],
    colors: Mapping[str, tuple[int, int, int]],
    current_step: int,
    title: str,
    *,
    force_zero_minimum: bool = False,
    dashed_labels: frozenset[str] = frozenset(),
) -> None:
    """Draw a complete time-series plot into a BGR canvas."""
    width = canvas.shape[1]
    left, right, bottom = 54, 18, 24
    legend_rows = (len(values_by_label) + 2) // 3
    x0, x1 = left, width - right
    y0, y1 = top + 22 + legend_rows * 17, top + height - bottom
    all_values = np.concatenate([values.reshape(-1) for values in values_by_label.values()])
    low, high = float(np.min(all_values)), float(np.max(all_values))
    if force_zero_minimum:
        low = min(0.0, low)
    margin = max((high - low) * 0.08, 1.0e-4)
    low -= margin
    high += margin
    step_count = next(iter(values_by_label.values())).size
    denominator = max(step_count - 1, 1)

    cv2.rectangle(canvas, (x0, y0), (x1, y1), (190, 190, 190), 1)
    cv2.putText(canvas, title, (8, top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{high:.3f}", (4, y0 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{low:.3f}", (4, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    marker_x = round(x0 + (x1 - x0) * current_step / denominator)
    cv2.line(canvas, (marker_x, y0), (marker_x, y1), (60, 60, 220), 1)
    for legend_index, (label, values) in enumerate(values_by_label.items()):
        color = colors[label]
        x = np.rint(np.linspace(x0, x1, step_count)).astype(np.int32)
        y = np.rint(y1 - (values - low) / (high - low) * (y1 - y0)).astype(np.int32)
        points = np.column_stack((x, y)).reshape(-1, 1, 2)
        if label in dashed_labels:
            for segment_start in range(0, step_count - 1, 6):
                segment_end = min(segment_start + 3, step_count - 1)
                cv2.polylines(canvas, [points[segment_start : segment_end + 1]], False, color, 2, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        legend_x = x0 + (legend_index % 3) * 150
        legend_y = top + 35 + (legend_index // 3) * 17
        cv2.line(canvas, (legend_x, legend_y - 4), (legend_x + 14, legend_y - 4), color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (legend_x + 19, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _diagnostic_frame(
    streams: Mapping[str, np.ndarray],
    signals: Mapping[str, np.ndarray],
    frame_index: int,
    policy_step: int,
) -> np.ndarray:
    mosaic = _camera_mosaic({key: value[frame_index] for key, value in streams.items()})
    chart_height = 530
    charts = np.full((chart_height, mosaic.shape[1], 3), 248, dtype=np.uint8)
    _plot_rect(
        charts,
        0,
        136,
        {"left force": signals["contact_force_n"][:, 0], "right force": signals["contact_force_n"][:, 1]},
        {"left force": (255, 120, 0), "right force": (0, 180, 0)},
        policy_step,
        "Contact force after action [N]",
        force_zero_minimum=True,
    )
    _plot_rect(
        charts,
        142,
        220,
        {
            "X true": signals["rma_cube_xy"][:, 0],
            "X Student": signals["student_predicted_cube_xy"][:, 0],
            "Y true": signals["rma_cube_xy"][:, 1],
            "Y Student": signals["student_predicted_cube_xy"][:, 1],
        },
        _POSITION_COLORS,
        policy_step,
        "Cube XY before action [m] - true solid, Student dashed",
        dashed_labels=frozenset(("X Student", "Y Student")),
    )
    _plot_rect(
        charts,
        368,
        150,
        {
            "left >=1N": (signals["rma_contact_force"][:, 0] >= 1.0).astype(np.float32),
            "right >=1N": (signals["rma_contact_force"][:, 1] >= 1.0).astype(np.float32),
        },
        _CONTACT_STATE_COLORS,
        policy_step,
        "Contact before action [0/1] from force threshold",
        force_zero_minimum=True,
    )
    cv2.putText(
        charts,
        f"policy step {policy_step}",
        (mosaic.shape[1] - 110, chart_height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (60, 60, 220),
        1,
    )
    return np.concatenate((mosaic, charts), axis=0)


def _write_diagnostic_video(
    path: Path,
    streams: Mapping[str, np.ndarray],
    frame_step_indices: np.ndarray,
    signals: Mapping[str, np.ndarray],
    fps: float,
) -> None:
    """Atomically write the only media artifact: a synchronized diagnostic MP4."""
    first = _diagnostic_frame(streams, signals, 0, int(frame_step_indices[0]))
    height, width = first.shape[:2]
    temporary = _atomic_path(path)
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}")
    try:
        writer.write(first)
        for frame_index in range(1, frame_step_indices.size):
            writer.write(
                _diagnostic_frame(
                    streams, signals, frame_index, int(frame_step_indices[frame_index])
                )
            )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        writer.release()
    os.replace(temporary, path)


def _infer_policy_fps(episode_path: Path) -> float:
    manifest_path = episode_path.parent / "manifest.json"
    if manifest_path.is_file():
        try:
            value = float(json.loads(manifest_path.read_text(encoding="utf-8"))["policy_frequency_hz"])
            if value > 0.0:
                return value
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return 30.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="Path to episode_XXXX.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for rollout_diagnostics.mp4; defaults to <episode>_media beside the NPZ",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS; default preserves simulation time using policy_frequency_hz and frame stride",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    episode_path = args.episode.expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Episode NPZ not found: {episode_path}")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = episode_path.with_suffix("").with_name(f"{episode_path.stem}_media")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    streams, frame_step_indices, signals = load_rollout_diagnostics(episode_path)
    sample_stride = int(np.median(np.diff(frame_step_indices))) if frame_step_indices.size > 1 else 1
    sample_stride = max(sample_stride, 1)
    fps = float(args.fps) if args.fps is not None else _infer_policy_fps(episode_path) / sample_stride
    if fps <= 0.0:
        raise ValueError("--fps must be positive")
    output = output_dir / "rollout_diagnostics.mp4"
    print(
        f"[INFO] Exporting {frame_step_indices.size} frame(s) at {fps:g} fps "
        f"(policy stride {sample_stride}) from {episode_path}",
        flush=True,
    )
    _write_diagnostic_video(output, streams, frame_step_indices, signals, fps)
    print(f"[INFO] Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
