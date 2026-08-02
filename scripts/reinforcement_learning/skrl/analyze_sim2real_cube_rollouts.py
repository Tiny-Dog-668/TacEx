#!/usr/bin/env python3
"""Convert a sim2real Cube rollout NPZ into readable tables and plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


ACTION_NAMES = ("x", "y", "z", "gripper")
STAGES = ((0, 0), (0, 4), (0, 9), (10, 29), (30, 74), (75, 149))


def _stage_summary(data: np.lib.npyio.NpzFile) -> list[dict]:
    action = data["action"]
    processed = data["processed_action"]
    episode_step = data["episode_step"]
    stages = []
    for low, high in STAGES:
        mask = (episode_step >= low) & (episode_step <= high)
        if not np.any(mask):
            continue
        stages.append(
            {
                "episode_step_range": [low, high],
                "sample_count": int(np.count_nonzero(mask)),
                "action_mean": np.mean(action[mask], axis=0).tolist(),
                "action_std": np.std(action[mask], axis=0).tolist(),
                "action_fraction_abs_ge_0_95": np.mean(
                    np.abs(action[mask]) >= 0.95, axis=0
                ).tolist(),
                "processed_action_mean_m": np.mean(processed[mask], axis=0).tolist(),
            }
        )
    return stages


def _episode_summary(data: np.lib.npyio.NpzFile) -> dict:
    terminal_mask = data["done"].astype(bool)
    terminal_steps = data["episode_step"][terminal_mask]
    terminal_lift = data["next_cube_lift_delta"][terminal_mask]
    # Reward magnitude is not a robust classifier because a simultaneous table
    # collision subtracts 10 points. Success is sustained center lift.
    successful = terminal_lift >= (0.035 - 1e-6)
    return {
        "completed_episode_count": int(np.count_nonzero(terminal_mask)),
        "successful_episode_count": int(np.count_nonzero(successful)),
        "successful_episode_fraction": (
            float(np.mean(successful)) if successful.size else float("nan")
        ),
        "terminal_episode_step_min": (
            int(np.min(terminal_steps)) if terminal_steps.size else None
        ),
        "terminal_episode_step_max": (
            int(np.max(terminal_steps)) if terminal_steps.size else None
        ),
    }


def _write_csv(data: np.lib.npyio.NpzFile, path: Path) -> None:
    action = data["action"]
    processed = data["processed_action"]
    count_steps, count_envs, _ = action.shape
    distance = np.linalg.norm(data["next_cube_to_gripper"], axis=-1)
    header = [
        "global_step",
        "env_id",
        "episode_id",
        "episode_step",
        "action_x",
        "action_y",
        "action_z",
        "action_gripper",
        "processed_x_m",
        "processed_y_m",
        "processed_z_m",
        "requested_gripper_delta_m",
        "desired_gripper_width_m",
        "measured_gripper_width_m",
        "cube_to_gripper_distance_m",
        "cube_lift_delta_m",
        "cube_tilt_deg",
        "table_collision_force_n",
        "table_collision",
        "table_collision_penalty",
        "reward",
        "privileged_dz_gate_active",
        "terminated",
        "truncated",
        "done",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for step in range(count_steps):
            for env_id in range(count_envs):
                writer.writerow(
                    [
                        int(data["step"][step, env_id]),
                        env_id,
                        int(data["episode_id"][step, env_id]),
                        int(data["episode_step"][step, env_id]),
                        *action[step, env_id].tolist(),
                        *processed[step, env_id].tolist(),
                        float(data["next_desired_gripper_width"][step, env_id]),
                        float(data["next_measured_gripper_width"][step, env_id]),
                        float(distance[step, env_id]),
                        float(data["next_cube_lift_delta"][step, env_id]),
                        float(data["next_cube_tilt_deg"][step, env_id]),
                        float(data["next_table_collision_force_n"][step, env_id]),
                        bool(data["next_table_collision"][step, env_id]),
                        float(data["next_table_collision_penalty"][step, env_id]),
                        float(data["reward"][step, env_id]),
                        bool(data["privileged_dz_gate_active"][step, env_id]),
                        bool(data["terminated"][step, env_id]),
                        bool(data["truncated"][step, env_id]),
                        bool(data["done"][step, env_id]),
                    ]
                )


def _episode_step_bands(
    episode_step: np.ndarray, values: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.full(horizon, np.nan, dtype=np.float64)
    low = np.full(horizon, np.nan, dtype=np.float64)
    high = np.full(horizon, np.nan, dtype=np.float64)
    for index in range(horizon):
        selected = values[episode_step == index]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            mean[index] = float(np.mean(selected))
            low[index] = float(np.percentile(selected, 10))
            high[index] = float(np.percentile(selected, 90))
    return mean, low, high


def _write_action_plot(data: np.lib.npyio.NpzFile, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    episode_step = data["episode_step"]
    action = data["action"]
    horizon = int(np.max(episode_step)) + 1
    x_axis = np.arange(horizon)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for action_index, (axis, name) in enumerate(zip(axes.flat, ACTION_NAMES)):
        mean, low, high = _episode_step_bands(
            episode_step, action[..., action_index], horizon
        )
        axis.fill_between(x_axis, low, high, alpha=0.22, label="10-90 percentile")
        axis.plot(x_axis, mean, linewidth=1.8, label="mean")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.axhline(0.95, color="tab:red", linewidth=0.7, linestyle="--")
        axis.axhline(-0.95, color="tab:red", linewidth=0.7, linestyle="--")
        axis.set_title(name)
        axis.set_ylim(-1.05, 1.05)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="lower right")
    for axis in axes[-1, :]:
        axis.set_xlabel("episode policy step")
    for axis in axes[:, 0]:
        axis.set_ylabel("normalized action")
    fig.suptitle("Sim2Real Cube policy action by episode step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_state_plot(data: np.lib.npyio.NpzFile, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    episode_step = data["episode_step"]
    done = data["done"].astype(bool)
    distance = np.linalg.norm(data["next_cube_to_gripper"], axis=-1) * 1000.0
    series = (
        ("desired gripper width", data["next_desired_gripper_width"] * 1000.0, "mm"),
        ("measured gripper width", data["next_measured_gripper_width"] * 1000.0, "mm"),
        ("cube-to-gripper distance", distance, "mm"),
        ("cube lift delta", data["next_cube_lift_delta"] * 1000.0, "mm"),
    )
    horizon = int(np.max(episode_step)) + 1
    x_axis = np.arange(horizon)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (name, values, unit) in zip(axes.flat, series):
        # DirectRLEnv auto-resets completed environments inside env.step().
        # Their post-step snapshots therefore belong to the next episode.
        values = np.asarray(values, dtype=np.float64).copy()
        values[done] = np.nan
        mean, low, high = _episode_step_bands(episode_step, values, horizon)
        axis.fill_between(x_axis, low, high, alpha=0.22)
        axis.plot(x_axis, mean, linewidth=1.8)
        axis.set_title(name)
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("episode policy step")
    fig.suptitle("Sim2Real Cube rollout state by episode step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Collector NPZ path.")
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Output prefix; defaults to the NPZ path without its suffix.",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    output_prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else npz_path.with_suffix("")
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path) as data:
        required = {
            "action",
            "processed_action",
            "episode_step",
            "next_cube_to_gripper",
            "next_cube_lift_delta",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Rollout NPZ is missing required arrays: {missing}")

        csv_path = output_prefix.with_suffix(".actions.csv")
        report_path = output_prefix.with_suffix(".analysis.json")
        action_plot_path = output_prefix.with_suffix(".actions.png")
        state_plot_path = output_prefix.with_suffix(".states.png")
        _write_csv(data, csv_path)
        _write_action_plot(data, action_plot_path)
        _write_state_plot(data, state_plot_path)

        action = data["action"]
        first_step_mask = data["episode_step"] == 0
        report = {
            "npz": str(npz_path),
            "shape": list(action.shape),
            "action_names": list(ACTION_NAMES),
            "first_episode_step": {
                "sample_count": int(np.count_nonzero(first_step_mask)),
                "action_mean": np.mean(action[first_step_mask], axis=0).tolist(),
                "action_std": np.std(action[first_step_mask], axis=0).tolist(),
                "action_fraction_abs_ge_0_95": np.mean(
                    np.abs(action[first_step_mask]) >= 0.95, axis=0
                ).tolist(),
            },
            "all_steps": {
                "action_mean": np.mean(action, axis=(0, 1)).tolist(),
                "action_std": np.std(action, axis=(0, 1)).tolist(),
                "action_fraction_abs_ge_0_95": np.mean(
                    np.abs(action) >= 0.95, axis=(0, 1)
                ).tolist(),
                "desired_width_fraction_at_0_mm": float(
                    np.mean(data["next_desired_gripper_width"] <= 1e-6)
                ),
                "desired_width_fraction_at_80_mm": float(
                    np.mean(data["next_desired_gripper_width"] >= 0.079999)
                ),
                "privileged_dz_gate_fraction": float(
                    np.mean(data["privileged_dz_gate_active"])
                ),
                "table_collision_transition_fraction": float(
                    np.mean(data["next_table_collision"])
                ),
                "table_collision_force_max_n": float(
                    np.max(data["next_table_collision_force_n"])
                ),
            },
            "episode": _episode_summary(data),
            "stages": _stage_summary(data),
            "outputs": {
                "csv": str(csv_path),
                "action_plot": str(action_plot_path),
                "state_plot": str(state_plot_path),
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[INFO] Saved action CSV: {csv_path}")
    print(f"[INFO] Saved analysis JSON: {report_path}")
    print(f"[INFO] Saved action plot: {action_plot_path}")
    print(f"[INFO] Saved state plot: {state_plot_path}")


if __name__ == "__main__":
    main()
