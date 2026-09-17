"""Evaluate the six Large Drawer fusion Students on one matched XY schedule."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    root = Path(__file__).resolve().parents[4]
    for relative in ("source/tacex_tasks", "source/tacex", "source/tacex_assets"):
        value = str(root / relative)
        if value not in sys.path:
            sys.path.insert(0, value)


_extend_repo_pythonpath()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--schedule_csv", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma import rma_gelsight_pulled_drawer_artifacts as artifacts
from tacex_tasks.sim2real_gelsight_rma.large_drawer_bbox import (
    LargeDrawerBBoxReader,
    enable_target_object_semantics,
)
from tacex_tasks.sim2real_gelsight_rma.large_drawer_fusion_runtime import (
    initial_recurrent_state,
    run_student_model,
    update_recurrent_state,
)
from tacex_tasks.sim2real_gelsight_rma.large_drawer_occlusion import (
    DEFAULT_OCCLUSION_BINS,
    occlusion_bin_index,
    wilson_interval,
)


def _atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_schedule(path: Path) -> list[dict[str, float | int]]:
    rows = []
    required = {
        "geometry_seed", "grid_index", "x", "y", "occlusion_ratio",
        "bbox_found", "env_id", "scale_bucket_id", "reset_seed",
    }
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Schedule CSV is missing columns: {sorted(required.difference(reader.fieldnames or ())) }")
        for row in reader:
            if int(float(row["bbox_found"])) != 1:
                continue
            parsed = {
                "geometry_seed": int(float(row["geometry_seed"])),
                "grid_index": int(float(row["grid_index"])),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "occlusion_ratio": float(row["occlusion_ratio"]),
                "env_id": int(float(row["env_id"])),
                "scale_bucket_id": int(float(row["scale_bucket_id"])),
                "reset_seed": int(float(row["reset_seed"])),
            }
            rows.append(parsed)
    if not rows:
        raise RuntimeError("Schedule contains no valid bbox rows")
    return rows


def _validate_checkpoints(paths: list[Path]):
    payloads = [artifacts.load_student_checkpoint(path, device="cpu") for path in paths]
    by_variant = {}
    for path, payload in zip(paths, payloads):
        task = str(payload["task"])
        variant = artifacts.LARGE_DRAWER_FUSION_VARIANT_BY_TASK.get(task)
        if variant is None:
            raise RuntimeError(f"Checkpoint is not a Large Drawer fusion Student: {path}")
        if variant in by_variant:
            raise RuntimeError(f"Duplicate fusion variant: {variant}")
        by_variant[variant] = (path, payload)
    expected = set(artifacts.LARGE_DRAWER_FUSION_VARIANT_BY_TASK.values())
    if set(by_variant) != expected:
        raise RuntimeError(
            f"Exactly the six fusion variants are required; missing={sorted(expected.difference(by_variant))}"
        )
    teacher_hashes = {payload["teacher_checkpoint_sha256"] for _, payload in by_variant.values()}
    encoder_hashes = {payload["encoder_init_checkpoint_sha256"] for _, payload in by_variant.values()}
    if len(teacher_hashes) != 1 or len(encoder_hashes) != 1:
        raise RuntimeError("All six checkpoints must share one Teacher and one vision encoder initialization")
    return by_variant, teacher_hashes.pop(), encoder_hashes.pop()


def _write_rows(rows: list[dict], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _seed_reset_rng(seed: int) -> None:
    """Replay reset-time Python/NumPy/Torch randomization across policies."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_variant(variant: str, checkpoint: Path, payload: dict, schedule: list[dict]) -> tuple[list[dict], dict]:
    task = str(payload["task"])
    grouped = defaultdict(list)
    for row in schedule:
        grouped[int(row["geometry_seed"])].append(row)
    result_rows = []
    inference_seconds = 0.0
    inference_env_steps = 0
    parameter_count = 0
    for geometry_seed, seed_rows in sorted(grouped.items()):
        seed_rows.sort(key=lambda row: int(row["grid_index"]))
        cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        cfg.seed = geometry_seed
        enable_target_object_semantics(cfg)
        env = gym.make(task, cfg=cfg)
        reader = LargeDrawerBBoxReader(env.unwrapped)
        try:
            base_env = env.unwrapped
            reader.initialize()
            device = torch.device(base_env.device)
            model = artifacts.make_student_model_for_checkpoint(
                payload, pretrained_backbone=False
            ).to(device).eval()
            artifacts.load_student_model_state(model, payload["model"])
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            for start in range(0, len(seed_rows), args.num_envs):
                batch = seed_rows[start : start + args.num_envs]
                for local_index, row in enumerate(batch):
                    if int(row["env_id"]) != local_index:
                        raise RuntimeError(
                            "Schedule env_id order does not match the scanner batch protocol"
                        )
                    actual_bucket = int(base_env._active_cube_bucket_ids[local_index].item())
                    if int(row["scale_bucket_id"]) != actual_bucket:
                        raise RuntimeError(
                            "Schedule scale bucket does not match the seeded environment: "
                            f"env={local_index}, schedule={row['scale_bucket_id']}, actual={actual_bucket}"
                        )
                padded = batch + [batch[-1]] * (args.num_envs - len(batch))
                reset_seeds = {int(row["reset_seed"]) for row in batch}
                if len(reset_seeds) != 1:
                    raise RuntimeError("Each scanner batch must use one shared reset_seed")
                xy = torch.tensor(
                    [[row["x"], row["y"]] for row in padded],
                    dtype=torch.float32,
                    device=device,
                )
                base_env.set_next_reset_object_xy(xy)
                _seed_reset_rng(reset_seeds.pop())
                observations, _ = env.reset()
                reset_occlusion, bbox_found = reader.read()
                if not bool(bbox_found[: len(batch)].all().item()):
                    missing = torch.nonzero(
                        ~bbox_found[: len(batch)], as_tuple=False
                    ).flatten().tolist()
                    raise RuntimeError(
                        f"No reset-time bbox occlusion for active envs: {missing}"
                    )
                recurrent_state, reset_mask = initial_recurrent_state(task, args.num_envs, device)
                active = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                active[: len(batch)] = True
                success = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)
                contact_steps = torch.zeros((args.num_envs, 4), dtype=torch.float32, device=device)
                alpha_sum = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
                occlusion_sum = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
                diagnostic_count = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
                termination_reason = ["forced_limit"] * args.num_envs
                limit = args.max_steps if args.max_steps > 0 else int(base_env.max_episode_length) + 2
                with torch.inference_mode():
                    for _ in range(limit):
                        if not bool(active.any().item()):
                            break
                        obs = observations["policy"]
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        model_result = run_student_model(
                            model,
                            task,
                            obs,
                            recurrent_state=recurrent_state,
                            reset_mask=reset_mask,
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        inference_seconds += time.perf_counter() - started
                        # The model always evaluates the full vectorized batch,
                        # including env slots whose scheduled episode completed.
                        inference_env_steps += args.num_envs
                        actions = model_result["action"].clone()
                        actions[~active] = 0.0
                        pre_active = active.clone()
                        observations, _, terminated, truncated, _ = env.step(actions)
                        recurrent_state, reset_mask = update_recurrent_state(
                            model_result, terminated, truncated
                        )
                        lengths[pre_active] += 1
                        contact_steps[pre_active] += base_env._last_rma_contact_state[pre_active].float()
                        success[pre_active] |= base_env._last_rma_success_nonterminal[pre_active]
                        alpha = model_result["alpha"]
                        predicted_occlusion = model_result["occlusion_probability"]
                        if alpha is not None:
                            alpha_sum[pre_active] += alpha[pre_active, 0]
                        if predicted_occlusion is not None:
                            occlusion_sum[pre_active] += predicted_occlusion[pre_active, 0]
                        if alpha is not None or predicted_occlusion is not None:
                            diagnostic_count[pre_active] += 1
                        completed = pre_active & torch.logical_or(terminated, truncated)
                        for env_id in torch.nonzero(completed, as_tuple=False).flatten().tolist():
                            termination_reason[env_id] = (
                                "terminated_and_timeout" if terminated[env_id] and truncated[env_id]
                                else "terminated" if terminated[env_id] else "timeout"
                            )
                        active[completed] = False
                for env_id, schedule_row in enumerate(batch):
                    length = max(1, int(lengths[env_id].item()))
                    occ = float(reset_occlusion[env_id].item())
                    bin_index = occlusion_bin_index(occ)
                    lower = DEFAULT_OCCLUSION_BINS[bin_index]
                    upper = DEFAULT_OCCLUSION_BINS[bin_index + 1]
                    count = float(diagnostic_count[env_id].item())
                    result_rows.append(
                        {
                            "variant": variant,
                            "task": task,
                            "checkpoint": str(checkpoint),
                            "geometry_seed": geometry_seed,
                            "grid_index": int(schedule_row["grid_index"]),
                            "env_id": env_id,
                            "scale_bucket_id": int(schedule_row["scale_bucket_id"]),
                            "reset_seed": int(schedule_row["reset_seed"]),
                            "x": float(schedule_row["x"]),
                            "y": float(schedule_row["y"]),
                            "scan_reference_occlusion": float(
                                schedule_row["occlusion_ratio"]
                            ),
                            "initial_occlusion": occ,
                            "occlusion_bin": f"[{lower:.1f},{upper:.1f}{']' if upper == 1.0 else ')'}",
                            "success": int(success[env_id].item()),
                            "length_steps": int(lengths[env_id].item()),
                            "termination_reason": termination_reason[env_id],
                            **{
                                f"truth_contact_fraction_{sensor}": float(
                                    contact_steps[env_id, sensor].item() / length
                                )
                                for sensor in range(4)
                            },
                            "alpha_mean": (
                                float(alpha_sum[env_id].item() / count)
                                if count > 0.0 and "alpha" in variant else ""
                            ),
                            "predicted_occlusion_mean": (
                                float(occlusion_sum[env_id].item() / count)
                                if count > 0.0 and "aux" in variant else ""
                            ),
                        }
                    )
        finally:
            reader.close()
            env.close()
    runtime = {
        "parameter_count": parameter_count,
        "model_inference_seconds": inference_seconds,
        "model_inference_env_steps": inference_env_steps,
        "mean_model_latency_ms_per_env": (
            1000.0 * inference_seconds / inference_env_steps if inference_env_steps else None
        ),
    }
    return result_rows, runtime


def _summarize(rows: list[dict], runtime_by_variant: dict[str, dict]) -> dict:
    summary = {}
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        successes = sum(int(row["success"]) for row in selected)
        micro_low, micro_high = wilson_interval(successes, len(selected))
        bins = {}
        for label in sorted({row["occlusion_bin"] for row in selected}):
            bucket = [row for row in selected if row["occlusion_bin"] == label]
            bucket_successes = sum(int(row["success"]) for row in bucket)
            low, high = wilson_interval(bucket_successes, len(bucket))
            bins[label] = {
                "trials": len(bucket),
                "successes": bucket_successes,
                "success_rate": bucket_successes / len(bucket),
                "wilson_95": [low, high],
            }
        alpha_values = [
            float(row["alpha_mean"])
            for row in selected
            if row["alpha_mean"] != ""
        ]
        predicted_occlusion_values = [
            (float(row["predicted_occlusion_mean"]), float(row["initial_occlusion"]))
            for row in selected
            if row["predicted_occlusion_mean"] != ""
        ]
        summary[variant] = {
            "episodes": len(selected),
            "successes": successes,
            "micro_success_rate": successes / len(selected),
            "micro_wilson_95": [micro_low, micro_high],
            "macro_success_rate_across_nonempty_bins": sum(
                value["success_rate"] for value in bins.values()
            ) / len(bins),
            "bins": bins,
            "mean_truth_contact_fraction": [
                sum(float(row[f"truth_contact_fraction_{index}"]) for row in selected) / len(selected)
                for index in range(4)
            ],
            "alpha_mean_across_episodes": (
                sum(alpha_values) / len(alpha_values) if alpha_values else None
            ),
            "predicted_occlusion_mean_across_episodes": (
                sum(value for value, _ in predicted_occlusion_values)
                / len(predicted_occlusion_values)
                if predicted_occlusion_values else None
            ),
            "predicted_occlusion_mae_to_initial_bbox": (
                sum(abs(value - target) for value, target in predicted_occlusion_values)
                / len(predicted_occlusion_values)
                if predicted_occlusion_values else None
            ),
            **runtime_by_variant[variant],
        }
    return summary


def _plot(summary: dict, output: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    variants = list(summary)
    labels = [
        f"[{DEFAULT_OCCLUSION_BINS[index]:.1f},{DEFAULT_OCCLUSION_BINS[index + 1]:.1f}{']' if index == len(DEFAULT_OCCLUSION_BINS) - 2 else ')'}"
        for index in range(len(DEFAULT_OCCLUSION_BINS) - 1)
    ]
    width = 0.8 / len(variants)
    figure, axis = plt.subplots(figsize=(16, 7))
    x = torch.arange(len(labels), dtype=torch.float32).numpy()
    for index, variant in enumerate(variants):
        values = [summary[variant]["bins"].get(label, {}).get("success_rate", float("nan")) for label in labels]
        axis.bar(x + (index - (len(variants) - 1) / 2) * width, values, width, label=variant)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Initial bbox occlusion")
    axis.set_ylabel("Success rate")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return True


def main() -> None:
    if args.num_envs <= 0 or args.num_envs % 8 != 0:
        raise ValueError("--num_envs must be a positive multiple of 8")
    checkpoint_paths = [Path(value).expanduser().resolve() for value in args.checkpoints]
    by_variant, teacher_hash, encoder_hash = _validate_checkpoints(checkpoint_paths)
    schedule_path = Path(args.schedule_csv).expanduser().resolve()
    schedule = _load_schedule(schedule_path)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    rows = []
    runtime = {}
    for variant, (checkpoint, payload) in by_variant.items():
        variant_rows, variant_runtime = _evaluate_variant(
            variant, checkpoint, payload, schedule
        )
        rows.extend(variant_rows)
        runtime[variant] = variant_runtime
        print(f"[INFO] Evaluated {variant}: {len(variant_rows)} episodes", flush=True)
    _write_rows(rows, output / "episodes.csv")
    policy_summary = _summarize(rows, runtime)
    plotted = _plot(policy_summary, output / "success_by_initial_occlusion.png")
    _atomic_json(
        {
            "kind": "tacex_large_drawer_six_fusion_comparison",
            "version": 1,
            "teacher_checkpoint_sha256": teacher_hash,
            "encoder_init_checkpoint_sha256": encoder_hash,
            "schedule_csv": str(schedule_path),
            "schedule_csv_sha256": artifacts.sha256_file(schedule_path),
            "contact_order": ["left_inner", "right_inner", "left_down", "right_down"],
            "success_definition": "environment_success_ever",
            "initial_occlusion_source": "reset_time_replicator_bbox_not_aux_prediction",
            "reset_schedule_contract": {
                "geometry_seed": "environment construction and fixed geometry",
                "scale_bucket_id": "validated against each environment slot",
                "reset_seed": "reseed Python NumPy Torch before every batch reset",
                "object_xy": "one-shot full-vector root-frame override",
            },
            "plot_generated": plotted,
            "policies": policy_summary,
        },
        output / "summary.json",
    )
    print(f"[INFO] Saved six-policy comparison: {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
