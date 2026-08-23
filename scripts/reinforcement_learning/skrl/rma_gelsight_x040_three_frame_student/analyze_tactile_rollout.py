"""Run a GelSight X040 Student rollout and compare contact/no-contact RGB."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> Path:
    root = Path(__file__).resolve().parents[4]
    for relative in ("source/tacex_tasks", "source/tacex", "source/tacex_assets"):
        value = str(root / relative)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


ROOT = _extend_repo_pythonpath()


_METRIC_KEYS = (
    "rgb_mae_0_1",
    "rgb_max_abs_0_1",
    "rgb_changed_fraction_gt_1_u8",
    "rgb_changed_fraction_gt_5_u8",
    "contact_force_n",
    "contact_probability",
    "indentation_mm",
    "taxim_contact_fraction",
)


def _metric_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_rollout_metrics(
    metrics: Mapping[str, np.ndarray], *, full_contact_fraction: float
) -> dict[str, Any]:
    """Summarize aligned per-side rollout metrics without requiring Isaac Sim."""
    state = np.asarray(metrics["contact_state"], dtype=np.float32)
    bilateral = np.asarray(metrics["bilateral_contact"], dtype=np.bool_)
    coverage = np.asarray(metrics["taxim_contact_fraction"], dtype=np.float32)
    masks = {
        "no_contact": state < 0.5,
        "contact": state >= 0.5,
        "bilateral_contact": bilateral,
        "full_surface_contact": (state >= 0.5) & (coverage >= full_contact_fraction),
    }
    groups: dict[str, Any] = {}
    for name, mask in masks.items():
        groups[name] = {
            "count": int(mask.sum()),
            "metrics": {
                key: _metric_stats(np.asarray(metrics[key])[mask]) for key in _METRIC_KEYS
            },
        }

    no_contact_mae = groups["no_contact"]["metrics"]["rgb_mae_0_1"].get("mean")
    full_contact_mae = groups["full_surface_contact"]["metrics"]["rgb_mae_0_1"].get(
        "mean"
    )
    comparison: dict[str, Any] = {
        "full_contact_fraction_threshold": float(full_contact_fraction),
        "has_no_contact_samples": groups["no_contact"]["count"] > 0,
        "has_full_surface_contact_samples": groups["full_surface_contact"]["count"] > 0,
    }
    if no_contact_mae is not None and full_contact_mae is not None:
        comparison.update(
            {
                "mean_rgb_mae_difference_0_1": float(full_contact_mae - no_contact_mae),
                "mean_rgb_mae_ratio": (
                    float(full_contact_mae / no_contact_mae)
                    if no_contact_mae > 0.0
                    else None
                ),
            }
        )
    return {"groups": groups, "comparison": comparison}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--metrics_interval", type=int, default=100)
    parser.add_argument(
        "--full_contact_fraction",
        type=float,
        default=0.90,
        help="Minimum Taxim contact-mask coverage used for the full-surface group.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to a timestamped directory beside the source checkpoint.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0 or args.num_envs % 8 != 0:
        raise ValueError("--num_envs must be a positive multiple of 8 for size buckets")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.metrics_interval <= 0:
        raise ValueError("--metrics_interval must be positive")
    if not 0.0 < args.full_contact_fraction <= 1.0:
        raise ValueError("--full_contact_fraction must lie in (0, 1]")


def _default_output_dir(checkpoint: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return checkpoint.parent / f"tactile_rgb_rollout_{checkpoint.stem}_{timestamp}"


def _taxim_contact_fraction(sensor: Any):
    """Return Taxim's pre-deformation contact-mask fraction, shape ``[N]``.

    The sensor height map is ``[N,H,W]`` in millimetres. Taxim shifts each map
    by its closest point and scalar indentation, then defines contact as
    ``shifted_height_map < 0``. This reproduces that exact mask definition.
    """
    height_map_mm = sensor.data.output["height_map"]
    indentation_mm = sensor.optical_simulator._indentation_depth
    shifted_height_map_mm = (
        height_map_mm
        - height_map_mm.amin(dim=(-2, -1), keepdim=True)
        - indentation_mm[:, None, None]
    )
    return (shifted_height_map_mm < 0.0).float().mean(dim=(-2, -1)), indentation_mm


def _snapshot(obs: Mapping[str, Any], env_id: int, score: float, metadata: dict[str, Any]):
    return {
        "score": float(score),
        "metadata": metadata,
        "left_current": obs["gsmini_left_rgb"][env_id].detach().cpu().numpy(),
        "right_current": obs["gsmini_right_rgb"][env_id].detach().cpu().numpy(),
        "left_reference": obs["gsmini_left_reference_rgb"][env_id].detach().cpu().numpy(),
        "right_reference": obs["gsmini_right_reference_rgb"][env_id].detach().cpu().numpy(),
    }


def _update_representatives(
    representatives: dict[str, dict[str, Any]],
    *,
    obs: Mapping[str, Any],
    step: int,
    contact_state: Any,
    contact_force_n: Any,
    contact_probability: Any,
    contact_fraction: Any,
) -> None:
    import torch

    bilateral_none = torch.all(contact_state < 0.5, dim=-1)
    bilateral_contact = torch.all(contact_state >= 0.5, dim=-1)
    candidates = {
        "no_contact": (bilateral_none, -contact_force_n.max(dim=-1).values),
        "bilateral_contact": (bilateral_contact, contact_force_n.min(dim=-1).values),
        "max_taxim_contact_fraction": (
            contact_state >= 0.5,
            contact_fraction,
        ),
    }
    for name, (mask, score_tensor) in candidates.items():
        if name == "max_taxim_contact_fraction":
            flat_score = torch.where(mask, score_tensor, torch.full_like(score_tensor, -1.0))
            score, flat_index = flat_score.reshape(-1).max(dim=0)
            if score.item() < 0.0:
                continue
            env_id = int(flat_index.item() // 2)
            side = int(flat_index.item() % 2)
        else:
            masked_score = torch.where(
                mask, score_tensor, torch.full_like(score_tensor, float("-inf"))
            )
            score, index = masked_score.max(dim=0)
            if not torch.isfinite(score):
                continue
            env_id = int(index.item())
            side = None
        if name in representatives and representatives[name]["score"] >= score.item():
            continue
        representatives[name] = _snapshot(
            obs,
            env_id,
            score.item(),
            {
                "step": int(step),
                "env_id": env_id,
                "side": side,
                "contact_state_lr": contact_state[env_id].detach().cpu().tolist(),
                "contact_force_n_lr": contact_force_n[env_id].detach().cpu().tolist(),
                "contact_probability_lr": contact_probability[env_id].detach().cpu().tolist(),
                "taxim_contact_fraction_lr": contact_fraction[env_id].detach().cpu().tolist(),
            },
        )


def _save_representatives(
    representatives: Mapping[str, Mapping[str, Any]], output_dir: Path
) -> dict[str, Any]:
    from PIL import Image

    image_dir = output_dir / "representative_frames"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for group, sample in representatives.items():
        manifest[group] = dict(sample["metadata"])
        manifest[group]["score"] = float(sample["score"])
        for side in ("left", "right"):
            reference = np.asarray(sample[f"{side}_reference"], dtype=np.uint8)
            current = np.asarray(sample[f"{side}_current"], dtype=np.uint8)
            delta = np.abs(current.astype(np.int16) - reference.astype(np.int16))
            delta_visual = np.clip(delta * 4, 0, 255).astype(np.uint8)
            montage = np.concatenate((reference, current, delta_visual), axis=1)
            filename = f"{group}_{side}_reference_current_absdiff_x4.png"
            Image.fromarray(montage).save(image_dir / filename)
            manifest[group][f"{side}_image"] = str(Path("representative_frames") / filename)
    return manifest


def _print_conclusion(summary: Mapping[str, Any]) -> None:
    groups = summary["groups"]
    comparison = summary["comparison"]
    no_count = groups["no_contact"]["count"]
    full_count = groups["full_surface_contact"]["count"]
    print(f"[RESULT] no_contact_side_samples={no_count}", flush=True)
    print(f"[RESULT] full_surface_contact_side_samples={full_count}", flush=True)
    if no_count == 0:
        print("[RESULT] 结论不足：rollout 没有采到未接触样本。", flush=True)
        return
    if full_count == 0:
        print(
            "[RESULT] 结论不足：rollout 有接触，但没有任何一侧达到设定的 Taxim 整面接触覆盖率。",
            flush=True,
        )
        return
    no_mae = groups["no_contact"]["metrics"]["rgb_mae_0_1"]["mean"]
    full_mae = groups["full_surface_contact"]["metrics"]["rgb_mae_0_1"]["mean"]
    difference = comparison["mean_rgb_mae_difference_0_1"]
    ratio = comparison["mean_rgb_mae_ratio"]
    ratio_text = "undefined" if ratio is None else f"{ratio:.3f}x"
    print(
        "[RESULT] reference→current RGB MAE: "
        f"未接触={no_mae:.6f}, 整面接触={full_mae:.6f}, "
        f"差值={difference:.6f}, 比值={ratio_text}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import tacex_tasks  # noqa: F401
    from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_three_frame_artifacts import (
        GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
        load_student_checkpoint,
        load_student_model_state,
        make_student_model_for_checkpoint,
        sha256_file,
    )

    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(checkpoint)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = load_student_checkpoint(checkpoint, device="cpu")
    env_cfg = parse_env_cfg(
        GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
        device=args.device,
        num_envs=args.num_envs,
    )
    env = gym.make(GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK, cfg=env_cfg)
    records: dict[str, list[Any]] = {
        key: []
        for key in (
            "step",
            "env_id",
            "side",
            "contact_state",
            "bilateral_contact",
            *_METRIC_KEYS,
        )
    }
    representatives: dict[str, dict[str, Any]] = {}
    try:
        base_env = env.unwrapped
        device = torch.device(base_env.device)
        model = make_student_model_for_checkpoint(payload, pretrained_backbone=False).to(device).eval()
        load_student_model_state(model, payload["model"])
        observations, _ = env.reset()
        env_ids = torch.arange(args.num_envs, device=device).repeat_interleave(2)
        side_ids = torch.arange(2, device=device).repeat(args.num_envs)
        with torch.inference_mode():
            for step in range(args.steps):
                obs = observations["policy"]
                actions, contact_probability, _ = model(
                    obs["wrist_rgb_history"],
                    obs["proprio_obs"].float(),
                    obs["action_history"].float(),
                    obs["gsmini_left_rgb"],
                    obs["gsmini_right_rgb"],
                    obs["gsmini_left_reference_rgb"],
                    obs["gsmini_right_reference_rgb"],
                )
                contact_state = obs["rma_contact_state"].float()
                contact_force_n = base_env._compute_rma_contact_forces()
                left_fraction, left_indentation = _taxim_contact_fraction(base_env.gsmini_left)
                right_fraction, right_indentation = _taxim_contact_fraction(base_env.gsmini_right)
                contact_fraction = torch.stack((left_fraction, right_fraction), dim=-1)
                indentation_mm = torch.stack((left_indentation, right_indentation), dim=-1)
                left_delta_u8 = (
                    obs["gsmini_left_rgb"].to(torch.int16)
                    - obs["gsmini_left_reference_rgb"].to(torch.int16)
                ).abs()
                right_delta_u8 = (
                    obs["gsmini_right_rgb"].to(torch.int16)
                    - obs["gsmini_right_reference_rgb"].to(torch.int16)
                ).abs()
                delta_u8 = torch.stack((left_delta_u8, right_delta_u8), dim=1).float()
                rgb_mae = delta_u8.mean(dim=(-3, -2, -1)).div(255.0)
                rgb_max_abs = delta_u8.amax(dim=(-3, -2, -1)).div(255.0)
                changed_gt_1 = (delta_u8 > 1.0).float().mean(dim=(-3, -2, -1))
                changed_gt_5 = (delta_u8 > 5.0).float().mean(dim=(-3, -2, -1))
                bilateral = torch.all(contact_state >= 0.5, dim=-1)
                values = {
                    "step": torch.full((args.num_envs * 2,), step, device=device, dtype=torch.long),
                    "env_id": env_ids,
                    "side": side_ids,
                    "contact_state": contact_state.reshape(-1),
                    "bilateral_contact": bilateral.repeat_interleave(2),
                    "rgb_mae_0_1": rgb_mae.reshape(-1),
                    "rgb_max_abs_0_1": rgb_max_abs.reshape(-1),
                    "rgb_changed_fraction_gt_1_u8": changed_gt_1.reshape(-1),
                    "rgb_changed_fraction_gt_5_u8": changed_gt_5.reshape(-1),
                    "contact_force_n": contact_force_n.reshape(-1),
                    "contact_probability": contact_probability.reshape(-1),
                    "indentation_mm": indentation_mm.reshape(-1),
                    "taxim_contact_fraction": contact_fraction.reshape(-1),
                }
                for key, value in values.items():
                    records[key].append(value.detach().cpu())
                _update_representatives(
                    representatives,
                    obs=obs,
                    step=step,
                    contact_state=contact_state,
                    contact_force_n=contact_force_n,
                    contact_probability=contact_probability,
                    contact_fraction=contact_fraction,
                )
                observations, _, _, _, _ = env.step(actions)
                if (step + 1) % args.metrics_interval == 0 or step + 1 == args.steps:
                    full_count = int(
                        ((contact_state >= 0.5) & (contact_fraction >= args.full_contact_fraction))
                        .sum()
                        .item()
                    )
                    print(
                        f"[rollout] {step + 1}/{args.steps} "
                        f"contact_lr={contact_state.mean(dim=0).tolist()} "
                        f"taxim_fraction_max={contact_fraction.max().item():.3f} "
                        f"full_surface_sides_in_step={full_count}",
                        flush=True,
                    )
    finally:
        env.close()

    arrays = {key: torch.cat(value).numpy() for key, value in records.items()}
    np.savez_compressed(output_dir / "rollout_metrics.npz", **arrays)
    summary = summarize_rollout_metrics(
        arrays, full_contact_fraction=args.full_contact_fraction
    )
    summary["run"] = {
        "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_global_step": int(payload["global_step"]),
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "side_samples": int(arrays["contact_state"].size),
        "task": GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
        "taxim_contact_mask_definition": "shifted_height_map_mm < 0",
        "rgb_delta_definition": "abs(current_uint8-reference_uint8)/255",
    }
    summary["representatives"] = _save_representatives(representatives, output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _print_conclusion(summary)
    print(f"[INFO] Saved rollout analysis: {output_dir}", flush=True)


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app
    try:
        run(args)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
