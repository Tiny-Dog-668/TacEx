"""Collect and analyze four-tactile Pulled-Drawer Student rollouts.

The script is intentionally separate from ``play_drawer_zero_shot.py``: it
persists completed-episode statistics for experiment comparisons while keeping
the interactive evaluator lightweight.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> Path:
    root = Path(__file__).resolve().parents[4]
    for relative in ("source/tacex_tasks", "source/tacex", "source/tacex_assets"):
        value = str(root / relative)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


ROOT = _extend_repo_pythonpath()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--metrics_interval", type=int, default=100)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to a timestamped directory beside the source checkpoint.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0 or args.num_envs % 8 != 0:
        raise ValueError("--num_envs must be a positive multiple of 8")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.metrics_interval <= 0:
        raise ValueError("--metrics_interval must be positive")


def _default_output_dir(checkpoint: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return checkpoint.parent / f"four_tactile_rollout_{checkpoint.stem}_{timestamp}"


def _as_numpy(tensor: Any):
    return tensor.detach().to(device="cpu").numpy()


def _episode_csv_fields() -> list[str]:
    fields = [
        "env_id",
        "env_episode_index",
        "length_steps",
        "termination_reason",
        "success_ever",
        "first_success_step",
    ]
    for index in range(4):
        suffix = str(index)
        fields.extend(
            [
                f"truth_contact_steps_{suffix}",
                f"truth_contact_fraction_{suffix}",
                f"truth_contact_seen_{suffix}",
                f"prediction_mean_probability_{suffix}",
                f"prediction_positive_fraction_{suffix}",
                f"prediction_target_fraction_{suffix}",
                f"prediction_true_positive_{suffix}",
                f"prediction_false_positive_{suffix}",
                f"prediction_false_negative_{suffix}",
                f"prediction_true_negative_{suffix}",
                f"pre_first_success_contact_steps_{suffix}",
                f"pre_first_success_contact_fraction_{suffix}",
            ]
        )
    return fields


def _write_episodes_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_episode_csv_fields())
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import tacex_tasks  # noqa: F401
    from tacex_tasks.sim2real_gelsight_rma import (
        rma_gelsight_pulled_drawer_artifacts as drawer_artifacts,
    )
    from tacex_tasks.sim2real_gelsight_rma.four_tactile_rollout_metrics import (
        FourTactileEpisodeAccumulator,
        summarize_completed_episodes,
    )
    from tacex_tasks.sim2real_gelsight_rma.large_drawer_fusion_runtime import (
        initial_recurrent_state,
        run_student_model,
        update_recurrent_state,
    )
    from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_large_drawer_fusion_models import (
        VISION_ONLY_DOWNSAMPLE,
    )
    from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
        FOUR_TACTILE_CONTACT_ORDER,
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
    )
    from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_env import (
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
    )
    from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_downsample_env import (
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
    )
    from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
    )

    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = drawer_artifacts.load_student_checkpoint(checkpoint, device="cpu")
    task = str(payload["task"])
    supported_tasks = {
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
        *drawer_artifacts.LARGE_DRAWER_FUSION_STUDENT_TASKS,
    }
    if task not in supported_tasks:
        raise RuntimeError(
            "This collector requires a registered four-tactile Pulled-Drawer Student checkpoint; "
            f"got task={task!r}"
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(checkpoint)
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg)
    rows: list[dict[str, Any]] = []
    try:
        base_env = env.unwrapped
        device = torch.device(base_env.device)
        model = drawer_artifacts.make_student_model_for_checkpoint(
            payload, pretrained_backbone=False
        ).to(device).eval()
        drawer_artifacts.load_student_model_state(model, payload["model"])
        fusion_variant = drawer_artifacts.LARGE_DRAWER_FUSION_VARIANT_BY_TASK.get(task)
        prediction_available = fusion_variant != VISION_ONLY_DOWNSAMPLE
        accumulator = FourTactileEpisodeAccumulator(
            args.num_envs, prediction_available=prediction_available
        )
        observations, _ = env.reset()
        recurrent_state, reset_mask = initial_recurrent_state(task, args.num_envs, device)

        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                obs = observations["policy"]
                prediction_target = _as_numpy(obs["rma_contact_state"])
                result = run_student_model(
                    model,
                    task,
                    obs,
                    recurrent_state=recurrent_state,
                    reset_mask=reset_mask,
                )
                observations, _, terminated, truncated, _ = env.step(result["action"])
                recurrent_state, reset_mask = update_recurrent_state(
                    result, terminated, truncated
                )
                if not hasattr(base_env, "_last_rma_contact_state"):
                    raise RuntimeError("Environment did not retain post-transition RMA contact state")
                if not hasattr(base_env, "_last_rma_success_nonterminal"):
                    raise RuntimeError("Environment did not retain post-transition success state")
                rows.extend(
                    accumulator.record_transition(
                        post_transition_contact_state=_as_numpy(base_env._last_rma_contact_state),
                        prediction_probability=(
                            _as_numpy(result["contact_probability"])
                            if prediction_available else None
                        ),
                        prediction_target_state=prediction_target,
                        success_now=_as_numpy(base_env._last_rma_success_nonterminal),
                        terminated=_as_numpy(terminated),
                        truncated=_as_numpy(truncated),
                    )
                )
                if step % args.metrics_interval == 0 or step == args.steps:
                    rollout = accumulator.rollout_snapshot()
                    completed = summarize_completed_episodes(rows)
                    print(
                        f"[Four tactile rollout] {step}/{args.steps} "
                        f"truth_contact={rollout['truth_contact_fraction']} "
                        f"prediction={rollout['prediction']} "
                        f"success={completed['success_episode_count']}/"
                        f"{completed['completed_episode_count']}",
                        flush=True,
                    )
        rollout = accumulator.rollout_snapshot()
        completed = summarize_completed_episodes(rows)
        summary = {
            "run": {
                "student_checkpoint": str(checkpoint),
                "student_checkpoint_sha256": drawer_artifacts.sha256_file(checkpoint),
                "checkpoint_global_step": int(payload["global_step"]),
                "task": task,
                "num_envs": int(args.num_envs),
                "steps": int(args.steps),
                "contact_order": list(FOUR_TACTILE_CONTACT_ORDER),
                "truth_contact_definition": (
                    "post_transition max filtered cube-to-gelpad force over one policy step "
                    f">= {float(env_cfg.rma_contact_force_threshold_n):g} N"
                ),
                "prediction_definition": (
                    "Student contact_probability from the pre-action observation; "
                    "classification threshold is 0.5"
                    if prediction_available else "not available for Vision-Only"
                ),
                "fusion_variant": fusion_variant,
            },
            "all_rollout": rollout,
            "completed_episodes": completed,
            "incomplete_episode_count": accumulator.incomplete_episode_count(),
        }
        _write_episodes_csv(rows, output_dir / "episodes.csv")
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[INFO] Saved four-tactile rollout analysis: {output_dir}", flush=True)
        return output_dir
    finally:
        env.close()


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
