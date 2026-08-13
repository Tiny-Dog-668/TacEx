"""Replay an X040-Wide three-frame Student and record action/XYZ metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


root = Path(__file__).resolve().parents[4]
for name in ("tacex_tasks", "tacex", "tacex_assets", "tacex_uipc"):
    sys.path.insert(0, str(root / "source" / name))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--metrics_interval", type=int, default=100)
parser.add_argument("--output_dir", default=None)
parser.add_argument(
    "--audit_appearance_bindings",
    action="store_true",
    help="Fail if an Appearance material ID changes for an environment that did not reset.",
)
parser.add_argument(
    "--wait_for_textures",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Wait indefinitely for RTX textures after reset. Disabled by default because "
        "the GUI Appearance-DR task can otherwise wait forever for material streaming."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.rma_x040_wide_models import (
    RMAX040WideActorCore,
    extract_x040_wide_actor_core_state_dict,
)
from tacex_tasks.sim2real_grasp.rma_x040_wide_three_frame_artifacts import (
    load_teacher_policy_state,
    load_three_frame_student_checkpoint,
    load_three_frame_student_model_state,
    sha256_file,
    validate_live_teacher_contract,
)
from tacex_tasks.sim2real_grasp.rma_x040_wide_three_frame_models import (
    RMAX040WideThreeFrameDirectActionVisualStudent,
)


def _appearance_material_ids(base) -> tuple[torch.Tensor, ...] | None:
    names = (
        "_appearance_plate_material_ids",
        "_appearance_backdrop_material_ids",
        "_appearance_cube_material_ids",
        "_appearance_cube_bucket_ids",
    )
    if not all(hasattr(base, name) for name in names):
        return None
    return tuple(getattr(base, name).clone() for name in names)


def main() -> None:
    print("[INFO] Loading Student checkpoint metadata...", flush=True)
    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_three_frame_student_checkpoint(
        checkpoint,
        device="cpu",
        allow_appearance_v1_evaluation=True,
    )
    appearance_contract = payload.get("student_environment_contract", {}).get(
        "appearance_randomization", {}
    )
    if appearance_contract.get("profile") == "x040_three_frame_realistic_material_v1":
        print(
            "[INFO] Evaluating legacy Appearance v1 weights with the independent "
            "PreviewSurface runtime; visual-domain metrics are not directly comparable.",
            flush=True,
        )
    print(f"[INFO] Resolving replay task: {payload['task']}", flush=True)
    env_cfg = parse_env_cfg(
        payload["task"],
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.seed = args.seed
    # DirectRLEnv's default waits in an unbounded render loop for all texture
    # streaming. Keep replay startup bounded even though the current
    # Appearance task uses local PreviewSurface colors without texture assets.
    # Playback remains live; callers needing a fully settled first RGB frame
    # may explicitly opt back into the Isaac Lab default with
    # ``--wait_for_textures``.
    env_cfg.wait_for_textures = bool(args.wait_for_textures)
    validate_live_teacher_contract(
        env_cfg,
        payload["teacher_manifest"],
        allow_static_size_bucket_assignment=bool(
            getattr(env_cfg, "cube_size_assignment", None)
        ),
    )
    directory = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint.parent.parent / "metrics" / "three_frame_play"
    )
    directory.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = directory / f"play_metrics_{tag}.csv"

    print(
        "[INFO] Creating environment "
        f"(num_envs={args.num_envs}, wait_for_textures={env_cfg.wait_for_textures})...",
        flush=True,
    )
    env = gym.make(payload["task"], cfg=env_cfg)
    base = env.unwrapped
    print("[INFO] Loading Student and Teacher onto the simulation device...", flush=True)
    student = RMAX040WideThreeFrameDirectActionVisualStudent().to(base.device).eval()
    load_three_frame_student_model_state(student, payload["model"])
    teacher = RMAX040WideActorCore().to(base.device).eval()
    teacher.load_state_dict(
        extract_x040_wide_actor_core_state_dict(
            load_teacher_policy_state(payload["teacher_checkpoint"], base.device)
        ),
        strict=True,
    )
    print("[INFO] Resetting environments and acquiring the first RGB history...", flush=True)
    observations, _ = env.reset()
    print("[INFO] Replay rollout started.", flush=True)
    reward_sum = action_sq = position_sq = 0.0
    action_count = position_count = 0
    previous = None
    delta_sq = 0.0
    delta_count = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "step",
                    "mean_step_reward",
                    "teacher_action_rmse",
                    "student_action_delta_rmse",
                    "cube_position_rmse_m",
                    "cumulative_success_rate",
                    "recent_success_rate",
                    "tcp_clearance_violation_fraction",
                )
            )
            with torch.inference_mode():
                for step in range(1, args.steps + 1):
                    appearance_ids_before = (
                        _appearance_material_ids(base)
                        if args.audit_appearance_bindings
                        else None
                    )
                    obs = observations["policy"]
                    action, predicted_normalized = student.forward_with_position(
                        obs["wrist_rgb_history"],
                        obs["proprio_obs"].float(),
                        obs["action_history"].float(),
                    )
                    truth = obs["rma_cube_pos"].float()
                    teacher_action = teacher(
                        obs["proprio_obs"].float(),
                        obs["action_history"].float(),
                        truth,
                    )
                    predicted = student.normalizer.denormalize_position(
                        predicted_normalized
                    )
                    action_sq += float((action - teacher_action).square().sum())
                    action_count += action.numel()
                    position_sq += float((predicted - truth).square().sum())
                    position_count += predicted.numel()
                    if previous is not None:
                        delta_sq += float((action - previous).square().sum())
                        delta_count += action.numel()
                    previous = action
                    observations, rewards, terminated, truncated, _ = env.step(action)
                    if appearance_ids_before is not None:
                        appearance_ids_after = _appearance_material_ids(base)
                        reset_mask = terminated | truncated
                        changed_mask = torch.zeros_like(reset_mask)
                        for before, after in zip(
                            appearance_ids_before,
                            appearance_ids_after,
                        ):
                            changed_mask |= before != after
                        unexpected_mask = changed_mask & ~reset_mask
                        if torch.any(unexpected_mask):
                            unexpected_envs = torch.nonzero(
                                unexpected_mask, as_tuple=False
                            ).squeeze(-1)
                            raise RuntimeError(
                                "Appearance material changed outside reset for envs "
                                f"{unexpected_envs.detach().cpu().tolist()} at step {step}"
                            )
                        if torch.any(reset_mask):
                            reset_envs = torch.nonzero(
                                reset_mask, as_tuple=False
                            ).squeeze(-1)
                            changed_envs = torch.nonzero(
                                changed_mask, as_tuple=False
                            ).squeeze(-1)
                            print(
                                f"[AUDIT] step={step} reset_envs="
                                f"{reset_envs.detach().cpu().tolist()} material_changed_envs="
                                f"{changed_envs.detach().cpu().tolist()}",
                                flush=True,
                            )
                    reward_sum += float(rewards.mean())
                    if step % args.metrics_interval == 0 or step == args.steps:
                        stats = base._episode_success_statistics()
                        writer.writerow(
                            (
                                step,
                                reward_sum / step,
                                (action_sq / action_count) ** 0.5,
                                (delta_sq / delta_count) ** 0.5
                                if delta_count
                                else 0.0,
                                (position_sq / position_count) ** 0.5,
                                float(stats["cumulative_rate"]),
                                float(stats["window_rate"]),
                                float(
                                    base._last_tcp_table_clearance_violation.float().mean()
                                ),
                            )
                        )
                        stream.flush()
    finally:
        env.close()

    summary = {
        "kind": "tacex_rma_x040_wide_three_frame_direct_action_play",
        "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": sha256_file(checkpoint),
        "task": payload["task"],
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "audit_appearance_bindings": args.audit_appearance_bindings,
        "metrics_csv": str(csv_path),
        "student_runtime_inputs": [
            "wrist_rgb_history",
            "proprio_obs",
            "action_history",
        ],
        "teacher_action_rmse": (action_sq / action_count) ** 0.5
        if action_count
        else None,
        "cube_position_rmse_m": (position_sq / position_count) ** 0.5
        if position_count
        else None,
    }
    (directory / f"play_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
