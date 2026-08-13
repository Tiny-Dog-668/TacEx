"""Train the X040-Wide Size-Buckets Teacher, then distill its visual Student."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


TEACHER_TASK = "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Size-Buckets-Teacher-v0"
STUDENT_TASK = (
    "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Size-Buckets-"
    "Direct-Action-Student-DR-v0"
)
TEACHER_LOG_DIRECTORY = "sim2real_cube_real_alignment_rma_x040_wide_size_buckets_teacher"
STUDENT_LOG_DIRECTORY = (
    "sim2real_cube_real_alignment_rma_x040_wide_size_buckets_direct_action_student"
)
TEACHER_MANIFEST_FILENAME = "rma_x040_wide_manifest.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder_init_checkpoint",
        type=Path,
        required=True,
        help="Approved RMA XY Heatmap-DR Student checkpoint used to initialize ResNet18.",
    )
    parser.add_argument("--teacher_num_envs", type=_positive_int, default=64)
    parser.add_argument(
        "--teacher_max_iterations",
        type=_positive_int,
        default=None,
        help=(
            "Optional PPO update count passed to train.py. If omitted, the Teacher YAML's "
            "trainer.timesteps value is used."
        ),
    )
    parser.add_argument("--teacher_seed", type=int, default=42)
    parser.add_argument(
        "--teacher_checkpoint_kind",
        choices=("auto", "best", "final"),
        default="auto",
        help=(
            "Checkpoint used for distillation after the exact final checkpoint proves that "
            "training completed. auto prefers best_agent.pt and otherwise uses the final checkpoint."
        ),
    )
    parser.add_argument("--student_num_envs", type=_positive_int, default=4)
    parser.add_argument("--student_timesteps", type=_positive_int, default=100_000)
    parser.add_argument("--student_seed", type=int, default=42)
    parser.add_argument("--student_log_interval", type=_positive_int, default=100)
    parser.add_argument("--student_checkpoint_interval", type=_positive_int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--show_gui",
        action="store_true",
        help="Show Isaac Sim windows. Both training stages are headless by default.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required metadata file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Metadata is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Metadata must contain a JSON object: {path}")
    return value


def _snapshot_run_dirs(log_root: Path) -> set[Path]:
    if not log_root.is_dir():
        return set()
    return {path.resolve() for path in log_root.iterdir() if path.is_dir()}


def _resolve_single_new_run(log_root: Path, before: set[Path], *, stage: str) -> Path:
    created = sorted(_snapshot_run_dirs(log_root) - before)
    if len(created) != 1:
        formatted = ", ".join(str(path) for path in created) or "none"
        raise RuntimeError(
            f"Expected exactly one new {stage} run under {log_root}, found {len(created)}: "
            f"{formatted}"
        )
    return created[0]


def _select_completed_teacher_checkpoint(run_dir: Path, kind: str) -> tuple[Path, Path]:
    manifest_path = run_dir / "params" / TEACHER_MANIFEST_FILENAME
    manifest = _read_json(manifest_path)
    if manifest.get("kind") != "tacex_rma_x040_wide_teacher":
        raise RuntimeError(f"Teacher manifest kind mismatch: {manifest_path}")
    if manifest.get("task") != TEACHER_TASK:
        raise RuntimeError(f"Teacher manifest task mismatch: {manifest_path}")
    timesteps = manifest.get("trainer_timesteps")
    if not isinstance(timesteps, int) or timesteps <= 0:
        raise RuntimeError(f"Teacher manifest has invalid trainer_timesteps: {manifest_path}")

    # The exact final checkpoint is the completion marker. An interrupted run
    # can contain best/intermediate files but must never start Student training.
    final_checkpoint = run_dir / "checkpoints" / f"agent_{timesteps}.pt"
    if not final_checkpoint.is_file() or final_checkpoint.stat().st_size <= 0:
        raise RuntimeError(
            "Teacher process exited without its exact final checkpoint; Student will not start: "
            f"{final_checkpoint}"
        )
    best_checkpoint = run_dir / "checkpoints" / "best_agent.pt"
    if kind == "best":
        if not best_checkpoint.is_file() or best_checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"Requested Teacher best checkpoint is missing: {best_checkpoint}")
        selected = best_checkpoint
    elif kind == "final":
        selected = final_checkpoint
    elif kind == "auto":
        selected = (
            best_checkpoint
            if best_checkpoint.is_file() and best_checkpoint.stat().st_size > 0
            else final_checkpoint
        )
    else:
        raise ValueError(f"Unsupported Teacher checkpoint kind: {kind}")
    return selected.resolve(), final_checkpoint.resolve()


def _validate_completed_student_run(
    run_dir: Path,
    *,
    teacher_checkpoint: Path,
    expected_timesteps: int,
) -> Path:
    training_path = run_dir / "params" / "training.json"
    training = _read_json(training_path)
    if training.get("task") != STUDENT_TASK:
        raise RuntimeError(f"Student training task mismatch: {training_path}")
    if int(training.get("timesteps", -1)) != expected_timesteps:
        raise RuntimeError(f"Student training timesteps mismatch: {training_path}")
    recorded_teacher = Path(str(training.get("teacher_checkpoint", ""))).expanduser().resolve()
    if recorded_teacher != teacher_checkpoint.resolve():
        raise RuntimeError(f"Student recorded a different Teacher checkpoint: {training_path}")

    final_checkpoint = run_dir / "checkpoints" / f"student_{expected_timesteps:07d}.pt"
    latest_checkpoint = run_dir / "checkpoints" / "latest.pt"
    for checkpoint in (final_checkpoint, latest_checkpoint):
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"Completed Student checkpoint is missing: {checkpoint}")
    return final_checkpoint.resolve()


def _teacher_command(args: argparse.Namespace, repo_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "reinforcement_learning" / "skrl" / "train.py"),
        "--task",
        TEACHER_TASK,
        "--num_envs",
        str(args.teacher_num_envs),
        "--seed",
        str(args.teacher_seed),
        "--device",
        args.device,
        "--save_final_checkpoint",
    ]
    if args.teacher_max_iterations is not None:
        command.extend(("--max_iterations", str(args.teacher_max_iterations)))
    if not args.show_gui:
        command.append("--headless")
    return command


def _student_command(
    args: argparse.Namespace,
    repo_root: Path,
    teacher_checkpoint: Path,
    encoder_checkpoint: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(
            repo_root
            / "scripts"
            / "reinforcement_learning"
            / "skrl"
            / "rma_x040_wide_direct_action_student"
            / "train.py"
        ),
        "--task",
        STUDENT_TASK,
        "--teacher_checkpoint",
        str(teacher_checkpoint),
        "--encoder_init_checkpoint",
        str(encoder_checkpoint),
        "--num_envs",
        str(args.student_num_envs),
        "--timesteps",
        str(args.student_timesteps),
        "--seed",
        str(args.student_seed),
        "--log_interval",
        str(args.student_log_interval),
        "--checkpoint_interval",
        str(args.student_checkpoint_interval),
        "--device",
        args.device,
    ]
    if not args.show_gui:
        command.append("--headless")
    return command


def _run(command: list[str], *, repo_root: Path) -> None:
    print(f"[PIPELINE] Running: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    encoder_checkpoint = args.encoder_init_checkpoint.expanduser().resolve()
    if not encoder_checkpoint.is_file():
        print(f"[PIPELINE][ERROR] Encoder initialization checkpoint not found: {encoder_checkpoint}")
        return 2

    teacher_log_root = repo_root / "logs" / "skrl" / TEACHER_LOG_DIRECTORY
    teacher_runs_before = _snapshot_run_dirs(teacher_log_root)
    try:
        _run(_teacher_command(args, repo_root), repo_root=repo_root)
        teacher_run = _resolve_single_new_run(
            teacher_log_root, teacher_runs_before, stage="Teacher"
        )
        teacher_checkpoint, teacher_final_checkpoint = _select_completed_teacher_checkpoint(
            teacher_run, args.teacher_checkpoint_kind
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[PIPELINE][ERROR] Teacher stage failed; Student was not started: {exc}", flush=True)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1

    print(f"[PIPELINE] Teacher run completed: {teacher_run}", flush=True)
    print(f"[PIPELINE] Teacher completion checkpoint: {teacher_final_checkpoint}", flush=True)
    print(f"[PIPELINE] Teacher checkpoint selected for Student: {teacher_checkpoint}", flush=True)

    student_log_root = repo_root / "logs" / "skrl" / STUDENT_LOG_DIRECTORY
    student_runs_before = _snapshot_run_dirs(student_log_root)
    try:
        _run(
            _student_command(args, repo_root, teacher_checkpoint, encoder_checkpoint),
            repo_root=repo_root,
        )
        student_run = _resolve_single_new_run(
            student_log_root, student_runs_before, stage="Student"
        )
        student_checkpoint = _validate_completed_student_run(
            student_run,
            teacher_checkpoint=teacher_checkpoint,
            expected_timesteps=args.student_timesteps,
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[PIPELINE][ERROR] Student stage failed: {exc}", flush=True)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1

    print(f"[PIPELINE] Student training completed: {student_checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
