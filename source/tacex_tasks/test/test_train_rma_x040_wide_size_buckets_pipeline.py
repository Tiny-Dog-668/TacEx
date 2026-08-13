from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "reinforcement_learning"
    / "skrl"
    / "train_rma_x040_wide_size_buckets_pipeline.py"
)
_SPEC = importlib.util.spec_from_file_location("size_bucket_pipeline", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pipeline)


def _write_teacher_run(run_dir: Path, *, timesteps: int = 128) -> tuple[Path, Path]:
    params = run_dir / "params"
    checkpoints = run_dir / "checkpoints"
    params.mkdir(parents=True)
    checkpoints.mkdir()
    (params / pipeline.TEACHER_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "kind": "tacex_rma_x040_wide_teacher",
                "task": pipeline.TEACHER_TASK,
                "trainer_timesteps": timesteps,
            }
        ),
        encoding="utf-8",
    )
    final_checkpoint = checkpoints / f"agent_{timesteps}.pt"
    best_checkpoint = checkpoints / "best_agent.pt"
    final_checkpoint.write_bytes(b"final")
    best_checkpoint.write_bytes(b"best")
    return final_checkpoint, best_checkpoint


def test_pipeline_selects_only_from_a_completed_teacher_run(tmp_path):
    run_dir = tmp_path / "teacher_run"
    final_checkpoint, best_checkpoint = _write_teacher_run(run_dir)

    selected, completion = pipeline._select_completed_teacher_checkpoint(run_dir, "auto")
    assert selected == best_checkpoint.resolve()
    assert completion == final_checkpoint.resolve()
    selected, _ = pipeline._select_completed_teacher_checkpoint(run_dir, "final")
    assert selected == final_checkpoint.resolve()

    final_checkpoint.unlink()
    with pytest.raises(RuntimeError, match="exact final checkpoint"):
        pipeline._select_completed_teacher_checkpoint(run_dir, "auto")


def test_pipeline_rejects_ambiguous_new_runs(tmp_path):
    log_root = tmp_path / "logs"
    old = log_root / "old"
    old.mkdir(parents=True)
    before = pipeline._snapshot_run_dirs(log_root)
    (log_root / "new_a").mkdir()
    (log_root / "new_b").mkdir()
    with pytest.raises(RuntimeError, match="exactly one new Teacher run"):
        pipeline._resolve_single_new_run(log_root, before, stage="Teacher")


def test_pipeline_validates_student_provenance_and_final_checkpoint(tmp_path):
    teacher_checkpoint = tmp_path / "teacher" / "checkpoints" / "best_agent.pt"
    teacher_checkpoint.parent.mkdir(parents=True)
    teacher_checkpoint.write_bytes(b"teacher")
    run_dir = tmp_path / "student_run"
    (run_dir / "params").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "params" / "training.json").write_text(
        json.dumps(
            {
                "task": pipeline.STUDENT_TASK,
                "timesteps": 2,
                "teacher_checkpoint": str(teacher_checkpoint.resolve()),
            }
        ),
        encoding="utf-8",
    )
    final_checkpoint = run_dir / "checkpoints" / "student_0000002.pt"
    final_checkpoint.write_bytes(b"student")
    (run_dir / "checkpoints" / "latest.pt").write_bytes(b"student")
    assert pipeline._validate_completed_student_run(
        run_dir,
        teacher_checkpoint=teacher_checkpoint,
        expected_timesteps=2,
    ) == final_checkpoint.resolve()


def test_pipeline_commands_bind_the_size_bucket_tasks(tmp_path):
    encoder_checkpoint = tmp_path / "encoder.pt"
    args = pipeline.build_parser().parse_args(
        ["--encoder_init_checkpoint", str(encoder_checkpoint)]
    )
    teacher = pipeline._teacher_command(args, tmp_path)
    assert teacher[teacher.index("--task") + 1] == pipeline.TEACHER_TASK
    assert "--headless" in teacher
    assert "--save_final_checkpoint" in teacher
    student = pipeline._student_command(
        args,
        tmp_path,
        tmp_path / "teacher.pt",
        encoder_checkpoint,
    )
    assert student[student.index("--task") + 1] == pipeline.STUDENT_TASK
    assert student[student.index("--teacher_checkpoint") + 1] == str(tmp_path / "teacher.pt")
    assert student[student.index("--encoder_init_checkpoint") + 1] == str(encoder_checkpoint)
