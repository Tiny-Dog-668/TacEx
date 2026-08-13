"""Offline tests for the external-process training scheduler."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/reinforcement_learning/skrl/run_after_process.py"
)


def test_scheduler_launches_only_after_completion_file(tmp_path):
    completion = tmp_path / "student_0100000.pt"
    launched = tmp_path / "next_started.txt"
    waited = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib,time; time.sleep(0.2); "
                f"pathlib.Path({str(completion)!r}).write_text('done')"
            ),
        ]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--wait_pid",
            str(waited.pid),
            "--expected_cmd",
            "time.sleep",
            "--completion_file",
            str(completion),
            "--poll_interval",
            "0.05",
            "--completion_grace_seconds",
            "1",
            "--workdir",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            f"import pathlib; pathlib.Path({str(launched)!r}).write_text('started')",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    waited.wait(timeout=1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert launched.read_text() == "started"


def test_scheduler_refuses_launch_when_completion_file_is_missing(tmp_path):
    missing = tmp_path / "missing.pt"
    launched = tmp_path / "must_not_exist.txt"
    waited = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.1)"]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--wait_pid",
            str(waited.pid),
            "--completion_file",
            str(missing),
            "--poll_interval",
            "0.05",
            "--completion_grace_seconds",
            "0.1",
            "--workdir",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            f"import pathlib; pathlib.Path({str(launched)!r}).write_text('bad')",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    waited.wait(timeout=1)
    assert result.returncode == 1
    assert not launched.exists()
    assert "next command will not run" in result.stdout
