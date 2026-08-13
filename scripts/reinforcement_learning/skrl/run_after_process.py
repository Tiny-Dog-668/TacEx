"""Wait for an external process to finish successfully, then run one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait_pid", type=int, required=True)
    parser.add_argument(
        "--expected_cmd",
        default=None,
        help="Substring required in the waited process command line.",
    )
    parser.add_argument(
        "--completion_file",
        required=True,
        help="File that must exist after the waited process exits.",
    )
    parser.add_argument("--poll_interval", type=_positive_float, default=30.0)
    parser.add_argument(
        "--completion_grace_seconds",
        type=_positive_float,
        default=10.0,
    )
    parser.add_argument(
        "--workdir",
        default=str(Path(__file__).resolve().parents[3]),
        help="Working directory for the command launched after completion.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to launch, written after --.",
    )
    args = parser.parse_args()
    if args.wait_pid <= 0:
        parser.error("--wait_pid must be positive")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command must be provided after --")
    return args


def _log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _read_process_identity(pid: int) -> tuple[int, str] | None:
    """Return Linux process start ticks and command line, or None if it exited."""
    proc_dir = Path("/proc") / str(pid)
    try:
        stat = (proc_dir / "stat").read_text(encoding="utf-8")
        command_line_raw = (proc_dir / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    except ProcessLookupError:
        return None

    # /proc/<pid>/stat field 2 is parenthesized and may contain spaces. Fields
    # after its final ')' start at field 3; starttime is field 22 -> index 19.
    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        raise RuntimeError(f"Malformed /proc/{pid}/stat")
    remaining_fields = stat[closing_parenthesis + 2 :].split()
    if len(remaining_fields) <= 19:
        raise RuntimeError(f"Incomplete /proc/{pid}/stat")
    if remaining_fields[0] == "Z":
        return None
    start_ticks = int(remaining_fields[19])
    command_line = command_line_raw.replace(b"\0", b" ").decode(
        "utf-8",
        errors="replace",
    ).strip()
    return start_ticks, command_line


def _completion_exists(path: Path, grace_seconds: float) -> bool:
    deadline = time.monotonic() + grace_seconds
    while True:
        if path.is_file():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))


def main() -> int:
    args = _parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    if not workdir.is_dir():
        raise NotADirectoryError(f"Launch workdir does not exist: {workdir}")
    completion_file = Path(args.completion_file).expanduser()
    if not completion_file.is_absolute():
        completion_file = (workdir / completion_file).resolve()

    initial = _read_process_identity(args.wait_pid)
    if initial is None:
        if not completion_file.is_file():
            _log(
                f"PID {args.wait_pid} is not running and completion file is missing: "
                f"{completion_file}"
            )
            return 1
        _log(
            f"PID {args.wait_pid} already exited; verified completion file: "
            f"{completion_file}"
        )
    else:
        initial_start_ticks, initial_command = initial
        if args.expected_cmd and args.expected_cmd not in initial_command:
            _log(
                f"PID {args.wait_pid} command does not contain expected text "
                f"{args.expected_cmd!r}: {initial_command}"
            )
            return 2
        _log(f"Waiting for PID {args.wait_pid}: {initial_command}")
        _log(f"Required completion file: {completion_file}")

        while True:
            time.sleep(args.poll_interval)
            current = _read_process_identity(args.wait_pid)
            if current is None:
                break
            current_start_ticks, current_command = current
            if current_start_ticks != initial_start_ticks:
                _log(
                    f"PID {args.wait_pid} was reused; treating the original process as exited"
                )
                break
            if args.expected_cmd and args.expected_cmd not in current_command:
                _log(
                    f"PID {args.wait_pid} changed command unexpectedly; refusing to launch"
                )
                return 2

        if not _completion_exists(
            completion_file,
            args.completion_grace_seconds,
        ):
            _log(
                "Waited process exited without the required completion file; "
                "next command will not run"
            )
            return 1
        _log(f"Verified completion file: {completion_file}")

    _log(f"Launching in {workdir}: {' '.join(args.command)}")
    completed = subprocess.run(args.command, cwd=workdir, check=False)
    _log(f"Launched command exited with code {completed.returncode}")
    return int(completed.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _log("Scheduler interrupted; waited and next processes were not signaled")
        raise SystemExit(130)
