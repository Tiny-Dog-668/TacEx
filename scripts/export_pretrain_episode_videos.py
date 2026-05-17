#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "tacex_tasks"
    / "tacex_tasks"
    / "franka_four_tactile_third_person_pretrain"
)
sys.path.insert(0, str(MODULE_DIR))

from video_export import export_episode_videos


def main():
    parser = argparse.ArgumentParser(description="Export stored pretrain episode arrays to MP4 videos.")
    parser.add_argument("npz", nargs="+", help="Episode npz file(s) to export.")
    parser.add_argument("--fps", type=int, default=20, help="Output video fps.")
    parser.add_argument(
        "--include_per_sensor",
        action="store_true",
        default=False,
        help="Also export the eight per-sensor tactile videos in addition to the grid videos.",
    )
    args = parser.parse_args()

    for item in args.npz:
        export_dir = export_episode_videos(Path(item), fps=args.fps, include_per_sensor=args.include_per_sensor)
        print(f"exported {item} -> {export_dir}")


if __name__ == "__main__":
    main()
