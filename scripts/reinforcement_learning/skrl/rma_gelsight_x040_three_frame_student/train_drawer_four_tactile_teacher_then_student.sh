#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TEACHER_TASK="TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-Progress-Four-Tactile-Teacher-v0"
export STUDENT_TASK="TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-Progress-Four-Tactile-Three-Frame-Binary-Direct-Action-Student-DR-v0"
export TEACHER_LOG_DIR="sim2real_cube_real_alignment_rma_gelsight_pulled_drawer_four_tactile_progress_teacher"
exec "${SCRIPT_DIR}/train_drawer_progress_binary_teacher_then_student.sh" "$@"
