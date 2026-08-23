#!/usr/bin/env bash
set -Eeuo pipefail

# Train the drawer Teacher first, then distill the paired Student.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
ISAAC_ENV_NAME="${ISAAC_ENV_NAME:-isaaclab_2.1.1}"
TEACHER_NUM_ENVS="${TEACHER_NUM_ENVS:-1024}"
STUDENT_NUM_ENVS="${STUDENT_NUM_ENVS:-128}"
START_AT="${START_AT:-}"
TEACHER_TASK="TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-Teacher-v0"
STUDENT_TASK="TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Pulled-Drawer-Three-Frame-Direct-Action-Student-DR-v0"
ENCODER_INIT_CHECKPOINT="${ENCODER_INIT_CHECKPOINT:-}"

usage() {
    cat <<'EOF'
Usage: train_drawer_teacher_then_student.sh [options]
  --start-at HH:MM       Wait until local wall-clock time before Teacher.
  --teacher-envs N       Teacher env count (default: 1024).
  --student-envs N       Student env count (default: 128).
  --encoder-init PATH    Required approved Heatmap-DR encoder checkpoint.
  --isaac-env NAME       Conda env (default: isaaclab_2.1.1).
  -h, --help             Show this help.
EOF
}

while (($# > 0)); do
    case "$1" in
        --start-at) START_AT="${2:?--start-at requires HH:MM}"; shift 2 ;;
        --teacher-envs) TEACHER_NUM_ENVS="${2:?--teacher-envs requires N}"; shift 2 ;;
        --student-envs) STUDENT_NUM_ENVS="${2:?--student-envs requires N}"; shift 2 ;;
        --encoder-init) ENCODER_INIT_CHECKPOINT="${2:?--encoder-init requires PATH}"; shift 2 ;;
        --isaac-env) ISAAC_ENV_NAME="${2:?--isaac-env requires NAME}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$TEACHER_NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid Teacher env count: $TEACHER_NUM_ENVS" >&2; exit 2; }
[[ "$STUDENT_NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid Student env count: $STUDENT_NUM_ENVS" >&2; exit 2; }
((TEACHER_NUM_ENVS % 8 == 0 && STUDENT_NUM_ENVS % 8 == 0)) || {
    echo "Both Pulled-Drawer env counts must be multiples of 8" >&2
    exit 2
}
[[ -n "$ENCODER_INIT_CHECKPOINT" && -f "$ENCODER_INIT_CHECKPOINT" ]] || {
    echo "Encoder initialization checkpoint not found: $ENCODER_INIT_CHECKPOINT" >&2
    exit 2
}

wait_until_start() {
    [[ -z "$START_AT" ]] && return 0
    [[ "$START_AT" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]] || {
        echo "--start-at must use HH:MM: $START_AT" >&2
        exit 2
    }
    local target now delay
    target="$(date -d "today $START_AT" +%s)"
    now="$(date +%s)"
    ((target > now)) || target="$(date -d "tomorrow $START_AT" +%s)"
    delay=$((target - now))
    echo "[pipeline] Waiting ${delay}s until $(date -d "@$target" '+%F %T %Z')"
    while ((delay > 0)); do
        sleep "$((delay > 60 ? 60 : delay))"
        now="$(date +%s)"
        delay=$((target - now))
    done
}

run_isaac_python() {
    conda run -n "$ISAAC_ENV_NAME" --no-capture-output \
        env TERM=xterm ./tacex.sh -p "$@"
}

find_latest_teacher_checkpoint() {
    find "$REPO_ROOT/logs/skrl/sim2real_cube_real_alignment_rma_gelsight_pulled_drawer_teacher" \
        -type f -path '*/checkpoints/agent_*.pt' -newer "$TEACHER_MARKER" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | cut -d' ' -f2-
}

cd "$REPO_ROOT"
wait_until_start
TEACHER_MARKER="$(mktemp)"
trap 'rm -f "$TEACHER_MARKER"' EXIT
touch "$TEACHER_MARKER"

echo "[pipeline] Starting Teacher with ${TEACHER_NUM_ENVS} envs"
run_isaac_python \
    scripts/reinforcement_learning/skrl/train.py \
    --task "$TEACHER_TASK" \
    --num_envs "$TEACHER_NUM_ENVS" \
    --save_final_checkpoint \
    --headless

TEACHER_CHECKPOINT="$(find_latest_teacher_checkpoint)"
[[ -n "$TEACHER_CHECKPOINT" && -f "$TEACHER_CHECKPOINT" ]] || {
    echo "Teacher succeeded but no agent checkpoint was found" >&2
    exit 1
}
echo "[pipeline] Using Teacher checkpoint: $TEACHER_CHECKPOINT"

echo "[pipeline] Starting Student with ${STUDENT_NUM_ENVS} envs"
run_isaac_python \
    scripts/reinforcement_learning/skrl/rma_gelsight_x040_three_frame_student/train.py \
    --task "$STUDENT_TASK" \
    --teacher_checkpoint "$TEACHER_CHECKPOINT" \
    --encoder_init_checkpoint "$ENCODER_INIT_CHECKPOINT" \
    --num_envs "$STUDENT_NUM_ENVS"

echo "[pipeline] Teacher and Student training completed successfully"
