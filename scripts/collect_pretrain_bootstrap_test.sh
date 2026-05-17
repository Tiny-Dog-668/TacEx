#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTOR_SCRIPT="source/tacex_tasks/tacex_tasks/franka_four_tactile_third_person_pretrain/bootstrap_collector.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

NUM_EPISODES="${NUM_EPISODES:-2}"
OBJECT_ID="${OBJECT_ID:-box_cube_40}"
PRIMITIVE="${PRIMITIVE:-pinch_grasp}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pretrain_bootstrap_test}"
SEED="${SEED:-0}"
HEADLESS_FLAG="${HEADLESS_FLAG---headless}"
EXPORT_MP4="${EXPORT_MP4:-1}"
MP4_FPS="${MP4_FPS:-20}"
CONTACT_DURATION_SCALE="${CONTACT_DURATION_SCALE:-2.0}"
CONTACT_PLAN_MODE="${CONTACT_PLAN_MODE:-diverse}"
CENTER_PRIMARY_SENSOR="${CENTER_PRIMARY_SENSOR:-left_down}"
FORCE_CONTACT_MODE="${FORCE_CONTACT_MODE:-auto}"
PRESS_DEPTH_MM="${PRESS_DEPTH_MM:-}"

cd "${ROOT_DIR}"

echo "[run] bootstrap collector test"
echo "[run] object_id=${OBJECT_ID} primitive=${PRIMITIVE} num_episodes=${NUM_EPISODES}"
echo "[run] output_dir=${OUTPUT_DIR}"
echo "[run] contact_duration_scale=${CONTACT_DURATION_SCALE}"
echo "[run] contact_plan_mode=${CONTACT_PLAN_MODE} center_primary_sensor=${CENTER_PRIMARY_SENSOR}"
echo "[run] force_contact_mode=${FORCE_CONTACT_MODE}"
echo "[run] press_depth_mm=${PRESS_DEPTH_MM:-default}"

CMD=(
  "${PYTHON_BIN}"
  "${COLLECTOR_SCRIPT}"
  --num_episodes "${NUM_EPISODES}"
  --output_dir "${OUTPUT_DIR}"
  --object_id "${OBJECT_ID}"
  --primitive "${PRIMITIVE}"
  --seed "${SEED}"
  --mp4_fps "${MP4_FPS}"
  --contact_duration_scale "${CONTACT_DURATION_SCALE}"
  --contact_plan_mode "${CONTACT_PLAN_MODE}"
  --center_primary_sensor "${CENTER_PRIMARY_SENSOR}"
  --force_contact_mode "${FORCE_CONTACT_MODE}"
)

if [[ -n "${PRESS_DEPTH_MM}" ]]; then
  CMD+=(--press_depth_mm "${PRESS_DEPTH_MM}")
fi

if [[ -n "${HEADLESS_FLAG}" ]]; then
  CMD+=("${HEADLESS_FLAG}")
fi

if [[ "${EXPORT_MP4}" == "1" ]]; then
  CMD+=(--export_mp4)
fi

"${CMD[@]}"
