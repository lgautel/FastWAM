#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CHECKPOINT="${CHECKPOINT:-/mnt/r/share/zwy/Project/FastWAM/runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-26_02-36-01/checkpoints/weights/step_023850.pt}"
DATASET_STATS="${DATASET_STATS:-runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-23_02-39-01/dataset_stats.json}"
TASK="${TASK:-r1_pro_chassis_uncond_3cam_384_1e-4}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Open the door with a downward-press handle, go through it, and enter the room.}"

"${PYTHON:-python}" bt/fastwam_ws_server.py \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-stats "${DATASET_STATS}" \
  --action-horizon "${ACTION_HORIZON}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
