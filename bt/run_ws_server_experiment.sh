#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CHECKPOINT="${CHECKPOINT:-/mnt/r/share/zwy/Project/FastWAM/runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-26_02-36-01/checkpoints/weights/step_020000.pt}"
DATASET_STATS="${DATASET_STATS:-runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-23_02-39-01/dataset_stats.json}"
TASK="${TASK:-r1_pro_chassis_uncond_3cam_384_1e-4}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Open the door with a downward-press handle, go through it, and enter the room.}"
SAVE_CLIENT_FRAME_DIR="${SAVE_CLIENT_FRAME_DIR:-./logs/ws_client_frame}"
LOG_STATE_ACTION_CSV="${LOG_STATE_ACTION_CSV:-./logs/ws_state_action.csv}"
LOG_FILE="${LOG_FILE:-./logs/ws_server.log}"
# Color match raw real-robot frames to the training distribution: off | fixed | adaptive.
# Use 'fixed' (or 'adaptive') for real-robot deploy; keep 'off' when replaying dataset frames.
COLOR_MATCH="${COLOR_MATCH:-adaptive}"
COLOR_MATCH_STRENGTH="${COLOR_MATCH_STRENGTH:-0.9}"
args=(
  --task "${TASK}"
  --checkpoint "${CHECKPOINT}"
  --dataset-stats "${DATASET_STATS}"
  --action-horizon "${ACTION_HORIZON}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --default-prompt "${DEFAULT_PROMPT}"
  --host "${HOST}"
  --port "${PORT}"
  --log-file "${LOG_FILE}"
  --color-match "${COLOR_MATCH}"
  --color-match-strength "${COLOR_MATCH_STRENGTH}"
)

  # --mix-two-client-obs
  # --image-role image
  # --state-role state

if [[ -n "${SAVE_CLIENT_FRAME_DIR}" ]]; then
  args+=(--save-client-frame-dir "${SAVE_CLIENT_FRAME_DIR}")
fi

if [[ -n "${LOG_STATE_ACTION_CSV}" ]]; then
  args+=(--log-state-action-csv "${LOG_STATE_ACTION_CSV}")
fi

"${PYTHON:-python}" bt/fastwam_ws_server_experiment.py "${args[@]}" "$@"
