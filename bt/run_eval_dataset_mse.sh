#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CHECKPOINT="${CHECKPOINT:-/mnt/r/share/zwy/Project/FastWAM/runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-26_02-36-01/checkpoints/weights/step_023850.pt}"
DATASET_STATS="${DATASET_STATS:-runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-23_02-39-01/dataset_stats.json}"
TASK="${TASK:-r1_pro_chassis_uncond_3cam_384_1e-4}"
TEST_DATA="${TEST_DATA:-/mnt/r/share/zwy/datasets/r1_pro_data_v2/r1_pro_data_convert_chassis}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
SAVE_JSON="${SAVE_JSON:-tmp/fastwam_dataset_mse_sorted.json}"

"${PYTHON:-python3}" bt/fastwam_eval_dataset_mse.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-stats "${DATASET_STATS}" \
  --task "${TASK}" \
  --test-data "${TEST_DATA}" \
  --action-horizon "${ACTION_HORIZON}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --save-json "${SAVE_JSON}" \
  "$@"
