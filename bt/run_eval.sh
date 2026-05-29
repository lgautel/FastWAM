#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# "${PYTHON:-python}" bt/fastwam_eval_policy.py \
#   --checkpoint /mnt/r/share/zwy/Project/FastWAM/runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-26_02-36-01/checkpoints/weights/step_023850.pt \
#   --dataset-stats runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-23_02-39-01/dataset_stats.json \
#   --task r1_pro_chassis_uncond_3cam_384_1e-4 \
#   --test-data /mnt/r/share/zwy/datasets/r1_pro_data_v2/r1_pro_test_data \
#   --episode-index 2 \
#   --steps 1000 \
#   --action-horizon 16 \
#   --ylim-min -1.8 \
#   --ylim-max 1.8 \
#   --num-inference-steps 10 \
#   --save-plot ./tmp/fastwam_openloop2_16_0526_23850.png \
#   --save-csv ./tmp/fastwam_openloop2_16_0526_23850.csv

"${PYTHON:-python}" bt/fastwam_eval_policy.py \
  --checkpoint /mnt/r/share/zwy/Project/FastWAM/runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-26_02-36-01/checkpoints/weights/step_023850.pt \
  --dataset-stats runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-23_02-39-01/dataset_stats.json \
  --task r1_pro_chassis_uncond_3cam_384_1e-4 \
  --test-data /mnt/r/share/zwy/datasets/r1_pro_data_v2/r1_pro_data_convert_chassis \
  --episode-index 4 \
  --steps 1000 \
  --action-horizon 32 \
  --ylim-min -1.8 \
  --ylim-max 1.8 \
  --num-inference-steps 10 \
  --save-plot ./tmp/fastwam_openloop0_16_0528_23850_trainset.png \
  --save-csv ./tmp/fastwam_openloop0_16_0528_23850_trainset.csv