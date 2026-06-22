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

# "${PYTHON:-python}" bt/fastwam_eval_policy.py \
#   --checkpoint /mnt/r/share/zwy/Project/FastWAM/runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-26_02-36-01/checkpoints/weights/step_023850.pt \
#   --dataset-stats runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-23_02-39-01/dataset_stats.json \
#   --task r1_pro_chassis_uncond_3cam_384_1e-4 \
#   --test-data /mnt/r/share/zwy/datasets/r1_pro_data_v2/r1_pro_data_convert_chassis \
#   --episode-index 0 \
#   --steps 1000 \
#   --action-horizon 20 \
#   --ylim-min -1.8 \
#   --ylim-max 1.8 \
#   --num-inference-steps 10 \
#   --save-plot ./tmp/fastwam_openloop0_20_0528_23850_trainset.png \
#   --save-csv ./tmp/fastwam_openloop0_20_0528_23850_testset.csv \
#   --save-input-image-dir ./tmp/fastwam_openloop0_20_0528_23850_trainset_input_images

  "${PYTHON:-python}" bt/fastwam_eval_policy.py \
  --checkpoint /mnt/r/CKPT/VLA/FW/RUN/R1/phase1_medium/r1_medium/checkpoints/global_step_12000_HF/fastwam_native.pt \
  --dataset-stats /mnt/r/CKPT/VLA/FW/RUN/R1/phase1_medium/dataset_stats.json \
  --task r1_pro_chassis_uncond_3cam_384_1e-4 \
  --test-data /mnt/r/share/zwy/datasets/r1_pro_data_v2/r1_pro_test_data \
  --episode-index 2 \
  --steps 1000 \
  --action-horizon 16 \
  --ylim-min -1.8 \
  --ylim-max 1.8 \
  --num-inference-steps 10 \
  --save-plot /mnt/r/tmp/btfastwam_openloop2_0606.png \
  --save-csv /mnt/r/tmp/btfastwam_openloop2_0606.csv