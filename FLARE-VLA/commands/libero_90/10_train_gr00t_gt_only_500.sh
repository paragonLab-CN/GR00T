#!/usr/bin/env bash
set -euo pipefail

# Train the GT-only control model with the same base checkpoint and optimization
# settings as the 500-GT + 358-teacher FLARE run. The only intended difference
# is that this dataset contains the exact same 500 GT rows and zero teacher rows.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_10}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/lerobot_data/libero_90_lerobot}"
SOURCE_DISTILLATION_DIR="${SOURCE_DISTILLATION_DIR:-${GR00T_ROOT}/outputs/flare_distillation_filtered_direct}"
GT_ONLY_DATASET_DIR="${GT_ONLY_DATASET_DIR:-${GR00T_ROOT}/outputs/flare_distillation_gt_only_500}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/gr00t_gt_only_500_train_bs8_v1}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
export SHARD_SIZE="${SHARD_SIZE:-250}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-2}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
export USE_WANDB="${USE_WANDB:-0}"

# Rebuild from the current mixed sidecars so the control always follows the
# exact 500 GT rows used by the FLARE experiment.
SOURCE_DISTILLATION_DIR="${SOURCE_DISTILLATION_DIR}" \
GT_ONLY_DATASET_DIR="${GT_ONLY_DATASET_DIR}" \
GR00T_ROOT="${GR00T_ROOT}" \
bash "${SCRIPT_DIR}/09_build_gr00t_gt_only_500.sh"

cd "${GR00T_ROOT}"

python scripts/flare/check_distillation_sidecars.py \
  --distillation-dataset-dir "${GT_ONLY_DATASET_DIR}" \
  --expected-total 500 \
  --expected-gt 500 \
  --expected-teacher 0

bash examples/finetune.sh \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag libero_sim \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name gr00t_gt_only_500 \
  --distillation-dataset-dir "${GT_ONLY_DATASET_DIR}" \
  --allow-padding \
  --state-dropout-prob 0.0 \
  --save-only-model
