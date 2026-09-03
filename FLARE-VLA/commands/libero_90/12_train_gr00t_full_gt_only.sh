#!/usr/bin/env bash
set -euo pipefail

# Full LIBERO-90 GT-only control run. Keep all optimization settings aligned
# with 11_train_gr00t_full_gt_flare.sh except for the absence of teacher data.

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_10}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/lerobot_data/libero_90_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/gr00t_full_gt_only}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-3000}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
export SHARD_SIZE="${SHARD_SIZE:-1024}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
export USE_WANDB="${USE_WANDB:-0}"

cd "${GR00T_ROOT}"

bash examples/finetune.sh \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag libero_sim \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name gr00t_full_gt_only \
  --state-dropout-prob 0.2 \
  --save-only-model
