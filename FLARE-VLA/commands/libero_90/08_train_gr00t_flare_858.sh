#!/usr/bin/env bash
set -euo pipefail

# FLARE-VLA 858-record GR00T smoke/full training launcher.
# Defaults to 5 steps to verify that the new mixed index can be read. Override
# MAX_STEPS, OUTPUT_DIR, batch size, and W&B settings through environment vars.

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_10}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/lerobot_data/libero_90_lerobot}"
DISTILLATION_DATASET_DIR="${DISTILLATION_DATASET_DIR:-${GR00T_ROOT}/outputs/flare_distillation_filtered_direct}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/flare_distillation_858_smoke}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
export SHARD_SIZE="${SHARD_SIZE:-858}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-1}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
export USE_WANDB="${USE_WANDB:-0}"

cd "${GR00T_ROOT}"

python scripts/flare/check_distillation_sidecars.py \
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}" \
  --expected-total 858

bash examples/finetune.sh \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag libero_sim \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name flare_distillation_858 \
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}" \
  --allow-padding \
  --state-dropout-prob 0.0 \
  --save-only-model
