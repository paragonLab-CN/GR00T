#!/usr/bin/env bash
set -euo pipefail

# Train GR00T with the complete original LIBERO-90 GT source plus a separately
# sampled teacher-only FLARE source. The default teacher fraction is 20%.

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot}"
DISTILLATION_DATASET_DIR="${DISTILLATION_DATASET_DIR:-${GR00T_ROOT}/outputs/flare_full_teacher_high_medium_only}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/spatial/gr00t_spatial_gt46490_teacher8998_official}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export SHARD_SIZE="${SHARD_SIZE:-1024}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
export TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO:-0.162161188}"
export USE_WANDB="${USE_WANDB:-0}"

# Fail closed: this launcher is specifically for full-GT + teacher mixing. If the
# keep-full-GT flag is lost anywhere in the shell/Python argument chain, the
# Python launcher must abort instead of silently running teacher-only training.
export FLARE_REQUIRE_MIXED_DATA=1
export FLARE_EXPECTED_TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO}"

cd "${GR00T_ROOT}"

python scripts/flare/check_distillation_sidecars.py \
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}" \
  --expected-gt 0

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --embodiment-tag libero_sim
  --output-dir "${OUTPUT_DIR}"
  --experiment-name gr00t_spatial_gt46490_teacher8998_official
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}"
  --keep-full-gt-dataset
  --teacher-mix-ratio "${TEACHER_MIX_RATIO}"
  --allow-padding
  --state-dropout-prob 0.2
  --save-only-model
)

# Exact-chain preflight. This resolves the same shell and Tyro arguments used by
# the real run, verifies the 80/20 contract, prints both data sources, and exits
# before loading the model.
#bash examples/finetune.sh "${COMMON_ARGS[@]}" -- --validate_config_only
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --validate_config_only

# Only reached when the mixed-data contract passed.
#bash examples/finetune.sh "${COMMON_ARGS[@]}"
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
