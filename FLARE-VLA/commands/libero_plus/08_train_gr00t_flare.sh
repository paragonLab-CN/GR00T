#!/usr/bin/env bash
# Train GR00T with the complete LIBERO-Plus GT source plus the FLARE-selected
# teacher-only source (mixed).
#
# NOTE: TEACHER_MIX_RATIO = teacher_samples / total_gt_frames, recomputed from
# the 07 distillation index teacher count and the full GT dataset size
# (libero_plus_lerobot meta/info.json total_frames=2238036). Do NOT reuse the
# spatial value. Current: 268979 / 2238036 = 0.1202 -> 0.12.
#
# Usage: bash FLARE-VLA/commands/libero_plus/08_train_gr00t_flare.sh

set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot}"
DISTILLATION_DATASET_DIR="${DISTILLATION_DATASET_DIR:-/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/distillation_dataset_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare_v2}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
export SHARD_SIZE="${SHARD_SIZE:-1024}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
export TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO:-0.12}"
export USE_WANDB="${USE_WANDB:-0}"
export EXTRA_AUGMENTATION_CONFIG='{"gaussian_blur_params":{"blur_limit":[3,7],"p":0.5},"motion_blur_params":{"blur_limit":[3,7],"p":0.3}}'

# Fail closed: this launcher is specifically for full-GT + teacher mixing.
export FLARE_REQUIRE_MIXED_DATA=1
export FLARE_EXPECTED_TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO}"

cd "${GR00T_ROOT}"

# finetune.sh invokes `exec python`, so put the project venv on PATH.
export PATH="${GR00T_ROOT}/.venv/bin:${PATH}"

# Load the gated Cosmos-Reason2-2B backbone from the local HF cache (offline),
# since nvidia/Cosmos-Reason2-2B is a gated repo and hf-mirror returns 403.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GR00T_BACKBONE_MODEL_NAME="/root/autodl-tmp/Isaac-GR00T/checkpoints/Cosmos-Reason2-2B"

python scripts/flare/check_distillation_sidecars.py \
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}" \
  --expected-gt 0

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --embodiment-tag libero_sim
  --output-dir "${OUTPUT_DIR}"
  --experiment-name gr00t_libero_plus_flare_v2
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}"
  --keep-full-gt-dataset
  --teacher-mix-ratio "${TEACHER_MIX_RATIO}"
  --allow-padding
  --state-dropout-prob 0.2
  --save-only-model
)

# Exact-chain preflight: verify the mixed-data contract before loading the model.
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --validate_config_only

# Only reached when the mixed-data contract passed.
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
