#!/usr/bin/env bash
# Train GR00T-N1.7-3B on the LIBERO-Plus GT source ONLY (no teacher).
#
# This is the GT-only ablation for the FLARE distillation comparison:
#   - pretrained GR00T-N1.7-LIBERO          (zero-shot, 76.52%)
#   - GR00T-N1.7-3B + GT-only   <- this run  (isolates GT fine-tuning gain)
#   - GR00T-N1.7-3B + GT + teacher (FLARE)   (isolates teacher gain on top)
#
# Identical recipe to 08_train_gr00t_flare.sh except: no distillation dir,
# no --keep-full-gt-dataset, no --teacher-mix-ratio, no --allow-padding.
#
# Usage: bash FLARE-VLA/commands/libero_plus/09_train_gr00t_gt_only.sh

set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_gt_only}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
export SHARD_SIZE="${SHARD_SIZE:-1024}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
export USE_WANDB="${USE_WANDB:-0}"
export EXTRA_AUGMENTATION_CONFIG='{"gaussian_blur_params":{"blur_limit":[3,7],"p":0.5},"motion_blur_params":{"blur_limit":[3,7],"p":0.3}}'

cd "${GR00T_ROOT}"

# finetune.sh invokes `exec python`, so put the project venv on PATH.
export PATH="${GR00T_ROOT}/.venv/bin:${PATH}"

# Load the gated Cosmos-Reason2-2B backbone from the local HF cache (offline),
# since nvidia/Cosmos-Reason2-2B is a gated repo and hf-mirror returns 403.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GR00T_BACKBONE_MODEL_NAME="/root/autodl-tmp/Isaac-GR00T/checkpoints/Cosmos-Reason2-2B"

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --embodiment-tag libero_sim
  --output-dir "${OUTPUT_DIR}"
  --experiment-name gr00t_libero_plus_gt_only
  --state-dropout-prob 0.2
  --save-only-model
)

# Exact-chain preflight.
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --validate_config_only

bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
