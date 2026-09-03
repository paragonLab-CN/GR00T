#!/usr/bin/env bash
# Train GR00T-N1.7-3B on the FLARE-selected teacher source ONLY (no GT).
#
# Three-way ablation:
#   - 09: GT-only           (isolates GT fine-tuning gain)
#   - 10: teacher-only      (isolates teacher data alone)  <- this run
#   - 08: GT + teacher      (FLARE, the full method)
# All share the same base (GR00T-N1.7-3B), 10k steps, bs64/accum10/lr1e-5.
#
# teacher-only uses the flare_distillation dataset as the SOLE source
# (--distillation-dataset-dir without --keep-full-gt-dataset), so mix_ratio=1.0
# on the teacher index. --allow-padding is required: the teacher records are
# horizon-8 but the GR00T processor uses horizon-16.
#
# Usage: bash FLARE-VLA/commands/libero_plus/10_train_gr00t_teacher_only.sh

set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot}"
DISTILLATION_DATASET_DIR="${DISTILLATION_DATASET_DIR:-/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/distillation_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_teacher_only}"

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
  --experiment-name gr00t_libero_plus_teacher_only
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}"
  --allow-padding
  --state-dropout-prob 0.2
  --save-only-model
)

# Exact-chain preflight.
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --validate_config_only

bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
