#!/usr/bin/env bash
# Train GR00T with the complete LIBERO-Plus GT source plus the FLARE-selected
# teacher-only source (mixed), with the action-head projector FROZEN
# (tune_projector=false). Only the DiT + VLLN of the action head are trained.
#
# This is an ablation of the `gr00t_libero_plus_flare` baseline (v1, 78.48%).
# The ONLY intended difference from that baseline is `--no-tune_projector`.
# Data, hyperparameters, max_steps, seed and augmentation are all matched:
#   - v1 distillation dataset (not _v2)
#   - teacher SAMPLING SHARE matched to the baseline (~15%), see NOTE below
#   - color jitter restored to v1's 0.3/0.4/0.5/0.08 (finetune.sh now hardcodes
#     0.5/0.6/0.7/0.12, which was introduced AFTER v1 was trained on 2026-08-27;
#     we override it back via a trailing duplicate flag, which tyro resolves
#     last-wins).
#   - no extra augmentation (v1 used extra_augmentation_config=null)
#
# NOTE (teacher sampling share): v1 launched with teacher_mix_ratio=0.15 under
# the OLD sampling logic `weight = relative_length * mix_ratio`. The OLD code
# computes relative_length PER dataset_spec (each spec has exactly one dataset
# path), so relative_length is always 1.0 and weight = mix_ratio -> teacher
# ACTUAL share ~15%. The 2026-08-27 change to `weight = mix_ratio /
# supervised_steps` (GT=16, teacher=8) makes 0.15 mean ~26% teacher -- NOT
# comparable. Under the current logic the share is 2r/(1+r); solving
# 2r/(1+r)=0.15 gives r=0.0811.
#
# Usage: bash FLARE-VLA/commands/libero_plus/08c_train_gr00t_flare_frozen_projector.sh

set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot}"
DISTILLATION_DATASET_DIR="${DISTILLATION_DATASET_DIR:-/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/distillation_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare_frozen_projector}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
export SHARD_SIZE="${SHARD_SIZE:-1024}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
export TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO:-0.0811}"
export USE_WANDB="${USE_WANDB:-0}"

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
  --experiment-name gr00t_libero_plus_flare_frozen_projector
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}"
  --keep-full-gt-dataset
  --teacher-mix-ratio "${TEACHER_MIX_RATIO}"
  --allow-padding
  --state-dropout-prob 0.2
  --save-only-model
)

# The trailing --color_jitter_params overrides finetune.sh's hardcoded
# 0.5/0.6/0.7/0.12 back to v1's 0.3/0.4/0.5/0.08 (tyro resolves last-wins).
AUG_ARGS=(
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
)

# Exact-chain preflight: verify the mixed-data contract before loading the model.
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --no-tune_projector \
  "${AUG_ARGS[@]}" \
  --validate_config_only

# Only reached when the mixed-data contract passed.
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --no-tune_projector \
  "${AUG_ARGS[@]}"
