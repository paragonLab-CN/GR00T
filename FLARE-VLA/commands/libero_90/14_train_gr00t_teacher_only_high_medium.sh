#!/usr/bin/env bash
set -euo pipefail

# Reproduce the teacher-only high+medium GR00T ablation.
# This intentionally uses only the FLARE teacher sidecar and does NOT retain
# the full physical GT dataset.

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_10}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/lerobot_data/libero_90_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/gr00t_teacher_only_high_medium_steps3000_retrain_v2}"

# Prefer the original server-side directory name recorded by the deleted run.
# Fall back to the clearer renamed directory when that is the one available.
if [[ -z "${DISTILLATION_DATASET_DIR:-}" ]]; then
  if [[ -f "${GR00T_ROOT}/outputs/flare_full_teacher_only/distillation_index.jsonl" && \
        -f "${GR00T_ROOT}/outputs/flare_full_teacher_only/teacher_records_enriched.pt" ]]; then
    DISTILLATION_DATASET_DIR="${GR00T_ROOT}/outputs/flare_full_teacher_only"
  elif [[ -f "${GR00T_ROOT}/outputs/flare_full_teacher_high_medium_only/distillation_index.jsonl" && \
          -f "${GR00T_ROOT}/outputs/flare_full_teacher_high_medium_only/teacher_records_enriched.pt" ]]; then
    DISTILLATION_DATASET_DIR="${GR00T_ROOT}/outputs/flare_full_teacher_high_medium_only"
  else
    echo "Could not find the teacher-only FLARE sidecars." >&2
    echo "Set DISTILLATION_DATASET_DIR explicitly." >&2
    exit 1
  fi
fi

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

# Do not inherit the fail-closed contract used by the 80/20 mixed launcher.
unset FLARE_REQUIRE_MIXED_DATA || true
unset FLARE_EXPECTED_TEACHER_MIX_RATIO || true

cd "${GR00T_ROOT}"

for required_path in \
  "${BASE_MODEL_PATH}" \
  "${DATASET_PATH}" \
  "${DISTILLATION_DATASET_DIR}/distillation_index.jsonl" \
  "${DISTILLATION_DATASET_DIR}/teacher_records_enriched.pt"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required input: ${required_path}" >&2
    exit 1
  fi
done

python scripts/flare/check_distillation_sidecars.py \
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}" \
  --expected-total 15635 \
  --expected-gt 0 \
  --expected-teacher 15635

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --embodiment-tag libero_sim
  --output-dir "${OUTPUT_DIR}"
  --experiment-name gr00t_full_gt_flare_h8_s8
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}"
  --allow-padding
  --state-dropout-prob 0.2
  --save-only-model
)

# Resolve the exact configuration first. The output must contain exactly one
# flare_distillation source with mix_ratio 1.0 and no physical GT source.
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- --validate_config_only

# Start the full retraining run only after the sidecar and configuration checks pass.
bash examples/finetune.sh "${COMMON_ARGS[@]}"
