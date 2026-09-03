#!/usr/bin/env bash
set -euo pipefail

# Train the 50% full LIBERO-90 GT + 50% high/medium FLARE teacher ablation.
# This reuses the validated mixed-data launcher and changes only the source ratio
# and output directory relative to the 80/20 experiment.

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
FLARE_ROOT="${FLARE_ROOT:-/root/autodl-tmp/FLARE-VLA}"

# Use the exact same teacher-only sidecar as the 20% and 100% teacher runs.
if [[ -z "${DISTILLATION_DATASET_DIR:-}" ]]; then
  if [[ -f "${GR00T_ROOT}/outputs/flare_full_teacher_only/distillation_index.jsonl" && \
        -f "${GR00T_ROOT}/outputs/flare_full_teacher_only/teacher_records_enriched.pt" ]]; then
    DISTILLATION_DATASET_DIR="${GR00T_ROOT}/outputs/flare_full_teacher_only"
  elif [[ -f "${GR00T_ROOT}/outputs/flare_full_teacher_high_medium_only/distillation_index.jsonl" && \
          -f "${GR00T_ROOT}/outputs/flare_full_teacher_high_medium_only/teacher_records_enriched.pt" ]]; then
    DISTILLATION_DATASET_DIR="${GR00T_ROOT}/outputs/flare_full_teacher_high_medium_only"
  else
    echo "Could not find the high+medium teacher-only FLARE sidecars." >&2
    echo "Set DISTILLATION_DATASET_DIR explicitly." >&2
    exit 1
  fi
fi

export GR00T_ROOT
export FLARE_ROOT
export DISTILLATION_DATASET_DIR
export TEACHER_MIX_RATIO=0.5
export MAX_STEPS="${MAX_STEPS:-3000}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/gr00t_full_gt_flare_mix50_50_steps3000_v1}"

# The delegated launcher performs a fail-closed config preflight and must resolve:
# physical_embodiment=0.5 and flare_distillation=0.5 before training starts.
bash "${FLARE_ROOT}/commands/11_train_gr00t_full_gt_flare.sh"
