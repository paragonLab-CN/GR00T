#!/usr/bin/env bash
set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gr00t_libero_plus_flare_image_aug_gt_replay_54_10}"
OUTPUT_DIR="${GR00T_ROOT}/outputs/libero_plus/${EXPERIMENT_NAME}"
RUN_DIR="${OUTPUT_DIR}/${EXPERIMENT_NAME}"
REFERENCE_CONFIG="${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare/gr00t_libero_plus_flare/experiment_cfg/conf.yaml"
SIDECAR_DIR="/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/distillation_dataset"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"

if [[ "${SAVE_ONLY_MODEL}" != "0" && "${SAVE_ONLY_MODEL}" != "1" ]]; then
    echo "SAVE_ONLY_MODEL must be 0 or 1, got: ${SAVE_ONLY_MODEL}" >&2
    exit 1
fi
CHECKPOINT_SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL}"
# examples/finetune.sh treats any non-empty inherited SAVE_ONLY_MODEL as true.
# Keep the user-facing 0/1 setting locally and only pass its explicit CLI flag.
unset SAVE_ONLY_MODEL

mapfile -t EXISTING_CHECKPOINTS < <(
    find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V
)
if (( ${#EXISTING_CHECKPOINTS[@]} > 0 )); then
    if [[ "${ALLOW_RESUME}" != "1" ]]; then
        echo "Refusing to reuse existing checkpoints without ALLOW_RESUME=1: ${RUN_DIR}" >&2
        exit 1
    fi
    LATEST_CHECKPOINT="${EXISTING_CHECKPOINTS[-1]}"
    LATEST_STEP="${LATEST_CHECKPOINT##*-}"
    if [[ ! "${LATEST_STEP}" =~ ^[0-9]+$ ]]; then
        echo "Invalid checkpoint step in path: ${LATEST_CHECKPOINT}" >&2
        exit 1
    fi
    "${GR00T_ROOT}/.venv/bin/python" \
        "${GR00T_ROOT}/scripts/flare/check_image_aug_checkpoint.py" \
        "${LATEST_CHECKPOINT}" \
        --expected-step "${LATEST_STEP}" \
        --fixed-gt-samples-per-batch 54 \
        --fixed-teacher-samples-per-batch 10 \
        --require-full-training-state
    echo "Validated resumable checkpoint: ${LATEST_CHECKPOINT}"
elif [[ "${ALLOW_RESUME}" == "1" ]]; then
    echo "ALLOW_RESUME=1 was requested but no checkpoint exists: ${RUN_DIR}" >&2
    exit 1
elif [[ -e "${RUN_DIR}" ]]; then
    echo "Refusing to overwrite an existing run directory without a checkpoint: ${RUN_DIR}" >&2
    exit 1
fi

cd "${GR00T_ROOT}"

export OMP_NUM_THREADS=1
export PATH="${GR00T_ROOT}/.venv/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GR00T_BACKBONE_MODEL_NAME="${GR00T_ROOT}/checkpoints/Cosmos-Reason2-2B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NUM_GPUS="${NUM_GPUS:-1}"
export USE_WANDB=0
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
export GLOBAL_BATCH_SIZE=64
export LEARNING_RATE=1e-5
export SHARD_SIZE=1024
export NUM_SHARDS_PER_EPOCH=100000
export EPISODE_SAMPLING_RATE=1.0
export EXTRA_AUGMENTATION_CONFIG='{"gaussian_blur_params":{"blur_limit":[3,7],"p":0.5},"motion_blur_params":{"blur_limit":[3,7],"p":0.3}}'

export FLARE_REQUIRE_MIXED_DATA=1
export FLARE_EXPECTED_TEACHER_MIX_RATIO=0.15
export FLARE_REFERENCE_CONFIG_PATH="${REFERENCE_CONFIG}"
export FLARE_EXPECTED_COLOR_JITTER_PARAMS='{"brightness":0.5,"contrast":0.6,"saturation":0.7,"hue":0.12}'
export FLARE_EXPECTED_EXTRA_AUGMENTATION_CONFIG="${EXTRA_AUGMENTATION_CONFIG}"
export FLARE_EXPECTED_SAVE_ONLY_MODEL="${CHECKPOINT_SAVE_ONLY_MODEL}"
if [[ "${FLARE_SMOKE_TEST:-0}" == "1" ]]; then
    unset FLARE_REFERENCE_CONFIG_PATH
fi

COMMON_ARGS=(
    --base-model-path "${GR00T_ROOT}/checkpoints/GR00T-N1.7-3B"
    --dataset-path "${GR00T_ROOT}/examples/LIBERO/libero_plus_lerobot"
    --embodiment-tag libero_sim
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --distillation-dataset-dir "${SIDECAR_DIR}"
    --keep-full-gt-dataset
    --teacher-mix-ratio 0.15
    --fixed-gt-samples-per-batch 54
    --fixed-teacher-samples-per-batch 10
    --allow-padding
    --state-dropout-prob 0.2
)
if [[ "${CHECKPOINT_SAVE_ONLY_MODEL}" == "1" ]]; then
    COMMON_ARGS+=(--save-only-model)
fi

echo "Validating fixed 54:10 resolved configuration..."
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
    --gradient_accumulation_steps 10 \
    --validate_config_only
if [[ "${RUN_MODE:-all}" == "validate" ]]; then
    exit 0
fi

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Running ten-batch real-data preflight..."
    .venv/bin/python scripts/flare/check_fixed_gt_replay_batches.py
fi
if [[ "${RUN_MODE:-all}" == "preflight" ]]; then
    exit 0
fi

echo "Starting training from the original GR00T base for ${MAX_STEPS} steps..."
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
    --gradient_accumulation_steps 10

