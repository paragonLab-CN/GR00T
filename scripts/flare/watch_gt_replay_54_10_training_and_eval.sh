#!/usr/bin/env bash
set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
TRAIN_SESSION="${TRAIN_SESSION:-gt_replay_54_10_rerun_20260911}"
POLL_SECONDS="${POLL_SECONDS:-300}"
EVAL_PORT="${EVAL_PORT:-5555}"

OUTPUT_ROOT="${GR00T_ROOT}/outputs/libero_plus/${EXPERIMENT_NAME}"
RUN_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
TRAIN_LOG="${OUTPUT_ROOT}/train.log"
CHECKPOINT_1000="${RUN_DIR}/checkpoint-1000"
CHECKPOINT_10000="${RUN_DIR}/checkpoint-10000"
VALIDATION_LOG="${OUTPUT_ROOT}/checkpoint_1000_validation.log"

training_is_running() {
    tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null
}

training_completed() {
    grep -Fq "Training completed!" "${TRAIN_LOG}" 2>/dev/null
}

checkpoint_1000_is_written() {
    local required
    for required in \
        trainer_state.json \
        optimizer.pt \
        scheduler.pt \
        rng_state.pth \
        processor_config.json; do
        [[ -f "${CHECKPOINT_1000}/${required}" ]] || return 1
    done
    [[ -f "${CHECKPOINT_1000}/experiment_cfg/conf.yaml" ]] || return 1
    [[ -f "${CHECKPOINT_1000}/experiment_cfg/final_processor_config.json" ]] || \
        return 1
    [[ -f "${CHECKPOINT_1000}/model.safetensors" ]] || \
        [[ -f "${CHECKPOINT_1000}/model.safetensors.index.json" ]]
}

fail_if_training_stopped() {
    if ! training_is_running; then
        echo "Training session stopped before the expected artifact was ready: ${TRAIN_SESSION}" >&2
        tail -n 100 "${TRAIN_LOG}" >&2 || true
        exit 1
    fi
}

echo "Waiting for a complete checkpoint-1000: ${CHECKPOINT_1000}"
until checkpoint_1000_is_written; do
    if training_completed; then
        echo "Training completed without a complete checkpoint-1000" >&2
        exit 1
    fi
    fail_if_training_stopped
    sleep "${POLL_SECONDS}"
done

"${GR00T_ROOT}/.venv/bin/python" \
    "${GR00T_ROOT}/scripts/flare/check_image_aug_checkpoint.py" \
    "${CHECKPOINT_1000}" \
    --expected-step 1000 \
    --fixed-gt-samples-per-batch 54 \
    --fixed-teacher-samples-per-batch 10 \
    --require-full-training-state \
    >"${VALIDATION_LOG}" 2>&1
echo "checkpoint-1000 full-state validation passed: ${VALIDATION_LOG}"

echo "Waiting for training completion..."
until training_completed; do
    fail_if_training_stopped
    sleep "${POLL_SECONDS}"
done

"${GR00T_ROOT}/.venv/bin/python" \
    "${GR00T_ROOT}/scripts/flare/check_image_aug_checkpoint.py" \
    "${CHECKPOINT_10000}" \
    --expected-step 10000 \
    --fixed-gt-samples-per-batch 54 \
    --fixed-teacher-samples-per-batch 10 \
    --require-full-training-state

echo "Starting fixed-noise 112-task evaluation..."
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    bash "${GR00T_ROOT}/scripts/flare/run_eval_libero_plus_gt_replay_54_10_fixed_noise.sh" \
    "${EVAL_PORT}"
echo "Training and fixed-noise evaluation completed."
