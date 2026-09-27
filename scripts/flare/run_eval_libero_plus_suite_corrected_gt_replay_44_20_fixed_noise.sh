#!/usr/bin/env bash
set -euo pipefail

# Evaluate the suite-corrected LIBERO-Plus GT-replay 44/20 checkpoint on the
# frozen 112-task protocol. Both environment randomness and server-side
# diffusion noise are fixed to seed 2026 for paired comparison with 54/10.

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly GR00T_ROOT="${GR00T_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
readonly GR00T_PYTHON="${GR00T_ROOT}/.venv/bin/python"
readonly EXPERIMENT_NAME="gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20"
readonly BASELINE_EXPERIMENT_NAME="gr00t_libero_plus_flare_suite_corrected_gt_replay_54_10"
readonly CHECKPOINT_STEP="10000"
readonly PORT="${1:-5555}"

readonly FIXED_SEED="2026"
readonly DIFFUSION_SEED="${DIFFUSION_SEED:-${FIXED_SEED}}"
readonly EVAL_SEED="${EVAL_SEED:-${FIXED_SEED}}"
readonly EXPECTED_TASK_COUNT="112"
readonly EXPECTED_TASKS_SHA256="6412991d21db24aaacd402cdd35e716962df331112cdfa1a6201c20f306c3b3f"

readonly MODEL_PATH="${MODEL_PATH:-${GR00T_ROOT}/outputs/libero_plus/${EXPERIMENT_NAME}/checkpoint-${CHECKPOINT_STEP}}"
readonly MODEL_TAG="${EXPERIMENT_NAME}_checkpoint_${CHECKPOINT_STEP}_fixed_noise_seed${FIXED_SEED}"
readonly BASELINE_TAG="${BASELINE_EXPERIMENT_NAME}_checkpoint_${CHECKPOINT_STEP}_fixed_noise_seed${FIXED_SEED}"
readonly TASKS_PATH="${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_flare_checkpoint_10000_fixed_noise_seed2026/tasks.tsv"
readonly OUTPUT_ROOT="${GR00T_ROOT}/outputs/eval_libero_plus/${MODEL_TAG}"
readonly REPLAY_SUMMARY="${OUTPUT_ROOT}/summary_all.json"
readonly BASELINE_SUMMARY="${GR00T_ROOT}/outputs/eval_libero_plus/${BASELINE_TAG}/summary_all.json"
readonly COMPARISON_PATH="${OUTPUT_ROOT}/gt_replay_44_20_vs_54_10_comparison.json"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "${PORT}" =~ ^[0-9]+$ ]] || die "port must be an integer: ${PORT}"
(( PORT >= 1 && PORT <= 65535 )) || die "port must be in [1, 65535]: ${PORT}"

if [[ "${DIFFUSION_SEED}" != "${FIXED_SEED}" || "${EVAL_SEED}" != "${FIXED_SEED}" ]]; then
    die "this paired protocol is frozen at EVAL_SEED=DIFFUSION_SEED=${FIXED_SEED}"
fi

[[ -x "${GR00T_PYTHON}" ]] || die "project Python is unavailable: ${GR00T_PYTHON}"
if pgrep -af '[l]aunch_finetune.py' | grep -F -- "--experiment_name ${EXPERIMENT_NAME}" >/dev/null; then
    die "training is still running for ${EXPERIMENT_NAME}; start evaluation after it exits"
fi

[[ -d "${MODEL_PATH}" ]] || die "checkpoint not found: ${MODEL_PATH}"
[[ -f "${TASKS_PATH}" ]] || die "112-task manifest not found: ${TASKS_PATH}"
[[ -f "${BASELINE_SUMMARY}" ]] || die "54/10 baseline summary not found: ${BASELINE_SUMMARY}"

actual_task_count="$(awk 'NF {count++} END {print count+0}' "${TASKS_PATH}")"
[[ "${actual_task_count}" == "${EXPECTED_TASK_COUNT}" ]] || \
    die "task count mismatch: expected ${EXPECTED_TASK_COUNT}, got ${actual_task_count}"

actual_tasks_sha256="$(sha256sum "${TASKS_PATH}" | awk '{print $1}')"
[[ "${actual_tasks_sha256}" == "${EXPECTED_TASKS_SHA256}" ]] || \
    die "task manifest changed: expected sha256 ${EXPECTED_TASKS_SHA256}, got ${actual_tasks_sha256}"

cd "${GR00T_ROOT}"

# Fail before starting the simulator if the final checkpoint is partial or if
# its augmentation / GT-replay metadata does not match this evaluation target.
"${GR00T_PYTHON}" scripts/flare/check_image_aug_checkpoint.py \
    "${MODEL_PATH}" \
    --expected-step "${CHECKPOINT_STEP}" \
    --fixed-gt-samples-per-batch 44 \
    --fixed-teacher-samples-per-batch 20 \
    --require-full-training-state

export OMP_NUM_THREADS=1
export N_EPISODES=10
export N_ENVS=1
export SEED="${EVAL_SEED}"
export MAX_EPISODE_STEPS=720
export N_ACTION_STEPS=8
export DIFFUSION_SEED
export TASK_FILE_SOURCE="${TASKS_PATH}"
export EXPECTED_TASK_COUNT
export TASK_LIMIT=0
export FULL_EVAL=0
export KEEP_VIDEOS="${KEEP_VIDEOS:-0}"
export RESUME=1

echo "Starting frozen LIBERO-Plus evaluation"
echo "  model:            ${MODEL_PATH}"
echo "  tasks:            ${EXPECTED_TASK_COUNT} (sha256=${EXPECTED_TASKS_SHA256})"
echo "  episodes/task:    ${N_EPISODES}"
echo "  environment seed: ${SEED}"
echo "  diffusion seed:   ${DIFFUSION_SEED}"
echo "  output:           ${OUTPUT_ROOT}"
echo "  resume:           enabled"

bash scripts/flare/run_eval_libero_plus_single_model.sh \
    "${MODEL_PATH}" \
    "${MODEL_TAG}" \
    "${PORT}"

[[ -f "${REPLAY_SUMMARY}" ]] || die "evaluation finished without summary: ${REPLAY_SUMMARY}"
[[ -f "${OUTPUT_ROOT}/tasks.tsv" ]] || die "evaluation output has no task manifest"

output_tasks_sha256="$(sha256sum "${OUTPUT_ROOT}/tasks.tsv" | awk '{print $1}')"
[[ "${output_tasks_sha256}" == "${EXPECTED_TASKS_SHA256}" ]] || \
    die "evaluated task manifest does not match the frozen 112-task protocol"

"${GR00T_PYTHON}" scripts/flare/summarize_fixed_noise_pair.py \
    "${BASELINE_SUMMARY}" \
    "${REPLAY_SUMMARY}" \
    --baseline-label "gt_replay_54_10" \
    --candidate-label "gt_replay_44_20" \
    --output "${COMPARISON_PATH}"

echo "Evaluation complete"
echo "  summary:    ${REPLAY_SUMMARY}"
echo "  comparison: ${COMPARISON_PATH}"
