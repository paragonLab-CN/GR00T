#!/usr/bin/env bash
set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
PORT="${1:-5555}"
DIFFUSION_SEED="${DIFFUSION_SEED:-2026}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gr00t_libero_plus_flare_image_aug_gt_replay_54_10}"
MODEL_PATH="${GR00T_ROOT}/outputs/libero_plus/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}/checkpoint-10000"
MODEL_TAG="${EXPERIMENT_NAME}_checkpoint_10000_fixed_noise_seed${DIFFUSION_SEED}"
TASKS_PATH="${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_flare_checkpoint_10000_fixed_noise_seed2026/tasks.tsv"

FLARE_SUMMARY="${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_flare_checkpoint_10000_fixed_noise_seed2026/summary_all.json"
IMAGE_AUG_SUMMARY="${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_flare_image_aug_checkpoint_10000_fixed_noise_seed2026/summary_all.json"
GT_ONLY_SUMMARY="${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_gt_only_checkpoint_10000_fixed_noise_seed2026/summary_all.json"
REPLAY_ROOT="${GR00T_ROOT}/outputs/eval_libero_plus/${MODEL_TAG}"
REPLAY_SUMMARY="${REPLAY_ROOT}/summary_all.json"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Checkpoint directory does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
for path in "${TASKS_PATH}" "${FLARE_SUMMARY}" "${IMAGE_AUG_SUMMARY}" "${GT_ONLY_SUMMARY}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required paired-evaluation artifact does not exist: ${path}" >&2
        exit 1
    fi
done

cd "${GR00T_ROOT}"
export OMP_NUM_THREADS=1
export N_EPISODES=10
export N_ENVS=1
export SEED=2026
export MAX_EPISODE_STEPS=720
export N_ACTION_STEPS=8
export DIFFUSION_SEED
export TASK_FILE_SOURCE="${TASKS_PATH}"
export EXPECTED_TASK_COUNT=112
export TASK_LIMIT=0
export FULL_EVAL=0
export KEEP_VIDEOS=0
export RESUME=1

.venv/bin/python scripts/flare/check_image_aug_checkpoint.py \
    "${MODEL_PATH}" \
    --fixed-gt-samples-per-batch 54 \
    --fixed-teacher-samples-per-batch 10 \
    --require-full-training-state

echo "============================================================"
echo "GT replay 54:10 fixed-noise LIBERO-Plus evaluation"
echo "Checkpoint          : ${MODEL_PATH}"
echo "Tasks / episodes    : 112 x 10"
echo "Environment seed    : ${SEED}"
echo "Diffusion master    : ${DIFFUSION_SEED}"
echo "Task manifest SHA256: $(sha256sum "${TASKS_PATH}" | cut -d ' ' -f 1)"
echo "============================================================"

bash scripts/flare/run_eval_libero_plus_single_model.sh \
    "${MODEL_PATH}" "${MODEL_TAG}" "${PORT}"

.venv/bin/python scripts/flare/summarize_fixed_noise_four.py \
    "${FLARE_SUMMARY}" \
    "${IMAGE_AUG_SUMMARY}" \
    "${GT_ONLY_SUMMARY}" \
    "${REPLAY_SUMMARY}" \
    --output "${REPLAY_ROOT}/four_model_comparison.json"
