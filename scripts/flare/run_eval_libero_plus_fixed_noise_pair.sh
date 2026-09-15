#!/usr/bin/env bash
set -euo pipefail

# Paired 112-task LIBERO-Plus evaluation with identical explicit diffusion
# initial noise for the historical FLARE checkpoint and image-aug checkpoint.
# The request seed is keyed by task, episode, and action-chunk indices, so the
# pairing remains valid when episode lengths differ or the run is resumed.

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
PORT="${1:-5555}"
DIFFUSION_SEED="${DIFFUSION_SEED:-2026}"

FLARE_MODEL_PATH="${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare/gr00t_libero_plus_flare/checkpoint-10000"
IMAGE_AUG_MODEL_PATH="${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare_image_aug/gr00t_libero_plus_flare_image_aug/checkpoint-10000"
FLARE_TAG="gr00t_libero_plus_flare_checkpoint_10000_fixed_noise_seed${DIFFUSION_SEED}"
IMAGE_AUG_TAG="gr00t_libero_plus_flare_image_aug_checkpoint_10000_fixed_noise_seed${DIFFUSION_SEED}"
BASELINE_TASKS="${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_flare/tasks.tsv"

for model_path in "${FLARE_MODEL_PATH}" "${IMAGE_AUG_MODEL_PATH}"; do
    if [[ ! -d "${model_path}" ]]; then
        echo "Checkpoint directory does not exist: ${model_path}" >&2
        exit 1
    fi
done
if [[ ! -f "${BASELINE_TASKS}" ]]; then
    echo "Historical 112-task manifest does not exist: ${BASELINE_TASKS}" >&2
    exit 1
fi

cd "${GR00T_ROOT}"
.venv/bin/python scripts/flare/check_image_aug_checkpoint.py "${IMAGE_AUG_MODEL_PATH}"

export N_EPISODES=10
export N_ENVS=1
export SEED=2026
export MAX_EPISODE_STEPS=720
export N_ACTION_STEPS=8
export DIFFUSION_SEED
export TASK_FILE_SOURCE="${BASELINE_TASKS}"
export EXPECTED_TASK_COUNT=112
export TASK_LIMIT=0
export FULL_EVAL=0
export KEEP_VIDEOS=0
export RESUME=1

echo "============================================================"
echo "Paired fixed-noise LIBERO-Plus evaluation"
echo "Tasks              : 112"
echo "Episodes/task      : 10"
echo "Environment seed   : ${SEED}"
echo "Diffusion seed     : ${DIFFUSION_SEED}"
echo "FLARE output tag   : ${FLARE_TAG}"
echo "Image-aug output   : ${IMAGE_AUG_TAG}"
echo "============================================================"

bash scripts/flare/run_eval_libero_plus_single_model.sh \
    "${FLARE_MODEL_PATH}" "${FLARE_TAG}" "${PORT}"

bash scripts/flare/run_eval_libero_plus_single_model.sh \
    "${IMAGE_AUG_MODEL_PATH}" "${IMAGE_AUG_TAG}" "${PORT}"

.venv/bin/python scripts/flare/summarize_fixed_noise_pair.py \
    "${GR00T_ROOT}/outputs/eval_libero_plus/${FLARE_TAG}/summary_all.json" \
    "${GR00T_ROOT}/outputs/eval_libero_plus/${IMAGE_AUG_TAG}/summary_all.json" \
    --output "${GR00T_ROOT}/outputs/eval_libero_plus/${IMAGE_AUG_TAG}/image_aug_comparison.json"
