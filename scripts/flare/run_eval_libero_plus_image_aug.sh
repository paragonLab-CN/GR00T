#!/usr/bin/env bash
set -euo pipefail

# Evaluate only the controlled image-augmentation checkpoint-10000 on the exact
# 112-task manifest used by the historical v1 FLARE baseline.

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
MODEL_TAG="gr00t_libero_plus_flare_image_aug_checkpoint_10000"
MODEL_PATH="${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare_image_aug/\
gr00t_libero_plus_flare_image_aug/checkpoint-10000"
BASELINE_TASKS="${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_flare/tasks.tsv"
PORT="${1:-5555}"

cd "${GR00T_ROOT}"

.venv/bin/python scripts/flare/check_image_aug_checkpoint.py "${MODEL_PATH}"

export N_EPISODES=10
export N_ENVS=1
export SEED=2026
export MAX_EPISODE_STEPS=720
export N_ACTION_STEPS=8
export TASK_FILE_SOURCE="${BASELINE_TASKS}"
export EXPECTED_TASK_COUNT=112
export TASK_LIMIT=0
export FULL_EVAL=0
export KEEP_VIDEOS=0
export RESUME=1

bash scripts/flare/run_eval_libero_plus_single_model.sh \
    "${MODEL_PATH}" "${MODEL_TAG}" "${PORT}"

.venv/bin/python scripts/flare/summarize_image_aug_eval.py \
    "${GR00T_ROOT}/outputs/eval_libero_plus/${MODEL_TAG}/summary_all.json"
