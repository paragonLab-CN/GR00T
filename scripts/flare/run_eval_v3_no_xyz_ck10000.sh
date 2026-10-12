#!/usr/bin/env bash
set -euo pipefail

# Evaluate the v3_no_xyz checkpoint-10000 on the same 112-task LIBERO-Plus
# stratified benchmark used by gr00t_libero_plus_flare_image_aug_gt_replay_54_10_rerun_20260911.
#
# Two passes with different diffusion-noise protocols:
#   Pass 1: DIFFUSION_SEED=2026 (server-side fixed seed)
#   Pass 2: DIFFUSION_SEED=""   (unfixed, run-to-run variable)
#
# Server port 5555 for Pass 1, 5556 for Pass 2 (avoids collision if Pass 1
# leaves anything bound). Each pass writes its own output directory so the
# two evaluations are isolated and reproducible.
#
# Output roots:
#   ${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_flare_v3_no_xyz_44_20_ck10000_fixed_noise_seed2026/
#   ${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_flare_v3_no_xyz_44_20_ck10000_unfixed_noise/
#
# Usage:
#   bash scripts/flare/run_eval_v3_no_xyz_ck10000.sh
#
# Environment overrides:
#   N_EPISODES  per-task episodes (default 10)

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
CHECKPOINT="${CHECKPOINT:-${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare_v3_no_xyz_44_20/gr00t_libero_plus_flare_v3_no_xyz_44_20/checkpoint-10000}"
TASKS_TSV="${TASKS_TSV:-${GR00T_ROOT}/outputs/eval_libero_plus/gr00t_libero_plus_flare_image_aug_gt_replay_54_10_rerun_20260911_checkpoint_10000_fixed_noise_seed2026/tasks.tsv}"
N_EPISODES="${N_EPISODES:-10}"

TASK_LINES="$(wc -l < "${TASKS_TSV}")"
if [[ "${TASK_LINES}" != "112" ]]; then
    echo "tasks.tsv line count = ${TASK_LINES}, expected 112" >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "Checkpoint missing: ${CHECKPOINT}" >&2
    exit 1
fi

# ---------- Pass 1: fixed noise seed 2026 ----------
MODEL_TAG_FIXED="gr00t_libero_plus_flare_v3_no_xyz_44_20_ck10000_fixed_noise_seed2026"
echo "============================================"
echo " Pass 1: fixed diffusion_noise_seed=2026"
echo " Tag : ${MODEL_TAG_FIXED}"
echo "============================================"
TASK_FILE_SOURCE="${TASKS_TSV}" \
    DIFFUSION_SEED=2026 \
    N_EPISODES="${N_EPISODES}" \
    SUITES=libero_spatial,libero_object,libero_goal,libero_10 \
    EXPECTED_TASK_COUNT=112 \
    FULL_EVAL=0 \
    KEEP_VIDEOS=0 \
    RESUME=1 \
    bash scripts/flare/run_eval_libero_plus_single_model.sh \
            "${CHECKPOINT}" "${MODEL_TAG_FIXED}" 5555

# ---------- Pass 2: unfixed ----------
MODEL_TAG_UNFIXED="gr00t_libero_plus_flare_v3_no_xyz_44_20_ck10000_unfixed_noise"
echo "============================================"
echo " Pass 2: unfixed (random) diffusion_noise"
echo " Tag : ${MODEL_TAG_UNFIXED}"
echo "============================================"
TASK_FILE_SOURCE="${TASKS_TSV}" \
    DIFFUSION_SEED="" \
    N_EPISODES="${N_EPISODES}" \
    SUITES=libero_spatial,libero_object,libero_goal,libero_10 \
    EXPECTED_TASK_COUNT=112 \
    FULL_EVAL=0 \
    KEEP_VIDEOS=0 \
    RESUME=1 \
    bash scripts/flare/run_eval_libero_plus_single_model.sh \
            "${CHECKPOINT}" "${MODEL_TAG_UNFIXED}" 5556

echo "============================================"
echo " Both passes complete:"
echo "   Fixed   : ${GR00T_ROOT}/outputs/eval_libero_plus/${MODEL_TAG_FIXED}/summary_all.json"
echo "   Unfixed : ${GR00T_ROOT}/outputs/eval_libero_plus/${MODEL_TAG_UNFIXED}/summary_all.json"
echo "============================================"