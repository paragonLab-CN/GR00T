#!/usr/bin/env bash
# Checkpoint sweep: re-extract GT latents + score a 5,000-window subset for
# LeWM epochs 20/30/50, then compare oracle win rate / teacher agreement /
# oracle_gap / 8-step rollout MSE growth to select the best scoring checkpoint.
#
# Usage: bash FLARE-VLA/commands/libero_plus/11_checkpoint_sweep.sh [epochs]
#
# Outputs:
#   FLARE-VLA/outputs/libero_plus/gt_latents/libero_plus_epoch{N}_h8_framebank_trainpipe.h5
#   FLARE-VLA/outputs/libero_plus/checkpoint_sweep/epoch{N}/*   (scoring subset)
#   FLARE-VLA/outputs/libero_plus/checkpoint_sweep/comparison.json
#
# Roughly 2h per epoch for extraction (313 fps over 2.24M frames) + ~1min scoring.

set -euo pipefail

PY=/root/autodl-tmp/envs/LeWorldModel/bin/python
export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

EPOCHS="${1:-20,30,50}"
CKPT_DIR=/root/autodl-tmp/le-wm/stablewm_data/checkpoints/libero_plus_lewm_full
GT_DIR=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/gt_latents
MANIFEST=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/manifests/spatial_sparse_manifest.jsonl
CAND_DIR=/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/sparse_candidates/candidates
CAND_SUMMARY=/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/sparse_candidates/candidate_generation_summary.json
SWEEP_DIR=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/checkpoint_sweep
EXTRACT=/root/autodl-tmp/FLARE-VLA/scripts/libero_plus/03_extract_gt_latent_libero_plus.py
SCORE=/root/autodl-tmp/FLARE-VLA/scripts/libero_plus/05_score_libero_plus.py
COMPARE=/root/autodl-tmp/FLARE-VLA/scripts/libero_plus/07_compare_checkpoint_sweep.py

mkdir -p "${SWEEP_DIR}"

IFS=',' read -ra EPOCH_LIST <<< "${EPOCHS}"

for EPOCH in "${EPOCH_LIST[@]}"; do
    CKPT="${CKPT_DIR}/weights_epoch_${EPOCH}.pt"
    GT_H5="${GT_DIR}/libero_plus_epoch${EPOCH}_h8_framebank_trainpipe.h5"
    SHA=$(sha256sum "${CKPT}" | awk '{print $1}')

    echo
    echo "============================================================"
    echo "  EPOCH ${EPOCH}: extract GT latents"
    echo "  checkpoint: ${CKPT}"
    echo "  output:     ${GT_H5}"
    echo "============================================================"
    "${PY}" -u "${EXTRACT}" \
        --checkpoint-path "${CKPT}" \
        --output-path "${GT_H5}"

    echo
    echo "============================================================"
    echo "  EPOCH ${EPOCH}: score 5,000-window subset"
    echo "============================================================"
    "${PY}" -u "${SCORE}" \
        --manifest "${MANIFEST}" \
        --candidate-dir "${CAND_DIR}" \
        --candidate-summary "${CAND_SUMMARY}" \
        --gt-h5 "${GT_H5}" \
        --checkpoint "${CKPT}" \
        --expected-checkpoint-sha256 "${SHA}" \
        --output-dir "${SWEEP_DIR}/epoch${EPOCH}" \
        --batch-size 32 \
        --max-samples 5000
done

echo
echo "============================================================"
echo "  COMPARISON"
echo "============================================================"
"${PY}" "${COMPARE}" \
    --sweep-dir "${SWEEP_DIR}" \
    --output "${SWEEP_DIR}/comparison.json"

echo
echo "DONE. Sweep dir: ${SWEEP_DIR}"
