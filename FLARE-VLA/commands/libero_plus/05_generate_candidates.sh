#!/usr/bin/env bash
# Generate GR00T K=4 candidate action chunks for the sparse LIBERO-Plus manifest,
# per suite (each suite's windows are proposed by that suite's GR00T-N1.7-LIBERO
# checkpoint — the "teacher" model), then merge into a single candidate set.
#
# Produces:
#   Isaac-GR00T/outputs/libero_plus/sparse_candidates/<suite>/candidates/
#   Isaac-GR00T/outputs/libero_plus/sparse_candidates/candidates/            (merged)
#   Isaac-GR00T/outputs/libero_plus/sparse_candidates/candidate_generation_summary.json
#
# Scale: 489,632 windows × 4 candidates; ~0.175 s/window → ~24h total.
#
# Usage: bash FLARE-VLA/commands/libero_plus/05_generate_candidates.sh

set -euo pipefail

DATASET=/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot
CHECKPOINT_ROOT=/root/autodl-tmp/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO
MANIFEST_DIR=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/manifests/per_suite
FULL_MANIFEST=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/manifests/spatial_sparse_manifest.jsonl
OUTPUT_ROOT=/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/sparse_candidates

cd /root/autodl-tmp/Isaac-GR00T

export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export UV_LINK_MODE=copy

for SUITE in libero_spatial libero_object libero_goal libero_10; do
    MANIFEST="${MANIFEST_DIR}/manifest_${SUITE}.jsonl"
    MODEL="${CHECKPOINT_ROOT}/${SUITE}"
    OUT="${OUTPUT_ROOT}/${SUITE}"
    mkdir -p "${OUT}"

    echo
    echo "============================================"
    echo "  Candidate generation: ${SUITE}"
    echo "  Manifest: ${MANIFEST}"
    echo "  Model   : ${MODEL}"
    echo "  Output  : ${OUT}"
    echo "============================================"

    uv run --no-sync python \
      scripts/flare/04_generate_validation_candidates.py \
      --manifest "${MANIFEST}" \
      --dataset-path "${DATASET}" \
      --model-path "${MODEL}" \
      --output-dir "${OUT}" \
      --embodiment-tag LIBERO_PANDA \
      --video-backend torchcodec \
      --action-horizon 8 \
      2>&1 | tee "${OUT}/candidate_generation.log"
done

echo
echo "============================================"
echo "  Merging per-suite candidates"
echo "============================================"
/root/autodl-tmp/envs/LeWorldModel/bin/python \
  /root/autodl-tmp/FLARE-VLA/scripts/libero_plus/04_merge_candidate_summaries.py \
  --manifest "${FULL_MANIFEST}" \
  --suite-dir "${OUTPUT_ROOT}"

echo ""
echo "Done. Merged candidates at: ${OUTPUT_ROOT}/candidates/"
