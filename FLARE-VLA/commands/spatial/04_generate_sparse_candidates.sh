#!/usr/bin/env bash
# Generate GR00T K=4 candidate action chunks for the sparse LIBERO-Spatial manifest.
# Uses libero_spatial weight (NOT libero_10).
#
# Produces:
#   spatial/sparse_candidates/candidates/        12,599 x {sample_id}.pt
#   spatial/sparse_candidates/candidate_generation_summary.json
#
# Spatial-specific. Does NOT read or modify any LIBERO-90 outputs.
#
# Usage:
#   bash FLARE-VLA/commands/spatial/04_generate_sparse_candidates.sh

set -euo pipefail

MANIFEST=/root/autodl-tmp/FLARE-VLA/outputs/spatial/manifests/spatial_sparse_manifest.jsonl
DATASET=/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot
MODEL=/root/autodl-tmp/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_spatial
OUTPUT_DIR=/root/autodl-tmp/Isaac-GR00T/outputs/spatial/sparse_candidates

cd /root/autodl-tmp/Isaac-GR00T

export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export UV_LINK_MODE=copy

echo "============================================"
echo "  Spatial Sparse Candidate Generation"
echo "============================================"
echo "  Manifest  : $MANIFEST"
echo "  Model     : libero_spatial"
echo "  Output    : $OUTPUT_DIR"
echo "  Windows   : 12,599"
echo "============================================"

uv run --no-sync python \
  scripts/flare/04_generate_validation_candidates.py \
  --manifest "$MANIFEST" \
  --dataset-path "$DATASET" \
  --model-path "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --embodiment-tag LIBERO_PANDA \
  --video-backend torchcodec \
  --action-horizon 8 \
  2>&1 | tee "$OUTPUT_DIR"/candidate_generation.log

echo ""
echo "Done. Candidates at: $OUTPUT_DIR/candidates/"
