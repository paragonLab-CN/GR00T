#!/usr/bin/env bash
# Score sparse LIBERO-Spatial candidates with LeWM-Spatial.
# Reuses 05_score_spatial.py with sparse manifest/candidates.
#
# Produces (le-wm/outputs/spatial/scoring/):
#   score_parts/           12,599 x {sample_id}.pt
#   validation_scores.csv  per-sample scoring
#   validation_scores.parquet
#   validation_teacher_records.pt
#   validation_summary.json
#   scoring.log
#
# Spatial-specific. Does NOT read or modify any LIBERO-90 outputs.
#
# Usage:
#   bash FLARE-VLA/commands/spatial/06_score_sparse_candidates.sh

set -euo pipefail

MANIFEST=/root/autodl-tmp/FLARE-VLA/outputs/spatial/manifests/spatial_sparse_manifest.jsonl
CANDIDATE_DIR=/root/autodl-tmp/Isaac-GR00T/outputs/spatial/sparse_candidates/candidates
CANDIDATE_SUMMARY=/root/autodl-tmp/Isaac-GR00T/outputs/spatial/sparse_candidates/candidate_generation_summary.json
GT_H5=/root/autodl-tmp/FLARE-VLA/outputs/spatial/gt_latents/spatial_epoch50_h8_framebank_trainpipe.h5
CHECKPOINT=/root/autodl-tmp/le-wm/stablewm_data/checkpoints/spatial_lewm_full/weights_epoch_50.pt
OUTPUT_DIR=/root/autodl-tmp/le-wm/outputs/spatial/scoring

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Spatial Sparse LeWM Scoring"
echo "============================================"
echo "  Manifest   : $MANIFEST"
echo "  Candidates : $CANDIDATE_DIR"
echo "  GT H5      : $GT_H5"
echo "  Checkpoint : $CHECKPOINT"
echo "  Output     : $OUTPUT_DIR"
echo "  Windows    : 12,599"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python -u \
  /root/autodl-tmp/FLARE-VLA/scripts/spatial/05_score_spatial.py \
  --manifest "$MANIFEST" \
  --candidate-dir "$CANDIDATE_DIR" \
  --candidate-summary "$CANDIDATE_SUMMARY" \
  --gt-h5 "$GT_H5" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 32 \
  2>&1 | tee "$OUTPUT_DIR"/scoring.log

echo ""
echo "Done. Scoring outputs at: $OUTPUT_DIR/"
