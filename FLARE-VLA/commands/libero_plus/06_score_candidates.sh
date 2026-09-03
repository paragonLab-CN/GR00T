#!/usr/bin/env bash
# Score K=4 GR00T candidates with the libero_plus LeWM and select teachers.
#
# NOTE: Update --checkpoint to the actual weights_epoch_N.pt from
# 02_train_lewm.sh before running.
#
# Produces (in output dir):
#   validation_summary.json / scores CSV+parquet / teacher records
#
# Usage: bash FLARE-VLA/commands/libero_plus/06_score_candidates.sh

set -euo pipefail

MANIFEST=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/manifests/spatial_sparse_manifest.jsonl
CANDIDATE_DIR=/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/sparse_candidates/candidates
CANDIDATE_SUMMARY=/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/sparse_candidates/candidate_generation_summary.json
GT_H5=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/gt_latents/libero_plus_epoch9_h8_framebank_trainpipe.h5
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/le-wm/stablewm_data/checkpoints/libero_plus_lewm_full/weights_epoch_9.pt}"
OUTPUT_DIR=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/teacher_scoring

echo "============================================"
echo "  Libero-Plus Teacher Scoring"
echo "============================================"
echo "  Manifest  : $MANIFEST"
echo "  Candidates: $CANDIDATE_DIR"
echo "  GT H5     : $GT_H5"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output    : $OUTPUT_DIR"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python -u \
  /root/autodl-tmp/FLARE-VLA/scripts/libero_plus/05_score_libero_plus.py \
  --manifest "$MANIFEST" \
  --candidate-dir "$CANDIDATE_DIR" \
  --candidate-summary "$CANDIDATE_SUMMARY" \
  --gt-h5 "$GT_H5" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 32

echo ""
echo "Done. Outputs in: $OUTPUT_DIR"
