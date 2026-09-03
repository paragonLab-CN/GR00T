#!/usr/bin/env bash
# Run diagnostic comparisons on sparse scoring results.
# Checks A-D validate LeWM teacher selection signal quality.
#
# Usage:
#   bash FLARE-VLA/commands/spatial/07_diagnostic_comparisons.sh

set -euo pipefail

SCORES_CSV=/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_scores.csv
TEACHER_RECORDS=/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_teacher_records.pt
STEPWISE_NPZ=/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_stepwise_metrics.npz
VALIDATION_SUMMARY=/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_summary.json
CANDIDATE_DIR=/root/autodl-tmp/Isaac-GR00T/outputs/spatial/sparse_candidates/candidates
GT_H5=/root/autodl-tmp/FLARE-VLA/outputs/spatial/gt_latents/spatial_epoch50_h8_framebank_trainpipe.h5
CHECKPOINT=/root/autodl-tmp/le-wm/stablewm_data/checkpoints/lewm/weights_epoch_50.pt
MANIFEST=/root/autodl-tmp/FLARE-VLA/outputs/spatial/manifests/spatial_sparse_manifest.jsonl
OUTPUT_DIR=/root/autodl-tmp/FLARE-VLA/outputs/spatial/10_teacher_scoring

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Sparse Teacher Diagnostic Comparisons"
echo "============================================"
echo "  Scores    : $SCORES_CSV"
echo "  Output    : $OUTPUT_DIR"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python -u \
  /root/autodl-tmp/FLARE-VLA/scripts/spatial/07_diagnostic_comparisons.py \
  --scores-csv "$SCORES_CSV" \
  --teacher-records "$TEACHER_RECORDS" \
  --stepwise-npz "$STEPWISE_NPZ" \
  --validation-summary "$VALIDATION_SUMMARY" \
  --candidate-dir "$CANDIDATE_DIR" \
  --gt-h5 "$GT_H5" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR"/diagnostic_comparisons.log

echo ""
echo "Done. Report at: $OUTPUT_DIR/diagnostic_comparisons.json"
