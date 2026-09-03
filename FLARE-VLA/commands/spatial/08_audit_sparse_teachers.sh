#!/usr/bin/env bash
# Audit sparse candidate teachers using FLARE-Filtered Direct Augmentation.
# Reuses 06_audit_teacher_quality.py with sparse scoring outputs.
#
# Produces (FLARE-VLA/outputs/spatial/10_teacher_scoring/):
#   teacher_quality_audit.parquet
#   teacher_quality_selected.jsonl
#   teacher_quality_summary.json
#   teacher_task_summary.csv
#
# Spatial-specific. Does NOT read or modify any LIBERO-90 outputs.
#
# Usage:
#   bash FLARE-VLA/commands/spatial/08_audit_sparse_teachers.sh

set -euo pipefail

SCORES_CSV=/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_scores.csv
TEACHER_RECORDS=/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_teacher_records.pt
STEPWISE_NPZ=/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_stepwise_metrics.npz
VALIDATION_SUMMARY=/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_summary.json
CANDIDATE_DIR=/root/autodl-tmp/Isaac-GR00T/outputs/spatial/sparse_candidates/candidates
OUTPUT_DIR=/root/autodl-tmp/FLARE-VLA/outputs/spatial/10_teacher_scoring

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Sparse Teacher Quality Audit"
echo "============================================"
echo "  Scores    : $SCORES_CSV"
echo "  Output    : $OUTPUT_DIR"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python -u \
  /root/autodl-tmp/FLARE-VLA/scripts/04_scoring/06_audit_teacher_quality.py \
  --scores-csv "$SCORES_CSV" \
  --teacher-records "$TEACHER_RECORDS" \
  --stepwise-npz "$STEPWISE_NPZ" \
  --validation-summary "$VALIDATION_SUMMARY" \
  --candidate-dir "$CANDIDATE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --primary-metric mse \
  --agreement-policy require \
  2>&1 | tee "$OUTPUT_DIR"/teacher_audit.log

echo ""
echo "Done. Audit outputs at: $OUTPUT_DIR/"
