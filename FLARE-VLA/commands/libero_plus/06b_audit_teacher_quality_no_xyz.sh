#!/usr/bin/env bash
# Variant of 06b_audit_teacher_quality.sh that disables flag_xyz_distance_ok
# to retain teacher trajectories beyond the GT xyz distance quantile boundary.
# Otherwise identical to 06b.
#
# Produces (FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_no_xyz/):
#   teacher_quality_audit.parquet
#   teacher_quality_selected.jsonl
#   teacher_quality_summary.json
#   teacher_quality_direct_records.pt
#   teacher_task_summary.csv
#
# Usage: bash FLARE-VLA/commands/libero_plus/06b_audit_teacher_quality_no_xyz.sh

set -euo pipefail

SCORING_DIR=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/teacher_scoring
OUTPUT_DIR=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_no_xyz

SCORES_CSV="${SCORING_DIR}/validation_scores.csv"
TEACHER_RECORDS="${SCORING_DIR}/validation_teacher_records.pt"
STEPWISE_NPZ="${SCORING_DIR}/validation_stepwise_metrics.npz"
VALIDATION_SUMMARY="${SCORING_DIR}/validation_summary.json"
CANDIDATE_DIR=/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/sparse_candidates/candidates

mkdir -p "$OUTPUT_DIR"

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data

echo "============================================"
echo "  Libero-Plus Teacher Quality Audit (no xyz distance filter)"
echo "============================================"
echo "  Scores    : $SCORES_CSV"
echo "  Output    : $OUTPUT_DIR"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python -u \
  /root/autodl-tmp/FLARE-VLA/scripts/libero_plus/06_audit_teacher_quality.py \
  --scores-csv "$SCORES_CSV" \
  --teacher-records "$TEACHER_RECORDS" \
  --stepwise-npz "$STEPWISE_NPZ" \
  --validation-summary "$VALIDATION_SUMMARY" \
  --candidate-dir "$CANDIDATE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --primary-metric mse \
  --agreement-policy require \
  --gripper-min-match-rate 0.3 \
  --no-xyz-distance-filter \
  --no-candidate-actions \
  2>&1 | tee "$OUTPUT_DIR"/teacher_audit.log

echo ""
echo "Done. Audit outputs at: $OUTPUT_DIR/"