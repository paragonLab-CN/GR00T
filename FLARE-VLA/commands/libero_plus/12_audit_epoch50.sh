#!/usr/bin/env bash
# Audit epoch-50 teacher quality. Reads the NEW epoch-50 scoring dir and writes
# to a NEW audit dir — does NOT touch the epoch-9 artifacts.
#
# Outputs (FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_v2/):
#   teacher_quality_audit.parquet / .csv
#   teacher_quality_selected.jsonl
#   teacher_quality_summary.json
#   teacher_quality_direct_records.pt / _filtered_records.pt
#   teacher_quality_task_summary.csv / .parquet
#
# Usage: bash FLARE-VLA/commands/libero_plus/12_audit_epoch50.sh

set -euo pipefail

SCORING_DIR=/root/autodl-tmp/Isaac-GR00T/FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_fixed
OUTPUT_DIR=/root/autodl-tmp/Isaac-GR00T/FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_fixed_v2
CANDIDATE_DIR=/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/sparse_candidates_epoch50_fixed/candidates

mkdir -p "$OUTPUT_DIR"

export STABLEWM_HOME=/root/autodl-tmp/le-wm/stablewm_data
export LOCAL_DATASET_DIR=/root/autodl-tmp/le-wm/stablewm_data

echo "============================================"
echo "  Libero-Plus Teacher Quality Audit (epoch 50, fixed manifest)"
echo "============================================"
echo "  Scores : $SCORING_DIR"
echo "  Output : $OUTPUT_DIR"
echo "============================================"

/root/autodl-tmp/envs/LeWorldModel/bin/python -u \
  /root/autodl-tmp/Isaac-GR00T/FLARE-VLA/scripts/libero_plus/06_audit_teacher_quality.py \
  --scores-csv "$SCORING_DIR/validation_scores.csv" \
  --teacher-records "$SCORING_DIR/validation_teacher_records.pt" \
  --stepwise-npz "$SCORING_DIR/validation_stepwise_metrics.npz" \
  --validation-summary "$SCORING_DIR/validation_summary.json" \
  --candidate-dir "$CANDIDATE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --primary-metric mse \
  --agreement-policy require \
  --gripper-min-match-rate 0.3 \
  --no-candidate-actions \
  2>&1 | tee "$OUTPUT_DIR"/teacher_audit.log

echo ""
echo "Done. Audit outputs at: $OUTPUT_DIR/"
