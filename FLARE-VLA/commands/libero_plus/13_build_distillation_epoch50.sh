#!/usr/bin/env bash
# Build the epoch-50 teacher distillation dataset index. Reads the epoch-50
# audit dir and writes to a NEW distillation dir — does NOT touch epoch-9 files.
#
# Outputs (FLARE-VLA/outputs/libero_plus/distillation_dataset_epoch50/):
#   distillation_index.jsonl / teacher_index.jsonl
#   teacher_records_enriched.pt
#   distillation_summary.json
#
# Usage: bash FLARE-VLA/commands/libero_plus/13_build_distillation_epoch50.sh

set -euo pipefail

REPO_ROOT=/root/autodl-tmp/Isaac-GR00T/FLARE-VLA

MANIFEST="${REPO_ROOT}/outputs/libero_plus/manifests_epoch50_fixed/libero_plus_sparse_manifest.jsonl"
MANIFEST_SUMMARY="${REPO_ROOT}/outputs/libero_plus/manifests_epoch50_fixed/libero_plus_sparse_manifest_summary.json"
TEACHER_SELECTED="${REPO_ROOT}/outputs/libero_plus/teacher_scoring_epoch50_fixed_v2/teacher_quality_selected.jsonl"
TEACHER_RECORDS="${REPO_ROOT}/outputs/libero_plus/teacher_scoring_epoch50_fixed_v2/teacher_quality_direct_records.pt"
TEACHER_SUMMARY="${REPO_ROOT}/outputs/libero_plus/teacher_scoring_epoch50_fixed_v2/teacher_quality_summary.json"
CHECKPOINT_CONFIG="/root/autodl-tmp/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_spatial/config.json"
PROCESSOR_CONFIG="/root/autodl-tmp/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_spatial/processor_config.json"
OUTPUT_DIR="${REPO_ROOT}/outputs/libero_plus/distillation_dataset_epoch50_fixed"

EXPECTED_GT_COUNT="${EXPECTED_GT_COUNT:-487880}"

echo "=============================================="
echo "Building epoch-50 distillation dataset index (fixed manifest)"
echo "=============================================="
echo "Teacher records: ${TEACHER_RECORDS}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Expected GT:     ${EXPECTED_GT_COUNT}"
echo "=============================================="

cd /root/autodl-tmp/Isaac-GR00T

/root/autodl-tmp/envs/LeWorldModel/bin/python \
    FLARE-VLA/scripts/spatial/08_build_spatial_distillation_dataset.py \
    --manifest "${MANIFEST}" \
    --manifest-summary "${MANIFEST_SUMMARY}" \
    --teacher-selected "${TEACHER_SELECTED}" \
    --teacher-records "${TEACHER_RECORDS}" \
    --teacher-summary "${TEACHER_SUMMARY}" \
    --checkpoint-config "${CHECKPOINT_CONFIG}" \
    --processor-config "${PROCESSOR_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --embodiment-key libero_sim \
    --index-mode teacher_only \
    --quality-levels high medium \
    --expected-gt-count "${EXPECTED_GT_COUNT}"

echo ""
echo "Done. Outputs in: ${OUTPUT_DIR}"
