#!/bin/bash
# Build spatial teacher distillation dataset index.
#
# Produces:
#   distillation_index.jsonl      mixed GT + teacher index
#   teacher_index.jsonl           teacher-only index (for weighted mixing)
#   teacher_records_enriched.pt   v1 enriched teacher records
#   distillation_summary.json     provenance and counts
#
# After this, GR00T training can use:
#   --keep-full-gt-dataset --teacher-mix-ratio 0.2
#   --distillation-index-path .../teacher_index.jsonl
#   --teacher-records-path .../teacher_records_enriched.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MANIFEST="${REPO_ROOT}/outputs/spatial/manifests/spatial_sparse_manifest.jsonl"
MANIFEST_SUMMARY="${REPO_ROOT}/outputs/spatial/manifests/spatial_sparse_manifest_summary.json"
TEACHER_SELECTED="${REPO_ROOT}/outputs/spatial/10_teacher_scoring/teacher_quality_selected.jsonl"
TEACHER_RECORDS="${REPO_ROOT}/outputs/spatial/10_teacher_scoring/teacher_quality_direct_records.pt"
TEACHER_SUMMARY="${REPO_ROOT}/outputs/spatial/10_teacher_scoring/teacher_quality_summary.json"
CHECKPOINT_CONFIG="/root/autodl-tmp/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_spatial/config.json"
PROCESSOR_CONFIG="/root/autodl-tmp/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_spatial/processor_config.json"
OUTPUT_DIR="${REPO_ROOT}/outputs/spatial/11_distillation_dataset"

echo "=============================================="
echo "Building spatial distillation dataset index"
echo "=============================================="
echo "Manifest:        ${MANIFEST}"
echo "Teacher records: ${TEACHER_RECORDS}"
echo "Checkpoint:      ${CHECKPOINT_CONFIG}"
echo "Output dir:      ${OUTPUT_DIR}"
echo ""

cd "${REPO_ROOT}"

python scripts/spatial/08_build_spatial_distillation_dataset.py \
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
    --expected-gt-count 12599 \
    --seed 42

echo ""
echo "Done. Outputs in: ${OUTPUT_DIR}"
