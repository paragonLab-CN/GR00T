#!/usr/bin/env bash
# Build validated teacher-only LIBERO-Plus FLARE sidecars.
#
# The complete GT LeRobot dataset remains the primary training source; these
# sidecars contain only selected teacher windows for weighted mixing.
#
# Usage: bash FLARE-VLA/commands/libero_plus/07_build_distillation_dataset.sh

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly FLARE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly ISAAC_ROOT="$(cd -- "${FLARE_ROOT}/.." && pwd)"
readonly WORKSPACE_ROOT="$(cd -- "${ISAAC_ROOT}/.." && pwd)"

readonly BUILD_PYTHON="${BUILD_PYTHON:-${WORKSPACE_ROOT}/envs/LeWorldModel/bin/python}"
readonly CHECK_PYTHON="${CHECK_PYTHON:-${ISAAC_ROOT}/.venv/bin/python}"
readonly BUILDER="${FLARE_ROOT}/scripts/libero_plus/08_build_distillation_dataset.py"
readonly MANIFEST="${MANIFEST:-${FLARE_ROOT}/outputs/libero_plus/manifests/libero_plus_sparse_manifest.jsonl}"
readonly MANIFEST_SUMMARY="${MANIFEST_SUMMARY:-${FLARE_ROOT}/outputs/libero_plus/manifests/libero_plus_sparse_manifest_summary.json}"
readonly AUDIT_DIR="${AUDIT_DIR:-${FLARE_ROOT}/outputs/libero_plus/teacher_scoring_v2}"
readonly TEACHER_SELECTED="${TEACHER_SELECTED:-${AUDIT_DIR}/teacher_quality_selected.jsonl}"
readonly TEACHER_RECORDS="${TEACHER_RECORDS:-${AUDIT_DIR}/teacher_quality_direct_records.pt}"
readonly TEACHER_SUMMARY="${TEACHER_SUMMARY:-${AUDIT_DIR}/teacher_quality_summary.json}"
readonly CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ISAAC_ROOT}/checkpoints/GR00T-N1.7-LIBERO}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-${FLARE_ROOT}/outputs/libero_plus/distillation_dataset_v2}"
readonly EXPECTED_GT_COUNT="${EXPECTED_GT_COUNT:-487880}"
readonly EXPECTED_TEACHER_COUNT="${EXPECTED_TEACHER_COUNT:-0}"

for integer_var in EXPECTED_GT_COUNT EXPECTED_TEACHER_COUNT; do
  value="${!integer_var}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${integer_var} must be a non-negative integer" >&2
    exit 1
  fi
done
if [[ "${EXPECTED_GT_COUNT}" -lt 1 ]]; then
  echo "ERROR: EXPECTED_GT_COUNT must be positive" >&2
  exit 1
fi

if pgrep -f '[0]5_score_libero_plus.py|[0]6_audit_teacher_quality.py' >/dev/null; then
  echo "ERROR: scoring or teacher audit is still running." >&2
  echo "Wait for 06 and 06b to finish before building distillation sidecars." >&2
  exit 1
fi

required_files=(
  "${BUILD_PYTHON}"
  "${CHECK_PYTHON}"
  "${BUILDER}"
  "${MANIFEST}"
  "${MANIFEST_SUMMARY}"
  "${TEACHER_SELECTED}"
  "${TEACHER_RECORDS}"
  "${TEACHER_SUMMARY}"
)
for suite in libero_spatial libero_object libero_goal libero_10; do
  required_files+=(
    "${CHECKPOINT_ROOT}/${suite}/config.json"
    "${CHECKPOINT_ROOT}/${suite}/processor_config.json"
  )
done
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file is missing: ${required_file}" >&2
    exit 1
  fi
done

echo "=============================================="
echo "  Build LIBERO-Plus distillation sidecars"
echo "=============================================="
echo "  Manifest       : ${MANIFEST}"
echo "  Audit directory: ${AUDIT_DIR}"
echo "  Checkpoints    : ${CHECKPOINT_ROOT}"
echo "  Output         : ${OUTPUT_DIR}"
echo "  Expected GT    : ${EXPECTED_GT_COUNT}"
echo "=============================================="

build_args=(
  --manifest "${MANIFEST}"
  --manifest-summary "${MANIFEST_SUMMARY}"
  --teacher-selected "${TEACHER_SELECTED}"
  --teacher-records "${TEACHER_RECORDS}"
  --teacher-summary "${TEACHER_SUMMARY}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --embodiment-key libero_sim
  --index-mode teacher_only
  --quality-levels high medium
  --expected-gt-count "${EXPECTED_GT_COUNT}"
)
if [[ "${EXPECTED_TEACHER_COUNT}" -gt 0 ]]; then
  build_args+=(--expected-teacher-count "${EXPECTED_TEACHER_COUNT}")
fi

"${BUILD_PYTHON}" -u "${BUILDER}" "${build_args[@]}"

cd "${ISAAC_ROOT}"
"${CHECK_PYTHON}" scripts/flare/check_distillation_sidecars.py \
  --distillation-dataset-dir "${OUTPUT_DIR}" \
  --expected-gt 0

printf '\nDone. Outputs in: %s\n' "${OUTPUT_DIR}"
