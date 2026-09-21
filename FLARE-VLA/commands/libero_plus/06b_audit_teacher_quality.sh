#!/usr/bin/env bash
# Audit the complete LIBERO-Plus scoring result and select teachers.
#
# Usage: bash FLARE-VLA/commands/libero_plus/06b_audit_teacher_quality.sh

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly FLARE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly ISAAC_ROOT="$(cd -- "${FLARE_ROOT}/.." && pwd)"
readonly WORKSPACE_ROOT="$(cd -- "${ISAAC_ROOT}/.." && pwd)"

readonly AUDIT_PYTHON="${AUDIT_PYTHON:-${WORKSPACE_ROOT}/envs/LeWorldModel/bin/python}"
readonly AUDITOR="${FLARE_ROOT}/scripts/libero_plus/06_audit_teacher_quality.py"
readonly MANIFEST="${MANIFEST:-${FLARE_ROOT}/outputs/libero_plus/manifests/libero_plus_sparse_manifest.jsonl}"
readonly SCORING_DIR="${SCORING_DIR:-${FLARE_ROOT}/outputs/libero_plus/teacher_scoring}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-${FLARE_ROOT}/outputs/libero_plus/teacher_scoring_v2}"
readonly SCORING_RUN_SUMMARY="${SCORING_DIR}/scoring_run_summary.json"
readonly SCORES_CSV="${SCORING_DIR}/validation_scores.csv"
readonly TEACHER_RECORDS="${SCORING_DIR}/validation_teacher_records.pt"
readonly STEPWISE_NPZ="${SCORING_DIR}/validation_stepwise_metrics.npz"
readonly VALIDATION_SUMMARY="${SCORING_DIR}/validation_summary.json"
readonly CANDIDATE_DIR="${CANDIDATE_DIR:-${ISAAC_ROOT}/outputs/libero_plus/sparse_candidates/candidates}"
readonly INCLUDE_CANDIDATE_ACTIONS="${INCLUDE_CANDIDATE_ACTIONS:-0}"

if [[ ! "${INCLUDE_CANDIDATE_ACTIONS}" =~ ^[01]$ ]]; then
  echo "ERROR: INCLUDE_CANDIDATE_ACTIONS must be 0 or 1" >&2
  exit 1
fi

# A resumed scorer can coexist with stale aggregate files from an older run.
# Never audit those files while score parts are still being rewritten.
if pgrep -f '[0]5_score_libero_plus.py' >/dev/null; then
  echo "ERROR: 05_score_libero_plus.py is still running." >&2
  echo "Wait for 06_score_candidates.sh to finish before launching this audit." >&2
  exit 1
fi

required_files=(
  "${AUDIT_PYTHON}"
  "${AUDITOR}"
  "${MANIFEST}"
  "${SCORING_RUN_SUMMARY}"
  "${SCORES_CSV}"
  "${TEACHER_RECORDS}"
  "${STEPWISE_NPZ}"
  "${VALIDATION_SUMMARY}"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file is missing: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================"
echo "  LIBERO-Plus Teacher Quality Audit"
echo "============================================"
echo "  Manifest  : ${MANIFEST}"
echo "  Score run : ${SCORING_RUN_SUMMARY}"
echo "  Scores    : ${SCORES_CSV}"
echo "  Output    : ${OUTPUT_DIR}"
echo "============================================"

audit_args=(
  --manifest "${MANIFEST}"
  --scoring-run-summary "${SCORING_RUN_SUMMARY}"
  --scores-csv "${SCORES_CSV}"
  --teacher-records "${TEACHER_RECORDS}"
  --stepwise-npz "${STEPWISE_NPZ}"
  --validation-summary "${VALIDATION_SUMMARY}"
  --output-dir "${OUTPUT_DIR}"
  --primary-metric mse
  --agreement-policy require
  --gripper-min-match-rate 0.3
)
if [[ "${INCLUDE_CANDIDATE_ACTIONS}" == "1" ]]; then
  audit_args+=(--candidate-dir "${CANDIDATE_DIR}")
else
  audit_args+=(--no-candidate-actions)
fi

"${AUDIT_PYTHON}" -u "${AUDITOR}" "${audit_args[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/teacher_audit.log"

printf '\nDone. Audit outputs at: %s\n' "${OUTPUT_DIR}"
