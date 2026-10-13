#!/usr/bin/env bash
# Wrapper for the round-3 refilter (06c_refilter_teacher_quality_v3.py).
#
# Reads the existing audit inputs from teacher_scoring_epoch50_fixed/ and
# Pol_base's teacher_quality_summary.json, runs the budget-constrained top-K
# refilter, and writes teacher_scoring_v3_budget/.
#
# Usage: bash FLARE-VLA/commands/libero_plus/06c_refilter_teacher_quality_v3.sh

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly FLARE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly ISAAC_ROOT="$(cd -- "${FLARE_ROOT}/.." && pwd)"
readonly WORKSPACE_ROOT="$(cd -- "${ISAAC_ROOT}/.." && pwd)"

readonly PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_ROOT}/envs/LeWorldModel/bin/python}"
readonly REFITTER="${FLARE_ROOT}/scripts/libero_plus/06c_refilter_teacher_quality_v3.py"
readonly SCORING_DIR="${SCORING_DIR:-${FLARE_ROOT}/outputs/libero_plus/teacher_scoring_epoch50_fixed}"
readonly POL_BASE_SUMMARY="${POL_BASE_SUMMARY:-${FLARE_ROOT}/outputs/libero_plus/teacher_scoring_epoch50_fixed_v2/teacher_quality_summary.json}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-${FLARE_ROOT}/outputs/libero_plus/teacher_scoring_v3_budget}"

if pgrep -f '[0]5_score_libero_plus.py' >/dev/null; then
  echo "ERROR: 05_score_libero_plus.py is still running." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "=============================================="
echo "  Round-3 Refilter (Budget Top-K)"
echo "=============================================="
echo "  Scoring dir        : ${SCORING_DIR}"
echo "  Pol_base summary   : ${POL_BASE_SUMMARY}"
echo "  Output dir         : ${OUTPUT_DIR}"
echo "=============================================="

"${PYTHON_BIN}" -u "${REFITTER}" \
  --scoring-dir "${SCORING_DIR}" \
  --pol-base-summary "${POL_BASE_SUMMARY}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/refilter.log"

printf '\nDone. Outputs at: %s\n' "${OUTPUT_DIR}"
