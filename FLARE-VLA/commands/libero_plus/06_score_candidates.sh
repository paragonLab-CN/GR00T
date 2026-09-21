#!/usr/bin/env bash
# Score the merged K=4 LIBERO-Plus GR00T candidates with LeWM.
#
# Inputs:
#   FLARE-VLA/outputs/libero_plus/manifests/libero_plus_sparse_manifest.jsonl
#   outputs/libero_plus/sparse_candidates/candidates/
#   outputs/libero_plus/sparse_candidates/candidate_generation_summary.json
#   FLARE-VLA/outputs/libero_plus/gt_latents/libero_plus_epoch9_h8_framebank_trainpipe.h5
#
# Output:
#   FLARE-VLA/outputs/libero_plus/teacher_scoring/
#
# Usage:
#   bash FLARE-VLA/commands/libero_plus/06_score_candidates.sh

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly FLARE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly ISAAC_ROOT="$(cd -- "${FLARE_ROOT}/.." && pwd)"
readonly WORKSPACE_ROOT="$(cd -- "${ISAAC_ROOT}/.." && pwd)"

readonly LEWM_ROOT="${LEWM_ROOT:-${WORKSPACE_ROOT}/le-wm}"
readonly LEWM_PYTHON="${LEWM_PYTHON:-${WORKSPACE_ROOT}/envs/LeWorldModel/bin/python}"
readonly SCORER="${FLARE_ROOT}/scripts/libero_plus/05_score_libero_plus.py"
readonly MANIFEST="${MANIFEST:-${FLARE_ROOT}/outputs/libero_plus/manifests/libero_plus_sparse_manifest.jsonl}"
readonly CANDIDATE_ROOT="${CANDIDATE_ROOT:-${ISAAC_ROOT}/outputs/libero_plus/sparse_candidates}"
readonly CANDIDATE_DIR="${CANDIDATE_DIR:-${CANDIDATE_ROOT}/candidates}"
readonly CANDIDATE_SUMMARY="${CANDIDATE_SUMMARY:-${CANDIDATE_ROOT}/candidate_generation_summary.json}"
readonly GT_H5="${GT_H5:-${FLARE_ROOT}/outputs/libero_plus/gt_latents/libero_plus_epoch9_h8_framebank_trainpipe.h5}"
readonly CHECKPOINT="${CHECKPOINT:-${LEWM_ROOT}/stablewm_data/checkpoints/libero_plus_lewm_full/weights_epoch_9.pt}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-${FLARE_ROOT}/outputs/libero_plus/teacher_scoring}"
readonly EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-ceb52c6b525ca34b5bbdc395e20f6e2b0663bb9d86995e9c00a7ae443ba1d3da}"
readonly BATCH_SIZE="${BATCH_SIZE:-32}"
readonly DEVICE="${DEVICE:-cuda}"
readonly MAX_SAMPLES="${MAX_SAMPLES:-}"

if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BATCH_SIZE must be a positive integer" >&2
  exit 1
fi
if [[ -n "${MAX_SAMPLES}" && ! "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_SAMPLES must be empty or a positive integer" >&2
  exit 1
fi
if [[ ! "${EXPECTED_CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: EXPECTED_CHECKPOINT_SHA256 must contain 64 lowercase hex characters" >&2
  exit 1
fi

required_files=(
  "${LEWM_PYTHON}"
  "${SCORER}"
  "${MANIFEST}"
  "${CANDIDATE_SUMMARY}"
  "${GT_H5}"
  "${CHECKPOINT}"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file is missing: ${required_file}" >&2
    if [[ "${required_file}" == "${CANDIDATE_SUMMARY}" ]]; then
      echo "Candidate generation/merge is not complete; let 05_generate_candidates.sh finish first." >&2
    fi
    exit 1
  fi
done
if [[ ! -d "${CANDIDATE_DIR}" ]]; then
  echo "ERROR: merged candidate directory is missing: ${CANDIDATE_DIR}" >&2
  echo "Let 05_generate_candidates.sh finish its final merge first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export LEWM_ROOT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================"
echo "  LIBERO-Plus Teacher Scoring"
echo "============================================"
echo "  Manifest  : ${MANIFEST}"
echo "  Candidates: ${CANDIDATE_DIR}"
echo "  GT H5     : ${GT_H5}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Device    : ${DEVICE}"
echo "============================================"

score_args=(
  --manifest "${MANIFEST}"
  --candidate-dir "${CANDIDATE_DIR}"
  --candidate-summary "${CANDIDATE_SUMMARY}"
  --gt-h5 "${GT_H5}"
  --checkpoint "${CHECKPOINT}"
  --expected-checkpoint-sha256 "${EXPECTED_CHECKPOINT_SHA256}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  score_args+=(--max-samples "${MAX_SAMPLES}")
fi

"${LEWM_PYTHON}" -u "${SCORER}" "${score_args[@]}" 2>&1 | tee "${OUTPUT_DIR}/score_candidates.log"

printf '\nDone. Outputs in: %s\n' "${OUTPUT_DIR}"
