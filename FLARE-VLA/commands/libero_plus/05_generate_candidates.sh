#!/usr/bin/env bash
# Generate K=4 GR00T candidate action chunks for LIBERO-Plus.
#
# Each per-suite manifest is paired with its matching GR00T-N1.7-LIBERO
# checkpoint. Existing valid candidate files are reused. After all four suites
# complete, candidates are merged with full-manifest provenance for LeWM
# scoring.
#
# Inputs:
#   FLARE-VLA/outputs/libero_plus/manifests_epoch50_fixed/libero_plus_sparse_manifest.jsonl
#   FLARE-VLA/outputs/libero_plus/manifests_epoch50_fixed/per_suite/manifest_<suite>.jsonl
#
# Outputs:
#   outputs/libero_plus/sparse_candidates_epoch50_fixed/<suite>/candidates/
#   outputs/libero_plus/sparse_candidates_epoch50_fixed/candidates/
#   outputs/libero_plus/sparse_candidates_epoch50_fixed/candidate_generation_summary.json
#
# Current manifest scale: 487,880 windows x 4 candidates.
#
# Usage:
#   bash FLARE-VLA/commands/libero_plus/05_generate_candidates.sh

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly FLARE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly ISAAC_ROOT="$(cd -- "${FLARE_ROOT}/.." && pwd)"
readonly DATASET="${ISAAC_ROOT}/examples/LIBERO/libero_plus_lerobot"
readonly CHECKPOINT_ROOT="${ISAAC_ROOT}/checkpoints/GR00T-N1.7-LIBERO"
readonly BACKBONE_MODEL="${ISAAC_ROOT}/checkpoints/Cosmos-Reason2-2B"
readonly MANIFEST_DIR="${FLARE_ROOT}/outputs/libero_plus/manifests_epoch50_fixed/per_suite"
readonly FULL_MANIFEST="${FLARE_ROOT}/outputs/libero_plus/manifests_epoch50_fixed/libero_plus_sparse_manifest.jsonl"
readonly OUTPUT_ROOT="${ISAAC_ROOT}/outputs/libero_plus/sparse_candidates_epoch50_fixed"
readonly GENERATOR="${ISAAC_ROOT}/scripts/flare/04_generate_validation_candidates.py"
readonly MERGER="${FLARE_ROOT}/scripts/libero_plus/04_merge_candidate_summaries.py"
readonly MANIFEST_FORMAT="libero_plus_per_suite_manifest_v2"
readonly MERGE_WORKERS="${MERGE_WORKERS:-12}"
readonly -a SUITES=(
  libero_spatial
  libero_object
  libero_goal
  libero_10
)

for required_file in "${FULL_MANIFEST}" "${GENERATOR}" "${MERGER}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file is missing: ${required_file}" >&2
    exit 1
  fi
done
for backbone_file in config.json model.safetensors preprocessor_config.json tokenizer_config.json; do
  if [[ ! -f "${BACKBONE_MODEL}/${backbone_file}" ]]; then
    echo "ERROR: local Cosmos backbone is incomplete: ${BACKBONE_MODEL}/${backbone_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${DATASET}" ]]; then
  echo "ERROR: LIBERO-Plus dataset is missing: ${DATASET}" >&2
  exit 1
fi
if [[ ! "${MERGE_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MERGE_WORKERS must be a positive integer" >&2
  exit 1
fi

for suite in "${SUITES[@]}"; do
  manifest="${MANIFEST_DIR}/manifest_${suite}.jsonl"
  model="${CHECKPOINT_ROOT}/${suite}"
  if [[ ! -f "${manifest}" ]]; then
    echo "ERROR: per-suite manifest is missing: ${manifest}" >&2
    exit 1
  fi
  if [[ ! -f "${model}/config.json" ]]; then
    echo "ERROR: checkpoint config is missing: ${model}/config.json" >&2
    exit 1
  fi
  if [[ ! -f "${model}/model.safetensors.index.json" ]]; then
    echo "ERROR: checkpoint weights index is missing: ${model}" >&2
    exit 1
  fi
done

cd "${ISAAC_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export UV_LINK_MODE=copy

# The Cosmos backbone is gated on Hugging Face. All required files already
# live in this checkout, so keep both the model and processor strictly local.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export GR00T_BACKBONE_MODEL_NAME="${BACKBONE_MODEL}"

for suite in "${SUITES[@]}"; do
  manifest="${MANIFEST_DIR}/manifest_${suite}.jsonl"
  model="${CHECKPOINT_ROOT}/${suite}"
  output="${OUTPUT_ROOT}/${suite}"
  mkdir -p "${output}"

  echo
  echo "============================================"
  echo "  Candidate generation: ${suite}"
  echo "  Manifest: ${manifest}"
  echo "  Model   : ${model}"
  echo "  Output  : ${output}"
  echo "============================================"

  uv run --no-sync python "${GENERATOR}" \
    --manifest "${manifest}" \
    --dataset-path "${DATASET}" \
    --model-path "${model}" \
    --output-dir "${output}" \
    --expected-suite "${suite}" \
    --manifest-format "${MANIFEST_FORMAT}" \
    --embodiment-tag LIBERO_PANDA \
    --video-backend torchcodec \
    --action-horizon 8 \
    2>&1 | tee "${output}/candidate_generation.log"
done

echo
echo "============================================"
echo "  Validating and merging per-suite candidates"
echo "============================================"
uv run --no-sync python "${MERGER}" \
  --manifest "${FULL_MANIFEST}" \
  --per-suite-manifest-dir "${MANIFEST_DIR}" \
  --suite-dir "${OUTPUT_ROOT}" \
  --workers "${MERGE_WORKERS}"

printf '\nDone.\n  Merged candidates: %s\n  Summary: %s\n' \
  "${OUTPUT_ROOT}/candidates" \
  "${OUTPUT_ROOT}/candidate_generation_summary.json"
