#!/usr/bin/env bash
# Build and split the sparse LIBERO-Plus candidate-window manifest.
#
# The LIBERO-Plus-specific builder creates the full sparse manifest with
# official BDDL-derived suite labels. This entry point then creates the four
# per-suite manifests used by suite-specific teacher checkpoints.
#
# Produces:
#   outputs/libero_plus/manifests/libero_plus_sparse_manifest.jsonl
#   outputs/libero_plus/manifests/libero_plus_sparse_manifest_summary.json
#   outputs/libero_plus/manifests/libero_plus_sparse_manifest_episode_summary.csv
#   outputs/libero_plus/manifests/libero_plus_sparse_manifest_task_summary.csv
#   outputs/libero_plus/manifests/per_suite/manifest_libero_spatial.jsonl
#   outputs/libero_plus/manifests/per_suite/manifest_libero_object.jsonl
#   outputs/libero_plus/manifests/per_suite/manifest_libero_goal.jsonl
#   outputs/libero_plus/manifests/per_suite/manifest_libero_10.jsonl
#   outputs/libero_plus/manifests/per_suite/suite_split_summary.json
#
# Usage:
#   bash FLARE-VLA/commands/libero_plus/04_build_sparse_manifest.sh

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly FLARE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly ISAAC_ROOT="$(cd -- "${FLARE_ROOT}/.." && pwd)"
readonly PYTHON_BIN="/root/autodl-tmp/envs/LeWorldModel/bin/python3"
readonly DATASET_ROOT="${ISAAC_ROOT}/examples/LIBERO/libero_plus_lerobot"
readonly GT_H5="${FLARE_ROOT}/outputs/libero_plus/gt_latents/libero_plus_epoch50_h8_framebank_trainpipe.h5"
readonly MANIFEST_DIR="${FLARE_ROOT}/outputs/libero_plus/manifests_epoch50_fixed"
readonly MANIFEST="${MANIFEST_DIR}/libero_plus_sparse_manifest.jsonl"
readonly PER_SUITE_DIR="${MANIFEST_DIR}/per_suite"
readonly BDDL_ROOT="${ISAAC_ROOT}/external_dependencies/LIBERO-plus/libero/libero/bddl_files"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python interpreter is missing or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: LIBERO-Plus dataset directory is missing: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${GT_H5}" ]]; then
  echo "ERROR: GT latent H5 is missing: ${GT_H5}" >&2
  echo "The epoch-50 GT latent should already exist from the previous run." >&2
  exit 1
fi
if [[ ! -d "${BDDL_ROOT}" ]]; then
  echo "ERROR: official BDDL directory is missing: ${BDDL_ROOT}" >&2
  exit 1
fi

echo "Building the full sparse LIBERO-Plus manifest..."
"${PYTHON_BIN}" -u \
  "${FLARE_ROOT}/scripts/libero_plus/03_build_sparse_manifest.py" \
  --dataset-root "${DATASET_ROOT}" \
  --gt-h5 "${GT_H5}" \
  --output-dir "${MANIFEST_DIR}" \
  --bddl-root "${BDDL_ROOT}" \
  --history-size 3 \
  --horizon 8 \
  --stride 8 \
  --candidate-k 4 \
  --base-seed 2026 \
  --gripper-radius 2 \
  --action-change-topk 3 \
  --action-change-radius 1 \
  --max-extra-windows-per-episode 20

echo
echo "Splitting the manifest using official BDDL suite membership..."
"${PYTHON_BIN}" -u \
  "${FLARE_ROOT}/scripts/libero_plus/03_split_manifest_by_suite.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${PER_SUITE_DIR}" \
  --tasks-jsonl "${DATASET_ROOT}/meta/tasks.jsonl" \
  --bddl-root "${BDDL_ROOT}"

printf '\nDone.\n  Full manifest: %s\n  Per-suite manifests: %s\n' \
  "${MANIFEST}" \
  "${PER_SUITE_DIR}"
