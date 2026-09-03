#!/usr/bin/env bash
set -euo pipefail

# Build the exact 500-GT control group by filtering the GT rows from the
# 858-record FLARE mixed index. No observation window is resampled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
SOURCE_DISTILLATION_DIR="${SOURCE_DISTILLATION_DIR:-${GR00T_ROOT}/outputs/flare_distillation_filtered_direct}"
GT_ONLY_DATASET_DIR="${GT_ONLY_DATASET_DIR:-${GR00T_ROOT}/outputs/flare_distillation_gt_only_500}"

python "${FLARE_ROOT}/scripts/05_distillation/10_build_gt_only_dataset.py" \
  --source-dir "${SOURCE_DISTILLATION_DIR}" \
  --output-dir "${GT_ONLY_DATASET_DIR}" \
  --expected-gt-count 500

cd "${GR00T_ROOT}"
python scripts/flare/check_distillation_sidecars.py \
  --distillation-dataset-dir "${GT_ONLY_DATASET_DIR}" \
  --expected-total 500 \
  --expected-gt 500 \
  --expected-teacher 0
