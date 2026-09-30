#!/usr/bin/env bash
# Train GR00T-N1.7-3B on LIBERO-Plus GT + epoch-50 FLARE teacher (fixed manifest).
# Identical recipe to 08c_train_gr00t_flare_44_20.sh (44:20 GT:teacher ratio),
# except it reads the epoch-50 fixed distillation index and writes to a NEW output dir.
#
# Usage: bash FLARE-VLA/commands/libero_plus/14_train_gr00t_flare_epoch50.sh

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"

# Override paths to use epoch-50 fixed artifacts
export EXPERIMENT_NAME="gr00t_libero_plus_flare_epoch50_fixed_44_20"
export DISTILLATION_DATASET_DIR="/root/autodl-tmp/Isaac-GR00T/FLARE-VLA/outputs/libero_plus/distillation_dataset_epoch50_fixed"
export MANIFEST="/root/autodl-tmp/Isaac-GR00T/FLARE-VLA/outputs/libero_plus/manifests_epoch50_fixed/libero_plus_sparse_manifest.jsonl"
export MANIFEST_SUMMARY="/root/autodl-tmp/Isaac-GR00T/FLARE-VLA/outputs/libero_plus/manifests_epoch50_fixed/libero_plus_sparse_manifest_summary.json"
export OUTPUT_DIR="/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/gr00t_libero_plus_flare_epoch50_fixed_44_20"

# Fixed 44:20 GT:teacher ratio (same as 08c)
export FIXED_GT_SAMPLES_PER_BATCH=44
export FIXED_TEACHER_SAMPLES_PER_BATCH=20
export TEACHER_MIX_RATIO=0.3125

# Training steps (user specified 10000, not 20000)
export MAX_STEPS=10000

# Delegate to the shared training script
exec bash "${SCRIPT_DIR}/08_train_gr00t_flare.sh"
