#!/usr/bin/env bash
# Launch the suite-corrected LIBERO-Plus GT-replay comparison with a strict
# 44:20 per-device batch composition. All other settings are inherited from
# 08_train_gr00t_flare.sh so this run differs from the 54:10 baseline only in
# experiment identity and GT/teacher sampling composition.

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"

export EXPERIMENT_NAME="gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20"
export FIXED_GT_SAMPLES_PER_BATCH=44
export FIXED_TEACHER_SAMPLES_PER_BATCH=20
export TEACHER_MIX_RATIO=0.3125

exec bash "${SCRIPT_DIR}/08_train_gr00t_flare.sh"
