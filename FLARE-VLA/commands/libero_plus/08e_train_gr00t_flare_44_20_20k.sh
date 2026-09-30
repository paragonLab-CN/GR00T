#!/usr/bin/env bash
# Launch an independent 20,000-step suite-corrected LIBERO-Plus GT-replay run
# with a strict 44:20 per-device batch composition. All other settings are
# inherited from 08_train_gr00t_flare.sh.

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"

export EXPERIMENT_NAME="gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20_steps_20000"
export MAX_STEPS=20000
export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=10
export FIXED_GT_SAMPLES_PER_BATCH=44
export FIXED_TEACHER_SAMPLES_PER_BATCH=20
export TEACHER_MIX_RATIO=0.3125

exec bash "${SCRIPT_DIR}/08_train_gr00t_flare.sh"
