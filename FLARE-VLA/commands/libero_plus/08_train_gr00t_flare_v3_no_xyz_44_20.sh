#!/usr/bin/env bash
# Launch actual GR00T training (v3_no_xyz variant, fixed 44+20 batch, lr=1e-4).
#
# Compared to 08_train_gr00t_flare_v3_no_xyz_preflight.sh:
#   - LEARNING_RATE: 1e-5 -> 1e-4
#   - Removes --validate_config_only (launches real training)
#   - Runs in a detached tmux session "flare_v3_no_xyz"
#   - Stdout/stderr tee'd to $OUTPUT_DIR/train.log
#
# Inputs:
#   DISTILLATION_DATASET_DIR = /root/autodl-tmp/FLARE-VLA/outputs/libero_plus/distillation_dataset_v3_no_xyz
#   teacher_samples = 284320 (high:11536, medium:272784)
#
# Expected runtime: 14-42 hours single GPU (lr=1e-4 is 10x larger than baseline).
#
# Usage:
#   bash FLARE-VLA/commands/libero_plus/08_train_gr00t_flare_v3_no_xyz_44_20.sh
#   # then:
#   tmux attach -t flare_v3_no_xyz   # monitor
#   tmux kill-session -t flare_v3_no_xyz   # stop

set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot}"
DISTILLATION_DATASET_DIR="${DISTILLATION_DATASET_DIR:-/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/distillation_dataset_v3_no_xyz}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare_v3_no_xyz_44_20}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-6}"
export SHARD_SIZE="${SHARD_SIZE:-1024}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
export TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO:-0.3125}"
export USE_WANDB="${USE_WANDB:-0}"
export EXTRA_AUGMENTATION_CONFIG='{"gaussian_blur_params":{"blur_limit":[3,7],"p":0.5},"motion_blur_params":{"blur_limit":[3,7],"p":0.3}}'

# Fixed per-micro-batch GT/teacher counts (must sum to global_batch_size).
export FIXED_GT_SAMPLES_PER_BATCH="${FIXED_GT_SAMPLES_PER_BATCH:-44}"
export FIXED_TEACHER_SAMPLES_PER_BATCH="${FIXED_TEACHER_SAMPLES_PER_BATCH:-20}"

export FLARE_REQUIRE_MIXED_DATA=1
export FLARE_EXPECTED_TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO}"

cd "${GR00T_ROOT}"
export PATH="${GR00T_ROOT}/.venv/bin:${PATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GR00T_BACKBONE_MODEL_NAME="/root/autodl-tmp/Isaac-GR00T/checkpoints/Cosmos-Reason2-2B"

mkdir -p "${OUTPUT_DIR}"

# Step 1: sidecar sanity (PASS print is informational only).
python scripts/flare/check_distillation_sidecars.py \
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}" \
  --expected-gt 0

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --embodiment-tag libero_sim
  --output-dir "${OUTPUT_DIR}"
  --experiment-name gr00t_libero_plus_flare_v3_no_xyz_44_20
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}"
  --keep-full-gt-dataset
  --teacher-mix-ratio "${TEACHER_MIX_RATIO}"
  --allow-padding
  --state-dropout-prob 0.2
  --save-only-model
)

# Step 2: launch actual training in a detached tmux session.
# Write a wrapper script that exports env vars then exec finetune.sh. tmux
# can then `bash wrapper.sh` as a single argument and the wrapper inherits
# the launcher's env via `export -p`.
SESSION_NAME="${TMUX_SESSION_NAME:-flare_v3_no_xyz}"
: > "${OUTPUT_DIR}/train.log"
WRAPPER="${OUTPUT_DIR}/run_training.sh"
cat > "${WRAPPER}" <<WRAPPER_EOF
#!/usr/bin/env bash
# Wrapper written by 08_train_gr00t_flare_v3_no_xyz_44_20.sh.
set -euo pipefail
export PATH="${GR00T_ROOT}/.venv/bin:\${PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GR00T_BACKBONE_MODEL_NAME="${GR00T_BACKBONE_MODEL_NAME}"
export FLARE_REQUIRE_MIXED_DATA=1
export FLARE_EXPECTED_TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
export TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO}"
export NUM_GPUS="${NUM_GPUS}"
export MAX_STEPS="${MAX_STEPS}"
export SAVE_STEPS="${SAVE_STEPS}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
export LEARNING_RATE="${LEARNING_RATE}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}"
export SHARD_SIZE="${SHARD_SIZE}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE}"
export USE_WANDB="${USE_WANDB}"
export EXTRA_AUGMENTATION_CONFIG='${EXTRA_AUGMENTATION_CONFIG}'
cd "${GR00T_ROOT}"
exec bash examples/finetune.sh ${COMMON_ARGS[@]} \
    --fixed-gt-samples-per-batch ${FIXED_GT_SAMPLES_PER_BATCH} \
    --fixed-teacher-samples-per-batch ${FIXED_TEACHER_SAMPLES_PER_BATCH} \
    -- \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    > ${OUTPUT_DIR}/train.log 2>&1
WRAPPER_EOF
chmod +x "${WRAPPER}"

tmux new-session -d -s "${SESSION_NAME}" "${WRAPPER}"

echo "=============================================="
echo "Detached tmux session: ${SESSION_NAME}"
echo "Training log:        ${OUTPUT_DIR}/train.log"
echo "Output dir:          ${OUTPUT_DIR}"
echo ""
echo "Monitor:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "Stop:"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo "=============================================="