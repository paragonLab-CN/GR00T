#!/usr/bin/env bash
# Train GR00T with strict 54:10 GT replay on corrected four-suite teachers.
#
# Every per-device micro-batch contains exactly 54 full-GT samples and 10
# teacher samples. TEACHER_MIX_RATIO mirrors the fixed teacher sample fraction
# (10 / 64 = 0.15625); the fixed counts, not this ratio, control sampling.
#
# RESUME_MODE=auto resumes an existing complete checkpoint and starts fresh only
# when the run directory does not exist. New checkpoints retain optimizer,
# scheduler, and RNG state so subsequent resumes restore full training state.
#
# Usage: bash FLARE-VLA/commands/libero_plus/08_train_gr00t_flare.sh

set -euo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly FLARE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly DEFAULT_GR00T_ROOT="$(cd -- "${FLARE_ROOT}/.." && pwd)"

GR00T_ROOT="${GR00T_ROOT:-${DEFAULT_GR00T_ROOT}}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${GR00T_ROOT}/.venv/bin/python}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-3B}"
BACKBONE_PATH="${BACKBONE_PATH:-${GR00T_ROOT}/checkpoints/Cosmos-Reason2-2B}"
DATASET_PATH="${DATASET_PATH:-${GR00T_ROOT}/examples/LIBERO/libero_plus_lerobot}"
DISTILLATION_DATASET_DIR="${DISTILLATION_DATASET_DIR:-${FLARE_ROOT}/outputs/libero_plus/distillation_dataset_v2}"
MANIFEST="${MANIFEST:-${FLARE_ROOT}/outputs/libero_plus/manifests/libero_plus_sparse_manifest.jsonl}"
MANIFEST_SUMMARY="${MANIFEST_SUMMARY:-${FLARE_ROOT}/outputs/libero_plus/manifests/libero_plus_sparse_manifest_summary.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-LIBERO}"
PREFLIGHT_SCRIPT="${FLARE_ROOT}/scripts/libero_plus/09_validate_training_inputs.py"
BATCH_PREFLIGHT_SCRIPT="${GR00T_ROOT}/scripts/flare/check_fixed_gt_replay_batches.py"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gr00t_libero_plus_flare_suite_corrected_gt_replay_54_10}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/libero_plus}"
RUN_DIR="${OUTPUT_DIR}/${EXPERIMENT_NAME}"
RUN_MODE="${RUN_MODE:-train}"
RESUME_MODE="${RESUME_MODE:-auto}"
ALLOW_WEIGHTS_ONLY_RESUME="${ALLOW_WEIGHTS_ONLY_RESUME:-0}"

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
export FIXED_GT_SAMPLES_PER_BATCH="${FIXED_GT_SAMPLES_PER_BATCH:-54}"
export FIXED_TEACHER_SAMPLES_PER_BATCH="${FIXED_TEACHER_SAMPLES_PER_BATCH:-10}"
export TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO:-0.15625}"
export PREFLIGHT_BATCHES="${PREFLIGHT_BATCHES:-10}"
export USE_WANDB="${USE_WANDB:-0}"
export EXTRA_AUGMENTATION_CONFIG='{"gaussian_blur_params":{"blur_limit":[3,7],"p":0.5},"motion_blur_params":{"blur_limit":[3,7],"p":0.3}}'

if [[ "${RUN_MODE}" != "train" && "${RUN_MODE}" != "validate" ]]; then
  echo "ERROR: RUN_MODE must be train or validate, got: ${RUN_MODE}" >&2
  exit 1
fi
if [[ "${RESUME_MODE}" != "auto" && "${RESUME_MODE}" != "required" && \
      "${RESUME_MODE}" != "never" ]]; then
  echo "ERROR: RESUME_MODE must be auto, required, or never; got: ${RESUME_MODE}" >&2
  exit 1
fi
if [[ "${ALLOW_WEIGHTS_ONLY_RESUME}" != "0" && \
      "${ALLOW_WEIGHTS_ONLY_RESUME}" != "1" ]]; then
  echo "ERROR: ALLOW_WEIGHTS_ONLY_RESUME must be 0 or 1" >&2
  exit 1
fi
for integer_var in NUM_GPUS MAX_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT GLOBAL_BATCH_SIZE \
  GRADIENT_ACCUMULATION_STEPS DATALOADER_NUM_WORKERS SHARD_SIZE NUM_SHARDS_PER_EPOCH \
  FIXED_GT_SAMPLES_PER_BATCH FIXED_TEACHER_SAMPLES_PER_BATCH PREFLIGHT_BATCHES; do
  value="${!integer_var}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${integer_var} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
done
if [[ "${NUM_GPUS}" -ne 1 || "${GLOBAL_BATCH_SIZE}" -ne 64 ]]; then
  echo "ERROR: strict 54:10 replay requires NUM_GPUS=1 and GLOBAL_BATCH_SIZE=64" >&2
  exit 1
fi
if [[ "${FIXED_GT_SAMPLES_PER_BATCH}" -ne 54 || \
      "${FIXED_TEACHER_SAMPLES_PER_BATCH}" -ne 10 ]]; then
  echo "ERROR: this launcher requires fixed GT/teacher counts 54/10" >&2
  exit 1
fi
if (( FIXED_GT_SAMPLES_PER_BATCH + FIXED_TEACHER_SAMPLES_PER_BATCH != 64 )); then
  echo "ERROR: fixed GT and teacher counts must sum to 64" >&2
  exit 1
fi

find_latest_checkpoint() {
  local candidate checkpoint_name checkpoint_step
  local latest_checkpoint=""
  local latest_step=-1

  if [[ ! -d "${RUN_DIR}" ]]; then
    return 0
  fi
  while IFS= read -r -d '' candidate; do
    checkpoint_name="${candidate##*/}"
    if [[ "${checkpoint_name}" =~ ^checkpoint-([0-9]+)$ ]]; then
      checkpoint_step="${BASH_REMATCH[1]}"
      if (( 10#${checkpoint_step} > latest_step )); then
        latest_step=$((10#${checkpoint_step}))
        latest_checkpoint="${candidate}"
      fi
    fi
  done < <(
    find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d \
      -name 'checkpoint-*' -print0
  )
  printf '%s' "${latest_checkpoint}"
}

LATEST_CHECKPOINT="$(find_latest_checkpoint)"
RESUME_KIND="fresh"

if [[ -e "${RUN_DIR}" && ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run path exists but is not a directory: ${RUN_DIR}" >&2
  exit 1
fi
if [[ "${RESUME_MODE}" == "never" ]]; then
  if [[ -e "${RUN_DIR}" ]]; then
    echo "ERROR: RESUME_MODE=never refuses existing run directory: ${RUN_DIR}" >&2
    echo "Choose a new EXPERIMENT_NAME or use RESUME_MODE=auto/required." >&2
    exit 1
  fi
elif [[ -z "${LATEST_CHECKPOINT}" ]]; then
  if [[ "${RESUME_MODE}" == "required" ]]; then
    echo "ERROR: RESUME_MODE=required but no checkpoint exists in: ${RUN_DIR}" >&2
    exit 1
  fi
  if [[ -d "${RUN_DIR}" ]]; then
    echo "ERROR: run directory exists but contains no checkpoint: ${RUN_DIR}" >&2
    echo "Refusing to overwrite an incomplete run." >&2
    exit 1
  fi
else
  required_checkpoint_files=(
    "${LATEST_CHECKPOINT}/trainer_state.json"
    "${LATEST_CHECKPOINT}/training_args.bin"
  )
  for required_checkpoint_file in "${required_checkpoint_files[@]}"; do
    if [[ ! -s "${required_checkpoint_file}" ]]; then
      echo "ERROR: latest checkpoint is incomplete: ${required_checkpoint_file}" >&2
      exit 1
    fi
  done
  if [[ ! -s "${LATEST_CHECKPOINT}/model.safetensors" && \
        ! -s "${LATEST_CHECKPOINT}/model.safetensors.index.json" ]]; then
    echo "ERROR: latest checkpoint has no model weights: ${LATEST_CHECKPOINT}" >&2
    exit 1
  fi

  missing_training_state=()
  for state_file in optimizer.pt scheduler.pt rng_state.pth; do
    if [[ ! -s "${LATEST_CHECKPOINT}/${state_file}" ]]; then
      missing_training_state+=("${state_file}")
    fi
  done
  if (( ${#missing_training_state[@]} == 0 )); then
    RESUME_KIND="full"
  elif [[ "${ALLOW_WEIGHTS_ONLY_RESUME}" == "1" ]]; then
    RESUME_KIND="weights-only"
    echo "WARNING: recovering from a legacy weights-only checkpoint:" >&2
    echo "  ${LATEST_CHECKPOINT}" >&2
    echo "  missing: ${missing_training_state[*]}" >&2
    echo "Optimizer, scheduler, and RNG state will restart; this is not exact resume." >&2
  else
    echo "ERROR: latest checkpoint is weights-only, not fully stateful:" >&2
    echo "  ${LATEST_CHECKPOINT}" >&2
    echo "  missing: ${missing_training_state[*]}" >&2
    echo "Set ALLOW_WEIGHTS_ONLY_RESUME=1 once to recover its model weights." >&2
    exit 1
  fi
fi

if [[ "${RESUME_KIND}" == "fresh" ]]; then
  echo "Resume plan: fresh run -> ${RUN_DIR}"
elif [[ "${RESUME_KIND}" == "full" ]]; then
  echo "Resume plan: full-state resume <- ${LATEST_CHECKPOINT}"
else
  echo "Resume plan: legacy weights-only recovery <- ${LATEST_CHECKPOINT}"
fi

if pgrep -f '[0]5_score_libero_plus.py|[0]6_audit_teacher_quality.py|[0]8_build_distillation_dataset.py' >/dev/null; then
  echo "ERROR: scoring, teacher audit, or sidecar construction is still running." >&2
  echo "Wait for 06, 06b, and 07 to finish before starting 08 training." >&2
  exit 1
fi

required_files=(
  "${TRAIN_PYTHON}"
  "${PREFLIGHT_SCRIPT}"
  "${BATCH_PREFLIGHT_SCRIPT}"
  "${GR00T_ROOT}/examples/finetune.sh"
  "${MANIFEST}"
  "${MANIFEST_SUMMARY}"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file is missing: ${required_file}" >&2
    exit 1
  fi
done

# Fail closed: this launcher is specifically for full-GT + teacher mixing.
export FLARE_REQUIRE_MIXED_DATA=1
export FLARE_EXPECTED_TEACHER_MIX_RATIO="${TEACHER_MIX_RATIO}"
export FLARE_EXPECTED_SAVE_ONLY_MODEL=0
unset SAVE_ONLY_MODEL
export PATH="${GR00T_ROOT}/.venv/bin:${PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export GR00T_BACKBONE_MODEL_NAME="${BACKBONE_PATH}"

cd "${GR00T_ROOT}"

"${TRAIN_PYTHON}" "${PREFLIGHT_SCRIPT}" \
  --dataset-path "${DATASET_PATH}" \
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}" \
  --manifest "${MANIFEST}" \
  --manifest-summary "${MANIFEST_SUMMARY}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --base-model-path "${BASE_MODEL_PATH}" \
  --backbone-path "${BACKBONE_PATH}" \
  --teacher-mix-ratio "${TEACHER_MIX_RATIO}" \
  --fixed-gt-samples-per-batch "${FIXED_GT_SAMPLES_PER_BATCH}" \
  --fixed-teacher-samples-per-batch "${FIXED_TEACHER_SAMPLES_PER_BATCH}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num-gpus "${NUM_GPUS}"

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --embodiment-tag libero_sim
  --output-dir "${OUTPUT_DIR}"
  --experiment-name "${EXPERIMENT_NAME}"
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}"
  --keep-full-gt-dataset
  --teacher-mix-ratio "${TEACHER_MIX_RATIO}"
  --fixed-gt-samples-per-batch "${FIXED_GT_SAMPLES_PER_BATCH}"
  --fixed-teacher-samples-per-batch "${FIXED_TEACHER_SAMPLES_PER_BATCH}"
  --allow-padding
  --state-dropout-prob 0.2
)

# Exact-chain preflight: verify the mixed-data contract before loading the model.
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --validate_config_only

"${TRAIN_PYTHON}" "${BATCH_PREFLIGHT_SCRIPT}" \
  --dataset-path "${DATASET_PATH}" \
  --distillation-dataset-dir "${DISTILLATION_DATASET_DIR}" \
  --base-model-path "${BASE_MODEL_PATH}" \
  --backbone-path "${BACKBONE_PATH}" \
  --fixed-gt-samples-per-batch "${FIXED_GT_SAMPLES_PER_BATCH}" \
  --fixed-teacher-samples-per-batch "${FIXED_TEACHER_SAMPLES_PER_BATCH}" \
  --num-batches "${PREFLIGHT_BATCHES}"

if [[ "${RUN_MODE}" == "validate" ]]; then
  echo "RUN_MODE=validate: all preflight checks passed; training was not started."
  exit 0
fi
if [[ "${RESUME_KIND}" == "full" ]]; then
  echo "Resuming corrected four-suite 54:10 GT replay with full state."
elif [[ "${RESUME_KIND}" == "weights-only" ]]; then
  echo "Recovering corrected four-suite 54:10 GT replay from model weights."
else
  echo "Starting new corrected four-suite 54:10 GT-replay training."
fi
echo "Run directory: ${RUN_DIR}"
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
