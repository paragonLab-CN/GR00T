#!/usr/bin/env bash
# gt-only control for gr00t_libero_plus_flare_epoch50_v2.
# 与 08b_train_gr00t_flare_epoch50.sh 完全一致，仅去掉 FLARE 相关参数。
#
# 区别清单（相对 08b）：
#   - 去掉 --distillation-dataset-dir / --keep-full-gt-dataset / --teacher-mix-ratio
#   - 去掉 --allow-padding（对 physical_embodiment 无副作用：effective length 已裁剪）
#   - 去掉 env FLARE_REQUIRE_MIXED_DATA / FLARE_EXPECTED_TEACHER_MIX_RATIO
#   - 去掉 check_distillation_sidecars.py 预检
#   - SAVE_TOTAL_LIMIT 显式设为 10，与 flare 对齐（旧 09 脚本漏掉）
#
# Usage: bash FLARE-VLA/commands/libero_plus/09b_train_gr00t_gt_only_v2.sh

set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_gt_only_v2}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"          # 对齐 flare
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
export SHARD_SIZE="${SHARD_SIZE:-1024}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
export USE_WANDB="${USE_WANDB:-0}"
export EXTRA_AUGMENTATION_CONFIG='{"gaussian_blur_params":{"blur_limit":[3,7],"p":0.5},"motion_blur_params":{"blur_limit":[3,7],"p":0.3}}'

cd "${GR00T_ROOT}"

# finetune.sh 里用 `exec python`，把项目 venv 放到 PATH 最前面。
export PATH="${GR00T_ROOT}/.venv/bin:${PATH}"

# 从本地 HF 缓存离线加载 gated 的 Cosmos-Reason2-2B 骨干（hf-mirror 会 403）。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GR00T_BACKBONE_MODEL_NAME="/root/autodl-tmp/Isaac-GR00T/checkpoints/Cosmos-Reason2-2B"

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --embodiment-tag libero_sim
  --output-dir "${OUTPUT_DIR}"
  --experiment-name gr00t_libero_plus_gt_only_v2
  --state-dropout-prob 0.2
  --save-only-model
)

# 预检：确认解析出的是单个 physical_embodiment / mix_ratio 1.0。
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --validate_config_only

# 正式训练
bash examples/finetune.sh "${COMMON_ARGS[@]}" -- \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
