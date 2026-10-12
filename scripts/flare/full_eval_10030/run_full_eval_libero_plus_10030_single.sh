#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LIBERO-Plus 全量评测（10,030 扰动任务）—— 单模型 wrapper
#
# 用法：
#   bash scripts/flare/full_eval_10030/run_full_eval_libero_plus_10030_single.sh \
#       <model_path> <model_tag> [port]
#
# 环境变量：
#   N_EPISODES  每任务评测次数（默认 1）
#   TASK_LIMIT  只取前 N 个任务（0=全部 10,030，用于 smoke test）
#
# 协议：FULL_EVAL=1 枚举 task_classification.json 全部任务；环境 seed 固定
# 2026；diffusion 噪声 unfixed（与 random_diffusion_noise 系列一致）；
# RESUME=1 按任务 log 断点续跑；KEEP_VIDEOS=0 防止爆盘。
# ============================================================================

MODEL_PATH="${1:?请提供模型路径，如 outputs/libero_plus/.../checkpoint-10000}"
MODEL_TAG="${2:?请提供模型标签，如 gr00t_libero_plus_flare_..._full10030}"
PORT="${3:-5555}"

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly GR00T_ROOT="${GR00T_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
readonly GR00T_PYTHON="${GR00T_ROOT}/.venv/bin/python"
readonly EXPECTED_FULL_TASK_COUNT="10030"

N_EPISODES="${N_EPISODES:-1}"
TASK_LIMIT="${TASK_LIMIT:-0}"

readonly OUTPUT_ROOT="${GR00T_ROOT}/outputs/eval_libero_plus/${MODEL_TAG}"
readonly SUMMARY_PATH="${OUTPUT_ROOT}/summary_all.json"
readonly TASKS_PATH="${OUTPUT_ROOT}/tasks.tsv"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "${PORT}" =~ ^[0-9]+$ ]] || die "port must be an integer: ${PORT}"
(( PORT >= 1 && PORT <= 65535 )) || die "port must be in [1, 65535]: ${PORT}"
[[ "${N_EPISODES}" =~ ^[1-9][0-9]*$ ]] || die "N_EPISODES must be a positive integer: ${N_EPISODES}"
[[ "${TASK_LIMIT}" =~ ^[0-9]+$ ]] || die "TASK_LIMIT must be a non-negative integer: ${TASK_LIMIT}"

[[ -x "${GR00T_PYTHON}" ]] || die "project Python is unavailable: ${GR00T_PYTHON}"
[[ -d "${MODEL_PATH}" ]] || die "checkpoint not found: ${MODEL_PATH}"

# 并行场景下允许多个服务器共存，只要求目标端口空闲
if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    die "port ${PORT} is already in use"
fi

cd "${GR00T_ROOT}"

# 可选：把服务器钉到指定 GPU（CUDA_VISIBLE_DEVICES 为空则不做限制）
if [[ -n "${GPU_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export DEVICE=cuda

# 与 random_diffusion_noise 系列保持同协议：不固定 server 端 diffusion 噪声
unset DIFFUSION_SEED

export OMP_NUM_THREADS=1
export N_EPISODES
export N_ENVS=1
export SEED=2026
export MAX_EPISODE_STEPS=720
export N_ACTION_STEPS=8
export TASK_LIMIT
export FULL_EVAL=1
export KEEP_VIDEOS=0
export RESUME=1

echo "============================================================"
echo "LIBERO-Plus FULL evaluation (10,030 tasks)"
echo "  model:             ${MODEL_PATH}"
echo "  model tag:         ${MODEL_TAG}"
echo "  episodes/task:     ${N_EPISODES}"
echo "  task limit:        ${TASK_LIMIT} (0 = all)"
echo "  environment seed:  ${SEED}"
echo "  diffusion seed:    unfixed"
echo "  gpu:               ${GPU_ID:-unset (all visible)}"
echo "  output:            ${OUTPUT_ROOT}"
echo "  resume:            enabled"
echo "============================================================"

bash scripts/flare/run_eval_libero_plus_single_model.sh \
    "${MODEL_PATH}" \
    "${MODEL_TAG}" \
    "${PORT}"

[[ -f "${SUMMARY_PATH}" ]] || die "evaluation finished without summary: ${SUMMARY_PATH}"
[[ -f "${TASKS_PATH}" ]] || die "evaluation output has no task manifest: ${TASKS_PATH}"

actual_task_count="$(awk 'NF {count++} END {print count+0}' "${TASKS_PATH}")"
if [[ "${TASK_LIMIT}" == "0" ]]; then
    [[ "${actual_task_count}" == "${EXPECTED_FULL_TASK_COUNT}" ]] || \
        die "task count mismatch: expected ${EXPECTED_FULL_TASK_COUNT}, got ${actual_task_count}"
fi

"${GR00T_PYTHON}" - "${SUMMARY_PATH}" "${actual_task_count}" "${N_EPISODES}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_tasks = int(sys.argv[2])
n_episodes = int(sys.argv[3])

summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("num_tasks") != expected_tasks:
    raise RuntimeError(
        f"Expected {expected_tasks} tasks, got {summary.get('num_tasks')}"
    )
if summary.get("episodes_per_task") != n_episodes:
    raise RuntimeError(
        f"Expected {n_episodes} episodes/task, got {summary.get('episodes_per_task')}"
    )
if summary.get("total_episodes") != expected_tasks * n_episodes:
    raise RuntimeError(
        f"Expected {expected_tasks * n_episodes} episodes, "
        f"got {summary.get('total_episodes')}"
    )
if summary.get("diffusion_noise_protocol") != {"mode": "unfixed"}:
    raise RuntimeError("Summary does not record unfixed diffusion noise")
seeded_tasks = [
    row["task"]
    for row in summary.get("tasks", [])
    if row.get("diffusion_noise_base_seed") is not None
]
if seeded_tasks:
    raise RuntimeError(f"Unexpected fixed diffusion seeds: {seeded_tasks[:3]}")

print("[PASS] summary validation")
print(f"  tasks:                {summary['num_tasks']}")
print(f"  episodes/task:        {summary['episodes_per_task']}")
print(f"  total episodes:       {summary['total_episodes']}")
print(f"  overall success rate: {summary['overall_success_rate']:.4f}")
print(f"  macro success rate:   {summary['macro_success_rate']:.4f}")
PY

echo "Evaluation complete"
echo "  summary: ${SUMMARY_PATH}"
