#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 单模型 LIBERO-Plus 评测：一个权重跑全部 112 任务（4 套件 × 7 扰动 × 4 任务）。
#
# 与基线的区别：基线（run_eval_libero_plus_all_suites.sh）按 4 个套件各自加载
# 对应权重；而蒸馏/消融模型是单一权重（在混合 libero_plus 数据上微调），所以
# 这里只起一个服务器、用同一个权重跑全部 112 任务。
#
# 用法：
#   bash scripts/flare/run_eval_libero_plus_single_model.sh <model_path> <model_tag> [port]
#
# 环境变量：
#   N_EPISODES  每任务评测次数（默认 10）
#   TASK_LIMIT  只取前 N 个任务（0=全部，用于 smoke test）
#
# 输出：outputs/eval_libero_plus/<model_tag>/summary_all.json
# ============================================================================

MODEL_PATH="${1:?请提供模型路径，如 outputs/libero_plus/.../checkpoint-10000}"
MODEL_TAG="${2:?请提供模型标签，如 gr00t_libero_plus_flare}"
PORT="${3:-5555}"

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
DEVICE="${DEVICE:-cuda}"
N_EPISODES="${N_EPISODES:-10}"
SUITES="libero_spatial,libero_object,libero_goal,libero_10"

OUTPUT_ROOT="${GR00T_ROOT}/outputs/eval_libero_plus/${MODEL_TAG}"
mkdir -p "${OUTPUT_ROOT}"

SERVER_LOG="${OUTPUT_ROOT}/server_single.log"
SERVER_PID=""

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT INT TERM

wait_for_server() {
    for _ in $(seq 1 300); do
        grep -q "Server ready" "${SERVER_LOG}" 2>/dev/null && return 0
        sleep 1
    done
    echo "服务器启动超时" >&2
    tail -n 100 "${SERVER_LOG}" >&2 || true
    return 1
}

cd "${GR00T_ROOT}"

echo "============================================"
echo "  单模型 LIBERO-Plus 评测"
echo "  模型: ${MODEL_PATH}"
echo "  标签: ${MODEL_TAG}"
echo "  Episodes/任务: ${N_EPISODES}"
echo "============================================"

# 加载 gated 骨干模型 Cosmos-Reason2-2B 需要指向本地路径
# （见 gr00t/model/gr00t_n1d7/gr00t_n1d7.py 的 GR00T_BACKBONE_MODEL_NAME 覆盖）
export GR00T_BACKBONE_MODEL_NAME="${GR00T_ROOT}/checkpoints/Cosmos-Reason2-2B"

PYTHONUNBUFFERED=1 uv run --no-sync python -u gr00t/eval/run_gr00t_server.py \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag LIBERO_PANDA \
    --device "${DEVICE}" \
    --port "${PORT}" \
    --use-sim-policy-wrapper \
    >"${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
wait_for_server

N_EPISODES="${N_EPISODES}" SUITES="${SUITES}" FULL_EVAL="${FULL_EVAL:-0}" TASK_LIMIT="${TASK_LIMIT:-0}" \
    bash scripts/flare/eval_libero_plus_client.sh "${MODEL_TAG}" "${PORT}"

echo ""
echo "评测完成。结果: ${OUTPUT_ROOT}/summary_all.json"
