#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LIBERO-Plus 全量基线评测总控：四个套件依次换权重跑，最后合并 summary。
#
# GR00T-N1.7-LIBERO 每个套件有独立权重目录（libero_spatial/libero_object/
# libero_goal/libero_10），推理服务器一次只能加载一个，因此按套件分批评测。
#
# 用法：
#   bash scripts/flare/run_eval_libero_plus_all_suites.sh [model_tag] [port]
#
# 输出：
#   outputs/eval_libero_plus/<model_tag>/summary_<suite>.json
#   outputs/eval_libero_plus/<model_tag>/summary_all.json    （合并后）
# ============================================================================

MODEL_TAG="${1:-GR00T-N1.7-LIBERO}"
PORT="${2:-5555}"

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${GR00T_ROOT}/checkpoints/GR00T-N1.7-LIBERO}"
DEVICE="${DEVICE:-cuda}"

N_EPISODES="${N_EPISODES:-10}"
SUITES="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"

OUTPUT_ROOT="${GR00T_ROOT}/outputs/eval_libero_plus/${MODEL_TAG}"
mkdir -p "${OUTPUT_ROOT}"

SERVER_PID=""

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap cleanup_server EXIT INT TERM

wait_for_server() {
    local server_log="$1"

    for _ in $(seq 1 300); do
        if grep -q "Server ready" "${server_log}" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for inference server." >&2
    tail -n 100 "${server_log}" >&2 || true
    return 1
}

cd "${GR00T_ROOT}"

IFS=',' read -ra SUITE_LIST <<< "${SUITES}"

for SUITE in "${SUITE_LIST[@]}"; do
    MODEL_PATH="${CHECKPOINT_ROOT}/${SUITE}"
    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "[ERROR] checkpoint dir not found: ${MODEL_PATH}" >&2
        exit 1
    fi

    echo
    echo "============================================================"
    echo "Suite : ${SUITE}"
    echo "Model : ${MODEL_PATH}"
    echo "============================================================"

    SERVER_LOG="${OUTPUT_ROOT}/server_${SUITE}.log"

    PYTHONUNBUFFERED=1 uv run --no-sync python -u gr00t/eval/run_gr00t_server.py \
        --model-path "${MODEL_PATH}" \
        --embodiment-tag LIBERO_PANDA \
        --device "${DEVICE}" \
        --port "${PORT}" \
        --use-sim-policy-wrapper \
        >"${SERVER_LOG}" 2>&1 &

    SERVER_PID=$!
    wait_for_server "${SERVER_LOG}"

    N_EPISODES="${N_EPISODES}" SUITES="${SUITE}" \
        bash scripts/flare/eval_libero_plus_client.sh "${MODEL_TAG}" "${PORT}"

    cleanup_server
    sleep 3
done

echo
echo "============================================================"
echo "All suites done. Merging per-suite summaries..."
echo "============================================================"

SUITES="${SUITES}" OUTPUT_ROOT="${OUTPUT_ROOT}" MODEL_TAG="${MODEL_TAG}" \
    python - <<'PY'
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT"])
model_tag = os.environ["MODEL_TAG"]
suites = [s.strip() for s in os.environ["SUITES"].split(",") if s.strip()]

all_rows = []
for suite in suites:
    summary_path = output_root / f"summary_{suite}.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_rows.extend(summary["tasks"])

n_episodes = all_rows[0]["episodes"]
total_successes = sum(row["successes"] for row in all_rows)
total_episodes = sum(row["episodes"] for row in all_rows)


def group_mean(key):
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row[key]].append(row["success_rate"])
    return {
        group: sum(values) / len(values)
        for group, values in sorted(grouped.items())
    }


merged = {
    "model_tag": model_tag,
    "benchmark": "libero_plus",
    "num_tasks": len(all_rows),
    "episodes_per_task": n_episodes,
    "total_successes": total_successes,
    "total_episodes": total_episodes,
    "overall_success_rate": total_successes / total_episodes,
    "macro_success_rate": sum(r["success_rate"] for r in all_rows) / len(all_rows),
    "per_suite": group_mean("suite"),
    "per_category": group_mean("category"),
    "per_difficulty": group_mean("difficulty"),
    "tasks": all_rows,
}

merged_path = output_root / "summary_all.json"
merged_path.write_text(
    json.dumps(merged, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Merged {len(all_rows)} tasks -> {merged_path}")
print(f"Overall success rate : {merged['overall_success_rate']:.4f}")
print(f"Macro success rate   : {merged['macro_success_rate']:.4f}")
print("Per suite:")
for suite, rate in merged["per_suite"].items():
    print(f"  {suite:<20} {rate:.4f}")
print("Per perturbation category:")
for category, rate in merged["per_category"].items():
    print(f"  {category:<24} {rate:.4f}")
PY
