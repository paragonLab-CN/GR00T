#!/usr/bin/env bash
set -euo pipefail

# Sequential closed-loop LIBERO-10 evaluation for the two full-data models.
# Both models are evaluated with the same tasks, seeds, episode count, action
# execution horizon, and environment settings.

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda}"
N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-1}"
SEED="${SEED:-2026}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"

GT_MODEL_PATH="${GT_MODEL_PATH:-${GR00T_ROOT}/outputs/gr00t_full_gt_only_steps3000/gr00t_full_gt_only}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${GR00T_ROOT}/outputs/gr00t_teacher_only_high_medium_steps3000/gr00t_full_gt_flare_h8_s8}"

GT_TAG="${GT_TAG:-gt_only_full_steps3000}"
TEACHER_TAG="${TEACHER_TAG:-teacher_only_high_medium_steps3000}"

SERVER_PID=""

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap cleanup_server EXIT INT TERM

check_model_dir() {
    local model_path="$1"

    python - "${model_path}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.is_dir():
    raise FileNotFoundError(f"Model directory not found: {root}")

for name in ["config.json", "model.safetensors.index.json"]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required model file: {path}")

processor_root = root if (root / "processor_config.json").is_file() else root / "processor"
for name in ["processor_config.json", "statistics.json", "embodiment_id.json"]:
    path = processor_root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required processor file: {path}")

index_path = root / "model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise RuntimeError(f"No weight shards referenced by {index_path}")

missing = [name for name in shards if not (root / name).is_file()]
if missing:
    raise FileNotFoundError(
        f"Checkpoint is incomplete at {root}; missing shards: {missing}"
    )

print(f"PASS model check: {root}")
print(f"  processor path: {processor_root}")
print(f"  weight shards : {len(shards)}")
PY
}

wait_for_server() {
    local server_pid="$1"
    local server_log="$2"

    for _ in $(seq 1 300); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "Inference server exited before becoming ready." >&2
            tail -n 100 "${server_log}" >&2 || true
            return 1
        fi

        if grep -q "Server ready" "${server_log}"; then
            return 0
        fi

        sleep 1
    done

    echo "Timed out waiting for inference server." >&2
    tail -n 100 "${server_log}" >&2 || true
    return 1
}

run_one_model() {
    local model_tag="$1"
    local model_path="$2"
    local output_root="${GR00T_ROOT}/outputs/eval_libero10/${model_tag}"
    local server_log="${output_root}/server.log"

    mkdir -p "${output_root}"
    check_model_dir "${model_path}"

    echo
    echo "============================================================"
    echo "Starting model evaluation"
    echo "Tag                 : ${model_tag}"
    echo "Model path          : ${model_path}"
    echo "Episodes per task   : ${N_EPISODES}"
    echo "Seed                : ${SEED}"
    echo "Action steps        : ${N_ACTION_STEPS}"
    echo "Max episode steps   : ${MAX_EPISODE_STEPS}"
    echo "============================================================"

    cd "${GR00T_ROOT}"

    PYTHONUNBUFFERED=1 uv run --no-sync python -u gr00t/eval/run_gr00t_server.py \
        --model-path "${model_path}" \
        --embodiment-tag LIBERO_PANDA \
        --device "${DEVICE}" \
        --port "${PORT}" \
        --use-sim-policy-wrapper \
        >"${server_log}" 2>&1 &

    SERVER_PID=$!
    wait_for_server "${SERVER_PID}" "${server_log}"

    N_EPISODES="${N_EPISODES}" \
    N_ENVS="${N_ENVS}" \
    SEED="${SEED}" \
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
    N_ACTION_STEPS="${N_ACTION_STEPS}" \
        bash scripts/flare/eval_libero10_client.sh "${model_tag}" "${PORT}"

    cleanup_server
    sleep 3
}

cd "${GR00T_ROOT}"
run_one_model "${GT_TAG}" "${GT_MODEL_PATH}"
run_one_model "${TEACHER_TAG}" "${TEACHER_MODEL_PATH}"

python - \
    "${GR00T_ROOT}/outputs/eval_libero10/${GT_TAG}/summary.json" \
    "${GR00T_ROOT}/outputs/eval_libero10/${TEACHER_TAG}/summary.json" \
    "${GR00T_ROOT}/outputs/eval_libero10/compare_${GT_TAG}_vs_${TEACHER_TAG}.json" <<'PY'
import json
import sys
from pathlib import Path

left_path = Path(sys.argv[1])
right_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

left = json.loads(left_path.read_text(encoding="utf-8"))
right = json.loads(right_path.read_text(encoding="utf-8"))

left_tasks = {row["task"]: row for row in left["tasks"]}
right_tasks = {row["task"]: row for row in right["tasks"]}
if set(left_tasks) != set(right_tasks):
    raise RuntimeError("The two evaluations do not contain the same task set")

rows = []
wins = ties = losses = 0
for task in sorted(left_tasks):
    gt_rate = float(left_tasks[task]["success_rate"])
    teacher_rate = float(right_tasks[task]["success_rate"])
    delta = teacher_rate - gt_rate
    if delta > 0:
        result = "teacher_only_win"
        wins += 1
    elif delta < 0:
        result = "gt_only_win"
        losses += 1
    else:
        result = "tie"
        ties += 1
    rows.append(
        {
            "task": task,
            "gt_only_success_rate": gt_rate,
            "teacher_only_success_rate": teacher_rate,
            "teacher_minus_gt": delta,
            "result": result,
        }
    )

comparison = {
    "gt_only_summary": str(left_path),
    "teacher_only_summary": str(right_path),
    "episodes_per_task": left["episodes_per_task"],
    "num_tasks": left["num_tasks"],
    "gt_only_overall_success_rate": left["overall_success_rate"],
    "teacher_only_overall_success_rate": right["overall_success_rate"],
    "teacher_minus_gt_overall": (
        right["overall_success_rate"] - left["overall_success_rate"]
    ),
    "teacher_task_wins": wins,
    "ties": ties,
    "gt_task_wins": losses,
    "tasks": rows,
}

out_path.write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n========== Two-model comparison ==========")
print(f"GT-only success rate      : {left['overall_success_rate']:.4f}")
print(f"Teacher-only success rate : {right['overall_success_rate']:.4f}")
print(f"Teacher - GT              : {comparison['teacher_minus_gt_overall']:+.4f}")
print(f"Task wins / ties / losses : {wins} / {ties} / {losses}")
print(f"Comparison JSON           : {out_path}")
PY
