#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LIBERO Spatial 闭环评测脚本
#
# 用法：
#   bash scripts/flare/eval_spatial_client.sh <model_tag> [port]
#
# 示例：
#   bash scripts/flare/eval_spatial_client.sh gr00t_spatial_gt46490_teacher8998_official 5555
#
# 需要先在一个终端中启动推理服务器：
#   uv run python gr00t/eval/run_gr00t_server.py \
#       --model-path <path-to-checkpoint> \
#       --embodiment-tag LIBERO_PANDA \
#       --use-sim-policy-wrapper
# ============================================================================

MODEL_TAG="${1:?Please provide model tag, e.g. gr00t_spatial_gt46490_teacher8998_official}"
PORT="${2:-5555}"

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
LIBERO_PY="${GR00T_ROOT}/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"

# ---- 和官方 eval_libero10_client.sh 保持一致的参数 ----
N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-1}"
SEED="${SEED:-2026}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"

OUTPUT_ROOT="${GR00T_ROOT}/outputs/eval_spatial/${MODEL_TAG}"
LOG_DIR="${OUTPUT_ROOT}/logs"
VIDEO_DIR="${OUTPUT_ROOT}/videos"

mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ---- LIBERO Spatial 10 个任务 ----
TASKS=(
    "libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
)

cd "${GR00T_ROOT}"

for TASK in "${TASKS[@]}"; do
    TASK_NAME="${TASK#libero_sim/}"
    LOG_PATH="${LOG_DIR}/${TASK_NAME}.log"
    TASK_VIDEO_DIR="${VIDEO_DIR}/${TASK_NAME}"

    echo
    echo "============================================================"
    echo "Model   : ${MODEL_TAG}"
    echo "Task    : ${TASK}"
    echo "Episodes: ${N_EPISODES}"
    echo "Port    : ${PORT}"
    echo "============================================================"

    "${LIBERO_PY}" gr00t/eval/rollout_policy.py \
        --n-episodes "${N_EPISODES}" \
        --policy-client-host 127.0.0.1 \
        --policy-client-port "${PORT}" \
        --max-episode-steps "${MAX_EPISODE_STEPS}" \
        --env-name "${TASK}" \
        --n-action-steps "${N_ACTION_STEPS}" \
        --n-envs "${N_ENVS}" \
        --seed "${SEED}" \
        --candidate-k 1 \
        --video-dir "${TASK_VIDEO_DIR}" \
        2>&1 | tee "${LOG_PATH}"
done

# ---- 汇总结果：从日志中提取 success rate，输出 CSV 和 JSON ----
"${LIBERO_PY}" - "${OUTPUT_ROOT}" "${N_EPISODES}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
n_episodes = int(sys.argv[2])
log_dir = output_root / "logs"

rows = []

for log_path in sorted(log_dir.glob("*.log")):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"success rate:\s*([0-9eE+\-.]+)",
        text,
    )
    if not matches:
        raise RuntimeError(f"No success rate found in {log_path}")

    success_rate = float(matches[-1])
    successes = int(round(success_rate * n_episodes))

    rows.append(
        {
            "task": log_path.stem,
            "successes": successes,
            "episodes": n_episodes,
            "success_rate": success_rate,
        }
    )

macro_success_rate = sum(r["success_rate"] for r in rows) / len(rows)
total_successes = sum(r["successes"] for r in rows)
total_episodes = sum(r["episodes"] for r in rows)
overall_success_rate = total_successes / total_episodes

csv_path = output_root / "task_success_rates.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["task", "successes", "episodes", "success_rate"],
    )
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "suite": "LIBERO Spatial",
    "model_tag": str(output_root.parent.name)
    if output_root.parent.name != "eval_spatial"
    else str(output_root.name),
    "num_tasks": len(rows),
    "episodes_per_task": n_episodes,
    "total_successes": total_successes,
    "total_episodes": total_episodes,
    "overall_success_rate": overall_success_rate,
    "macro_success_rate": macro_success_rate,
    "tasks": rows,
}

summary_path = output_root / "summary.json"
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("=" * 60)
print("          LIBERO Spatial Evaluation Summary")
print("=" * 60)
print(f"  Tasks                : {len(rows)}")
print(f"  Episodes per task    : {n_episodes}")
print(f"  Total successes      : {total_successes}")
print(f"  Total episodes       : {total_episodes}")
print(f"  Overall success rate : {overall_success_rate:.4f}  ({total_successes}/{total_episodes})")
print(f"  Macro success rate   : {macro_success_rate:.4f}")
print(f"  CSV                  : {csv_path}")
print(f"  JSON                 : {summary_path}")
print("=" * 60)

for r in rows:
    bar = "█" * int(r["success_rate"] * 20)
    print(f"  {r['success_rate']:.2f}  {bar}  {r['task']}")
PY
