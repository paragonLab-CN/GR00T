#!/usr/bin/env bash
set -euo pipefail

# 用法：
# bash scripts/flare/eval_libero10_client.sh gt_only_500 5555
# bash scripts/flare/eval_libero10_client.sh flare_858 5555

MODEL_TAG="${1:?Please provide model tag, such as gt_only_500 or flare_858}"
PORT="${2:-5555}"

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
LIBERO_PY="${GR00T_ROOT}/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"

N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-1}"
SEED="${SEED:-2026}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"

OUTPUT_ROOT="${GR00T_ROOT}/outputs/eval_libero10/${MODEL_TAG}"
LOG_DIR="${OUTPUT_ROOT}/logs"
VIDEO_DIR="${OUTPUT_ROOT}/videos"

mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

TASKS=(
"libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
"libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
"libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
"libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
"libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
"libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
"libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
"libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
"libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
"libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
)

cd "${GR00T_ROOT}"

for TASK in "${TASKS[@]}"; do
    TASK_NAME="${TASK#libero_sim/}"
    LOG_PATH="${LOG_DIR}/${TASK_NAME}.log"
    TASK_VIDEO_DIR="${VIDEO_DIR}/${TASK_NAME}"

    echo
    echo "============================================================"
    echo "Model: ${MODEL_TAG}"
    echo "Task : ${TASK}"
    echo "Episodes: ${N_EPISODES}"
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

print("\n========== Evaluation Summary ==========")
print(f"Tasks                : {len(rows)}")
print(f"Total successes      : {total_successes}")
print(f"Total episodes       : {total_episodes}")
print(f"Overall success rate : {overall_success_rate:.4f}")
print(f"Macro success rate   : {macro_success_rate:.4f}")
print(f"CSV                   : {csv_path}")
print(f"JSON                  : {summary_path}")
PY