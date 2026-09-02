#!/usr/bin/env bash
set -euo pipefail

# Fixed three-task LIBERO-90 diagnostic.
# Usage:
#   MODEL_PATH_RECORD=/path/to/model \
#   N_EPISODES=10 SEED=2026 KEEP_VIDEOS=1 RESUME=0 \
#   bash scripts/flare/eval_libero90_diagnostic_client.sh <model_tag> <port>

MODEL_TAG="${1:?Please provide model tag}"
PORT="${2:-5555}"

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
LIBERO_PY="${GR00T_ROOT}/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"

N_EPISODES="${N_EPISODES:-10}"
N_ENVS="${N_ENVS:-1}"
SEED="${SEED:-2026}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
KEEP_VIDEOS="${KEEP_VIDEOS:-1}"
RESUME="${RESUME:-0}"
MODEL_PATH_RECORD="${MODEL_PATH_RECORD:-unknown}"

OUTPUT_ROOT="${GR00T_ROOT}/outputs/eval_libero90_diagnostic/${MODEL_TAG}"
LOG_DIR="${OUTPUT_ROOT}/logs"
VIDEO_DIR="${OUTPUT_ROOT}/videos"
TASK_FILE="${OUTPUT_ROOT}/tasks.tsv"

mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GR00T_EVAL_SEED="${SEED}"

cd "${GR00T_ROOT}"

cat > "${TASK_FILE}" <<'EOF'
libero_sim/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet	close the top drawer of the cabinet
libero_sim/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet	open the bottom drawer of the cabinet
libero_sim/LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket	pick up the alphabet soup and put it in the basket
EOF

cat > "${OUTPUT_ROOT}/run_metadata.json" <<EOF
{
  "model_tag": "${MODEL_TAG}",
  "model_path": "${MODEL_PATH_RECORD}",
  "server_port": ${PORT},
  "seed": ${SEED},
  "episodes_per_task": ${N_EPISODES},
  "n_envs": ${N_ENVS},
  "max_episode_steps": ${MAX_EPISODE_STEPS},
  "n_action_steps": ${N_ACTION_STEPS},
  "keep_videos": ${KEEP_VIDEOS}
}
EOF

NUM_TASKS="$(wc -l < "${TASK_FILE}")"

echo "============================================================"
echo "LIBERO-90 three-task diagnostic"
echo "Model tag          : ${MODEL_TAG}"
echo "Model path         : ${MODEL_PATH_RECORD}"
echo "Tasks              : ${NUM_TASKS}"
echo "Episodes per task  : ${N_EPISODES}"
echo "Seed               : ${SEED}"
echo "Server port        : ${PORT}"
echo "Output             : ${OUTPUT_ROOT}"
echo "============================================================"

TASK_NUMBER=0
while IFS=$'\t' read -r TASK TASK_LANGUAGE; do
    TASK_NUMBER=$((TASK_NUMBER + 1))
    TASK_NAME="${TASK#libero_sim/}"
    LOG_PATH="${LOG_DIR}/${TASK_NAME}.log"
    TASK_VIDEO_DIR="${VIDEO_DIR}/${TASK_NAME}"

    if [[ "${RESUME}" == "1" && -f "${LOG_PATH}" ]] && \
       grep -q "success rate:" "${LOG_PATH}"; then
        echo "[SKIP ${TASK_NUMBER}/${NUM_TASKS}] ${TASK_NAME}"
        continue
    fi

    rm -f "${LOG_PATH}"
    rm -rf "${TASK_VIDEO_DIR}"
    mkdir -p "${TASK_VIDEO_DIR}"

    echo
    echo "============================================================"
    echo "[${TASK_NUMBER}/${NUM_TASKS}] ${TASK_NAME}"
    echo "Instruction : ${TASK_LANGUAGE}"
    echo "Episodes    : ${N_EPISODES}"
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

    if [[ "${KEEP_VIDEOS}" != "1" ]]; then
        rm -rf "${TASK_VIDEO_DIR}"
    fi
done < "${TASK_FILE}"

"${LIBERO_PY}" - \
    "${OUTPUT_ROOT}" \
    "${N_EPISODES}" \
    "${MODEL_TAG}" \
    "${MODEL_PATH_RECORD}" \
    "${SEED}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
n_episodes = int(sys.argv[2])
model_tag = sys.argv[3]
model_path = sys.argv[4]
seed = int(sys.argv[5])

rows = []
for line in (output_root / "tasks.tsv").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    env_name, language = line.split("\t", 1)
    task_name = env_name.split("/", 1)[1]
    log_path = output_root / "logs" / f"{task_name}.log"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"success rate:\s*([0-9eE+\-.]+)", text)
    if not matches:
        raise RuntimeError(f"No success rate found in {log_path}")
    success_rate = float(matches[-1])
    successes = int(round(success_rate * n_episodes))
    rows.append(
        {
            "task": task_name,
            "language": language,
            "successes": successes,
            "episodes": n_episodes,
            "success_rate": success_rate,
        }
    )

total_successes = sum(row["successes"] for row in rows)
total_episodes = sum(row["episodes"] for row in rows)
summary = {
    "model_tag": model_tag,
    "model_path": model_path,
    "seed": seed,
    "num_tasks": len(rows),
    "episodes_per_task": n_episodes,
    "total_successes": total_successes,
    "total_episodes": total_episodes,
    "overall_success_rate": total_successes / total_episodes,
    "tasks": rows,
}

(output_root / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
with (output_root / "task_success_rates.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n============================================================")
print("Diagnostic summary")
print(f"Model     : {model_tag}")
print(f"Path      : {model_path}")
print(f"Successes : {total_successes}/{total_episodes}")
print(f"Rate      : {total_successes / total_episodes:.4f}")
for row in rows:
    print(f"  {row['task']}: {row['successes']}/{row['episodes']} ({row['success_rate']:.3f})")
print(f"Summary   : {output_root / 'summary.json'}")
print("============================================================")
PY
