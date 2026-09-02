#!/usr/bin/env bash
set -euo pipefail

# 用法：
# bash scripts/flare/eval_libero90_client.sh <model_tag> <port> <scope>
#
# scope:
#   dataset74  只评测当前 libero_90_lerobot/meta/tasks.jsonl 中的74个任务
#   all90      评测 LIBERO benchmark 注册的全部90个任务
#
# 示例：
# bash scripts/flare/eval_libero90_client.sh gt_only_500 5555 dataset74
# bash scripts/flare/eval_libero90_client.sh flare_858 5555 dataset74

MODEL_TAG="${1:?Please provide model tag}"
PORT="${2:-5555}"
SCOPE="${3:-dataset74}"

if [[ "${SCOPE}" != "dataset74" && "${SCOPE}" != "all90" ]]; then
    echo "Invalid scope: ${SCOPE}"
    echo "Supported scopes: dataset74, all90"
    exit 1
fi

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
LIBERO_PY="${GR00T_ROOT}/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"

DATASET_TASKS_JSONL="${DATASET_TASKS_JSONL:-/root/autodl-tmp/lerobot_data/libero_90_lerobot/meta/tasks.jsonl}"

N_EPISODES="${N_EPISODES:-5}"
N_ENVS="${N_ENVS:-1}"
SEED="${SEED:-2026}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"

# 0表示评测全部任务；正数表示只取前N个任务，用于smoke test
TASK_LIMIT="${TASK_LIMIT:-0}"

# 0表示每个任务结束后删除视频，避免74个任务占用大量磁盘
KEEP_VIDEOS="${KEEP_VIDEOS:-0}"

# 1表示已经完成且日志中存在success rate的任务自动跳过
RESUME="${RESUME:-1}"

OUTPUT_ROOT="${GR00T_ROOT}/outputs/eval_libero90_${SCOPE}/${MODEL_TAG}"
LOG_DIR="${OUTPUT_ROOT}/logs"
VIDEO_DIR="${OUTPUT_ROOT}/videos"
TASK_FILE="${OUTPUT_ROOT}/tasks.tsv"

mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "${GR00T_ROOT}"

echo "============================================================"
echo "Generating LIBERO-90 task list"
echo "Scope              : ${SCOPE}"
echo "Dataset tasks file : ${DATASET_TASKS_JSONL}"
echo "Task limit         : ${TASK_LIMIT}"
echo "============================================================"

"${LIBERO_PY}" - \
    "${SCOPE}" \
    "${DATASET_TASKS_JSONL}" \
    "${TASK_FILE}" \
    "${TASK_LIMIT}" <<'PY'
import json
import re
import sys
from pathlib import Path

from libero.libero import benchmark


scope = sys.argv[1]
dataset_tasks_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
task_limit = int(sys.argv[4])


def normalize(text: str) -> str:
    """Normalize language instructions for robust matching."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


benchmark_dict = benchmark.get_benchmark_dict()
suite = benchmark_dict["libero_90"]()

official_tasks = []
for task_id in range(suite.get_num_tasks()):
    task = suite.get_task(task_id)
    official_tasks.append(
        {
            "task_id": task_id,
            "name": task.name,
            "language": task.language,
            "normalized_language": normalize(task.language),
        }
    )

if scope == "all90":
    selected = official_tasks

elif scope == "dataset74":
    if not dataset_tasks_path.is_file():
        raise FileNotFoundError(
            f"Dataset task metadata not found: {dataset_tasks_path}"
        )

    dataset_languages = []
    with dataset_tasks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            language = str(record["task"])
            dataset_languages.append(
                {
                    "task_index": int(record["task_index"]),
                    "language": language,
                    "normalized_language": normalize(language),
                }
            )

    official_by_language = {}
    for task in official_tasks:
        key = task["normalized_language"]
        if key in official_by_language:
            raise RuntimeError(
                "Duplicate normalized LIBERO language instruction: "
                f"{task['language']}"
            )
        official_by_language[key] = task

    selected = []
    missing = []

    # 按LeRobot tasks.jsonl中的task_index顺序保存
    for dataset_task in sorted(
        dataset_languages,
        key=lambda x: x["task_index"],
    ):
        key = dataset_task["normalized_language"]

        if key not in official_by_language:
            missing.append(dataset_task)
            continue

        official_task = dict(official_by_language[key])
        official_task["dataset_task_index"] = dataset_task["task_index"]
        selected.append(official_task)

    if missing:
        print(
            "\nThe following dataset tasks could not be mapped to "
            "LIBERO-90 benchmark tasks:",
            file=sys.stderr,
        )
        for item in missing:
            print(
                f"  task_index={item['task_index']}: "
                f"{item['language']}",
                file=sys.stderr,
            )
        raise RuntimeError(
            f"Failed to map {len(missing)} dataset tasks to LIBERO-90"
        )

else:
    raise ValueError(f"Unsupported scope: {scope}")

if task_limit > 0:
    selected = selected[:task_limit]

if not selected:
    raise RuntimeError("No tasks selected")

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    for task in selected:
        env_name = f"libero_sim/{task['name']}"
        f.write(f"{env_name}\t{task['language']}\n")

print(f"LIBERO-90 benchmark task count : {len(official_tasks)}")
print(f"Selected task count            : {len(selected)}")
print(f"Saved task list                : {output_path}")

for index, task in enumerate(selected[:5]):
    print(
        f"[{index:02d}] libero_sim/{task['name']} "
        f"| {task['language']}"
    )
PY

NUM_TASKS="$(wc -l < "${TASK_FILE}")"

echo
echo "============================================================"
echo "Starting LIBERO-90 evaluation"
echo "Model tag          : ${MODEL_TAG}"
echo "Scope              : ${SCOPE}"
echo "Tasks              : ${NUM_TASKS}"
echo "Episodes per task  : ${N_EPISODES}"
echo "Total episodes     : $((NUM_TASKS * N_EPISODES))"
echo "Seed               : ${SEED}"
echo "Max episode steps  : ${MAX_EPISODE_STEPS}"
echo "Action steps       : ${N_ACTION_STEPS}"
echo "Server port        : ${PORT}"
echo "============================================================"

TASK_NUMBER=0

while IFS=$'\t' read -r TASK TASK_LANGUAGE; do
    TASK_NUMBER=$((TASK_NUMBER + 1))
    TASK_NAME="${TASK#libero_sim/}"

    LOG_PATH="${LOG_DIR}/${TASK_NAME}.log"
    TASK_VIDEO_DIR="${VIDEO_DIR}/${TASK_NAME}"

    if [[ "${RESUME}" == "1" && -f "${LOG_PATH}" ]] &&
       grep -q "success rate:" "${LOG_PATH}"; then
        echo
        echo "[SKIP ${TASK_NUMBER}/${NUM_TASKS}] ${TASK_NAME}"
        echo "Existing completed log: ${LOG_PATH}"
        continue
    fi

    rm -f "${LOG_PATH}"
    rm -rf "${TASK_VIDEO_DIR}"
    mkdir -p "${TASK_VIDEO_DIR}"

    echo
    echo "============================================================"
    echo "[${TASK_NUMBER}/${NUM_TASKS}]"
    echo "Model       : ${MODEL_TAG}"
    echo "Task        : ${TASK}"
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
    "${SCOPE}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path


output_root = Path(sys.argv[1])
n_episodes = int(sys.argv[2])
model_tag = sys.argv[3]
scope = sys.argv[4]

log_dir = output_root / "logs"
task_file = output_root / "tasks.tsv"

rows = []

for line in task_file.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    env_name, language = line.split("\t", 1)
    task_name = env_name.split("/", 1)[1]
    log_path = log_dir / f"{task_name}.log"

    if not log_path.is_file():
        raise FileNotFoundError(f"Missing evaluation log: {log_path}")

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    matches = re.findall(
        r"success rate:\s*([0-9eE+\-.]+)",
        text,
    )

    if not matches:
        raise RuntimeError(
            f"No success rate found in {log_path}"
        )

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

if not rows:
    raise RuntimeError("No completed evaluation rows found")

total_successes = sum(row["successes"] for row in rows)
total_episodes = sum(row["episodes"] for row in rows)

overall_success_rate = total_successes / total_episodes
macro_success_rate = (
    sum(row["success_rate"] for row in rows) / len(rows)
)

csv_path = output_root / "task_success_rates.csv"

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "task",
            "language",
            "successes",
            "episodes",
            "success_rate",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "model_tag": model_tag,
    "suite": "libero_90",
    "scope": scope,
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
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("\n========== LIBERO-90 Evaluation Summary ==========")
print(f"Model                 : {model_tag}")
print(f"Scope                 : {scope}")
print(f"Tasks                 : {len(rows)}")
print(f"Episodes per task     : {n_episodes}")
print(f"Total successes       : {total_successes}")
print(f"Total episodes        : {total_episodes}")
print(f"Overall success rate  : {overall_success_rate:.4f}")
print(f"Macro success rate    : {macro_success_rate:.4f}")
print(f"CSV                    : {csv_path}")
print(f"JSON                   : {summary_path}")
PY
