#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LIBERO-Plus 闭环评测脚本（分层抽样任务）
#
# 用法：
#   bash scripts/flare/eval_libero_plus_client.sh <model_tag> [port]
#
# 需要先在另一个终端启动推理服务器（按套件用对应权重）：
#   uv run python gr00t/eval/run_gr00t_server.py \
#       --model-path checkpoints/GR00T-N1.7-LIBERO/<suite> \
#       --embodiment-tag LIBERO_PANDA \
#       --use-sim-policy-wrapper
#
# 任务抽样协议：每个套件 × 每个扰动类别（7 类）抽 TASKS_PER_CATEGORY 个任务，
# 难度随机（固定 seed），默认 4 任务/类 → 28 任务/套件，共 112 任务。
# 扰动类别/难度来自 LIBERO-plus fork 的 task_classification.json。
#
# 与 eval_libero90_client.sh 保持一致的评测参数：
#   N_EPISODES / SEED / MAX_EPISODE_STEPS / N_ACTION_STEPS / candidate-k 1
# ============================================================================

MODEL_TAG="${1:?Please provide model tag, e.g. GR00T-N1.7-LIBERO}"
PORT="${2:-5555}"

GR00T_ROOT="${GR00T_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
LIBERO_PLUS_PY="${GR00T_ROOT}/gr00t/eval/sim/LIBERO/libero_plus_uv/.venv/bin/python"
LIBERO_PLUS_CONFIG_DIR="${GR00T_ROOT}/gr00t/eval/sim/LIBERO/libero_plus_config"
CLASSIFICATION_JSON="${GR00T_ROOT}/external_dependencies/LIBERO-plus/libero/libero/benchmark/task_classification.json"

N_EPISODES="${N_EPISODES:-10}"
N_ENVS="${N_ENVS:-1}"
SEED="${SEED:-2026}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
DIFFUSION_SEED="${DIFFUSION_SEED:-}"
DIFFUSION_TASK_STRIDE=1000000

# 每个套件 × 每个扰动类别抽多少任务
TASKS_PER_CATEGORY="${TASKS_PER_CATEGORY:-4}"
# 1=评测全部任务（完整评测集，忽略抽样）；0=分层抽样（默认）
FULL_EVAL="${FULL_EVAL:-0}"

# 逗号分隔的套件列表；按套件分批评测时只跑指定套件
# （GR00T-N1.7-LIBERO 每个套件有独立权重，服务器一次只能加载一个套件）
SUITES="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"

# 0表示评测全部任务；正数表示只取前N个任务，用于 smoke test
TASK_LIMIT="${TASK_LIMIT:-0}"

# 可选：直接复用既有任务清单，避免重新抽样。
TASK_FILE_SOURCE="${TASK_FILE_SOURCE:-}"
# 非零时校验最终任务数；本实验固定为历史 112 任务。
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-0}"

# 0表示每个任务结束后删除视频，避免大量视频占用磁盘
KEEP_VIDEOS="${KEEP_VIDEOS:-0}"

# 1表示已经完成且日志中存在success rate的任务自动跳过
RESUME="${RESUME:-1}"

OUTPUT_ROOT="${GR00T_ROOT}/outputs/eval_libero_plus/${MODEL_TAG}"
LOG_DIR="${OUTPUT_ROOT}/logs"
VIDEO_DIR="${OUTPUT_ROOT}/videos"
TASK_FILE="${OUTPUT_ROOT}/tasks.tsv"

mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# 关键：让 fork 使用自己的 bddl/init_states 路径，不读 vanilla 的 ~/.libero/config.yaml
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_DIR}"

cd "${GR00T_ROOT}"

if [[ -n "${DIFFUSION_SEED}" && "${N_ENVS}" != "1" ]]; then
    echo "Fixed paired diffusion noise requires N_ENVS=1" >&2
    exit 1
fi

echo "============================================================"
echo "Generating LIBERO-Plus task list (stratified sampling)"
echo "Classification     : ${CLASSIFICATION_JSON}"
echo "Tasks per category : ${TASKS_PER_CATEGORY}"
echo "Seed               : ${SEED}"
echo "Task limit         : ${TASK_LIMIT}"
echo "============================================================"

if [[ -n "${TASK_FILE_SOURCE}" ]]; then
    if [[ ! -f "${TASK_FILE_SOURCE}" ]]; then
        echo "Task manifest does not exist: ${TASK_FILE_SOURCE}" >&2
        exit 1
    fi
    cp "${TASK_FILE_SOURCE}" "${TASK_FILE}"
    echo "Reused task list   : ${TASK_FILE_SOURCE}"
else
"${LIBERO_PLUS_PY}" - \
    "${CLASSIFICATION_JSON}" \
    "${TASK_FILE}" \
    "${TASKS_PER_CATEGORY}" \
    "${TASK_LIMIT}" \
    "${SEED}" \
    "${SUITES}" \
    "${FULL_EVAL}" <<'PY'
import json
import random
import sys
from pathlib import Path

from libero.libero import benchmark

classification_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
tasks_per_category = int(sys.argv[3])
task_limit = int(sys.argv[4])
seed = int(sys.argv[5])
suites = [s.strip() for s in sys.argv[6].split(",") if s.strip()]
full_eval = int(sys.argv[7])

SUITES = suites

classification = json.loads(classification_path.read_text(encoding="utf-8"))
benchmark_dict = benchmark.get_benchmark_dict()
rng = random.Random(seed)

selected = []
for suite_name in SUITES:
    entries = classification.get(suite_name, [])
    suite = benchmark_dict[suite_name]()  # task_order_index=0 → identity order

    by_category: dict[str, list[dict]] = {}
    for entry in entries:
        by_category.setdefault(entry["category"], []).append(entry)

    for category, cat_entries in sorted(by_category.items()):
        rng.shuffle(cat_entries)  # 固定 seed 洗牌 → 难度随机且可复现
        limit = None if full_eval == 1 else tasks_per_category
        for entry in (cat_entries if limit is None else cat_entries[:limit]):
            task = suite.get_task(entry["id"] - 1)  # id 是 1 起始，get_task 是 0 起始
            selected.append(
                {
                    "suite": suite_name,
                    "category": category,
                    "difficulty": entry["difficulty_level"],
                    "name": task.name,
                    "language": task.language,
                }
            )

if task_limit > 0:
    selected = selected[:task_limit]

if not selected:
    raise RuntimeError("No tasks selected")

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    for task in selected:
        env_name = f"libero_sim/{task['name']}"
        difficulty = (
            str(task["difficulty"])
            if task["difficulty"] is not None
            else "unknown"
        )
        f.write(
            "\t".join(
                [
                    env_name,
                    task["language"],
                    task["suite"],
                    task["category"],
                    difficulty,
                ]
            )
            + "\n"
        )

print(f"Selected task count : {len(selected)}")
for suite_name in SUITES:
    n = sum(1 for t in selected if t["suite"] == suite_name)
    print(f"  {suite_name}: {n} tasks")
print(f"Saved task list     : {output_path}")
PY
fi

NUM_TASKS="$(wc -l < "${TASK_FILE}")"
if [[ "${EXPECTED_TASK_COUNT}" != "0" && "${NUM_TASKS}" != "${EXPECTED_TASK_COUNT}" ]]; then
    echo "Task manifest count mismatch: ${NUM_TASKS} != ${EXPECTED_TASK_COUNT}" >&2
    exit 1
fi
echo "Task manifest SHA256: $(sha256sum "${TASK_FILE}" | cut -d ' ' -f 1)"

echo
echo "============================================================"
echo "Starting LIBERO-Plus evaluation"
echo "Model tag          : ${MODEL_TAG}"
echo "Tasks              : ${NUM_TASKS}"
echo "Episodes per task  : ${N_EPISODES}"
echo "Total episodes     : $((NUM_TASKS * N_EPISODES))"
echo "Seed               : ${SEED}"
echo "Max episode steps  : ${MAX_EPISODE_STEPS}"
echo "Action steps       : ${N_ACTION_STEPS}"
echo "Diffusion seed     : ${DIFFUSION_SEED:-unfixed}"
echo "Server port        : ${PORT}"
echo "============================================================"

TASK_NUMBER=0

while IFS=$'\t' read -r TASK TASK_LANGUAGE TASK_SUITE TASK_CATEGORY TASK_DIFFICULTY; do
    TASK_NUMBER=$((TASK_NUMBER + 1))
    TASK_NAME="${TASK#libero_sim/}"
    TASK_DIFFUSION_SEED=""
    if [[ -n "${DIFFUSION_SEED}" ]]; then
        TASK_DIFFUSION_SEED=$((DIFFUSION_SEED + (TASK_NUMBER - 1) * DIFFUSION_TASK_STRIDE))
    fi

    LOG_PATH="${LOG_DIR}/${TASK_NAME}.log"
    TASK_VIDEO_DIR="${VIDEO_DIR}/${TASK_NAME}"

    LOG_COMPLETE=0
    if [[ "${RESUME}" == "1" && -f "${LOG_PATH}" ]] &&
       grep -q "success rate:" "${LOG_PATH}" &&
       grep -q "^results:" "${LOG_PATH}"; then
        LOG_COMPLETE=1
        if [[ -n "${TASK_DIFFUSION_SEED}" ]] &&
           ! grep -Fq "task_base=${TASK_DIFFUSION_SEED}," "${LOG_PATH}"; then
            LOG_COMPLETE=0
        fi
    fi
    if [[ "${LOG_COMPLETE}" == "1" ]]; then
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
    echo "Suite       : ${TASK_SUITE}"
    echo "Category    : ${TASK_CATEGORY}"
    echo "Difficulty  : ${TASK_DIFFICULTY}"
    echo "Task        : ${TASK}"
    echo "Instruction : ${TASK_LANGUAGE}"
    echo "Episodes    : ${N_EPISODES}"
    echo "Noise seed  : ${TASK_DIFFUSION_SEED:-unfixed}"
    echo "============================================================"

    DIFFUSION_ARGS=()
    if [[ -n "${TASK_DIFFUSION_SEED}" ]]; then
        DIFFUSION_ARGS=(--diffusion-noise-base-seed "${TASK_DIFFUSION_SEED}")
    fi

    "${LIBERO_PLUS_PY}" gr00t/eval/rollout_policy.py \
        --n-episodes "${N_EPISODES}" \
        --policy-client-host 127.0.0.1 \
        --policy-client-port "${PORT}" \
        --max-episode-steps "${MAX_EPISODE_STEPS}" \
        --env-name "${TASK}" \
        --n-action-steps "${N_ACTION_STEPS}" \
        --n-envs "${N_ENVS}" \
        --seed "${SEED}" \
        --candidate-k 1 \
        "${DIFFUSION_ARGS[@]}" \
        --video-dir "${TASK_VIDEO_DIR}" \
        2>&1 | tee "${LOG_PATH}"

    if [[ "${KEEP_VIDEOS}" != "1" ]]; then
        rm -rf "${TASK_VIDEO_DIR}"
    fi

done < "${TASK_FILE}"

# 分批评测时每批（套件）写独立 summary，最后用 merge 脚本合并
if [[ "${SUITES}" == *","* ]]; then
    SUMMARY_SUFFIX="all"
else
    SUMMARY_SUFFIX="${SUITES}"
fi

"${LIBERO_PLUS_PY}" - \
    "${OUTPUT_ROOT}" \
    "${N_EPISODES}" \
    "${MODEL_TAG}" \
    "${SUMMARY_SUFFIX}" \
    "${DIFFUSION_SEED}" <<'PY'
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


output_root = Path(sys.argv[1])
n_episodes = int(sys.argv[2])
model_tag = sys.argv[3]
summary_suffix = sys.argv[4]
diffusion_seed = None if sys.argv[5] == "" else int(sys.argv[5])

log_dir = output_root / "logs"
task_file = output_root / "tasks.tsv"

rows = []

for task_index, line in enumerate(task_file.read_text(encoding="utf-8").splitlines()):
    if not line.strip():
        continue

    env_name, language, suite, category, difficulty = line.split("\t", 5)
    task_name = env_name.split("/", 1)[1]
    log_path = log_dir / f"{task_name}.log"

    if not log_path.is_file():
        raise FileNotFoundError(f"Missing evaluation log: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")

    matches = re.findall(r"success rate:\s*([0-9eE+\-.]+)", text)

    if not matches:
        raise RuntimeError(f"No success rate found in {log_path}")

    success_rate = float(matches[-1])
    successes = int(round(success_rate * n_episodes))

    result_matches = re.findall(r"^results:\s*(.+)$", text, flags=re.MULTILINE)
    if not result_matches:
        raise RuntimeError(f"No episode results found in {log_path}")
    try:
        rollout_result = ast.literal_eval(result_matches[-1])
        episode_successes = [bool(value) for value in rollout_result[1]]
    except (SyntaxError, ValueError, TypeError, IndexError) as error:
        raise RuntimeError(f"Invalid episode results in {log_path}") from error
    if len(episode_successes) != n_episodes:
        raise RuntimeError(
            f"Expected {n_episodes} episode outcomes in {log_path}, "
            f"got {len(episode_successes)}"
        )
    if sum(episode_successes) != successes:
        raise RuntimeError(
            f"Episode outcomes disagree with success rate in {log_path}"
        )


    rows.append(
        {
            "task": task_name,
            "language": language,
            "suite": suite,
            "category": category,
            "difficulty": difficulty,
            "successes": successes,
            "episode_successes": episode_successes,
            "episodes": n_episodes,
            "success_rate": success_rate,
            "diffusion_noise_base_seed": (
                None if diffusion_seed is None else diffusion_seed + task_index * 1_000_000
            ),
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


def group_mean(key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row["success_rate"])
    return {
        group: sum(values) / len(values)
        for group, values in sorted(grouped.items())
    }


per_suite = group_mean("suite")
per_category = group_mean("category")
per_difficulty = group_mean("difficulty")

csv_path = output_root / f"task_success_rates_{summary_suffix}.csv"

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "task",
            "language",
            "suite",
            "category",
            "difficulty",
            "successes",
            "episode_successes",
            "episodes",
            "success_rate",
            "diffusion_noise_base_seed",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "model_tag": model_tag,
    "benchmark": "libero_plus",
    "num_tasks": len(rows),
    "episodes_per_task": n_episodes,
    "total_successes": total_successes,
    "total_episodes": total_episodes,
    "overall_success_rate": overall_success_rate,
    "macro_success_rate": macro_success_rate,
    "diffusion_noise_protocol": (
        {"mode": "unfixed"}
        if diffusion_seed is None
        else {
            "mode": "explicit_per_request_seed",
            "master_seed": diffusion_seed,
            "task_stride": 1_000_000,
            "episode_stride": 10_000,
            "request_seed_formula": (
                "master_seed + task_index*1000000 + "
                "episode_index*10000 + action_chunk_index"
            ),
        }
    ),
    "per_suite": per_suite,
    "per_category": per_category,
    "per_difficulty": per_difficulty,
    "tasks": rows,
}

summary_path = output_root / f"summary_{summary_suffix}.json"
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n========== LIBERO-Plus Evaluation Summary ==========")
print(f"Model                 : {model_tag}")
print(f"Tasks                 : {len(rows)}")
print(f"Episodes per task     : {n_episodes}")
print(f"Total successes       : {total_successes}")
print(f"Total episodes        : {total_episodes}")
print(f"Overall success rate  : {overall_success_rate:.4f}")
print(f"Macro success rate    : {macro_success_rate:.4f}")
print("\nPer suite:")
for suite, rate in per_suite.items():
    print(f"  {suite:<20} {rate:.4f}")
print("\nPer perturbation category:")
for category, rate in per_category.items():
    print(f"  {category:<24} {rate:.4f}")
print("\nPer difficulty level:")
for difficulty, rate in per_difficulty.items():
    print(f"  {difficulty:<8} {rate:.4f}")
print(f"\nCSV  : {csv_path}")
print(f"JSON : {summary_path}")
PY
