#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LIBERO-Plus 全量评测（10,030 扰动任务）—— 双卡并行
#
# GPU0 评 epoch50_fixed_44_20/checkpoint-10000（端口 5555）
# GPU1 评 suite_corrected_gt_replay_44_20/checkpoint-10000（端口 5556）
# 两个模型同时跑，总时长 ≈ 单模型时长（~1.3 天 @ N_EPISODES=1）。
# 中断后重跑本脚本即可按 RESUME=1 续跑。
#
# 推荐在 tmux 里运行：
#   tmux new-session -d -s full_eval_10030 \
#     'bash scripts/flare/full_eval_10030/run_full_eval_libero_plus_10030_parallel2.sh \
#        > outputs/eval_libero_plus/full_eval_10030_parallel2.log 2>&1'
#
# 环境变量：
#   N_EPISODES  每任务评测次数（默认 1）
#   TASK_LIMIT  只取前 N 个任务（0=全部 10,030，用于 smoke test）
# ============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly GR00T_ROOT="${GR00T_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${GR00T_ROOT}"

readonly TAG1="gr00t_libero_plus_flare_epoch50_fixed_44_20_checkpoint_10000_full10030"
readonly TAG2="gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20_checkpoint_10000_full10030"
readonly CKPT1="${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare_epoch50_fixed_44_20/gr00t_libero_plus_flare_epoch50_fixed_44_20/checkpoint-10000"
readonly CKPT2="${GR00T_ROOT}/outputs/libero_plus/gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20/checkpoint-10000"

mkdir -p "${GR00T_ROOT}/outputs/eval_libero_plus"

echo "############################################################"
echo "# Launching parallel full evaluations"
echo "#   GPU0 : ${TAG1}"
echo "#   GPU1 : ${TAG2}"
echo "############################################################"

GPU_ID=0 bash scripts/flare/full_eval_10030/run_full_eval_libero_plus_10030_single.sh \
    "${CKPT1}" "${TAG1}" 5555 \
    > "${GR00T_ROOT}/outputs/eval_libero_plus/full10030_epoch50.log" 2>&1 &
PID1=$!

GPU_ID=1 bash scripts/flare/full_eval_10030/run_full_eval_libero_plus_10030_single.sh \
    "${CKPT2}" "${TAG2}" 5556 \
    > "${GR00T_ROOT}/outputs/eval_libero_plus/full10030_gt_replay.log" 2>&1 &
PID2=$!

echo "epoch50    PID=${PID1}  log: outputs/eval_libero_plus/full10030_epoch50.log"
echo "gt_replay  PID=${PID2}  log: outputs/eval_libero_plus/full10030_gt_replay.log"

FAIL=0
wait "${PID1}" || { echo "epoch50 evaluation FAILED"; FAIL=1; }
wait "${PID2}" || { echo "gt_replay evaluation FAILED"; FAIL=1; }

if [[ "${FAIL}" != "0" ]]; then
    echo "At least one evaluation failed; see logs above." >&2
    exit 1
fi

echo
echo "############################################################"
echo "# Comparison"
echo "############################################################"
"${GR00T_ROOT}/.venv/bin/python" - \
    "${GR00T_ROOT}/outputs/eval_libero_plus/${TAG1}/summary_all.json" \
    "${GR00T_ROOT}/outputs/eval_libero_plus/${TAG2}/summary_all.json" <<'PY'
import json
import sys
from pathlib import Path

s1 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
s2 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))


def row(label, v1, v2):
    print(f"  {label:<26} {v1:>10.4f} {v2:>10.4f}  {v1 - v2:+.4f}")


print(f"{'':<28}{'epoch50':>10} {'gt_replay':>10}  delta")
row("overall_success_rate", s1["overall_success_rate"], s2["overall_success_rate"])
row("macro_success_rate", s1["macro_success_rate"], s2["macro_success_rate"])
print("  -- per suite --")
for suite in sorted(set(s1["per_suite"]) | set(s2["per_suite"])):
    row(suite, s1["per_suite"].get(suite, float("nan")), s2["per_suite"].get(suite, float("nan")))
print("  -- per category --")
for cat in sorted(set(s1["per_category"]) | set(s2["per_category"])):
    row(cat, s1["per_category"].get(cat, float("nan")), s2["per_category"].get(cat, float("nan")))
PY

echo
echo "All evaluations complete."
