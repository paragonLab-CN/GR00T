#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LIBERO-Plus 全量评测（10,030 扰动任务）—— 两个 checkpoint 串行
#
# 机器只有 1 张 GPU，因此两个权重严格串行：第一个评测完整结束（summary
# 校验通过、服务器关闭）后才启动第二个。中途失败可直接重跑本脚本，
# RESUME=1 会跳过已完成任务。
#
# 用法：
#   nohup bash scripts/flare/full_eval_10030/run_full_eval_libero_plus_10030_both.sh \
#       > outputs/eval_libero_plus/full_eval_10030_both.log 2>&1 &
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

echo "############################################################"
echo "# [1/2] epoch50_fixed_44_20 checkpoint-10000"
echo "############################################################"
bash scripts/flare/full_eval_10030/run_full_eval_libero_plus_10030_single.sh \
    "${CKPT1}" \
    "${TAG1}" \
    5555

echo
echo "############################################################"
echo "# [2/2] suite_corrected_gt_replay_44_20 checkpoint-10000"
echo "############################################################"
bash scripts/flare/full_eval_10030/run_full_eval_libero_plus_10030_single.sh \
    "${CKPT2}" \
    "${TAG2}" \
    5555

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
