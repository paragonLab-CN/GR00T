#!/usr/bin/env python3
"""Paired episode-level statistics for the controlled four-model LIBERO-Plus comparison.

All four models were evaluated under the frozen protocol: 112 tasks x 10 episodes,
environment_seed=2026, explicit per-request diffusion seeds
(master_seed 2026 + task_index*1e6 + episode_index*1e4 + action_chunk_index), and a
byte-identical 112-task manifest. Because every run shares the same manifest order AND
the same seed formula, episode i of task t is the SAME environment instance across all
models, so episode-level McNemar pairing is exact (not approximate).

This script recomputes, from the summary_all.json per-episode fields:
  - overall success rates
  - pairwise episode-level McNemar (exact two-sided binomial on discordant pairs)
  - pairwise task-level sign test (per-task success-count comparison, n=112)
  - a tasks.tsv sha256 check across all runs (frozen-protocol integrity)

Usage:
  python scripts/flare/paired_stats_controlled_four.py
  python scripts/flare/paired_stats_controlled_four.py --out outputs/eval_libero_plus/paired_stats_controlled_four.json

All numbers are derived from the cited artifacts; nothing is hardcoded except the
four eval directory tags and the (verified) frozen-protocol constants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

GR00T_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = GR00T_ROOT / "outputs" / "eval_libero_plus"

# tag -> eval directory name (all fixed diffusion-noise seed 2026, checkpoint-10000)
RUNS = {
    "gt_only_lr1e4_no_aug": "gr00t_libero_plus_gt_only_lr1e4_no_aug_checkpoint_10000_fixed_noise_seed2026",
    "epoch50_fixed_44_20": "gr00t_libero_plus_flare_epoch50_fixed_44_20_checkpoint_10000_fixed_noise_seed2026",
    "deployed_44_20": "gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20_checkpoint_10000_fixed_noise_seed2026",
    "v3_no_xyz_44_20": "gr00t_libero_plus_flare_v3_no_xyz_44_20_ck10000_fixed_noise_seed2026",
}
# ordered pairs reported (A vs B)
PAIRS = [
    ("v3_no_xyz_44_20", "deployed_44_20"),
    ("v3_no_xyz_44_20", "gt_only_lr1e4_no_aug"),
    ("v3_no_xyz_44_20", "epoch50_fixed_44_20"),
    ("epoch50_fixed_44_20", "deployed_44_20"),
    ("epoch50_fixed_44_20", "gt_only_lr1e4_no_aug"),
    ("deployed_44_20", "gt_only_lr1e4_no_aug"),
]


def load_run(tag: str, dirname: str) -> dict:
    summary = EVAL_ROOT / dirname / "summary_all.json"
    d = json.loads(summary.read_text(encoding="utf-8"))
    episodes = {t["task"]: [bool(x) for x in t["episode_successes"]] for t in d["tasks"]}
    tasks_tsv = EVAL_ROOT / dirname / "tasks.tsv"
    return {
        "tag": tag,
        "dir": dirname,
        "overall_success_rate": d["overall_success_rate"],
        "total_successes": d["total_successes"],
        "total_episodes": d["total_episodes"],
        "per_suite": d["per_suite"],
        "per_category": d["per_category"],
        "per_difficulty": d["per_difficulty"],
        "diffusion_noise_protocol": d["diffusion_noise_protocol"],
        "tasks_tsv_sha256": hashlib.sha256(tasks_tsv.read_bytes()).hexdigest(),
        "episodes": episodes,
    }


def mcnemar_exact(a: dict, b: dict) -> dict:
    """Episode-level McNemar, exact two-sided binomial on discordant pairs."""
    b01 = c01 = 0  # b01: A success & B fail ; c01: A fail & B success
    common = set(a) & set(b)
    for t in common:
        for x, y in zip(a[t], b[t]):
            if x and not y:
                b01 += 1
            elif y and not x:
                c01 += 1
    n = b01 + c01
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b01, c01) + 1)) / 2**n)
    return {"b_A_success_B_fail": b01, "c_A_fail_B_success": c01, "net": b01 - c01, "n_discordant": n, "p_exact": p}


def task_sign_test(a: dict, b: dict) -> dict:
    """Task-level sign test on per-task success counts (n = number of tasks)."""
    better = worse = 0
    for t in set(a) & set(b):
        sa, sb = sum(a[t]), sum(b[t])
        if sa > sb:
            better += 1
        elif sb > sa:
            worse += 1
    n = better + worse
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(better, worse) + 1)) / 2**n)
    return {"A_better_tasks": better, "A_worse_tasks": worse, "n_untied_tasks": n, "p_exact": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    runs = {tag: load_run(tag, dn) for tag, dn in RUNS.items()}

    # frozen-protocol integrity: all task manifests must be byte-identical
    hashes = {tag: r["tasks_tsv_sha256"] for tag, r in runs.items()}
    protocol_ok = len(set(hashes.values())) == 1

    summary_table = {
        tag: {k: r[k] for k in ("overall_success_rate", "total_successes", "total_episodes", "per_suite", "per_category", "per_difficulty", "diffusion_noise_protocol", "tasks_tsv_sha256")}
        for tag, r in runs.items()
    }

    mcnemar, sign = {}, {}
    for A, B in PAIRS:
        key = f"{A}__vs__{B}"
        mcnemar[key] = mcnemar_exact(runs[A]["episodes"], runs[B]["episodes"])
        sign[key] = task_sign_test(runs[A]["episodes"], runs[B]["episodes"])

    result = {
        "description": "Paired stats for the controlled four-model LIBERO-Plus comparison (fixed diffusion-noise seed 2026, 112 tasks x 10 episodes, env seed 2026)",
        "frozen_protocol": {
            "n_tasks": 112,
            "episodes_per_task": 10,
            "environment_seed": 2026,
            "diffusion_noise_mode": runs["v3_no_xyz_44_20"]["diffusion_noise_protocol"],
            "tasks_tsv_sha256_all_identical": protocol_ok,
            "tasks_tsv_sha256": hashes,
        },
        "models": summary_table,
        "mcnemar_episode_level": mcnemar,
        "sign_test_task_level": sign,
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
