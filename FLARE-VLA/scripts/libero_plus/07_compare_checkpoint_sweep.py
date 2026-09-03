#!/usr/bin/env python3
"""Compare LeWM checkpoint sweep results (oracle win rate, teacher agreement,
oracle_gap distribution, 8-step rollout MSE growth slope).

Reads each ``epoch*/validation_summary.json`` + ``validation_stepwise_metrics.npz``
under a sweep dir and prints a side-by-side table plus a JSON summary.

Selection heuristics (matching the plan):
  * lower mean oracle rollout MSE  -> better latent prediction
  * flatter stepwise growth        -> less rollout compounding
  * higher teacher agreement       -> cosine/MSE more consistent
  * higher oracle-vs-candidate gap -> more discriminative scoring
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def stepwise_growth(per_step: np.ndarray) -> dict[str, float]:
    """per_step: [N, H]. Returns mean per-step values plus growth stats."""
    step_means = per_step.mean(axis=0)
    x = np.arange(len(step_means), dtype=np.float64)
    slope = float(np.polyfit(x, step_means, 1)[0]) if len(step_means) >= 2 else np.nan
    first = float(step_means[0])
    last = float(step_means[-1])
    return {
        "step_means": [float(v) for v in step_means],
        "slope": slope,
        "last_over_first": (last / first) if first > 0 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for sub in sorted(args.sweep_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("epoch"):
            continue
        summary_path = sub / "validation_summary.json"
        stepwise_path = sub / "validation_stepwise_metrics.npz"
        if not summary_path.is_file() or not stepwise_path.is_file():
            print(f"[skip] {sub.name}: missing summary or stepwise npz")
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stepwise = np.load(stepwise_path, allow_pickle=False)

        oracle_mse_step = np.asarray(stepwise["oracle_mse_per_step"], dtype=np.float64)
        teacher_mse_step = np.asarray(stepwise["teacher_mse_per_step"], dtype=np.float64)
        oracle_cos_step = np.asarray(stepwise["oracle_cosine_per_step"], dtype=np.float64)
        teacher_cos_step = np.asarray(stepwise["teacher_cosine_per_step"], dtype=np.float64)

        oracle_mse_growth = stepwise_growth(oracle_mse_step)
        teacher_mse_growth = stepwise_growth(teacher_mse_step)

        # Candidate discrimination: spread of per-sample mean MSE across the K
        # candidates (larger = the scorer separates candidates more).
        mse_per_step = np.asarray(stepwise["mse_per_step"], dtype=np.float64)  # [N,K,H]
        cand_mean_mse = mse_per_step.mean(axis=-1)  # [N,K]
        candidate_mse_std = float(cand_mean_mse.std(axis=-1).mean())

        dist = summary.get("metric_distributions", {})
        row = {
            "epoch": int(sub.name.replace("epoch", "")),
            "samples": int(summary.get("scored_samples", 0)),
            "teacher_agreement_rate": summary["teacher_agreement"]["rate"],
            "oracle_better_cosine_rate": summary["oracle_better_than_all_candidates"]["cosine_rate"],
            "oracle_better_mse_rate": summary["oracle_better_than_all_candidates"]["mse_rate"],
            "oracle_mse_mean": dist["oracle_mse_score"]["mean"],
            "oracle_mse_median": dist["oracle_mse_score"]["median"],
            "oracle_cosine_mean": dist["oracle_cosine_score"]["mean"],
            "best_mse_mean": dist["best_mse_score"]["mean"],
            "oracle_gap_mse_mean": dist["oracle_mse_gap"]["mean"],
            "oracle_gap_mse_median": dist["oracle_mse_gap"]["median"],
            "oracle_gap_mse_p95": dist["oracle_mse_gap"]["p95"],
            "candidate_mse_std_mean": candidate_mse_std,
            "oracle_mse_step_slope": oracle_mse_growth["slope"],
            "oracle_mse_step_last_over_first": oracle_mse_growth["last_over_first"],
            "oracle_mse_step_means": oracle_mse_growth["step_means"],
            "teacher_mse_step_slope": teacher_mse_growth["slope"],
            "teacher_mse_step_last_over_first": teacher_mse_growth["last_over_first"],
            "teacher_mse_step_means": teacher_mse_growth["step_means"],
        }
        rows.append(row)

    if not rows:
        raise SystemExit(f"No epoch* result dirs found under {args.sweep_dir}")

    # Print table
    header = (
        f"{'epoch':>5} {'samples':>8} {'agree':>6} {'orcl>cos':>8} {'orcl>mse':>8} "
        f"{'oracleMSE':>10} {'orclMSE_med':>11} {'gapMean':>8} {'gapMed':>8} {'candStd':>8} "
        f"{'orcl_slope':>10} {'orcl_ratio':>10} {'tch_slope':>9} {'tch_ratio':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['epoch']:>5} {r['samples']:>8} {r['teacher_agreement_rate']:>6.4f} "
            f"{r['oracle_better_cosine_rate']:>8.4f} {r['oracle_better_mse_rate']:>8.4f} "
            f"{r['oracle_mse_mean']:>10.6f} {r['oracle_mse_median']:>11.6f} "
            f"{r['oracle_gap_mse_mean']:>8.5f} {r['oracle_gap_mse_median']:>8.5f} {r['candidate_mse_std_mean']:>8.5f} "
            f"{r['oracle_mse_step_slope']:>10.6f} {r['oracle_mse_step_last_over_first']:>10.3f} "
            f"{r['teacher_mse_step_slope']:>9.6f} {r['teacher_mse_step_last_over_first']:>9.3f}"
        )

    print("\nStepwise oracle rollout MSE (mean per step, step1..step8):")
    for r in rows:
        steps = "  ".join(f"{v:.5f}" for v in r["oracle_mse_step_means"])
        print(f"  epoch {r['epoch']:>2}: [{steps}]")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote JSON: {args.output}")


if __name__ == "__main__":
    main()
