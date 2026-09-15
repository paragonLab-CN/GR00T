#!/usr/bin/env python3
"""Validate and compare four fixed-noise LIBERO-Plus evaluations."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import re

from summarize_fixed_noise_pair import APPEARANCE_CATEGORIES, model_metrics, validate_pair
from summarize_fixed_noise_three import metric_deltas, paired_task_outcomes


def load_episode_outcomes(summary_path: Path, summary: dict) -> list[list[bool]]:
    """Load and validate per-episode outcomes, with legacy-log fallback."""
    outcomes = []
    for row in summary["tasks"]:
        values = row.get("episode_successes")
        if values is None:
            log_path = summary_path.parent / "logs" / f"{row['task']}.log"
            if not log_path.is_file():
                raise FileNotFoundError(
                    f"Missing episode outcomes and legacy log: {log_path}"
                )
            text = log_path.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"^results:\s*(.+)$", text, flags=re.MULTILINE)
            if not matches:
                raise RuntimeError(f"No episode results found in {log_path}")
            try:
                values = ast.literal_eval(matches[-1])[1]
            except (SyntaxError, ValueError, TypeError, IndexError) as error:
                raise RuntimeError(f"Invalid episode results in {log_path}") from error

        values = [bool(value) for value in values]
        expected = int(row["episodes"])
        if len(values) != expected or sum(values) != int(row["successes"]):
            raise RuntimeError(
                f"Invalid episode outcomes for {summary['model_tag']}/{row['task']}"
            )
        outcomes.append(values)
    return outcomes


def exact_mcnemar_pvalue(candidate_only: int, reference_only: int) -> float:
    """Two-sided exact binomial p-value for discordant episode pairs."""
    discordant = candidate_only + reference_only
    if discordant == 0:
        return 1.0
    lower = min(candidate_only, reference_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_episode_outcomes(
    candidate: list[list[bool]], reference: list[list[bool]]
) -> dict:
    candidate_only = 0
    reference_only = 0
    both_success = 0
    both_failure = 0
    for candidate_task, reference_task in zip(candidate, reference, strict=True):
        for candidate_value, reference_value in zip(
            candidate_task, reference_task, strict=True
        ):
            if candidate_value and reference_value:
                both_success += 1
            elif candidate_value:
                candidate_only += 1
            elif reference_value:
                reference_only += 1
            else:
                both_failure += 1
    return {
        "candidate_only_success": candidate_only,
        "reference_only_success": reference_only,
        "both_success": both_success,
        "both_failure": both_failure,
        "net_candidate_successes": candidate_only - reference_only,
        "discordant_pairs": candidate_only + reference_only,
        "exact_mcnemar_two_sided_pvalue": exact_mcnemar_pvalue(
            candidate_only, reference_only
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("flare_summary", type=Path)
    parser.add_argument("image_aug_summary", type=Path)
    parser.add_argument("gt_only_summary", type=Path)
    parser.add_argument("replay_summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "flare": args.flare_summary,
        "image_aug": args.image_aug_summary,
        "gt_only": args.gt_only_summary,
        "gt_replay_54_10": args.replay_summary,
    }
    summaries = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }

    protocol = None
    for name in ("image_aug", "gt_only", "gt_replay_54_10"):
        candidate_protocol = validate_pair(summaries["flare"], summaries[name])
        if protocol is None:
            protocol = candidate_protocol
        elif candidate_protocol != protocol:
            raise RuntimeError("The four diffusion-noise protocols differ")

    metrics = {name: model_metrics(summary) for name, summary in summaries.items()}
    episodes = {
        name: load_episode_outcomes(paths[name], summary)
        for name, summary in summaries.items()
    }
    pair_specs = {
        "image_aug_minus_flare": ("image_aug", "flare"),
        "gt_only_minus_flare": ("gt_only", "flare"),
        "image_aug_minus_gt_only": ("image_aug", "gt_only"),
        "gt_replay_54_10_minus_flare": ("gt_replay_54_10", "flare"),
        "gt_replay_54_10_minus_image_aug": ("gt_replay_54_10", "image_aug"),
        "gt_replay_54_10_minus_gt_only": ("gt_replay_54_10", "gt_only"),
    }

    comparison = {
        "comparison_mode": "four_model_episode_paired_fixed_diffusion_noise",
        "protocol": {
            "tasks": 112,
            "episodes_per_task": 10,
            "total_episodes_per_model": 1120,
            "environment_seed": 2026,
            "diffusion_noise": protocol,
            "appearance_definition": sorted(APPEARANCE_CATEGORIES),
            "appearance_tasks": 48,
        },
        "models": metrics,
        "overall_ranking": sorted(
            (
                {
                    "model": name,
                    "successes": value["overall"]["successes"],
                    "episodes": value["overall"]["episodes"],
                    "success_rate": value["overall"]["success_rate"],
                }
                for name, value in metrics.items()
            ),
            key=lambda row: row["success_rate"],
            reverse=True,
        ),
        "pairwise_percentage_point_change": {
            label: metric_deltas(metrics[candidate], metrics[reference])
            for label, (candidate, reference) in pair_specs.items()
        },
        "paired_task_outcomes": {
            label: paired_task_outcomes(summaries[candidate], summaries[reference])
            for label, (candidate, reference) in pair_specs.items()
        },
        "paired_episode_outcomes": {
            label: paired_episode_outcomes(
                episodes[candidate], episodes[reference]
            )
            for label, (candidate, reference) in pair_specs.items()
        },
        "comparability_note": (
            "All four checkpoints use identical task order, environment seed, and "
            "explicit diffusion initial-noise seeds keyed by task, episode, and "
            "action-chunk indices. Episode outcomes are paired by task and episode."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    flare_rate = metrics["flare"]["overall"]["success_rate"]
    print("========== Four-model fixed-noise comparison ==========")
    for row in comparison["overall_ranking"]:
        change = 100 * (row["success_rate"] - flare_rate)
        print(
            f"{row['model']:<20} {row['successes']}/{row['episodes']} "
            f"= {row['success_rate']:.2%} ({change:+.2f} pp vs FLARE)"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
