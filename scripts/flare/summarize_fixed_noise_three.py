#!/usr/bin/env python3
"""Validate and summarize three paired fixed-noise LIBERO-Plus evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_fixed_noise_pair import APPEARANCE_CATEGORIES, model_metrics, validate_pair


def metric_deltas(candidate: dict, reference: dict) -> dict:
    """Return candidate-minus-reference percentage-point changes."""

    def delta(candidate_metric: dict, reference_metric: dict) -> float:
        return 100 * (
            candidate_metric["success_rate"] - reference_metric["success_rate"]
        )

    return {
        "overall": delta(candidate["overall"], reference["overall"]),
        "appearance": delta(candidate["appearance"], reference["appearance"]),
        "per_suite": {
            key: delta(candidate["per_suite"][key], reference["per_suite"][key])
            for key in candidate["per_suite"]
        },
        "per_category": {
            key: delta(candidate["per_category"][key], reference["per_category"][key])
            for key in candidate["per_category"]
        },
        "per_difficulty": {
            key: delta(
                candidate["per_difficulty"][key],
                reference["per_difficulty"][key],
            )
            for key in candidate["per_difficulty"]
        },
        "sensor_noise_subtypes": {
            key: delta(
                candidate["sensor_noise_subtypes"][key],
                reference["sensor_noise_subtypes"][key],
            )
            for key in candidate["sensor_noise_subtypes"]
        },
    }


def paired_task_outcomes(candidate: dict, reference: dict) -> dict:
    deltas = [
        int(candidate_row["successes"]) - int(reference_row["successes"])
        for reference_row, candidate_row in zip(
            reference["tasks"], candidate["tasks"], strict=True
        )
    ]
    return {
        "candidate_better": sum(delta > 0 for delta in deltas),
        "candidate_worse": sum(delta < 0 for delta in deltas),
        "tied": sum(delta == 0 for delta in deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("flare_summary", type=Path)
    parser.add_argument("image_aug_summary", type=Path)
    parser.add_argument("gt_only_summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = {
        "flare": json.loads(args.flare_summary.read_text(encoding="utf-8")),
        "image_aug": json.loads(args.image_aug_summary.read_text(encoding="utf-8")),
        "gt_only": json.loads(args.gt_only_summary.read_text(encoding="utf-8")),
    }
    protocol = validate_pair(summaries["flare"], summaries["image_aug"])
    gt_protocol = validate_pair(summaries["flare"], summaries["gt_only"])
    if protocol != gt_protocol:
        raise RuntimeError("The three diffusion-noise protocols differ")

    metrics = {name: model_metrics(summary) for name, summary in summaries.items()}
    pair_specs = {
        "image_aug_minus_flare": ("image_aug", "flare"),
        "gt_only_minus_flare": ("gt_only", "flare"),
        "image_aug_minus_gt_only": ("image_aug", "gt_only"),
    }
    comparison = {
        "comparison_mode": "three_model_paired_fixed_diffusion_noise",
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
            label: paired_task_outcomes(
                summaries[candidate], summaries[reference]
            )
            for label, (candidate, reference) in pair_specs.items()
        },
        "comparability_note": (
            "All three checkpoints used identical task order, environment seed, and "
            "explicit diffusion initial-noise seeds keyed by task, episode, and "
            "action-chunk indices."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("========== Three-model fixed-noise comparison ==========")
    flare_rate = metrics["flare"]["overall"]["success_rate"]
    for row in comparison["overall_ranking"]:
        change = 100 * (row["success_rate"] - flare_rate)
        print(
            f"{row['model']:<10} {row['successes']}/{row['episodes']} "
            f"= {row['success_rate']:.2%} ({change:+.2f} pp vs FLARE)"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
