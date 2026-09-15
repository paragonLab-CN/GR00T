#!/usr/bin/env python3
"""Validate and summarize a paired fixed-noise LIBERO-Plus evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


APPEARANCE_CATEGORIES = {
    "Background Textures",
    "Light Conditions",
    "Sensor Noise",
}


def noise_subtype(task_name: str) -> str:
    match = re.search(r"_noise_(\d+)$", task_name)
    if not match:
        raise ValueError(f"Sensor Noise task lacks a noise id: {task_name}")
    noise_id = int(match.group(1))
    if 1 <= noise_id <= 10:
        return "motion"
    if noise_id <= 20:
        return "gaussian"
    if noise_id <= 30:
        return "zoom"
    if noise_id <= 40:
        return "fog"
    if noise_id <= 50:
        return "glass"
    raise ValueError(f"Unsupported LIBERO-Plus noise id {noise_id}: {task_name}")


def aggregate(rows: list[dict]) -> dict:
    successes = sum(int(row["successes"]) for row in rows)
    episodes = sum(int(row["episodes"]) for row in rows)
    return {
        "tasks": len(rows),
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
    }


def model_metrics(summary: dict) -> dict:
    rows = summary["tasks"]
    appearance = [row for row in rows if row["category"] in APPEARANCE_CATEGORIES]
    sensor_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["category"] == "Sensor Noise":
            sensor_groups[noise_subtype(row["task"])].append(row)
    return {
        "model_tag": summary["model_tag"],
        "overall": aggregate(rows),
        "appearance": aggregate(appearance),
        "per_suite": {
            suite: aggregate([row for row in rows if row["suite"] == suite])
            for suite in sorted({row["suite"] for row in rows})
        },
        "per_category": {
            category: aggregate([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
        "per_difficulty": {
            difficulty: aggregate([row for row in rows if row["difficulty"] == difficulty])
            for difficulty in sorted({row["difficulty"] for row in rows})
        },
        "sensor_noise_subtypes": {
            subtype: aggregate(sensor_groups[subtype])
            for subtype in ("motion", "gaussian", "zoom", "fog", "glass")
        },
    }


def validate_pair(baseline: dict, candidate: dict) -> dict:
    for label, summary in (("baseline", baseline), ("candidate", candidate)):
        rows = summary.get("tasks", [])
        if len(rows) != 112 or summary.get("episodes_per_task") != 10:
            raise RuntimeError(
                f"{label} must use 112 tasks x 10 episodes; got "
                f"tasks={len(rows)}, episodes_per_task={summary.get('episodes_per_task')}"
            )
        protocol = summary.get("diffusion_noise_protocol", {})
        if protocol.get("mode") != "explicit_per_request_seed":
            raise RuntimeError(f"{label} does not record fixed per-request diffusion noise")

    baseline_tasks = [row["task"] for row in baseline["tasks"]]
    candidate_tasks = [row["task"] for row in candidate["tasks"]]
    if baseline_tasks != candidate_tasks:
        raise RuntimeError("Baseline and candidate task order differs")

    baseline_protocol = baseline["diffusion_noise_protocol"]
    candidate_protocol = candidate["diffusion_noise_protocol"]
    if baseline_protocol != candidate_protocol:
        raise RuntimeError(
            "Baseline and candidate diffusion-noise protocols differ: "
            f"{baseline_protocol!r} != {candidate_protocol!r}"
        )

    for index, (baseline_row, candidate_row) in enumerate(
        zip(baseline["tasks"], candidate["tasks"], strict=True)
    ):
        expected_seed = baseline_protocol["master_seed"] + index * baseline_protocol["task_stride"]
        if baseline_row.get("diffusion_noise_base_seed") != expected_seed:
            raise RuntimeError(f"Baseline task {index} has invalid diffusion seed")
        if candidate_row.get("diffusion_noise_base_seed") != expected_seed:
            raise RuntimeError(f"Candidate task {index} has invalid diffusion seed")

    return baseline_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_summary", type=Path)
    parser.add_argument("candidate_summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    protocol = validate_pair(baseline, candidate)
    baseline_metrics = model_metrics(baseline)
    candidate_metrics = model_metrics(candidate)

    category_changes = {
        category: 100
        * (
            candidate_metrics["per_category"][category]["success_rate"]
            - baseline_metrics["per_category"][category]["success_rate"]
        )
        for category in candidate_metrics["per_category"]
    }
    per_task_deltas = [
        int(candidate_row["successes"]) - int(baseline_row["successes"])
        for baseline_row, candidate_row in zip(
            baseline["tasks"], candidate["tasks"], strict=True
        )
    ]

    comparison = {
        "comparison_mode": "paired_fixed_diffusion_noise",
        "protocol": {
            "tasks": 112,
            "episodes_per_task": 10,
            "total_episodes_per_model": 1120,
            "environment_seed": 2026,
            "diffusion_noise": protocol,
            "appearance_definition": sorted(APPEARANCE_CATEGORIES),
            "appearance_tasks": 48,
        },
        "baseline": baseline_metrics,
        "new_model": candidate_metrics,
        "percentage_point_change": {
            "overall": 100
            * (
                candidate_metrics["overall"]["success_rate"]
                - baseline_metrics["overall"]["success_rate"]
            ),
            "appearance": 100
            * (
                candidate_metrics["appearance"]["success_rate"]
                - baseline_metrics["appearance"]["success_rate"]
            ),
            "per_category": category_changes,
        },
        "paired_task_outcomes": {
            "candidate_better": sum(delta > 0 for delta in per_task_deltas),
            "candidate_worse": sum(delta < 0 for delta in per_task_deltas),
            "tied": sum(delta == 0 for delta in per_task_deltas),
        },
        "comparability_note": (
            "Both checkpoints were rerun with identical environment seeds and explicit "
            "diffusion initial-noise seeds keyed by task, episode, and action-chunk indices."
        ),
    }

    output = args.output or args.candidate_summary.with_name("image_aug_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    new_overall = candidate_metrics["overall"]
    base_overall = baseline_metrics["overall"]
    change = comparison["percentage_point_change"]["overall"]
    print("========== Paired fixed-noise comparison ==========")
    print(
        f"Baseline : {base_overall['successes']}/{base_overall['episodes']} "
        f"= {base_overall['success_rate']:.2%}"
    )
    print(
        f"Image aug: {new_overall['successes']}/{new_overall['episodes']} "
        f"= {new_overall['success_rate']:.2%} ({change:+.2f} pp)"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
