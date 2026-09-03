#!/usr/bin/env python3
"""Summarize the fixed 112-task image-augmentation evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


APPEARANCE_CATEGORIES = {
    "Background Textures",
    "Light Conditions",
    "Sensor Noise",
}
HISTORICAL_BASELINE = {
    "overall": {"successes": 879, "episodes": 1120},
    "appearance": {"successes": 418, "episodes": 480},
    "Background Textures": {"rate": 0.88125},
    "Light Conditions": {"rate": 0.8625},
    "Sensor Noise": {"rate": 0.86875},
}


def _noise_subtype(task_name: str) -> str:
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


def _aggregate(rows):
    successes = sum(int(row["successes"]) for row in rows)
    episodes = sum(int(row["episodes"]) for row in rows)
    return {
        "tasks": len(rows),
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = summary["tasks"]
    if len(rows) != 112 or summary["episodes_per_task"] != 10:
        raise RuntimeError(
            "Expected the fixed 112-task x 10-episode protocol, got "
            f"tasks={len(rows)}, episodes_per_task={summary['episodes_per_task']}"
        )

    appearance_rows = [row for row in rows if row["category"] in APPEARANCE_CATEGORIES]
    if len(appearance_rows) != 48:
        raise RuntimeError(f"Expected 48 Appearance tasks, got {len(appearance_rows)}")

    sensor_by_subtype = defaultdict(list)
    for row in rows:
        if row["category"] == "Sensor Noise":
            sensor_by_subtype[_noise_subtype(row["task"])].append(row)
    expected_subtypes = {"motion", "gaussian", "zoom", "fog", "glass"}
    if set(sensor_by_subtype) != expected_subtypes:
        raise RuntimeError(
            f"Sensor Noise subtype coverage mismatch: {sorted(sensor_by_subtype)}"
        )

    overall = _aggregate(rows)
    appearance = _aggregate(appearance_rows)
    per_suite = {
        suite: _aggregate([row for row in rows if row["suite"] == suite])
        for suite in sorted({row["suite"] for row in rows})
    }
    per_category = {
        category: _aggregate([row for row in rows if row["category"] == category])
        for category in sorted({row["category"] for row in rows})
    }
    sensor_subtypes = {
        subtype: _aggregate(sensor_by_subtype[subtype])
        for subtype in ("motion", "gaussian", "zoom", "fog", "glass")
    }

    comparison = {
        "model_tag": summary["model_tag"],
        "protocol": {
            "tasks": 112,
            "episodes_per_task": 10,
            "total_episodes": 1120,
            "appearance_definition": sorted(APPEARANCE_CATEGORIES),
            "appearance_tasks": 48,
            "note": "Historical stratified sample, not all 10,030 LIBERO-Plus tasks.",
        },
        "new_model": {
            "overall": overall,
            "appearance": appearance,
            "per_suite": per_suite,
            "per_category": per_category,
            "sensor_noise_subtypes": sensor_subtypes,
        },
        "historical_baseline": HISTORICAL_BASELINE,
        "percentage_point_change": {
            "overall": 100
            * (
                overall["success_rate"]
                - HISTORICAL_BASELINE["overall"]["successes"]
                / HISTORICAL_BASELINE["overall"]["episodes"]
            ),
            "appearance": 100
            * (
                appearance["success_rate"]
                - HISTORICAL_BASELINE["appearance"]["successes"]
                / HISTORICAL_BASELINE["appearance"]["episodes"]
            ),
            **{
                category: 100
                * (
                    per_category[category]["success_rate"]
                    - HISTORICAL_BASELINE[category]["rate"]
                )
                for category in APPEARANCE_CATEGORIES
            },
        },
        "interpretation_caveat": (
            "The baseline was not rerun and historical server-side diffusion noise was "
            "not fixed. Differences are descriptive percentage-point changes, not paired "
            "or statistically significant effects."
        ),
    }

    output = args.output or args.summary.with_name("image_aug_comparison.json")
    output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    print("========== Image augmentation evaluation comparison ==========")
    print(
        f"Overall    : {overall['successes']}/{overall['episodes']} "
        f"= {overall['success_rate']:.2%} "
        f"({comparison['percentage_point_change']['overall']:+.2f} pp)"
    )
    print(
        f"Appearance : {appearance['successes']}/{appearance['episodes']} "
        f"= {appearance['success_rate']:.2%} "
        f"({comparison['percentage_point_change']['appearance']:+.2f} pp)"
    )
    print("Sensor Noise subtypes:")
    for subtype, result in sensor_subtypes.items():
        print(
            f"  {subtype:<8} {result['successes']}/{result['episodes']} "
            f"= {result['success_rate']:.2%}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
