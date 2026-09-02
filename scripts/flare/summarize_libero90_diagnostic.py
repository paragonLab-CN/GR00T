#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine fixed three-task LIBERO-90 diagnostic summaries."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root

    combined_models: list[dict] = []
    task_names: list[str] = []

    for model_tag in args.models:
        summary_path = root / model_tag / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing diagnostic summary: {summary_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tasks = summary["tasks"]
        current_task_names = [item["task"] for item in tasks]

        if not task_names:
            task_names = current_task_names
        elif current_task_names != task_names:
            raise RuntimeError(
                f"Task order mismatch for {model_tag}: {current_task_names} != {task_names}"
            )

        combined_models.append(summary)

    csv_rows: list[dict[str, object]] = []
    for summary in combined_models:
        row: dict[str, object] = {
            "model_tag": summary["model_tag"],
            "model_path": summary["model_path"],
            "seed": summary["seed"],
            "total_successes": summary["total_successes"],
            "total_episodes": summary["total_episodes"],
            "overall_success_rate": summary["overall_success_rate"],
        }
        for task in summary["tasks"]:
            row[f"{task['task']}__successes"] = task["successes"]
            row[f"{task['task']}__success_rate"] = task["success_rate"]
        csv_rows.append(row)

    combined_json = {
        "protocol": {
            "num_models": len(combined_models),
            "tasks": task_names,
            "same_seed_required": True,
        },
        "models": combined_models,
    }

    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "combined_summary.json"
    csv_path = root / "combined_summary.csv"

    json_path.write_text(
        json.dumps(combined_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n================ Combined diagnostic ================")
    for summary in combined_models:
        print(
            f"{summary['model_tag']:<32} "
            f"{summary['total_successes']:>2}/{summary['total_episodes']:<2} "
            f"rate={summary['overall_success_rate']:.4f}"
        )
        for task in summary["tasks"]:
            print(
                f"  {task['task']}: "
                f"{task['successes']}/{task['episodes']} "
                f"({task['success_rate']:.3f})"
            )
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
