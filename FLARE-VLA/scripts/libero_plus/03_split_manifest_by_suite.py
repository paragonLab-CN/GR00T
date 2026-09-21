#!/usr/bin/env python3
"""Split a LIBERO-Plus sparse manifest by the official source suite.

LIBERO-Plus task indices are interleaved, so blocks 0-9, 10-19, and so on
cannot be used as suite membership. This script derives the mapping from the
canonical (unperturbed) BDDL files and validates every task and manifest row
before replacing the output files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAAC_ROOT = PROJECT_ROOT.parent
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "libero_plus"
    / "manifests"
    / "libero_plus_sparse_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_MANIFEST.parent / "per_suite"
DEFAULT_TASKS_JSONL = (
    ISAAC_ROOT
    / "examples"
    / "LIBERO"
    / "libero_plus_lerobot"
    / "meta"
    / "tasks.jsonl"
)
DEFAULT_BDDL_ROOT = (
    ISAAC_ROOT
    / "external_dependencies"
    / "LIBERO-plus"
    / "libero"
    / "libero"
    / "bddl_files"
)


def normalize_task_name(value: str) -> str:
    """Make task text comparable with a BDDL filename stem."""
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Expected a JSON object in {path}:{line_number}"
                )
            yield line_number, row


def require_int(value: Any, field: str, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"{location}: {field} must be an integer, got {value!r}"
        )
    return value


def load_tasks(path: Path, expected_count: int) -> dict[int, str]:
    tasks: dict[int, str] = {}
    normalized_to_index: dict[str, int] = {}

    for line_number, row in iter_jsonl(path):
        location = f"{path}:{line_number}"
        task_index = require_int(row.get("task_index"), "task_index", location)
        task_text = row.get("task")
        if not isinstance(task_text, str) or not task_text.strip():
            raise RuntimeError(f"{location}: task must be a non-empty string")
        if task_index in tasks:
            raise RuntimeError(
                f"{location}: duplicate task_index {task_index}"
            )

        normalized = normalize_task_name(task_text)
        if normalized in normalized_to_index:
            other_index = normalized_to_index[normalized]
            raise RuntimeError(
                f"{location}: task text duplicates task_index {other_index}"
            )
        tasks[task_index] = task_text
        normalized_to_index[normalized] = task_index

    if expected_count > 0:
        expected_indices = set(range(expected_count))
        actual_indices = set(tasks)
        if actual_indices != expected_indices:
            raise RuntimeError(
                "tasks.jsonl has unexpected indices: "
                f"missing={sorted(expected_indices - actual_indices)}, "
                f"extra={sorted(actual_indices - expected_indices)}"
            )
    elif not tasks:
        raise RuntimeError(f"No tasks found in {path}")

    return tasks


def derive_official_mapping(
    tasks: dict[int, str],
    bddl_root: Path,
    expected_tasks_per_suite: int,
) -> tuple[dict[int, str], dict[int, str]]:
    """Match each task to exactly one canonical BDDL filename."""
    files_by_suite: dict[str, list[Path]] = {}
    for suite in SUITES:
        suite_dir = bddl_root / suite
        if not suite_dir.is_dir():
            raise FileNotFoundError(suite_dir)
        files = sorted(suite_dir.glob("*.bddl"))
        if not files:
            raise RuntimeError(f"No BDDL files found in {suite_dir}")
        files_by_suite[suite] = files

    task_to_suite: dict[int, str] = {}
    task_to_bddl: dict[int, str] = {}
    for task_index, task_text in sorted(tasks.items()):
        task_name = normalize_task_name(task_text)
        matches: list[tuple[str, Path]] = []
        for suite, paths in files_by_suite.items():
            for bddl_path in paths:
                stem = normalize_task_name(bddl_path.stem)
                if stem == task_name or stem.endswith(f"_{task_name}"):
                    matches.append((suite, bddl_path))

        if len(matches) != 1:
            rendered = [
                f"{suite}:{path.name}" for suite, path in matches
            ]
            raise RuntimeError(
                f"task_index {task_index} ({task_text!r}) must match exactly "
                f"one canonical BDDL file; found {len(matches)}: {rendered}"
            )

        suite, bddl_path = matches[0]
        task_to_suite[task_index] = suite
        task_to_bddl[task_index] = str(bddl_path.resolve())

    suite_counts = Counter(task_to_suite.values())
    if set(suite_counts) != set(SUITES):
        raise RuntimeError(
            f"Not all four official suites were matched: {dict(suite_counts)}"
        )
    if expected_tasks_per_suite > 0:
        invalid = {
            suite: suite_counts[suite]
            for suite in SUITES
            if suite_counts[suite] != expected_tasks_per_suite
        }
        if invalid:
            raise RuntimeError(
                f"Expected {expected_tasks_per_suite} tasks per suite, "
                f"got {invalid}"
            )

    return task_to_suite, task_to_bddl


def legacy_block_suite(task_index: int) -> str | None:
    block = task_index // 10
    return SUITES[block] if 0 <= block < len(SUITES) else None


def split_manifest(
    manifest: Path,
    output_dir: Path,
    tasks_jsonl: Path,
    bddl_root: Path,
    tasks: dict[int, str],
    task_to_suite: dict[int, str],
    task_to_bddl: dict[int, str],
) -> dict[str, Any]:
    """Validate, split, and atomically replace all per-suite outputs."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    suite_windows = Counter({suite: 0 for suite in SUITES})
    task_windows: Counter[int] = Counter()
    corrected_suites = 0
    corrected_latent_keys = 0
    seen_sample_ids: set[str] = set()
    total_windows = 0

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-",
        dir=output_dir.parent,
    ) as staging_name:
        staging_dir = Path(staging_name)
        with ExitStack() as stack:
            handles = {
                suite: stack.enter_context(
                    (
                        staging_dir / f"manifest_{suite}.jsonl"
                    ).open("w", encoding="utf-8")
                )
                for suite in SUITES
            }

            for line_number, row in iter_jsonl(manifest):
                location = f"{manifest}:{line_number}"
                source_index = require_int(
                    row.get("sample_index"), "sample_index", location
                )
                if source_index != total_windows:
                    raise RuntimeError(
                        f"{location}: non-contiguous sample_index; expected "
                        f"{total_windows}, got {source_index}"
                    )

                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise RuntimeError(
                        f"{location}: sample_id must be a non-empty string"
                    )
                if sample_id in seen_sample_ids:
                    raise RuntimeError(
                        f"{location}: duplicate sample_id {sample_id!r}"
                    )
                seen_sample_ids.add(sample_id)

                task_index = require_int(
                    row.get("task_index"), "task_index", location
                )
                if task_index not in task_to_suite:
                    raise RuntimeError(
                        f"{location}: unknown task_index {task_index}"
                    )

                task_text = row.get("task_text")
                if not isinstance(task_text, str):
                    raise RuntimeError(
                        f"{location}: task_text must be a string"
                    )
                if (
                    normalize_task_name(task_text)
                    != normalize_task_name(tasks[task_index])
                ):
                    raise RuntimeError(
                        f"{location}: task_text does not match task_index "
                        f"{task_index}; manifest={task_text!r}, "
                        f"metadata={tasks[task_index]!r}"
                    )

                suite = task_to_suite[task_index]
                corrected_suites += row.get("suite") != suite
                corrected_latent_keys += row.get("gt_latent_key") != suite

                output_row = dict(row)
                output_row["suite"] = suite
                output_row["gt_latent_key"] = suite
                output_row["sample_index"] = suite_windows[suite]
                handles[suite].write(
                    json.dumps(output_row, ensure_ascii=False) + "\n"
                )

                suite_windows[suite] += 1
                task_windows[task_index] += 1
                total_windows += 1

        if total_windows == 0:
            raise RuntimeError(f"No records found in {manifest}")
        missing_tasks = sorted(set(tasks) - set(task_windows))
        if missing_tasks:
            raise RuntimeError(
                "Source manifest does not cover every LIBERO-Plus task; "
                f"missing task indices: {missing_tasks}"
            )
        if sum(suite_windows.values()) != total_windows:
            raise AssertionError(
                "Internal error: suite window counts do not sum to total"
            )

        legacy_mismatches = [
            index
            for index, suite in sorted(task_to_suite.items())
            if legacy_block_suite(index) != suite
        ]
        summary: dict[str, Any] = {
            "format": "libero_plus_per_suite_manifest_v2",
            "mapping_source": "exact canonical BDDL filename match",
            "input_manifest": str(manifest.resolve()),
            "tasks_jsonl": str(tasks_jsonl.resolve()),
            "bddl_root": str(bddl_root.resolve()),
            "output_dir": str(output_dir.resolve()),
            "total_tasks": len(tasks),
            "total_windows": total_windows,
            "corrected_input_suite_labels": corrected_suites,
            "corrected_input_gt_latent_keys": corrected_latent_keys,
            "legacy_block_mismatched_task_count": len(
                legacy_mismatches
            ),
            "legacy_block_mismatched_task_indices": legacy_mismatches,
            "suites": {
                suite: {
                    "manifest": f"manifest_{suite}.jsonl",
                    "task_indices": sorted(
                        index
                        for index, mapped_suite in task_to_suite.items()
                        if mapped_suite == suite
                    ),
                    "task_count": sum(
                        mapped_suite == suite
                        for mapped_suite in task_to_suite.values()
                    ),
                    "window_count": suite_windows[suite],
                }
                for suite in SUITES
            },
            "tasks": [
                {
                    "task_index": index,
                    "task_text": tasks[index],
                    "suite": task_to_suite[index],
                    "canonical_bddl": task_to_bddl[index],
                    "window_count": task_windows[index],
                    "legacy_block_suite": legacy_block_suite(index),
                    "legacy_block_mismatch": index in legacy_mismatches,
                }
                for index in sorted(tasks)
            ],
        }
        (staging_dir / "suite_split_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        names = [f"manifest_{suite}.jsonl" for suite in SUITES]
        names.append("suite_split_summary.json")
        for name in names:
            os.replace(staging_dir / name, output_dir / name)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a LIBERO-Plus sparse manifest using official canonical "
            "BDDL suite membership."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default=DEFAULT_TASKS_JSONL,
    )
    parser.add_argument(
        "--bddl-root",
        type=Path,
        default=DEFAULT_BDDL_ROOT,
    )
    parser.add_argument(
        "--expected-task-count",
        type=int,
        default=40,
        help="Expected contiguous task count; 0 disables (default: 40).",
    )
    parser.add_argument(
        "--expected-tasks-per-suite",
        type=int,
        default=10,
        help="Expected tasks in each suite; 0 disables (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_task_count < 0:
        raise ValueError("expected-task-count must be >= 0")
    if args.expected_tasks_per_suite < 0:
        raise ValueError("expected-tasks-per-suite must be >= 0")

    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tasks_jsonl = args.tasks_jsonl.expanduser().resolve()
    bddl_root = args.bddl_root.expanduser().resolve()

    tasks = load_tasks(tasks_jsonl, args.expected_task_count)
    task_to_suite, task_to_bddl = derive_official_mapping(
        tasks,
        bddl_root,
        args.expected_tasks_per_suite,
    )
    summary = split_manifest(
        manifest,
        output_dir,
        tasks_jsonl,
        bddl_root,
        tasks,
        task_to_suite,
        task_to_bddl,
    )

    print("Official task-to-suite mapping validated:")
    for suite in SUITES:
        suite_summary = summary["suites"][suite]
        print(
            f"  {suite:<16} tasks={suite_summary['task_indices']} "
            f"windows={suite_summary['window_count']:,}"
        )
    print(
        "Legacy contiguous-block mapping would misassign "
        f"{summary['legacy_block_mismatched_task_count']} / "
        f"{summary['total_tasks']} tasks."
    )
    print(
        f"Wrote {summary['total_windows']:,} windows to "
        f"{summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
