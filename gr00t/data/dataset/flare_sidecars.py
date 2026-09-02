# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and validate FLARE-VLA distillation sidecar files."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DISTILLATION_INDEX_FILENAME = "distillation_index.jsonl"
TEACHER_RECORDS_FILENAME = "teacher_records_enriched.pt"
DISTILLATION_SUMMARY_FILENAME = "distillation_summary.json"
DISTILLATION_INDEX_FORMAT = "flare_distillation_index_v1"
_SUPPORTED_SOURCES = {"gt", "teacher"}


@dataclass(frozen=True)
class FlareDistillationSidecars:
    """Resolved FLARE files and the record counts discovered in the index."""

    index_path: Path
    teacher_records_path: Path
    summary_path: Path | None
    total_records: int
    source_counts: dict[str, int]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    return value


def _inspect_index(path: Path) -> tuple[int, dict[str, int]]:
    counts: Counter[str] = Counter()
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"Expected a JSON object at {path}:{line_no}, "
                    f"got {type(row).__name__}"
                )
            if str(row.get("format")) != DISTILLATION_INDEX_FORMAT:
                raise RuntimeError(
                    f"Unsupported FLARE index format at {path}:{line_no}: "
                    f"{row.get('format')!r}"
                )
            if int(row.get("distillation_index", -1)) != total:
                raise RuntimeError(
                    f"Non-contiguous distillation_index at {path}:{line_no}: "
                    f"stored={row.get('distillation_index')!r}, expected={total}"
                )
            source = str(row.get("source"))
            if source not in _SUPPORTED_SOURCES:
                raise RuntimeError(
                    f"Unsupported source at {path}:{line_no}: {source!r}; "
                    f"expected one of {sorted(_SUPPORTED_SOURCES)}"
                )
            counts[source] += 1
            total += 1

    if total == 0:
        raise RuntimeError(f"FLARE distillation index is empty: {path}")
    return total, dict(counts)


def _validate_summary(
    summary_path: Path,
    *,
    total_records: int,
    source_counts: dict[str, int],
) -> None:
    summary = _load_json(summary_path)
    if str(summary.get("status")) != "PASS":
        raise RuntimeError(
            f"FLARE distillation summary status is not PASS: "
            f"{summary.get('status')!r} in {summary_path}"
        )

    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise TypeError(f"Missing protocol object in {summary_path}")

    expected = {
        "total_samples": total_records,
        "gt_samples": source_counts.get("gt", 0),
        "teacher_samples": source_counts.get("teacher", 0),
    }
    for field, actual in expected.items():
        if field not in protocol:
            raise KeyError(f"Summary protocol is missing {field!r}: {summary_path}")
        stored = int(protocol[field])
        if stored != actual:
            raise RuntimeError(
                f"FLARE summary/index count mismatch for {field}: "
                f"summary={stored}, index={actual}"
            )


def resolve_flare_distillation_sidecars(
    *,
    distillation_dataset_dir: str | Path | None,
    distillation_index_path: str | Path | None,
    teacher_records_path: str | Path | None,
) -> FlareDistillationSidecars | None:
    """Resolve FLARE paths from one directory or an explicit file pair.

    Returns ``None`` when no FLARE option is supplied. A directory must contain
    ``distillation_index.jsonl`` and ``teacher_records_enriched.pt``. If a
    ``distillation_summary.json`` is present, its GT/teacher/total counts are
    checked against the index before model initialization begins.
    """

    has_dir = distillation_dataset_dir is not None
    has_index = distillation_index_path is not None
    has_teacher = teacher_records_path is not None

    if has_dir and (has_index or has_teacher):
        raise ValueError(
            "Use either --distillation_dataset_dir or the explicit "
            "--distillation_index_path/--teacher_records_path pair, not both"
        )
    if has_index != has_teacher:
        raise ValueError(
            "FLARE distillation requires both --distillation_index_path and "
            "--teacher_records_path"
        )
    if not has_dir and not has_index:
        return None

    if has_dir:
        dataset_dir = Path(distillation_dataset_dir).expanduser().resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"FLARE distillation directory does not exist: {dataset_dir}")
        index_path = dataset_dir / DISTILLATION_INDEX_FILENAME
        teacher_path = dataset_dir / TEACHER_RECORDS_FILENAME
        candidate_summary = dataset_dir / DISTILLATION_SUMMARY_FILENAME
    else:
        index_path = Path(distillation_index_path).expanduser().resolve()
        teacher_path = Path(teacher_records_path).expanduser().resolve()
        candidate_summary = (
            index_path.parent / DISTILLATION_SUMMARY_FILENAME
            if index_path.parent == teacher_path.parent
            else None
        )

    if not index_path.is_file():
        raise FileNotFoundError(f"FLARE distillation index does not exist: {index_path}")
    if not teacher_path.is_file():
        raise FileNotFoundError(f"FLARE teacher records do not exist: {teacher_path}")

    total_records, source_counts = _inspect_index(index_path)
    summary_path = (
        candidate_summary if candidate_summary is not None and candidate_summary.is_file() else None
    )
    if summary_path is not None:
        _validate_summary(
            summary_path,
            total_records=total_records,
            source_counts=source_counts,
        )

    return FlareDistillationSidecars(
        index_path=index_path,
        teacher_records_path=teacher_path,
        summary_path=summary_path,
        total_records=total_records,
        source_counts=source_counts,
    )
