#!/usr/bin/env python3
"""Validate and merge per-suite LIBERO-Plus candidate files.

Per-suite generation necessarily records a suite-local sample_index and the
per-suite manifest SHA-256. Scoring uses the full manifest, so merged copies
are rewritten with the full-manifest SHA and global sample_index. Source
candidate files remain unchanged and can still be used for generation resume.

Every manifest, summary, checkpoint-suite association, and candidate set is
validated before a PASS summary is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import torch


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CANDIDATE_FORMAT = "flare_groot_manifest_candidates_v1"
FULL_MANIFEST_FORMAT = "libero_plus_sparse_manifest_v1"
PER_SUITE_MANIFEST_FORMAT = "libero_plus_per_suite_manifest_v2"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAAC_ROOT = PROJECT_ROOT.parent
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "libero_plus"
    / "manifests"
    / "libero_plus_sparse_manifest.jsonl"
)
DEFAULT_PER_SUITE_MANIFEST_DIR = DEFAULT_MANIFEST.parent / "per_suite"
DEFAULT_SUITE_DIR = (
    ISAAC_ROOT / "outputs" / "libero_plus" / "sparse_candidates"
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a dictionary in {path}")
    return value


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
                    f"Expected JSON object in {path}:{line_number}"
                )
            yield line_number, row


def load_full_manifest(
    path: Path,
) -> tuple[dict[str, tuple[int, str]], Counter[str]]:
    sample_map: dict[str, tuple[int, str]] = {}
    suite_counts: Counter[str] = Counter()

    for expected_index, (line_number, row) in enumerate(iter_jsonl(path)):
        location = f"{path}:{line_number}"
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError(
                f"{location}: sample_id must be a non-empty string"
            )
        if sample_id in sample_map:
            raise RuntimeError(f"{location}: duplicate sample_id {sample_id}")

        sample_index = row.get("sample_index")
        if sample_index != expected_index:
            raise RuntimeError(
                f"{location}: expected sample_index {expected_index}, "
                f"got {sample_index!r}"
            )
        suite = row.get("suite")
        if suite not in SUITES:
            raise RuntimeError(f"{location}: invalid suite {suite!r}")

        sample_map[sample_id] = (expected_index, str(suite))
        suite_counts[str(suite)] += 1

    if not sample_map:
        raise RuntimeError(f"Full manifest is empty: {path}")
    if set(suite_counts) != set(SUITES):
        raise RuntimeError(
            f"Full manifest does not cover all suites: {dict(suite_counts)}"
        )
    return sample_map, suite_counts


def load_suite_manifest(
    path: Path,
    suite: str,
    full_sample_map: dict[str, tuple[int, str]],
    expected_count: int,
) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    seen: set[str] = set()

    for expected_local_index, (line_number, row) in enumerate(
        iter_jsonl(path)
    ):
        location = f"{path}:{line_number}"
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError(
                f"{location}: sample_id must be a non-empty string"
            )
        if sample_id in seen:
            raise RuntimeError(f"{location}: duplicate sample_id {sample_id}")
        seen.add(sample_id)

        if row.get("sample_index") != expected_local_index:
            raise RuntimeError(
                f"{location}: expected local sample_index "
                f"{expected_local_index}, got {row.get('sample_index')!r}"
            )
        if row.get("suite") != suite:
            raise RuntimeError(
                f"{location}: suite={row.get('suite')!r}, expected {suite!r}"
            )
        if row.get("gt_latent_key") != suite:
            raise RuntimeError(
                f"{location}: gt_latent_key="
                f"{row.get('gt_latent_key')!r}, expected {suite!r}"
            )
        if sample_id not in full_sample_map:
            raise RuntimeError(
                f"{location}: sample_id is absent from the full manifest"
            )

        global_index, full_suite = full_sample_map[sample_id]
        if full_suite != suite:
            raise RuntimeError(
                f"{location}: full manifest suite={full_suite!r}, "
                f"expected {suite!r}"
            )
        rows.append((sample_id, expected_local_index, global_index))

    if len(rows) != expected_count:
        raise RuntimeError(
            f"{path}: found {len(rows)} rows, expected {expected_count}"
        )
    return rows


def validate_suite_summary(
    summary_path: Path,
    suite_manifest: Path,
    suite: str,
    expected_count: int,
) -> tuple[dict[str, Any], str]:
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise RuntimeError(f"Expected JSON object in {summary_path}")
    if summary.get("status") != "PASS":
        raise RuntimeError(
            f"Suite {suite} generation status is "
            f"{summary.get('status')!r}, expected 'PASS'"
        )
    if summary.get("expected_suite") != suite:
        raise RuntimeError(
            f"Suite {suite} summary has expected_suite="
            f"{summary.get('expected_suite')!r}"
        )
    if Path(str(summary.get("model_path", ""))).name != suite:
        raise RuntimeError(
            f"Suite {suite} summary points to the wrong model: "
            f"{summary.get('model_path')!r}"
        )
    if Path(str(summary.get("manifest", ""))).name != suite_manifest.name:
        raise RuntimeError(
            f"Suite {suite} summary points to the wrong manifest: "
            f"{summary.get('manifest')!r}"
        )
    if summary.get("manifest_format") != PER_SUITE_MANIFEST_FORMAT:
        raise RuntimeError(
            f"Suite {suite} summary has manifest_format="
            f"{summary.get('manifest_format')!r}"
        )
    if int(summary.get("action_horizon", -1)) != 8:
        raise RuntimeError(
            f"Suite {suite} summary has an unexpected action horizon"
        )

    suite_manifest_sha = sha256_file(suite_manifest)
    if summary.get("manifest_sha256") != suite_manifest_sha:
        raise RuntimeError(
            f"Suite {suite} summary manifest SHA-256 mismatch"
        )

    counters = summary.get("counters", {})
    selected = int(counters.get("selected", -1))
    generated = int(counters.get("generated", -1))
    skipped = int(counters.get("skipped_valid", -1))
    failed = int(counters.get("failed", -1))
    if selected != expected_count:
        raise RuntimeError(
            f"Suite {suite}: selected={selected}, expected={expected_count}"
        )
    if failed != 0 or generated + skipped != expected_count:
        raise RuntimeError(
            f"Suite {suite} incomplete: generated={generated}, "
            f"skipped_valid={skipped}, failed={failed}, "
            f"expected={expected_count}"
        )
    return summary, suite_manifest_sha


def merged_candidate_is_valid(
    path: Path,
    sample_id: str,
    global_index: int,
    full_manifest_sha: str,
    suite: str,
    source_candidate_sha: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_torch(path)
        return (
            payload.get("format") == CANDIDATE_FORMAT
            and payload.get("sample_id") == sample_id
            and payload.get("sample_index") == global_index
            and payload.get("manifest_sha256") == full_manifest_sha
            and payload.get("manifest_format") == FULL_MANIFEST_FORMAT
            and payload.get("suite") == suite
            and payload.get("source_candidate_sha256")
            == source_candidate_sha
        )
    except Exception:
        return False


def merge_one(
    args: tuple[
        str,
        str,
        str,
        int,
        int,
        str,
        str,
        str,
        bool,
    ],
) -> tuple[str, str]:
    (
        source_name,
        destination_name,
        sample_id,
        local_index,
        global_index,
        suite_manifest_sha,
        full_manifest_sha,
        suite,
        overwrite,
    ) = args
    source = Path(source_name)
    destination = Path(destination_name)

    try:
        if not source.is_file():
            raise FileNotFoundError(source)
        source_candidate_sha = sha256_file(source)
        if (
            not overwrite
            and merged_candidate_is_valid(
                destination,
                sample_id,
                global_index,
                full_manifest_sha,
                suite,
                source_candidate_sha,
            )
        ):
            return "skipped_valid", ""

        payload = load_torch(source)
        if payload.get("format") != CANDIDATE_FORMAT:
            raise RuntimeError(
                f"Unexpected candidate format {payload.get('format')!r}"
            )
        if payload.get("sample_id") != sample_id:
            raise RuntimeError(
                f"sample_id={payload.get('sample_id')!r}, "
                f"expected {sample_id!r}"
            )
        if payload.get("sample_index") != local_index:
            raise RuntimeError(
                f"local sample_index={payload.get('sample_index')!r}, "
                f"expected {local_index}"
            )
        if payload.get("manifest_sha256") != suite_manifest_sha:
            raise RuntimeError("per-suite manifest SHA-256 mismatch")
        if payload.get("suite") != suite:
            raise RuntimeError(
                f"suite={payload.get('suite')!r}, expected {suite!r}"
            )
        if Path(str(payload.get("model_path", ""))).name != suite:
            raise RuntimeError(
                f"candidate model_path={payload.get('model_path')!r}, "
                f"expected suite checkpoint {suite!r}"
            )
        if payload.get("gt_latent_key") != suite:
            raise RuntimeError(
                f"gt_latent_key={payload.get('gt_latent_key')!r}, "
                f"expected {suite!r}"
            )

        payload["per_suite_manifest_sha256"] = suite_manifest_sha
        payload["per_suite_sample_index"] = local_index
        payload["source_candidate_sha256"] = source_candidate_sha
        payload["manifest_sha256"] = full_manifest_sha
        payload["manifest_format"] = FULL_MANIFEST_FORMAT
        payload["sample_index"] = global_index

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(
            destination.suffix + f".tmp.{os.getpid()}"
        )
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        return "written", ""
    except Exception as exc:
        return "failed", f"{source.name}: {exc}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--per-suite-manifest-dir",
        type=Path,
        default=DEFAULT_PER_SUITE_MANIFEST_DIR,
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=DEFAULT_SUITE_DIR,
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--overwrite-merged",
        action="store_true",
        help="Rewrite valid merged candidate files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    manifest = args.manifest.expanduser().resolve()
    per_suite_manifest_dir = (
        args.per_suite_manifest_dir.expanduser().resolve()
    )
    suite_dir = args.suite_dir.expanduser().resolve()
    merged_dir = suite_dir / "candidates"
    merged_dir.mkdir(parents=True, exist_ok=True)

    full_manifest_sha = sha256_file(manifest)
    full_sample_map, full_suite_counts = load_full_manifest(manifest)
    total_expected = len(full_sample_map)

    source_counters: Counter[str] = Counter()
    merge_counters: Counter[str] = Counter()
    per_suite_completed: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for suite in SUITES:
            expected_count = full_suite_counts[suite]
            suite_manifest = (
                per_suite_manifest_dir / f"manifest_{suite}.jsonl"
            )
            suite_rows = load_suite_manifest(
                suite_manifest,
                suite,
                full_sample_map,
                expected_count,
            )
            summary_path = (
                suite_dir / suite / "candidate_generation_summary.json"
            )
            summary, suite_manifest_sha = validate_suite_summary(
                summary_path,
                suite_manifest,
                suite,
                expected_count,
            )
            counters = summary["counters"]
            source_counters["selected"] += int(counters["selected"])
            source_counters["generated"] += int(counters["generated"])
            source_counters["skipped_valid"] += int(
                counters["skipped_valid"]
            )
            source_counters["regenerated_invalid"] += int(
                counters.get("regenerated_invalid", 0)
            )
            source_counters["failed"] += int(counters["failed"])

            source_candidate_dir = suite_dir / suite / "candidates"
            actual_file_count = sum(
                1 for _ in source_candidate_dir.glob("*.pt")
            )
            if actual_file_count != expected_count:
                raise RuntimeError(
                    f"Suite {suite}: candidate file count="
                    f"{actual_file_count}, expected={expected_count}"
                )

            jobs = [
                (
                    str(source_candidate_dir / f"{sample_id}.pt"),
                    str(merged_dir / f"{sample_id}.pt"),
                    sample_id,
                    local_index,
                    global_index,
                    suite_manifest_sha,
                    full_manifest_sha,
                    suite,
                    bool(args.overwrite_merged),
                )
                for sample_id, local_index, global_index in suite_rows
            ]
            processed = 0
            errors: list[str] = []
            for status, message in pool.map(
                merge_one,
                jobs,
                chunksize=64,
            ):
                merge_counters[status] += 1
                processed += 1
                if message and len(errors) < 20:
                    errors.append(message)
                if processed % 10_000 == 0:
                    print(
                        f"  {suite}: merged {processed:,}/"
                        f"{expected_count:,}"
                    )
            if merge_counters["failed"]:
                raise RuntimeError(
                    "Candidate merge failed; first errors:\n  "
                    + "\n  ".join(errors)
                )

            per_suite_completed[suite] = expected_count
            print(
                f"  {suite:<16} completed={expected_count:>7,}"
            )

    if source_counters["selected"] != total_expected:
        raise RuntimeError(
            f"Per-suite selected total={source_counters['selected']}, "
            f"full manifest total={total_expected}"
        )
    if (
        source_counters["generated"]
        + source_counters["skipped_valid"]
        != total_expected
    ):
        raise RuntimeError("Per-suite generation summaries are incomplete")

    merged_file_count = sum(1 for _ in merged_dir.glob("*.pt"))
    if merged_file_count != total_expected:
        raise RuntimeError(
            f"Merged candidate file count={merged_file_count}, "
            f"expected={total_expected}; remove stale files explicitly "
            "before retrying"
        )

    merged_summary = {
        "format": "flare_candidate_generation_summary_v1",
        "status": "PASS",
        "manifest": str(manifest),
        "manifest_format": FULL_MANIFEST_FORMAT,
        "manifest_sha256": full_manifest_sha,
        "output_dir": str(merged_dir),
        "note": (
            "Merged and provenance-normalized per-suite LIBERO-Plus "
            "candidate generation."
        ),
        "per_suite_completed": per_suite_completed,
        "source_generation_counters": dict(source_counters),
        "merge_counters": dict(merge_counters),
        "counters": {
            "selected": total_expected,
            "generated": int(source_counters["generated"]),
            "skipped_valid": int(source_counters["skipped_valid"]),
            "regenerated_invalid": int(
                source_counters["regenerated_invalid"]
            ),
            "failed": 0,
        },
    }
    summary_path = suite_dir / "candidate_generation_summary.json"
    atomic_write_json(summary_path, merged_summary)

    print(f"  TOTAL completed = {total_expected:,}")
    print(f"Merged candidates -> {merged_dir}")
    print(f"Merged summary    -> {summary_path}")


if __name__ == "__main__":
    main()
