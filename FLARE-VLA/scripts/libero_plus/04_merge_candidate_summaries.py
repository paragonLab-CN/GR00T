#!/usr/bin/env python3
"""
Merge per-suite candidate dirs + summaries into a single candidate set that
satisfies 05_score_libero_plus.py's validate_candidate_summary() contract.

Inputs (produced by the per-suite candidate generation loop):
    sparse_candidates/<suite>/candidates/*.pt
    sparse_candidates/<suite>/candidate_generation_summary.json

Outputs:
    sparse_candidates/candidates/*.pt              (all suites, unique sample_ids)
    sparse_candidates/candidate_generation_summary.json  (manifest_sha256 = FULL manifest)

Usage:
    python FLARE-VLA/scripts/libero_plus/04_merge_candidate_summaries.py \
        --manifest FLARE-VLA/outputs/libero_plus/manifests/spatial_sparse_manifest.jsonl \
        --suite-dir Isaac-GR00T/outputs/libero_plus/sparse_candidates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/manifests/"
            "spatial_sparse_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/sparse_candidates"
        ),
    )
    args = parser.parse_args()

    manifest_sha = sha256_file(args.manifest)

    merged_dir = args.suite_dir / "candidates"
    merged_dir.mkdir(parents=True, exist_ok=True)

    total_selected = 0
    total_generated = 0
    per_suite_counts = {}

    for suite in SUITES:
        suite_candidates = args.suite_dir / suite / "candidates"
        suite_summary_path = args.suite_dir / suite / "candidate_generation_summary.json"
        if not suite_summary_path.is_file():
            raise FileNotFoundError(f"Missing suite summary: {suite_summary_path}")
        summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise RuntimeError(f"Suite {suite} did not PASS: {summary.get('status')}")
        counters = summary["counters"]
        total_selected += int(counters["selected"])
        total_generated += int(counters["generated"])
        per_suite_counts[suite] = int(counters["generated"])

        n_copied = 0
        for pt in suite_candidates.glob("*.pt"):
            shutil.copy2(pt, merged_dir / pt.name)
            n_copied += 1
        if n_copied != int(counters["generated"]):
            raise RuntimeError(
                f"Suite {suite}: {n_copied} .pt files vs generated={counters['generated']}"
            )
        print(f"  {suite:<16} generated={counters['generated']:>7}")

    merged_summary = {
        "format": "flare_candidate_generation_summary_v1",
        "status": "PASS",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "output_dir": str(merged_dir),
        "note": "Merged per-suite candidate generation (spatial/object/goal/libero_10).",
        "per_suite_generated": per_suite_counts,
        "counters": {
            "selected": total_selected,
            "generated": total_generated,
            "skipped_valid": 0,
            "regenerated_invalid": 0,
            "failed": 0,
        },
    }
    out_summary = args.suite_dir / "candidate_generation_summary.json"
    out_summary.write_text(
        json.dumps(merged_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"  TOTAL generated = {total_generated}")
    print(f"Merged candidates -> {merged_dir}")
    print(f"Merged summary    -> {out_summary}")


if __name__ == "__main__":
    main()
