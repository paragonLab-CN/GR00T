#!/usr/bin/env python3
"""
Split the libero_plus sparse manifest into per-suite manifests.

libero_plus_lerobot concatenates 4 LIBERO suites × 10 base tasks, in order:
    task_index 0-9   -> libero_spatial
    task_index 10-19 -> libero_object
    task_index 20-29 -> libero_goal
    task_index 30-39 -> libero_10

Each suite's candidate windows must be scored with that suite's
GR00T-N1.7-LIBERO checkpoint (the teacher proposer), so the manifest is
split by task_index and each split is fed to a separate candidate-generation
pass.

Usage:
    python FLARE-VLA/scripts/libero_plus/03_split_manifest_by_suite.py \
        --manifest FLARE-VLA/outputs/libero_plus/manifests/spatial_sparse_manifest.jsonl \
        --output-dir FLARE-VLA/outputs/libero_plus/manifests/per_suite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUITE_BY_BLOCK = {
    0: "libero_spatial",
    1: "libero_object",
    2: "libero_goal",
    3: "libero_10",
}


def suite_of(task_index: int) -> str:
    return SUITE_BY_BLOCK[task_index // 10]


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
        "--output-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/manifests/per_suite"
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    handles = {
        suite: (args.output_dir / f"manifest_{suite}.jsonl").open("w")
        for suite in SUITE_BY_BLOCK.values()
    }
    counts = {suite: 0 for suite in SUITE_BY_BLOCK.values()}
    total = 0

    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            suite = suite_of(int(rec["task_index"]))
            # correct the hardcoded suite label left by the spatial manifest
            # builder (it always wrote "libero_spatial")
            rec["suite"] = suite
            rec["gt_latent_key"] = suite
            # candidate generation requires sample_index contiguous 0..N-1 per
            # manifest, but sample_id must stay globally unique (it names the
            # .pt files, which are merged across suites for scoring).
            rec["sample_index"] = counts[suite]
            handles[suite].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[suite] += 1
            total += 1

    for suite, h in handles.items():
        h.close()
        print(f"  {suite:<16} {counts[suite]:>8} windows -> manifest_{suite}.jsonl")
    print(f"  {'TOTAL':<16} {total:>8} windows")


if __name__ == "__main__":
    main()
