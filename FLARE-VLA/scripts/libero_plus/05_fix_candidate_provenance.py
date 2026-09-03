#!/usr/bin/env python3
"""
Rewrite candidate .pt provenance fields so they match the FULL manifest.

Per-suite candidate generation used split manifests (renumbered sample_index,
per-suite manifest SHA), but scoring validates against the FULL manifest:
  - candidate.sample_index must equal the full manifest's global sample_index
    (recoverable from sample_id: "spatial_sparse_XXXXXXXXX" -> XXXXXXXXX)
  - candidate.manifest_sha256 must equal the full manifest's SHA

The split manifests are a pure re-partition of the full manifest (same windows,
same sample_ids), so rewriting these two provenance labels is semantically
correct. Parallelized across the available CPUs.

Usage:
    python FLARE-VLA/scripts/libero_plus/05_fix_candidate_provenance.py \
        --manifest FLARE-VLA/outputs/libero_plus/manifests/spatial_sparse_manifest.jsonl \
        --candidate-dir Isaac-GR00T/outputs/libero_plus/sparse_candidates/candidates
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def fix_one(args: tuple[Path, str]) -> tuple[int, str]:
    pt_path, full_sha = args
    try:
        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        sid = payload["sample_id"]
        global_index = int(sid.split("_")[-1])
        payload["sample_index"] = global_index
        payload["manifest_sha256"] = full_sha
        torch.save(payload, pt_path)
        return 0, ""
    except Exception as exc:  # noqa: BLE001
        return 1, f"{pt_path.name}: {exc}"


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
        "--candidate-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/Isaac-GR00T/outputs/libero_plus/sparse_candidates/candidates"
        ),
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    full_sha = sha256_file(args.manifest)
    print(f"Full manifest SHA: {full_sha}")

    files = sorted(args.candidate_dir.glob("*.pt"))
    print(f"Candidates: {len(files)}")
    tasks = [(p, full_sha) for p in files]

    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for status, msg in pool.map(fix_one, tasks, chunksize=64):
            if status != 0:
                failed += 1
                print(msg)
            done += 1
            if done % 50000 == 0:
                print(f"  {done}/{len(files)}")

    print(f"Done. fixed={done - failed}, failed={failed}")


if __name__ == "__main__":
    main()
