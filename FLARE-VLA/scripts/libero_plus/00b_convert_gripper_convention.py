#!/usr/bin/env python3
"""
Convert libero_plus_lerobot action gripper from raw LIBERO convention to
LeRobot convention, in-place.

Why: Sylvest/libero_plus_lerobot stores actions with the raw LIBERO gripper
convention (-1 = open, +1 = close), while every vanilla LIBERO LeRobot
dataset used by GR00T (libero_spatial/object/goal/10) uses the LeRobot
convention (1 = open, 0 = close). GR00T's policy output and its sim env
wrapper (normalize_gripper_action + invert_gripper_action in libero_env.py)
assume the LeRobot convention end to end, so keeping the raw convention
would break distillation training statistics and rollout evaluation.

Mapping: LeRobot = (1 - raw) / 2   (raw -1 -> 1 open, raw +1 -> 0 close)

This script rewrites the action column of every parquet file and updates
norm_stats.json / meta/stats.json action statistics to match. A marker file
(meta/.gripper_converted) prevents accidental double conversion.

Usage:
    python FLARE-VLA/scripts/libero_plus/00b_convert_gripper_convention.py \
        --source-dir /root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE = Path(
    "/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot"
)
MARKER_NAME = ".gripper_converted"


def convert_parquet_actions(source_dir: Path) -> None:
    parquet_files = sorted((source_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {source_dir / 'data'}")

    for i, pq_path in enumerate(parquet_files):
        df = pd.read_parquet(pq_path)
        actions = np.stack(df["action"].to_numpy()).astype(np.float32)
        gripper = actions[..., 6]
        assert np.allclose(np.abs(gripper).round(2), 1.0, atol=1e-2), (
            f"Unexpected non-binary gripper values in {pq_path}: "
            f"{np.unique(np.round(gripper, 2))[:10]}"
        )
        # raw -1 (open) -> 1 (open); raw +1 (close) -> 0 (close)
        converted = (1.0 - gripper) / 2.0
        actions[..., 6] = converted
        df["action"] = list(actions)
        df.to_parquet(pq_path, index=False)

        if (i + 1) % 500 == 0 or i + 1 == len(parquet_files):
            print(f"  [{i + 1}/{len(parquet_files)}] converted")

    marker = source_dir / "meta" / MARKER_NAME
    marker.write_text("gripper converted from raw LIBERO to LeRobot convention\n")


def convert_stats(source_dir: Path) -> None:
    norm_stats_path = source_dir / "norm_stats.json"
    norm = json.loads(norm_stats_path.read_text(encoding="utf-8"))["norm_stats"]
    action_entry = norm["actions"]

    mean = np.asarray(action_entry["mean"], dtype=np.float64)
    std = np.asarray(action_entry["std"], dtype=np.float64)
    q01 = np.asarray(action_entry["q01"], dtype=np.float64)
    q99 = np.asarray(action_entry["q99"], dtype=np.float64)

    # L = (1 - raw) / 2
    action_entry["mean"] = ((1.0 - mean) / 2.0).tolist()
    action_entry["std"] = (std / 2.0).tolist()
    # quantile order flips under the decreasing transform
    action_entry["q01"] = ((1.0 - q99) / 2.0).tolist()
    action_entry["q99"] = ((1.0 - q01) / 2.0).tolist()

    norm_stats_path.write_text(
        json.dumps({"norm_stats": norm}, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] updated {norm_stats_path}")

    stats_path = source_dir / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if "action" in stats:
            stats["action"] = {
                "mean": action_entry["mean"],
                "std": action_entry["std"],
                "q01": action_entry["q01"],
                "q99": action_entry["q99"],
            }
            stats_path.write_text(
                json.dumps(stats, indent=4, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[DONE] updated {stats_path}")
    else:
        print("[WARN] meta/stats.json not found; run 00_prepare_meta_patch.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    marker = source_dir / "meta" / MARKER_NAME
    if marker.exists():
        raise RuntimeError(
            f"Marker {marker} exists — dataset already converted. Aborting."
        )

    if args.dry_run:
        parquet_files = sorted((source_dir / "data").glob("chunk-*/episode_*.parquet"))
        print(f"[dry-run] would convert {len(parquet_files)} parquet files")
        return

    convert_parquet_actions(source_dir)
    convert_stats(source_dir)
    print("Done. Re-run 00_prepare_meta_patch.py if meta/stats.json was stale.")


if __name__ == "__main__":
    main()
