#!/usr/bin/env python3
"""
Generate gr00t-required meta files for Sylvest/libero_plus_lerobot.

The dataset ships with meta/{info,episodes,tasks}.jsonl and a top-level
norm_stats.json but lacks the two files gr00t's LeRobotEpisodeLoader requires:

  - meta/modality.json   (state/action slicing + video original keys)
  - meta/stats.json      (per-modality mean/std/q01/q99 for normalization)

modality.json mirrors examples/LIBERO/modality.json, with video keys mapped
to the libero_plus camera names (front / wrist).

Usage:
    python FLARE-VLA/scripts/libero_plus/00_prepare_meta_patch.py \
        --source-dir /root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SOURCE = Path(
    "/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot"
)

MODALITY_JSON = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "image": {"original_key": "observation.images.front"},
        "wrist_image": {"original_key": "observation.images.wrist"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"}
    },
}


def build_stats_json(norm_stats_path: Path) -> dict:
    norm = json.loads(norm_stats_path.read_text(encoding="utf-8"))["norm_stats"]
    stats = {}
    # gr00t loader iterates whichever stat types exist (mean/std/q01/q99);
    # min/max are optional.
    for source_key, dest_key in [
        ("state", "observation.state"),
        ("actions", "action"),
    ]:
        entry = norm[source_key]
        stats[dest_key] = {
            stat_type: entry[stat_type] for stat_type in ["mean", "std", "q01", "q99"]
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    meta_dir = source_dir / "meta"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"Not a LeRobot dataset (no meta/): {source_dir}")

    modality_path = meta_dir / "modality.json"
    modality_path.write_text(
        json.dumps(MODALITY_JSON, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] {modality_path}")

    norm_stats_path = source_dir / "norm_stats.json"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"norm_stats.json not found in {source_dir}")

    stats_path = meta_dir / "stats.json"
    stats_path.write_text(
        json.dumps(build_stats_json(norm_stats_path), indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] {stats_path}")


if __name__ == "__main__":
    main()
