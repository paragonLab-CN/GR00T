#!/usr/bin/env python3
"""
Generate mapping files from converted H5 + source LeRobot parquet.

Adapted from scripts/spatial/01_generate_spatial_mapping.py for
Sylvest/libero_plus_lerobot: multi-chunk source paths and front/wrist
camera keys.

Inputs:
  - H5 file (front camera, for episode structure)
  - Source LeRobot dataset dir (libero_plus_lerobot)

Outputs:
  - libero_plus_lerobot_to_h5_episode_map.jsonl
  - libero_plus_lerobot_to_h5_frame_map.parquet
  - libero_plus_conversion_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# --- defaults ---
DEFAULT_H5 = Path(
    "/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/h5/"
    "libero_plus_lerobot__observation_images_front.h5"
)
DEFAULT_SOURCE = Path(
    "/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot"
)
DEFAULT_MAPPING_DIR = Path("/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/mapping")


def load_tasks(source_dir: Path) -> dict[int, str]:
    """Load task_index -> task_text mapping from tasks.jsonl."""
    tasks_path = source_dir / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        print(f"[WARN] No tasks.jsonl at {tasks_path}")
        return {}
    task_map = {}
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_map[int(rec["task_index"])] = rec["task"]
    return task_map


def load_episodes_meta(source_dir: Path) -> list[dict]:
    """Load episodes.jsonl."""
    ep_path = source_dir / "meta" / "episodes.jsonl"
    records = []
    with open(ep_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parquet_rel_path(episode_index: int, chunks_size: int) -> str:
    chunk = f"chunk-{episode_index // chunks_size:03d}"
    return f"data/{chunk}/episode_{episode_index:06d}.parquet"


def generate_mapping(
    h5_path: Path,
    source_dir: Path,
    mapping_dir: Path,
) -> dict:
    """Read H5 + source parquet, produce all 3 mapping files."""
    mapping_dir.mkdir(parents=True, exist_ok=True)

    # --- load metadata ---
    info = json.loads((source_dir / "meta" / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    task_map = load_tasks(source_dir)
    episodes_meta = load_episodes_meta(source_dir)

    total_episodes_expected = int(info["total_episodes"])
    total_frames_expected = int(info["total_frames"])
    suite_name = source_dir.name

    print(f"Suite:       {suite_name}")
    print(f"Episodes:    {total_episodes_expected}")
    print(f"Frames:      {total_frames_expected}")
    print(f"Tasks:       {len(task_map)}")
    print(f"Chunks size: {chunks_size}")
    print(f"H5:          {h5_path}")

    # --- read H5 structure ---
    with h5py.File(h5_path, "r") as f:
        h5_total_frames = int(f["action"].shape[0])
        h5_ep_len = np.asarray(f["ep_len"][:])
        h5_ep_offset = np.asarray(f["ep_offset"][:])
        h5_episode_idx = np.asarray(f["episode_idx"][:])
        h5_task_index = np.asarray(f["task_index"][:])
        h5_step_idx = np.asarray(f["step_idx"][:])

    num_h5_episodes = len(h5_ep_len)

    # --- validate counts ---
    assert h5_total_frames == total_frames_expected, (
        f"H5 frames={h5_total_frames}, info.json={total_frames_expected}"
    )
    assert num_h5_episodes == len(episodes_meta), (
        f"H5 episodes={num_h5_episodes}, episodes.jsonl={len(episodes_meta)}"
    )
    print(f"[PASS] H5 frame count matches: {h5_total_frames}")
    print(f"[PASS] H5 episode count matches: {num_h5_episodes}")

    # --- validate episode ordering ---
    expected_ep_indices = np.arange(num_h5_episodes, dtype=np.int64)
    unique_h5_episodes = np.unique(h5_episode_idx[:])
    if not np.array_equal(unique_h5_episodes, expected_ep_indices):
        print(
            f"[WARN] H5 episode_idx unique values: "
            f"{unique_h5_episodes[:10]}... vs expected 0..{num_h5_episodes - 1}"
        )
    else:
        print(f"[PASS] H5 episode_idx values: 0..{num_h5_episodes - 1}")

    # --- build episode-level mapping ---
    episode_map_records = []
    for h5_ep in range(num_h5_episodes):
        start = int(h5_ep_offset[h5_ep])
        length = int(h5_ep_len[h5_ep])
        end = start + length - 1

        meta = episodes_meta[h5_ep]
        lerobot_ep_index = int(meta["episode_index"])
        task_idx = int(h5_task_index[start])
        task_text = task_map.get(task_idx, "")

        record = {
            "suite_name": suite_name,
            "h5_episode_index": h5_ep,
            "lerobot_episode_index": lerobot_ep_index,
            "task_index": task_idx,
            "task_text": task_text,
            "num_frames": length,
            "h5_start_frame": start,
            "h5_end_frame": end,
            "source_parquet": parquet_rel_path(lerobot_ep_index, chunks_size),
        }
        episode_map_records.append(record)

    # write episode map
    ep_map_path = mapping_dir / "libero_plus_lerobot_to_h5_episode_map.jsonl"
    with open(ep_map_path, "w") as f:
        for rec in episode_map_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[DONE] Episode map: {ep_map_path}  ({len(episode_map_records)} episodes)")

    # --- build frame-level mapping ---
    frame_rows = []
    for h5_ep in range(num_h5_episodes):
        start = int(h5_ep_offset[h5_ep])
        length = int(h5_ep_len[h5_ep])
        meta = episodes_meta[h5_ep]
        lerobot_ep_index = int(meta["episode_index"])
        task_idx = int(h5_task_index[start])
        task_text = task_map.get(task_idx, "")
        pq_rel = parquet_rel_path(lerobot_ep_index, chunks_size)

        for local_step in range(length):
            global_row = start + local_step
            frame_rows.append({
                "lerobot_episode_index": lerobot_ep_index,
                "lerobot_local_step": int(h5_step_idx[global_row]),
                "lerobot_global_index": global_row,
                "task_index": task_idx,
                "task_text": task_text,
                "h5_episode_index": h5_ep,
                "h5_step": global_row,
                "source_parquet": pq_rel,
            })

    frame_df = pd.DataFrame(frame_rows)
    frame_map_path = mapping_dir / "libero_plus_lerobot_to_h5_frame_map.parquet"
    frame_df.to_parquet(frame_map_path, index=False)
    print(f"[DONE] Frame map: {frame_map_path}  ({len(frame_df)} rows)")

    # --- build conversion summary ---
    per_task = frame_df.groupby(["task_index", "task_text"]).size().reset_index(name="frame_count")
    per_task_records = []
    for _, row in per_task.iterrows():
        per_task_records.append({
            "task_index": int(row["task_index"]),
            "task_text": row["task_text"],
            "frame_count": int(row["frame_count"]),
        })

    summary = {
        "suite_name": suite_name,
        "source_dir": str(source_dir),
        "codebase_version": info.get("codebase_version", ""),
        "total_episodes": num_h5_episodes,
        "total_frames": h5_total_frames,
        "total_tasks": len(task_map),
        "fps": int(info.get("fps", 20)),
        "robot_type": info.get("robot_type", ""),
        "image_size": 224,
        "action_dim": 7,
        "state_dim": 8,
        "h5_files": [
            str(h5_path),
            str(h5_path).replace("observation_images_front", "observation_images_wrist"),
        ],
        "per_task_frame_counts": per_task_records,
        "mapping_files": {
            "episode_map": str(ep_map_path),
            "frame_map": str(frame_map_path),
        },
    }

    summary_path = mapping_dir / "libero_plus_conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Summary: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate mapping files from libero_plus H5 + source LeRobot"
    )
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mapping-dir", type=Path, default=DEFAULT_MAPPING_DIR)
    args = parser.parse_args()

    if not args.h5_path.exists():
        print(f"ERROR: H5 not found: {args.h5_path}")
        sys.exit(1)
    if not args.source_dir.is_dir():
        print(f"ERROR: source dir not found: {args.source_dir}")
        sys.exit(1)

    generate_mapping(
        h5_path=args.h5_path,
        source_dir=args.source_dir,
        mapping_dir=args.mapping_dir,
    )
    print("\nAll mapping files generated successfully.")


if __name__ == "__main__":
    main()
