#!/usr/bin/env python3
"""
Verify a GT-latent HDF5 file reordered into LeRobot episode order.

Checks:
1. Mapping is a complete bijection.
2. New ep_len equals LeRobot episodes.jsonl lengths.
3. Every frame_latent and frame_action episode block is exactly equal to the
   mapped block in the original latent file.
4. ep_offset and episode_id are correct.
5. All window indices match the canonical H-step construction and never cross
   episode boundaries.

Example:
python verify_reordered_gt_latent.py \
  --original-h5 outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe.h5 \
  --reordered-h5 outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe_lerobot_order.h5 \
  --mapping outputs/lerobot_to_train_episode_map_gripper_fixed.jsonl \
  --episodes-jsonl /root/autodl-tmp/lerobot_data/libero_90_lerobot/meta/episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
    return rows


def offsets_from_lengths(lengths: np.ndarray) -> np.ndarray:
    if lengths.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(lengths[:-1], dtype=np.int64),
        ]
    )


def first_mismatch(a: np.ndarray, b: np.ndarray) -> tuple[int, ...] | None:
    if a.shape != b.shape:
        return ()
    mask = a != b
    if not np.any(mask):
        return None
    return tuple(int(x) for x in np.argwhere(mask)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-h5", required=True)
    parser.add_argument("--reordered-h5", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--episodes-jsonl", required=True)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    original_path = Path(args.original_h5)
    reordered_path = Path(args.reordered_h5)
    mapping_records = load_jsonl(Path(args.mapping))
    episode_records = load_jsonl(Path(args.episodes_jsonl))

    mapping_records.sort(
        key=lambda item: int(item["lerobot_episode_index"])
    )
    episode_records.sort(
        key=lambda item: int(item["episode_index"])
    )

    n = len(mapping_records)
    if len(episode_records) != n:
        raise RuntimeError(
            f"Mapping episodes={n}, LeRobot episodes={len(episode_records)}"
        )

    source_ids = np.asarray(
        [int(item["train_episode_index"]) for item in mapping_records],
        dtype=np.int64,
    )
    if np.unique(source_ids).size != n:
        raise RuntimeError("Mapping is not a bijection")

    lerobot_lengths = np.asarray(
        [int(item["length"]) for item in episode_records],
        dtype=np.int64,
    )
    expected_offsets = offsets_from_lengths(lerobot_lengths)
    total_frames = int(lerobot_lengths.sum())

    with h5py.File(original_path, "r") as old, h5py.File(
        reordered_path, "r"
    ) as new:
        print("========== Metadata ==========")
        print("original :", original_path)
        print("reordered:", reordered_path)
        print("episodes :", n)
        print("frames   :", total_frames)

        if not bool(new.attrs.get("complete", False)):
            raise RuntimeError("Reordered file does not have complete=True")

        new_ep_len = np.asarray(new["ep_len"][:], dtype=np.int64)
        new_ep_offset = np.asarray(new["ep_offset"][:], dtype=np.int64)
        old_ep_len = np.asarray(old["ep_len"][:], dtype=np.int64)
        old_ep_offset = np.asarray(old["ep_offset"][:], dtype=np.int64)

        if not np.array_equal(new_ep_len, lerobot_lengths):
            bad = int(np.flatnonzero(new_ep_len != lerobot_lengths)[0])
            raise RuntimeError(
                f"ep_len mismatch at episode {bad}: "
                f"new={new_ep_len[bad]}, "
                f"LeRobot={lerobot_lengths[bad]}"
            )
        print("[PASS] ep_len exactly matches LeRobot episodes.jsonl")

        if not np.array_equal(new_ep_offset, expected_offsets):
            bad = int(np.flatnonzero(
                new_ep_offset != expected_offsets
            )[0])
            raise RuntimeError(
                f"ep_offset mismatch at episode {bad}: "
                f"new={new_ep_offset[bad]}, "
                f"expected={expected_offsets[bad]}"
            )
        print("[PASS] ep_offset is correct")

        if new["frame_latent"].shape[0] != total_frames:
            raise RuntimeError("frame_latent row count mismatch")
        if new["frame_action"].shape[0] != total_frames:
            raise RuntimeError("frame_action row count mismatch")
        print("[PASS] frame dataset row counts are correct")

        print("\n========== Exact episode-block comparison ==========")
        for new_ep, source_ep in enumerate(source_ids):
            source_ep = int(source_ep)
            length = int(lerobot_lengths[new_ep])

            old_start = int(old_ep_offset[source_ep])
            old_end = old_start + length
            new_start = int(new_ep_offset[new_ep])
            new_end = new_start + length

            expected_length = int(old_ep_len[source_ep])
            if expected_length != length:
                raise RuntimeError(
                    f"Mapped length mismatch for new episode {new_ep}: "
                    f"old episode {source_ep} has {expected_length}, "
                    f"LeRobot has {length}"
                )

            old_latent = old["frame_latent"][old_start:old_end]
            new_latent = new["frame_latent"][new_start:new_end]
            if not np.array_equal(old_latent, new_latent):
                idx = first_mismatch(old_latent, new_latent)
                raise RuntimeError(
                    f"frame_latent mismatch: new episode {new_ep}, "
                    f"old episode {source_ep}, first index={idx}"
                )

            old_action = old["frame_action"][old_start:old_end]
            new_action = new["frame_action"][new_start:new_end]
            if not np.array_equal(old_action, new_action):
                idx = first_mismatch(old_action, new_action)
                raise RuntimeError(
                    f"frame_action mismatch: new episode {new_ep}, "
                    f"old episode {source_ep}, first index={idx}"
                )

            episode_id_block = np.asarray(
                new["episode_id"][new_start:new_end],
                dtype=np.int64,
            )
            if not np.all(episode_id_block == new_ep):
                raise RuntimeError(
                    f"episode_id mismatch inside new episode {new_ep}"
                )

            if (
                new_ep == 0
                or new_ep + 1 == n
                or (new_ep + 1) % args.progress_every == 0
            ):
                print(
                    f"episodes {new_ep + 1:4d}/{n}: exact latent/action "
                    "blocks verified"
                )

        print("[PASS] all frame_latent blocks are exact copies")
        print("[PASS] all frame_action blocks are exact copies")
        print("[PASS] episode_id is correct for every frame")

        print("\n========== Window verification ==========")
        horizon = int(new["window_future_row"].shape[1])
        expected_windows = int(
            np.maximum(lerobot_lengths - horizon, 0).sum()
        )
        for name in (
            "window_current_row",
            "window_episode_id",
            "window_step_idx",
        ):
            if new[name].shape != (expected_windows,):
                raise RuntimeError(
                    f"{name} shape={new[name].shape}, "
                    f"expected=({expected_windows},)"
                )
        if new["window_future_row"].shape != (
            expected_windows,
            horizon,
        ):
            raise RuntimeError(
                f"window_future_row shape="
                f"{new['window_future_row'].shape}, "
                f"expected=({expected_windows},{horizon})"
            )

        cursor = 0
        future_delta = np.arange(
            1, horizon + 1, dtype=np.int64
        )[None, :]

        for ep, (start, length) in enumerate(
            zip(new_ep_offset, lerobot_lengths)
        ):
            start = int(start)
            length = int(length)
            count = max(length - horizon, 0)
            if count == 0:
                continue

            local = np.arange(count, dtype=np.int64)
            expected_current = start + local
            expected_future = expected_current[:, None] + future_delta
            end = cursor + count

            actual_current = np.asarray(
                new["window_current_row"][cursor:end],
                dtype=np.int64,
            )
            actual_episode = np.asarray(
                new["window_episode_id"][cursor:end],
                dtype=np.int64,
            )
            actual_step = np.asarray(
                new["window_step_idx"][cursor:end],
                dtype=np.int64,
            )
            actual_future = np.asarray(
                new["window_future_row"][cursor:end],
                dtype=np.int64,
            )

            if not np.array_equal(actual_current, expected_current):
                raise RuntimeError(
                    f"window_current_row mismatch at episode {ep}"
                )
            if not np.all(actual_episode == ep):
                raise RuntimeError(
                    f"window_episode_id mismatch at episode {ep}"
                )
            if not np.array_equal(actual_step, local):
                raise RuntimeError(
                    f"window_step_idx mismatch at episode {ep}"
                )
            if not np.array_equal(actual_future, expected_future):
                raise RuntimeError(
                    f"window_future_row mismatch at episode {ep}"
                )

            cursor = end

        if cursor != expected_windows:
            raise RuntimeError(
                f"Verified windows={cursor}, expected={expected_windows}"
            )

        print(
            f"[PASS] all {expected_windows:,} H={horizon} windows "
            "are correct and stay inside episode boundaries"
        )

    print("\n======================================================")
    print("[FINAL PASS] Reordered GT latent is aligned to LeRobot order.")
    print("You may now use:")
    print("global_row = ep_offset[lerobot_episode_index] + local_step")
    print("======================================================")


if __name__ == "__main__":
    main()
