#!/usr/bin/env python3
"""
Final direct verification:

LeRobot v2 episode order
    versus
reordered LeWM GT-latent HDF5 episode order.

Checks:
1. episodes.jsonl episode_index and length order
2. ep_len / ep_offset / episode_id in reordered H5
3. every LeRobot parquet action sequence versus reordered frame_action
   (using any4lerobot's known gripper conversion)
4. all H-step window indices

A full PASS proves that:
    LeRobot (episode_index, local_step)
maps to:
    reordered_h5 global_row = ep_offset[episode_index] + local_step

Example:
python verify_lerobot_vs_reordered_latent.py \
  --lerobot-root /root/autodl-tmp/lerobot_data/libero_90_lerobot \
  --latent-h5 /root/autodl-tmp/le-wm/outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe_lerobot_order.h5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq


EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
    return records


def offsets_from_lengths(lengths: np.ndarray) -> np.ndarray:
    if lengths.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(lengths[:-1], dtype=np.int64),
        ]
    )


def build_parquet_map(root: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}

    for path in (root / "data").rglob("episode_*.parquet"):
        match = EPISODE_RE.search(path.name)
        if match is None:
            continue

        episode_index = int(match.group(1))
        if episode_index in mapping:
            raise RuntimeError(
                f"Duplicate parquet for episode {episode_index}: "
                f"{mapping[episode_index]} and {path}"
            )
        mapping[episode_index] = path

    if not mapping:
        raise RuntimeError(
            f"No episode parquet files found under {root / 'data'}"
        )

    return mapping


def to_1d_float(value: Any) -> np.ndarray:
    if value is None:
        raise RuntimeError("Encountered null action value")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def read_lerobot_action(path: Path) -> np.ndarray:
    """
    Read an episode's action sequence as [T, D].

    Supports:
    - standard vector column: action
    - split columns: action.x, action.y, ...
    """
    schema_names = pq.read_schema(path).names

    if "action" in schema_names:
        values = (
            pq.read_table(path, columns=["action"])
            .column("action")
            .combine_chunks()
            .to_pylist()
        )
        action = np.stack([to_1d_float(v) for v in values], axis=0)
        return action.astype(np.float32, copy=False)

    split_columns = sorted(
        name for name in schema_names if name.startswith("action.")
    )
    if not split_columns:
        raise RuntimeError(
            f"No 'action' or 'action.*' columns in {path}. "
            f"Columns={schema_names}"
        )

    table = pq.read_table(path, columns=split_columns)
    column_values = {
        name: table.column(name).combine_chunks().to_pylist()
        for name in split_columns
    }

    rows: list[np.ndarray] = []
    for row_index in range(table.num_rows):
        pieces = [
            to_1d_float(column_values[name][row_index])
            for name in split_columns
        ]
        rows.append(np.concatenate(pieces, axis=0))

    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def lewm_action_to_lerobot(action: np.ndarray) -> np.ndarray:
    """
    Match any4lerobot's LIBERO action conversion:

    raw LIBERO:
        gripper -1=open, +1=close

    LeRobot:
        gripper 1=open, 0=close

    converted_gripper = 1 - clip(raw_gripper, 0, 1)
    """
    action = np.asarray(action, dtype=np.float32)

    if action.ndim != 2 or action.shape[1] != 7:
        raise RuntimeError(
            f"Expected LeWM frame_action shape [T,7], got {action.shape}"
        )

    converted = action.copy()
    converted[:, 6] = 1.0 - np.clip(action[:, 6], 0.0, 1.0)
    return converted


def validate_windows(
    latent_file: h5py.File,
    ep_len: np.ndarray,
    ep_offset: np.ndarray,
) -> None:
    required = [
        "window_current_row",
        "window_future_row",
        "window_episode_id",
        "window_step_idx",
    ]
    for key in required:
        if key not in latent_file:
            raise RuntimeError(f"Missing window dataset: {key}")

    future_ds = latent_file["window_future_row"]
    if future_ds.ndim != 2:
        raise RuntimeError(
            f"window_future_row must be 2D, got {future_ds.shape}"
        )

    horizon = int(future_ds.shape[1])
    expected_counts = np.maximum(ep_len - horizon, 0)
    expected_total = int(expected_counts.sum())

    if latent_file["window_current_row"].shape != (expected_total,):
        raise RuntimeError(
            "window_current_row shape mismatch: "
            f"{latent_file['window_current_row'].shape} "
            f"vs ({expected_total},)"
        )
    if latent_file["window_episode_id"].shape != (expected_total,):
        raise RuntimeError(
            "window_episode_id shape mismatch"
        )
    if latent_file["window_step_idx"].shape != (expected_total,):
        raise RuntimeError(
            "window_step_idx shape mismatch"
        )
    if future_ds.shape != (expected_total, horizon):
        raise RuntimeError(
            f"window_future_row shape mismatch: {future_ds.shape} "
            f"vs ({expected_total}, {horizon})"
        )

    future_delta = np.arange(
        1, horizon + 1, dtype=np.int64
    )[None, :]

    cursor = 0
    for episode_index, (start, length, count) in enumerate(
        zip(ep_offset, ep_len, expected_counts)
    ):
        start = int(start)
        length = int(length)
        count = int(count)

        if count == 0:
            continue

        end = cursor + count
        local_step = np.arange(count, dtype=np.int64)
        expected_current = start + local_step
        expected_future = expected_current[:, None] + future_delta

        current = np.asarray(
            latent_file["window_current_row"][cursor:end],
            dtype=np.int64,
        )
        future = np.asarray(
            latent_file["window_future_row"][cursor:end],
            dtype=np.int64,
        )
        episode_id = np.asarray(
            latent_file["window_episode_id"][cursor:end],
            dtype=np.int64,
        )
        step_idx = np.asarray(
            latent_file["window_step_idx"][cursor:end],
            dtype=np.int64,
        )

        if not np.array_equal(current, expected_current):
            raise RuntimeError(
                f"window_current_row mismatch at episode {episode_index}"
            )
        if not np.array_equal(future, expected_future):
            raise RuntimeError(
                f"window_future_row mismatch at episode {episode_index}"
            )
        if not np.all(episode_id == episode_index):
            raise RuntimeError(
                f"window_episode_id mismatch at episode {episode_index}"
            )
        if not np.array_equal(step_idx, local_step):
            raise RuntimeError(
                f"window_step_idx mismatch at episode {episode_index}"
            )

        # Strong boundary check.
        if future.size:
            last_valid_row = start + length - 1
            if int(future.min()) < start or int(future.max()) > last_valid_row:
                raise RuntimeError(
                    f"Window crosses episode boundary at episode {episode_index}"
                )

        cursor = end

    if cursor != expected_total:
        raise RuntimeError(
            f"Window cursor={cursor}, expected={expected_total}"
        )

    print(
        f"[PASS] all {expected_total:,} H={horizon} windows are correct "
        "and stay inside episode boundaries"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--latent-h5", required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="0 checks all episodes; use a positive number for a quick test.",
    )
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    lerobot_root = Path(args.lerobot_root).resolve()
    latent_path = Path(args.latent_h5).resolve()

    episodes_path = lerobot_root / "meta" / "episodes.jsonl"

    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)
    if not latent_path.exists():
        raise FileNotFoundError(latent_path)

    episodes = load_jsonl(episodes_path)
    if not episodes:
        raise RuntimeError("episodes.jsonl is empty")

    episodes.sort(key=lambda item: int(item["episode_index"]))

    expected_episode_ids = np.arange(
        len(episodes), dtype=np.int64
    )
    actual_episode_ids = np.asarray(
        [int(item["episode_index"]) for item in episodes],
        dtype=np.int64,
    )

    if not np.array_equal(actual_episode_ids, expected_episode_ids):
        bad = int(
            np.flatnonzero(actual_episode_ids != expected_episode_ids)[0]
        )
        raise RuntimeError(
            f"episodes.jsonl is not contiguous at position {bad}: "
            f"got {actual_episode_ids[bad]}"
        )

    lerobot_ep_len = np.asarray(
        [int(item["length"]) for item in episodes],
        dtype=np.int64,
    )
    expected_ep_offset = offsets_from_lengths(lerobot_ep_len)
    total_frames = int(lerobot_ep_len.sum())

    parquet_map = build_parquet_map(lerobot_root)
    if len(parquet_map) != len(episodes):
        raise RuntimeError(
            f"Parquet episode count={len(parquet_map)}, "
            f"episodes.jsonl count={len(episodes)}"
        )

    print("========== Dataset summary ==========")
    print("LeRobot root :", lerobot_root)
    print("GT latent H5 :", latent_path)
    print("episodes     :", len(episodes))
    print("total frames :", total_frames)
    print("parquet files:", len(parquet_map))

    with h5py.File(latent_path, "r") as latent_file:
        required = {
            "ep_len",
            "ep_offset",
            "episode_id",
            "frame_action",
            "frame_latent",
            "window_current_row",
            "window_future_row",
            "window_episode_id",
            "window_step_idx",
        }
        missing = required - set(latent_file.keys())
        if missing:
            raise RuntimeError(
                f"Latent H5 missing datasets: {sorted(missing)}"
            )

        print("\n========== 1. Metadata/order verification ==========")

        if not bool(latent_file.attrs.get("complete", False)):
            raise RuntimeError("Latent H5 complete attribute is not True")
        print("[PASS] latent file has complete=True")

        episode_order = latent_file.attrs.get("episode_order", None)
        print("episode_order attribute:", episode_order)

        latent_ep_len = np.asarray(
            latent_file["ep_len"][:], dtype=np.int64
        ).reshape(-1)
        latent_ep_offset = np.asarray(
            latent_file["ep_offset"][:], dtype=np.int64
        ).reshape(-1)

        if not np.array_equal(latent_ep_len, lerobot_ep_len):
            bad = int(
                np.flatnonzero(latent_ep_len != lerobot_ep_len)[0]
            )
            raise RuntimeError(
                f"ep_len mismatch at episode {bad}: "
                f"latent={latent_ep_len[bad]}, "
                f"LeRobot={lerobot_ep_len[bad]}"
            )
        print(
            f"[PASS] ep_len exactly matches episodes.jsonl "
            f"for all {len(episodes):,} episodes"
        )

        if not np.array_equal(
            latent_ep_offset, expected_ep_offset
        ):
            bad = int(
                np.flatnonzero(
                    latent_ep_offset != expected_ep_offset
                )[0]
            )
            raise RuntimeError(
                f"ep_offset mismatch at episode {bad}: "
                f"latent={latent_ep_offset[bad]}, "
                f"expected={expected_ep_offset[bad]}"
            )
        print("[PASS] ep_offset is correct")

        if latent_file["frame_latent"].shape[0] != total_frames:
            raise RuntimeError(
                "frame_latent row count does not match total frames"
            )
        if latent_file["frame_action"].shape != (total_frames, 7):
            raise RuntimeError(
                f"frame_action shape={latent_file['frame_action'].shape}, "
                f"expected=({total_frames}, 7)"
            )
        if latent_file["episode_id"].shape != (total_frames,):
            raise RuntimeError(
                f"episode_id shape={latent_file['episode_id'].shape}, "
                f"expected=({total_frames},)"
            )
        print("[PASS] frame dataset shapes are correct")

        print("\n========== 2. Per-frame episode_id verification ==========")
        for episode_index, (start, length) in enumerate(
            zip(latent_ep_offset, latent_ep_len)
        ):
            start = int(start)
            length = int(length)
            values = np.asarray(
                latent_file["episode_id"][start:start + length],
                dtype=np.int64,
            )
            if not np.all(values == episode_index):
                bad_local = int(
                    np.flatnonzero(values != episode_index)[0]
                )
                raise RuntimeError(
                    f"episode_id mismatch: episode={episode_index}, "
                    f"local_step={bad_local}, "
                    f"value={values[bad_local]}"
                )
        print(
            f"[PASS] episode_id is correct for all {total_frames:,} frames"
        )

        print("\n========== 3. Direct action/order verification ==========")
        total_to_check = (
            len(episodes)
            if args.max_episodes <= 0
            else min(args.max_episodes, len(episodes))
        )

        global_max_diff = np.zeros(7, dtype=np.float64)

        for episode_index in range(total_to_check):
            parquet_path = parquet_map[episode_index]
            lerobot_action = read_lerobot_action(parquet_path)

            start = int(latent_ep_offset[episode_index])
            length = int(latent_ep_len[episode_index])
            raw_lewm_action = np.asarray(
                latent_file["frame_action"][start:start + length],
                dtype=np.float32,
            )
            comparable_action = lewm_action_to_lerobot(
                raw_lewm_action
            )

            if lerobot_action.shape != comparable_action.shape:
                raise RuntimeError(
                    f"Episode {episode_index} action shape mismatch: "
                    f"LeRobot={lerobot_action.shape}, "
                    f"latent={comparable_action.shape}, "
                    f"path={parquet_path}"
                )

            diff = np.abs(
                lerobot_action.astype(np.float64)
                - comparable_action.astype(np.float64)
            )
            global_max_diff = np.maximum(
                global_max_diff,
                np.nanmax(diff, axis=0),
            )

            close = np.isclose(
                lerobot_action,
                comparable_action,
                atol=args.atol,
                rtol=args.rtol,
                equal_nan=True,
            )
            if not np.all(close):
                bad = np.argwhere(~close)[0]
                local_step = int(bad[0])
                dim = int(bad[1])
                raise RuntimeError(
                    f"Action mismatch at episode={episode_index}, "
                    f"local_step={local_step}, dim={dim}\n"
                    f"LeRobot={lerobot_action[local_step]}\n"
                    f"latent raw={raw_lewm_action[local_step]}\n"
                    f"latent converted={comparable_action[local_step]}\n"
                    f"abs_diff={diff[local_step]}\n"
                    f"path={parquet_path}"
                )

            if (
                episode_index == 0
                or episode_index + 1 == total_to_check
                or (episode_index + 1) % args.progress_every == 0
            ):
                print(
                    f"episodes {episode_index + 1:4d}/"
                    f"{total_to_check}: actions match"
                )

        label = (
            "all episodes"
            if total_to_check == len(episodes)
            else f"first {total_to_check} episodes"
        )
        print(
            f"[PASS] LeRobot parquet actions match reordered "
            f"frame_action for {label}"
        )
        print(
            "max_abs_diff_by_dim:",
            global_max_diff.tolist(),
        )

        print("\n========== 4. Window verification ==========")
        validate_windows(
            latent_file,
            ep_len=latent_ep_len,
            ep_offset=latent_ep_offset,
        )

    print("\n============================================================")
    print("[FINAL PASS] LeRobot v2 order and reordered GT latent order")
    print("are fully aligned.")
    print("")
    print("Safe mapping:")
    print("global_row = ep_offset[episode_index] + local_step")
    print("============================================================")


if __name__ == "__main__":
    main()
