#!/usr/bin/env python3
"""
Verify the full LIBERO90 row-order chain:

LeRobot v2.1 episode parquet
    == libero90_image_train_v2.h5[action]
    == GT-latent H5[frame_action]

It also checks:
- episodes.jsonl episode_index and length
- train H5 ep_len / ep_offset / episode_idx
- total frame counts
- frame_latent row count
- window_future_row index range

A full PASS proves that (episode_index, step) in LeRobot maps to the same
global frame row in the LeWM training H5 and GT-latent H5.

Example:
python verify_libero90_chain_alignment.py \
  --lerobot-root /root/autodl-tmp/lerobot_data/libero_90_lerobot \
  --train-h5 /root/autodl-tmp/le-wm/stablewm_data/datasets/libero90_image_train_v2.h5 \
  --latent-h5 /root/autodl-tmp/le-wm/outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe.h5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq


EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def fail(message: str) -> None:
    print(f"\n[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[PASS] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


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
                fail(f"Invalid JSON at {path}:{line_no}: {exc}")
    return records


def as_int_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def compare_int_arrays(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        fail(f"{name} shape mismatch: actual={actual.shape}, expected={expected.shape}")
    bad = np.flatnonzero(actual != expected)
    if bad.size:
        i = int(bad[0])
        fail(
            f"{name} mismatch at index {i}: "
            f"actual={int(actual[i])}, expected={int(expected[i])}; "
            f"total mismatches={bad.size}"
        )
    ok(f"{name} exactly matches ({actual.size} entries)")


def compare_float_blocks(
    name: str,
    a_ds: h5py.Dataset,
    b_ds: h5py.Dataset,
    chunk_rows: int,
    atol: float,
    rtol: float,
) -> None:
    if a_ds.shape != b_ds.shape:
        fail(f"{name} shape mismatch: {a_ds.shape} vs {b_ds.shape}")

    n = int(a_ds.shape[0])
    global_max = 0.0
    mismatch_rows = 0
    first_bad: tuple[int, np.ndarray, np.ndarray] | None = None

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        a = np.asarray(a_ds[start:end])
        b = np.asarray(b_ds[start:end])

        finite_equal = np.isfinite(a) == np.isfinite(b)
        close = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        row_ok = np.all(close & finite_equal, axis=tuple(range(1, a.ndim)))

        if not np.all(row_ok):
            local_bad = np.flatnonzero(~row_ok)
            mismatch_rows += int(local_bad.size)
            if first_bad is None:
                j = int(local_bad[0])
                first_bad = (start + j, a[j], b[j])

        finite_diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        finite_diff = finite_diff[np.isfinite(finite_diff)]
        if finite_diff.size:
            global_max = max(global_max, float(finite_diff.max()))

        if start == 0 or end == n or (start // chunk_rows) % 20 == 0:
            info(f"{name}: checked rows {start:,}:{end:,}/{n:,}")

    if mismatch_rows:
        assert first_bad is not None
        idx, av, bv = first_bad
        fail(
            f"{name}: {mismatch_rows:,} mismatched rows; first row={idx}\n"
            f"A={av}\nB={bv}\nmax_abs_diff={global_max}"
        )

    ok(
        f"{name} matches for all {n:,} rows "
        f"(atol={atol}, rtol={rtol}, max_abs_diff={global_max:.3e})"
    )


def build_parquet_map(root: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    files = list((root / "data").rglob("episode_*.parquet"))
    if not files:
        fail(f"No episode parquet files found under {root / 'data'}")

    for path in files:
        match = EPISODE_RE.search(path.name)
        if not match:
            continue
        episode_index = int(match.group(1))
        if episode_index in mapping:
            fail(
                f"Duplicate parquet for episode {episode_index}: "
                f"{mapping[episode_index]} and {path}"
            )
        mapping[episode_index] = path

    return mapping


def load_action_order(modality_path: Path) -> list[str]:
    if not modality_path.exists():
        return []

    obj = json.loads(modality_path.read_text(encoding="utf-8"))
    action = obj.get("action", {})

    # Common GR00T modality.json form: an insertion-ordered dict of action fields.
    if isinstance(action, dict):
        return list(action.keys())

    return []


def value_to_1d(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Encountered null action value")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def read_episode_action(
    parquet_path: Path,
    action_order: list[str],
) -> np.ndarray:
    schema_names = pq.read_schema(parquet_path).names

    # Standard LeRobot storage: one vector-valued "action" column.
    if "action" in schema_names:
        values = (
            pq.read_table(parquet_path, columns=["action"])
            .column("action")
            .combine_chunks()
            .to_pylist()
        )
        return np.stack([value_to_1d(v) for v in values], axis=0)

    # Some converted datasets may store split action fields.
    split_columns = [name for name in schema_names if name.startswith("action.")]
    if not split_columns:
        raise KeyError(
            f"No 'action' or 'action.*' columns in {parquet_path}. "
            f"Columns={schema_names}"
        )

    if action_order:
        desired = [f"action.{key}" for key in action_order]
        ordered = [name for name in desired if name in split_columns]
        remaining = sorted(set(split_columns) - set(ordered))
        split_columns = ordered + remaining
    else:
        split_columns = sorted(split_columns)

    table = pq.read_table(parquet_path, columns=split_columns)
    rows: list[np.ndarray] = []

    columns_as_lists = {
        name: table.column(name).combine_chunks().to_pylist()
        for name in split_columns
    }
    n = table.num_rows

    for i in range(n):
        pieces = [value_to_1d(columns_as_lists[name][i]) for name in split_columns]
        rows.append(np.concatenate(pieces, axis=0))

    return np.stack(rows, axis=0)


def compare_lerobot_actions_to_h5(
    episodes: list[dict[str, Any]],
    parquet_map: dict[int, Path],
    train_action: h5py.Dataset,
    offsets: np.ndarray,
    action_order: list[str],
    max_episodes: int,
    atol: float,
    rtol: float,
) -> None:
    total = len(episodes) if max_episodes <= 0 else min(max_episodes, len(episodes))
    global_max = 0.0

    for position in range(total):
        record = episodes[position]
        episode_index = int(record["episode_index"])
        expected_length = int(record["length"])

        if episode_index not in parquet_map:
            fail(f"Missing parquet for episode_index={episode_index}")

        parquet_path = parquet_map[episode_index]
        le_action = read_episode_action(parquet_path, action_order)

        start = int(offsets[position])
        end = start + expected_length
        h5_action = np.asarray(train_action[start:end], dtype=np.float32)

        if le_action.shape != h5_action.shape:
            fail(
                f"Episode {episode_index} action shape mismatch: "
                f"LeRobot={le_action.shape}, train H5 slice={h5_action.shape}, "
                f"path={parquet_path}"
            )

        close = np.isclose(le_action, h5_action, atol=atol, rtol=rtol, equal_nan=True)
        if not np.all(close):
            bad_rows = np.flatnonzero(~np.all(close, axis=1))
            local_row = int(bad_rows[0])
            diff = np.abs(
                le_action.astype(np.float64) - h5_action.astype(np.float64)
            )
            fail(
                f"Episode {episode_index} action mismatch at local step {local_row}, "
                f"global row {start + local_row}\n"
                f"LeRobot={le_action[local_row]}\n"
                f"train H5={h5_action[local_row]}\n"
                f"episode max_abs_diff={np.nanmax(diff)}\n"
                f"path={parquet_path}"
            )

        diff = np.abs(le_action.astype(np.float64) - h5_action.astype(np.float64))
        finite = diff[np.isfinite(diff)]
        if finite.size:
            global_max = max(global_max, float(finite.max()))

        if position == 0 or position + 1 == total or (position + 1) % 100 == 0:
            info(
                f"LeRobot vs train H5: episodes "
                f"{position + 1:,}/{total:,}, global rows through {end:,}"
            )

    label = "all episodes" if total == len(episodes) else f"first {total} episodes"
    ok(
        f"LeRobot action sequences match train H5 for {label}; "
        f"max_abs_diff={global_max:.3e}"
    )


def validate_window_rows(
    latent_file: h5py.File,
    episode_lengths: np.ndarray,
    offsets: np.ndarray,
    total_frames: int,
) -> None:
    if "window_future_row" not in latent_file:
        info("latent H5 has no window_future_row; skipped window mapping check")
        return

    rows = np.asarray(latent_file["window_future_row"])
    if rows.ndim != 2:
        fail(f"window_future_row must be 2D, got {rows.shape}")

    if rows.size and (rows.min() < 0 or rows.max() >= total_frames):
        fail(
            f"window_future_row contains out-of-range values: "
            f"min={rows.min()}, max={rows.max()}, total_frames={total_frames}"
        )

    # Check each window is consecutive and never crosses an episode boundary.
    bad_consecutive = np.flatnonzero(np.any(np.diff(rows, axis=1) != 1, axis=1))
    if bad_consecutive.size:
        i = int(bad_consecutive[0])
        fail(f"window_future_row row {i} is not consecutive: {rows[i]}")

    frame_episode = np.empty(total_frames, dtype=np.int64)
    for ep_i, (start, length) in enumerate(zip(offsets, episode_lengths)):
        frame_episode[int(start): int(start + length)] = ep_i

    first_ep = frame_episode[rows[:, 0]]
    same_ep = np.all(frame_episode[rows] == first_ep[:, None], axis=1)
    bad_cross = np.flatnonzero(~same_ep)
    if bad_cross.size:
        i = int(bad_cross[0])
        fail(f"window_future_row row {i} crosses episode boundary: {rows[i]}")

    ok(
        f"window_future_row valid: shape={rows.shape}, consecutive, "
        f"and no window crosses an episode boundary"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--latent-h5", required=True)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="0 checks all episodes; use a positive number for a quick test.",
    )
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    lerobot_root = Path(args.lerobot_root)
    episodes_path = lerobot_root / "meta" / "episodes.jsonl"
    modality_path = lerobot_root / "meta" / "modality.json"
    train_h5_path = Path(args.train_h5)
    latent_h5_path = Path(args.latent_h5)

    for path in [episodes_path, train_h5_path, latent_h5_path]:
        if not path.exists():
            fail(f"Path does not exist: {path}")

    print("========== 1. Load episode metadata ==========")
    episodes = load_jsonl(episodes_path)
    if not episodes:
        fail("episodes.jsonl is empty")

    episode_indices = np.asarray(
        [int(record["episode_index"]) for record in episodes],
        dtype=np.int64,
    )
    expected_indices = np.arange(len(episodes), dtype=np.int64)
    compare_int_arrays("episodes.jsonl episode_index", episode_indices, expected_indices)

    episode_lengths = np.asarray(
        [int(record["length"]) for record in episodes],
        dtype=np.int64,
    )
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(episode_lengths[:-1])]
    )
    total_frames = int(episode_lengths.sum())

    info(f"episodes={len(episodes):,}")
    info(f"total_frames from episodes.jsonl={total_frames:,}")
    info(f"first lengths={episode_lengths[:10].tolist()}")
    info(f"first offsets={offsets[:10].tolist()}")

    print("\n========== 2. Compare episode boundaries with train H5 ==========")
    with h5py.File(train_h5_path, "r") as train_f, h5py.File(
        latent_h5_path, "r"
    ) as latent_f:
        info(f"train H5 keys={list(train_f.keys())}")
        info(f"latent H5 keys={list(latent_f.keys())}")

        if "action" not in train_f:
            fail("train H5 does not contain root dataset 'action'")
        if "frame_action" not in latent_f:
            fail("latent H5 does not contain root dataset 'frame_action'")
        if "frame_latent" not in latent_f:
            fail("latent H5 does not contain root dataset 'frame_latent'")

        train_action = train_f["action"]
        latent_action = latent_f["frame_action"]
        frame_latent = latent_f["frame_latent"]

        if train_action.shape[0] != total_frames:
            fail(
                f"train H5 action rows={train_action.shape[0]:,}, "
                f"but episodes.jsonl total={total_frames:,}"
            )
        ok(f"train H5 action rows match total_frames ({total_frames:,})")

        if frame_latent.shape[0] != total_frames:
            fail(
                f"frame_latent rows={frame_latent.shape[0]:,}, "
                f"but total_frames={total_frames:,}"
            )
        ok(f"frame_latent rows match total_frames ({total_frames:,})")

        if "pixels" in train_f:
            if train_f["pixels"].shape[0] != total_frames:
                fail(
                    f"train H5 pixels rows={train_f['pixels'].shape[0]:,}, "
                    f"expected={total_frames:,}"
                )
            ok(f"train H5 pixels rows match total_frames ({total_frames:,})")

        if "ep_len" in train_f:
            compare_int_arrays(
                "train H5 ep_len",
                as_int_array(train_f["ep_len"][:]),
                episode_lengths,
            )
        else:
            info("train H5 has no ep_len; skipped")

        if "ep_offset" in train_f:
            compare_int_arrays(
                "train H5 ep_offset",
                as_int_array(train_f["ep_offset"][:]),
                offsets,
            )
        else:
            info("train H5 has no ep_offset; skipped")

        expected_episode_idx = np.repeat(expected_indices, episode_lengths)
        if "episode_idx" in train_f:
            compare_int_arrays(
                "train H5 episode_idx",
                as_int_array(train_f["episode_idx"][:]),
                expected_episode_idx,
            )
        else:
            info("train H5 has no episode_idx; skipped")

        print("\n========== 3. Compare train H5 action with latent frame_action ==========")
        compare_float_blocks(
            "train[action] vs latent[frame_action]",
            train_action,
            latent_action,
            chunk_rows=args.chunk_rows,
            atol=args.atol,
            rtol=args.rtol,
        )

        print("\n========== 4. Validate future-row mapping ==========")
        validate_window_rows(
            latent_f,
            episode_lengths=episode_lengths,
            offsets=offsets,
            total_frames=total_frames,
        )

        print("\n========== 5. Compare every LeRobot episode action with train H5 ==========")
        parquet_map = build_parquet_map(lerobot_root)
        if len(parquet_map) != len(episodes):
            fail(
                f"parquet episode count={len(parquet_map)}, "
                f"episodes.jsonl count={len(episodes)}"
            )
        ok(f"found exactly {len(parquet_map):,} episode parquet files")

        action_order = load_action_order(modality_path)
        if action_order:
            info(f"action field order from modality.json={action_order}")

        compare_lerobot_actions_to_h5(
            episodes=episodes,
            parquet_map=parquet_map,
            train_action=train_action,
            offsets=offsets,
            action_order=action_order,
            max_episodes=args.max_episodes,
            atol=args.atol,
            rtol=args.rtol,
        )

    print("\n============================================================")
    print("[FINAL PASS] The ordering chain is aligned:")
    print("LeRobot (episode_index, local_step)")
    print("  -> train H5 global row = ep_offset[episode_index] + local_step")
    print("  -> latent H5 frame_action/frame_latent at the same global row")
    print("============================================================")
    print(
        "\nNote: this proves row mapping through exact action alignment. "
        "For the strongest content check, also re-encode several train-H5 "
        "pixel rows—especially episode boundaries—and compare them with "
        "frame_latent at the same rows using the original checkpoint and "
        "training image preprocessor."
    )


if __name__ == "__main__":
    main()
