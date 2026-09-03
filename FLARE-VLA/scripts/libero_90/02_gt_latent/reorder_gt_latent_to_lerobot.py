#!/usr/bin/env python3
"""
Reorder an existing LeWM GT-latent HDF5 file from the old train-H5 episode
order into the LeRobot episode order using a verified bijective JSONL map.

The script:
1. Copies frame_latent and frame_action episode blocks without modifying values.
2. Rebuilds ep_len, ep_offset and per-frame episode_id in LeRobot order.
3. Rebuilds all H-step window indices so they reference the new global rows.
4. Copies file/dataset attributes and writes audit attributes.
5. Writes to a temporary file first, then atomically renames it.

Expected mapping JSONL fields:
    lerobot_episode_index
    train_episode_index
    length
    train_offset            (optional; checked if present)

Example:
python reorder_gt_latent_to_lerobot.py \
  --input-h5 outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe.h5 \
  --mapping outputs/lerobot_to_train_episode_map_gripper_fixed.jsonl \
  --output-h5 outputs/gt_latents/libero90_epoch50_h8_framebank_trainpipe_lerobot_order.h5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_KEYS = {
    "ep_len",
    "ep_offset",
    "episode_id",
    "frame_action",
    "frame_latent",
    "window_current_row",
    "window_episode_id",
    "window_future_row",
    "window_step_idx",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def copy_attrs(src: Any, dst: Any) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def dataset_creation_kwargs(src: h5py.Dataset, shape: tuple[int, ...]) -> dict[str, Any]:
    """Reuse safe storage properties from the source dataset."""
    kwargs: dict[str, Any] = {}

    if src.chunks is not None:
        # Output shapes are expected to be unchanged, but clamp defensively.
        kwargs["chunks"] = tuple(
            min(int(chunk), int(dim)) if int(dim) > 0 else int(chunk)
            for chunk, dim in zip(src.chunks, shape)
        )

    if src.compression is not None:
        kwargs["compression"] = src.compression
        if src.compression_opts is not None:
            kwargs["compression_opts"] = src.compression_opts

    if src.shuffle:
        kwargs["shuffle"] = True
    if src.fletcher32:
        kwargs["fletcher32"] = True
    if src.scaleoffset is not None:
        kwargs["scaleoffset"] = src.scaleoffset

    return kwargs


def create_like(
    dst_file: h5py.File,
    name: str,
    src: h5py.Dataset,
    shape: tuple[int, ...] | None = None,
) -> h5py.Dataset:
    target_shape = tuple(src.shape if shape is None else shape)
    ds = dst_file.create_dataset(
        name,
        shape=target_shape,
        dtype=src.dtype,
        **dataset_creation_kwargs(src, target_shape),
    )
    copy_attrs(src, ds)
    return ds


def offsets_from_lengths(lengths: np.ndarray) -> np.ndarray:
    if lengths.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(lengths[:-1], dtype=np.int64),
        ]
    )


def validate_mapping(
    records: list[dict[str, Any]],
    old_ep_len: np.ndarray,
    old_ep_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(old_ep_len)
    if len(records) != n:
        raise RuntimeError(
            f"Mapping rows={len(records)}, but latent episodes={n}"
        )

    required_fields = {
        "lerobot_episode_index",
        "train_episode_index",
        "length",
    }
    for row_no, record in enumerate(records, start=1):
        missing = required_fields - set(record)
        if missing:
            raise RuntimeError(
                f"Mapping row {row_no} missing fields: {sorted(missing)}"
            )
        if record["train_episode_index"] is None:
            raise RuntimeError(
                f"Mapping row {row_no} has train_episode_index=None"
            )

    records.sort(key=lambda item: int(item["lerobot_episode_index"]))

    lerobot_ids = np.asarray(
        [int(item["lerobot_episode_index"]) for item in records],
        dtype=np.int64,
    )
    expected_ids = np.arange(n, dtype=np.int64)
    if not np.array_equal(lerobot_ids, expected_ids):
        bad = np.flatnonzero(lerobot_ids != expected_ids)
        i = int(bad[0]) if bad.size else -1
        raise RuntimeError(
            f"LeRobot episode indices are not exactly 0..{n-1}; "
            f"first mismatch at position {i}"
        )

    source_ids = np.asarray(
        [int(item["train_episode_index"]) for item in records],
        dtype=np.int64,
    )
    if source_ids.min() < 0 or source_ids.max() >= n:
        raise RuntimeError(
            f"train_episode_index out of range: "
            f"min={source_ids.min()}, max={source_ids.max()}, n={n}"
        )
    if np.unique(source_ids).size != n:
        raise RuntimeError(
            "train_episode_index is not a complete one-to-one permutation"
        )

    new_ep_len = old_ep_len[source_ids]

    for new_ep, record in enumerate(records):
        source_ep = int(source_ids[new_ep])
        mapped_length = int(record["length"])
        actual_length = int(old_ep_len[source_ep])
        if mapped_length != actual_length:
            raise RuntimeError(
                f"Length mismatch for LeRobot episode {new_ep}: "
                f"mapping={mapped_length}, source latent episode "
                f"{source_ep}={actual_length}"
            )

        if record.get("train_offset") is not None:
            mapped_offset = int(record["train_offset"])
            actual_offset = int(old_ep_offset[source_ep])
            if mapped_offset != actual_offset:
                raise RuntimeError(
                    f"Offset mismatch for LeRobot episode {new_ep}: "
                    f"mapping train_offset={mapped_offset}, "
                    f"latent ep_offset[{source_ep}]={actual_offset}"
                )

    return source_ids, new_ep_len.astype(np.int64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N episodes.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_h5).resolve()
    mapping_path = Path(args.mapping).resolve()
    output_path = Path(args.output_h5).resolve()
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if input_path == output_path:
        raise RuntimeError("Input and output paths must be different")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use --overwrite only after checking the path."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_path.exists():
        temp_path.unlink()

    mapping_records = load_jsonl(mapping_path)

    try:
        with h5py.File(input_path, "r") as src:
            missing = REQUIRED_KEYS - set(src.keys())
            if missing:
                raise RuntimeError(
                    f"Input latent H5 is missing keys: {sorted(missing)}"
                )

            unexpected = set(src.keys()) - REQUIRED_KEYS
            if unexpected:
                raise RuntimeError(
                    "Input contains unexpected root datasets that this script "
                    f"does not know how to reorder safely: {sorted(unexpected)}"
                )

            old_ep_len = np.asarray(src["ep_len"][:], dtype=np.int64).reshape(-1)
            old_ep_offset = np.asarray(
                src["ep_offset"][:], dtype=np.int64
            ).reshape(-1)

            expected_old_offsets = offsets_from_lengths(old_ep_len)
            if not np.array_equal(old_ep_offset, expected_old_offsets):
                raise RuntimeError(
                    "Input latent ep_offset is inconsistent with ep_len"
                )

            total_frames = int(old_ep_len.sum())
            if int(src["frame_latent"].shape[0]) != total_frames:
                raise RuntimeError(
                    "frame_latent row count does not equal sum(ep_len)"
                )
            if int(src["frame_action"].shape[0]) != total_frames:
                raise RuntimeError(
                    "frame_action row count does not equal sum(ep_len)"
                )

            source_episode_ids, new_ep_len = validate_mapping(
                mapping_records,
                old_ep_len,
                old_ep_offset,
            )
            new_ep_offset = offsets_from_lengths(new_ep_len)

            horizon = int(src["window_future_row"].shape[1])
            if horizon <= 0:
                raise RuntimeError("Invalid future horizon")

            expected_windows = int(
                np.maximum(new_ep_len - horizon, 0).sum()
            )
            old_window_count = int(src["window_future_row"].shape[0])
            if expected_windows != old_window_count:
                raise RuntimeError(
                    f"Window count mismatch: rebuilt={expected_windows}, "
                    f"input={old_window_count}, horizon={horizon}"
                )

            print("========== Reorder plan ==========")
            print("input           :", input_path)
            print("mapping         :", mapping_path)
            print("output          :", output_path)
            print("episodes        :", len(new_ep_len))
            print("total frames    :", total_frames)
            print("future horizon  :", horizon)
            print("total windows   :", expected_windows)
            print("identity episodes:", int(np.sum(
                source_episode_ids == np.arange(len(source_episode_ids))
            )))
            print("first mappings  :")
            for new_ep in range(min(20, len(new_ep_len))):
                print(
                    f"  LeRobot {new_ep:4d} <- old latent "
                    f"{int(source_episode_ids[new_ep]):4d}, "
                    f"length={int(new_ep_len[new_ep])}"
                )

            with h5py.File(temp_path, "w") as dst:
                copy_attrs(src, dst)
                dst.attrs["complete"] = False
                dst.attrs["episode_order"] = "lerobot_v2.1_episode_index"
                dst.attrs["reordered_from"] = str(input_path)
                dst.attrs["episode_mapping_jsonl"] = str(mapping_path)
                dst.attrs["reorder_method"] = (
                    "copy whole episode blocks using "
                    "lerobot_episode_index->train_episode_index bijection"
                )
                dst.attrs["frame_values_modified"] = False
                dst.attrs["window_indices_rebuilt"] = True
                dst.attrs["future_horizon"] = horizon

                out_ep_len = create_like(
                    dst, "ep_len", src["ep_len"], shape=new_ep_len.shape
                )
                out_ep_offset = create_like(
                    dst, "ep_offset", src["ep_offset"], shape=new_ep_offset.shape
                )
                out_episode_id = create_like(
                    dst,
                    "episode_id",
                    src["episode_id"],
                    shape=(total_frames,),
                )
                out_frame_action = create_like(
                    dst,
                    "frame_action",
                    src["frame_action"],
                    shape=tuple(src["frame_action"].shape),
                )
                out_frame_latent = create_like(
                    dst,
                    "frame_latent",
                    src["frame_latent"],
                    shape=tuple(src["frame_latent"].shape),
                )

                out_window_current = create_like(
                    dst,
                    "window_current_row",
                    src["window_current_row"],
                    shape=(expected_windows,),
                )
                out_window_episode = create_like(
                    dst,
                    "window_episode_id",
                    src["window_episode_id"],
                    shape=(expected_windows,),
                )
                out_window_future = create_like(
                    dst,
                    "window_future_row",
                    src["window_future_row"],
                    shape=(expected_windows, horizon),
                )
                out_window_step = create_like(
                    dst,
                    "window_step_idx",
                    src["window_step_idx"],
                    shape=(expected_windows,),
                )

                out_ep_len[:] = new_ep_len.astype(
                    out_ep_len.dtype, copy=False
                )
                out_ep_offset[:] = new_ep_offset.astype(
                    out_ep_offset.dtype, copy=False
                )

                print("\n========== Copy episode blocks ==========")
                window_cursor = 0

                for new_ep, source_ep in enumerate(source_episode_ids):
                    source_ep = int(source_ep)
                    length = int(new_ep_len[new_ep])

                    old_start = int(old_ep_offset[source_ep])
                    old_end = old_start + length
                    new_start = int(new_ep_offset[new_ep])
                    new_end = new_start + length

                    # Values are copied exactly; no action/gripper conversion here.
                    out_frame_latent[new_start:new_end] = (
                        src["frame_latent"][old_start:old_end]
                    )
                    out_frame_action[new_start:new_end] = (
                        src["frame_action"][old_start:old_end]
                    )
                    out_episode_id[new_start:new_end] = np.asarray(
                        new_ep, dtype=out_episode_id.dtype
                    )

                    n_windows = max(length - horizon, 0)
                    if n_windows > 0:
                        local_step = np.arange(n_windows, dtype=np.int64)
                        current_rows = new_start + local_step
                        future_rows = (
                            current_rows[:, None]
                            + np.arange(
                                1, horizon + 1, dtype=np.int64
                            )[None, :]
                        )
                        w_start = window_cursor
                        w_end = w_start + n_windows

                        out_window_current[w_start:w_end] = (
                            current_rows.astype(
                                out_window_current.dtype, copy=False
                            )
                        )
                        out_window_episode[w_start:w_end] = np.asarray(
                            new_ep, dtype=out_window_episode.dtype
                        )
                        out_window_step[w_start:w_end] = local_step.astype(
                            out_window_step.dtype, copy=False
                        )
                        out_window_future[w_start:w_end] = (
                            future_rows.astype(
                                out_window_future.dtype, copy=False
                            )
                        )
                        window_cursor = w_end

                    if (
                        new_ep == 0
                        or new_ep + 1 == len(new_ep_len)
                        or (new_ep + 1) % args.progress_every == 0
                    ):
                        print(
                            f"episodes {new_ep + 1:4d}/"
                            f"{len(new_ep_len)}, "
                            f"frames through {new_end:,}, "
                            f"windows through {window_cursor:,}"
                        )

                if window_cursor != expected_windows:
                    raise RuntimeError(
                        f"Internal window cursor={window_cursor}, "
                        f"expected={expected_windows}"
                    )

                dst.attrs["frames_written"] = total_frames
                dst.attrs["windows_written"] = expected_windows
                dst.attrs["complete"] = True
                dst.flush()

        if output_path.exists():
            output_path.unlink()
        os.replace(temp_path, output_path)

        print("\n[PASS] Reordered latent file created:")
        print(output_path)
        print(
            "The original file was not modified. Run the separate verifier "
            "before using the new file."
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


if __name__ == "__main__":
    main()
