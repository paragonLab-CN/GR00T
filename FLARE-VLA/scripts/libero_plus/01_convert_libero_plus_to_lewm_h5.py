#!/usr/bin/env python3
"""
Convert Sylvest/libero_plus_lerobot (LeRobot v2.1) to LEWM-compatible HDF5.

Adapted from lerobot-dataset-to-HDF5/scripts/convert_lerobot_to_lewm_hdf5.py.
The only behavioral difference: libero_plus_lerobot spans 15 parquet/video
chunks (chunks_size=1000), while the original converter hardcodes chunk-000.
This version resolves per-episode chunk paths from meta/info.json templates.

LEWM HDF5 schema (per file):
    pixels:      [N, 224, 224, 3]  uint8      resized camera frames
    action:      [N, 7]            float32
    state:       [N, 8]            float32    (same as libero vanilla)
    proprio:     [N, 8]            float32    (same as state)
    episode_idx: [N]               int64      0-based episode id
    step_idx:    [N]               int64      0..L-1 within each episode
    done:        [N]               bool       True only at episode boundary
    timestamp:   [N]               float32
    index:       [N]               int64
    task_index:  [N]               int64
    ep_len:      [E]               int64      frames per episode
    ep_offset:   [E]               int64      starting frame offset per episode

Usage:
    /root/autodl-tmp/envs/LeWorldModel/bin/python \
        FLARE-VLA/scripts/libero_plus/01_convert_libero_plus_to_lewm_h5.py \
        --source-dir /root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot \
        --output-dir FLARE-VLA/outputs/libero_plus/h5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_SOURCE = Path("/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/h5")
DEFAULT_IMAGE_SIZE = 224
CAMERA_KEYS = ["observation.images.front", "observation.images.wrist"]


# ─── data reading ────────────────────────────────────────────────────────


def _parse_episode_range(split_spec: str, total_episodes: int) -> list[int]:
    start_s, end_s = split_spec.split(":")
    start = int(start_s)
    end = int(end_s) if int(end_s) > 0 else total_episodes
    return list(range(start, end))


def _read_parquet_files(
    source_dir: Path, episodes: list[int], chunks_size: int
) -> pd.DataFrame:
    """Read and concatenate parquet files for the given episodes (multi-chunk)."""
    dfs = []
    for ep in episodes:
        chunk = f"chunk-{ep // chunks_size:03d}"
        pq_path = source_dir / "data" / chunk / f"episode_{ep:06d}.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"Missing parquet: {pq_path}")
        dfs.append(pd.read_parquet(pq_path))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return df


def _column_to_numpy(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(df[column].to_numpy())


def _build_episode_metadata(
    episode_idx_arr: np.ndarray, total_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    ep_len = []
    ep_offset = []
    cursor = 0
    i = 0
    while i < total_rows:
        start = i
        current_ep = episode_idx_arr[i]
        while i < total_rows and episode_idx_arr[i] == current_ep:
            i += 1
        ep_len.append(i - start)
        ep_offset.append(start)
        cursor = i
    return np.asarray(ep_len, dtype=np.int64), np.asarray(ep_offset, dtype=np.int64)


# ─── video decode ────────────────────────────────────────────────────────


def _decode_video_ffmpeg(video_path: Path, native_h: int, native_w: int) -> np.ndarray:
    """Decode an MP4 (AV1) video to numpy array [T, H, W, 3] uint8 RGB."""
    frame_bytes = native_h * native_w * 3
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-c:v", "libdav1d",
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(
            f"Decoded bytes {len(raw)} not a multiple of frame size {frame_bytes} "
            f"for {video_path}"
        )
    n_frames = len(raw) // frame_bytes
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, native_h, native_w, 3)
    return frames


def _resize_frames_pil(frames: np.ndarray, image_size: int) -> np.ndarray:
    resized = np.empty(
        (len(frames), image_size, image_size, 3), dtype=np.uint8
    )
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img = img.resize((image_size, image_size), Image.BILINEAR)
        resized[i] = np.asarray(img, dtype=np.uint8)
    return resized


# ─── HDF5 writing ────────────────────────────────────────────────────────


def _create_h5_datasets(
    h5f: h5py.File,
    total_rows: int,
    state_dim: int,
    action_dim: int,
    pixel_shape: tuple[int, int, int],
    ep_len: np.ndarray,
    ep_offset: np.ndarray,
    chunk_rows: int,
    compression: str | None,
) -> dict:
    dsets = {}
    scalar_chunks = (chunk_rows,)
    matrix_chunks = (chunk_rows, action_dim) if action_dim > 0 else scalar_chunks
    state_chunks = (chunk_rows, state_dim)

    dsets["pixels"] = h5f.create_dataset(
        "pixels",
        shape=(total_rows,) + pixel_shape,
        dtype=np.uint8,
        chunks=(chunk_rows,) + pixel_shape,
        compression=compression,
    )
    dsets["action"] = h5f.create_dataset(
        "action",
        shape=(total_rows, action_dim),
        dtype=np.float32,
        chunks=matrix_chunks,
        compression=compression,
    )
    dsets["state"] = h5f.create_dataset(
        "state",
        shape=(total_rows, state_dim),
        dtype=np.float32,
        chunks=state_chunks,
        compression=compression,
    )
    dsets["proprio"] = h5f.create_dataset(
        "proprio",
        shape=(total_rows, state_dim),
        dtype=np.float32,
        chunks=state_chunks,
        compression=compression,
    )
    for name in [
        "episode_idx",
        "step_idx",
        "done",
        "timestamp",
        "index",
        "task_index",
    ]:
        dtype = np.bool_ if name == "done" else (
            np.float32 if name == "timestamp" else np.int64
        )
        dsets[name] = h5f.create_dataset(
            name,
            shape=(total_rows,),
            dtype=dtype,
            chunks=scalar_chunks,
            compression=compression,
        )
    dsets["ep_len"] = h5f.create_dataset("ep_len", data=ep_len)
    dsets["ep_offset"] = h5f.create_dataset("ep_offset", data=ep_offset)
    return dsets


def convert_one_camera(
    source_dir: Path,
    output_dir: Path,
    camera_key: str,
    image_size: int,
    compression: str | None,
    chunks_size: int,
    overwrite: bool,
) -> Path:
    """Convert all episodes for one camera into a single LEWM HDF5 file."""

    camera_slug = camera_key.replace(".", "_")
    out_path = output_dir / f"{source_dir.name}__{camera_slug}.h5"

    if out_path.exists():
        if overwrite:
            print(f"  Removing existing: {out_path}")
            out_path.unlink()
        else:
            raise FileExistsError(
                f"Output exists: {out_path}. Use --overwrite to replace."
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load metadata ---
    info = json.loads((source_dir / "meta" / "info.json").read_text())
    total_episodes = int(info["total_episodes"])
    total_frames_expected = int(info["total_frames"])
    split_spec = info["splits"]["train"]
    episodes = _parse_episode_range(split_spec, total_episodes)

    print(f"[{camera_key}] Reading {len(episodes)} episodes from parquet…")
    df = _read_parquet_files(source_dir, episodes, chunks_size)
    total_rows = len(df)
    assert total_rows == total_frames_expected, (
        f"Frame count mismatch: parquet={total_rows}, info.json={total_frames_expected}"
    )
    print(f"[{camera_key}] Total frames: {total_rows}  ({len(episodes)} episodes)")

    # --- dimensions ---
    action_arr = _column_to_numpy(df, "action")
    state_arr = _column_to_numpy(df, "observation.state")
    action_dim = int(action_arr.shape[1])
    state_dim = int(state_arr.shape[1])
    print(f"[{camera_key}] action_dim={action_dim}, state_dim={state_dim}")

    episode_idx_arr = _column_to_numpy(df, "episode_index").astype(np.int64)
    ep_len, ep_offset = _build_episode_metadata(episode_idx_arr, total_rows)
    num_eps = len(ep_len)

    # --- determine native video resolution ---
    video_path_0 = (
        source_dir / "videos" / "chunk-000" / camera_key / "episode_000000.mp4"
    )
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path_0),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=False)
    try:
        native_w, native_h = [int(x) for x in probe.stdout.strip().split(",")]
    except Exception:
        native_h, native_w = 256, 256  # known from info.json

    pixel_shape = (image_size, image_size, 3)
    print(
        f"[{camera_key}] Resolution: {native_w}x{native_h} → {image_size}x{image_size}"
    )

    # --- create HDF5 ---
    print(f"[{camera_key}] Creating: {out_path}")
    comp = None if compression == "none" else compression
    chunk_rows = max(1, min(total_rows, 64))

    h5f = h5py.File(out_path, "w")
    try:
        dsets = _create_h5_datasets(
            h5f=h5f,
            total_rows=total_rows,
            state_dim=state_dim,
            action_dim=action_dim,
            pixel_shape=pixel_shape,
            ep_len=ep_len,
            ep_offset=ep_offset,
            chunk_rows=chunk_rows,
            compression=comp,
        )

        # --- attributes ---
        h5f.attrs["source"] = str(source_dir)
        h5f.attrs["camera_key"] = camera_key
        h5f.attrs["image_size"] = image_size
        h5f.attrs["num_frames"] = total_rows
        h5f.attrs["num_episodes"] = num_eps
        h5f.attrs["fps"] = int(info["fps"])
        h5f.attrs["format"] = "lewm_hdf5_from_lerobot_v21"
        h5f.attrs["generated_by"] = "01_convert_libero_plus_to_lewm_h5.py"

        # --- write tabular data ---
        print(f"[{camera_key}] Writing tabular data…")
        dsets["action"][:] = action_arr
        dsets["state"][:] = state_arr
        dsets["proprio"][:] = state_arr
        dsets["episode_idx"][:] = episode_idx_arr
        dsets["step_idx"][:] = _column_to_numpy(df, "frame_index").astype(np.int64)
        dsets["timestamp"][:] = _column_to_numpy(df, "timestamp").astype(np.float32)
        dsets["index"][:] = _column_to_numpy(df, "index").astype(np.int64)
        dsets["task_index"][:] = _column_to_numpy(df, "task_index").astype(np.int64)

        # done = True only at last frame of each episode
        done = np.zeros(total_rows, dtype=np.bool_)
        ep_ends = (ep_offset + ep_len) - 1
        done[ep_ends] = True
        dsets["done"][:] = done

        h5f.flush()

        # --- decode videos & write pixels ---
        print(f"[{camera_key}] Decoding videos (ffmpeg libdav1d) & writing pixels…")
        t0 = time.perf_counter()
        cursor = 0
        heartbeat_s = 10.0
        last_hb = t0

        for ep_pos, ep_idx in enumerate(episodes):
            ep_count = int(ep_len[ep_pos])
            chunk = f"chunk-{ep_idx // chunks_size:03d}"
            video_path = (
                source_dir / "videos" / chunk
                / camera_key / f"episode_{ep_idx:06d}.mp4"
            )
            if not video_path.exists():
                raise FileNotFoundError(f"Missing video: {video_path}")

            frames = _decode_video_ffmpeg(video_path, native_h, native_w)

            # Handle frame count mismatch gracefully
            if len(frames) != ep_count:
                print(
                    f"  NOTE: ep {ep_idx}: video frames={len(frames)} "
                    f"!= parquet rows={ep_count}. Using min."
                )
                use_len = min(len(frames), ep_count)
                frames = frames[:use_len]
                actual_count = use_len
            else:
                actual_count = ep_count

            # Resize
            if image_size != native_h or image_size != native_w:
                frames = _resize_frames_pil(frames, image_size)

            # Write to HDF5
            row_end = cursor + actual_count
            dsets["pixels"][cursor:row_end] = frames
            cursor = row_end

            # Progress heartbeat
            now = time.perf_counter()
            if now - last_hb >= heartbeat_s or ep_pos == len(episodes) - 1:
                pct = 100.0 * (ep_pos + 1) / len(episodes)
                elapsed = now - t0
                rate = (ep_pos + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(episodes) - (ep_pos + 1)) / rate if rate > 0 else 0.0
                print(
                    f"[{camera_key}] {ep_pos + 1}/{len(episodes)} episodes "
                    f"({pct:.1f}%) | {rate:.2f} ep/s | ETA {eta:.0f}s"
                    + (f"  (cursor={cursor})" if ep_pos == len(episodes) - 1 else "")
                )
                last_hb = now

        elapsed = time.perf_counter() - t0
        print(
            f"[{camera_key}] Done! {total_rows} frames in {elapsed:.1f}s "
            f"({total_rows / elapsed:.1f} fps decode+resize+write)"
        )

    finally:
        h5f.close()

    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[{camera_key}] Output: {out_path}  ({file_mb:.1f} MiB)")
    return out_path


# ─── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert Sylvest/libero_plus_lerobot → LEWM HDF5 (multi-chunk)"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--camera-keys", type=str, nargs="*", default=None)
    parser.add_argument(
        "--compression", type=str, default="lzf",
        choices=["none", "lzf", "gzip"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source not found: {source_dir}")
    if not (source_dir / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Not a LeRobot dataset (no meta/info.json): {source_dir}")

    info = json.loads((source_dir / "meta" / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    print(f"Source:      {source_dir}")
    print(f"Codebase:    {info.get('codebase_version')}")
    print(f"Episodes:    {info['total_episodes']}")
    print(f"Frames:      {info['total_frames']}")
    print(f"FPS:         {info['fps']}")
    print(f"Chunks size: {chunks_size}")

    camera_keys = args.camera_keys or CAMERA_KEYS
    available = [
        c for c in camera_keys
        if (source_dir / "videos" / "chunk-000" / c).is_dir()
    ]
    if not available:
        raise RuntimeError(f"No camera dirs found. Tried: {camera_keys}")
    print(f"Cameras:     {available}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Image size:  {args.image_size}")
    print()

    for cam in available:
        print(f"{'=' * 65}")
        convert_one_camera(
            source_dir=source_dir,
            output_dir=args.output_dir,
            camera_key=cam,
            image_size=args.image_size,
            compression=args.compression if args.compression != "none" else None,
            chunks_size=chunks_size,
            overwrite=args.overwrite,
        )
        print()

    print("All done!")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
