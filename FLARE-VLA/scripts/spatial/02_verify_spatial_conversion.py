#!/usr/bin/env python3
"""
Verify LIBERO-Spatial conversion chain end-to-end.

Checks:
  1. H5 episode count == source episode count
  2. Total frames match
  3. First/middle/last frame traceable (action, state, pixels)
  4. Action element-wise exact match (all frames)
  5. State element-wise exact match (all frames)
  6. Image visual consistency (spot-check PSNR)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

# --- defaults ---
DEFAULT_H5 = Path(
    "/root/autodl-tmp/FLARE-VLA/outputs/spatial/h5/"
    "libero_spatial_no_noops_1.0.0_lerobot__observation_images_image.h5"
)
DEFAULT_SOURCE = Path(
    "/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/"
    "libero_spatial_no_noops_1.0.0_lerobot"
)
DEFAULT_MAPPING_DIR = Path("/root/autodl-tmp/FLARE-VLA/outputs/spatial/mapping")

PASS_COUNT = 0
FAIL_COUNT = 0


def fail(msg: str):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"\n  [FAIL] {msg}")


def ok(msg: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  [PASS] {msg}")


def info(msg: str):
    print(f"  [INFO] {msg}")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─── check 1 & 2: counts ─────────────────────────────────────────────────

def check_counts(h5f: h5py.File, info_json: dict, episodes_meta: list[dict]):
    section("1. Episode Count Check")
    n_h5 = len(np.asarray(h5f["ep_len"][:]))
    n_info = int(info_json["total_episodes"])
    n_meta = len(episodes_meta)
    if n_h5 == n_info == n_meta:
        ok(f"H5 episodes={n_h5}, info.json={n_info}, episodes.jsonl={n_meta} — all match")
    else:
        fail(f"Count mismatch: H5={n_h5}, info.json={n_info}, episodes.jsonl={n_meta}")

    section("2. Total Frame Count Check")
    n_h5_frames = int(h5f["action"].shape[0])
    n_info_frames = int(info_json["total_frames"])

    # sum from episodes.jsonl
    total_from_meta = sum(int(e["length"]) for e in episodes_meta)

    if n_h5_frames == n_info_frames == total_from_meta:
        ok(f"Total frames: H5={n_h5_frames}, info.json={n_info_frames}, episodes.jsonl={total_from_meta}")
    else:
        fail(f"Frame count mismatch: H5={n_h5_frames}, info.json={n_info_frames}, episodes.jsonl sum={total_from_meta}")

    return n_h5, n_h5_frames


# ─── check 3: first/middle/last frame traceable ────────────────────────────

def check_traceability(
    h5f: h5py.File,
    source_dir: Path,
    episodes_meta: list[dict],
    ep_offsets: np.ndarray,
    ep_lengths: np.ndarray,
    image_size: int,
):
    section("3. First/Middle/Last Frame Traceability")

    total_eps = len(episodes_meta)
    check_episodes = [0, total_eps // 2, total_eps - 1]
    check_labels = ["first", "middle", "last"]

    for ep_idx, label in zip(check_episodes, check_labels):
        meta = episodes_meta[ep_idx]
        lerobot_ep = int(meta["episode_index"])
        ep_len = int(meta["length"])
        h5_start = int(ep_offsets[ep_idx])

        # Check 3 frames within this episode: first, middle, last
        local_steps = [0, ep_len // 2, ep_len - 1]

        for ls in local_steps:
            ls_label = "first" if ls == 0 else ("last" if ls == ep_len - 1 else "mid")
            h5_row = h5_start + ls

            # 3a. Compare action with parquet
            pq_path = source_dir / "data" / "chunk-000" / f"episode_{lerobot_ep:06d}.parquet"
            pq_df = pd.read_parquet(pq_path)
            pq_action = np.stack(pq_df["action"].iloc[ls].tolist()).astype(np.float32) if pq_df["action"].dtype == object else np.asarray(pq_df["action"].iloc[ls], dtype=np.float32)
            pq_action = pq_action.reshape(-1)
            h5_action = np.asarray(h5f["action"][h5_row], dtype=np.float32).reshape(-1)
            if not np.allclose(pq_action, h5_action, atol=1e-6):
                diff = np.abs(pq_action.astype(np.float64) - h5_action.astype(np.float64))
                fail(f"Ep {ep_idx} ({label}) step {ls} ({ls_label}) action mismatch: max_diff={diff.max():.2e}")
            else:
                ok(f"Ep {ep_idx} ({label}) step {ls} ({ls_label}): action ok (h5_row={h5_row})")

            # 3b. Compare state with parquet
            pq_state = np.stack(pq_df["observation.state"].iloc[ls].tolist()).astype(np.float32) if pq_df["observation.state"].dtype == object else np.asarray(pq_df["observation.state"].iloc[ls], dtype=np.float32)
            pq_state = pq_state.reshape(-1)
            h5_state = np.asarray(h5f["state"][h5_row], dtype=np.float32).reshape(-1)
            if not np.allclose(pq_state, h5_state, atol=1e-6):
                diff = np.abs(pq_state.astype(np.float64) - h5_state.astype(np.float64))
                fail(f"Ep {ep_idx} ({label}) step {ls} ({ls_label}) state mismatch: max_diff={diff.max():.2e}")
            else:
                ok(f"Ep {ep_idx} ({label}) step {ls} ({ls_label}): state ok (h5_row={h5_row})")

            # 3c. Compare pixels with decoded MP4
            h5_pixels = np.asarray(h5f["pixels"][h5_row])
            # Decode single frame from video
            video_path = source_dir / "videos" / "chunk-000" / "observation.images.image" / f"episode_{lerobot_ep:06d}.mp4"
            native_h, native_w = 256, 256
            frame_bytes = native_h * native_w * 3
            # Use ffmpeg to extract single frame
            cmd = [
                "ffmpeg", "-v", "error",
                "-i", str(video_path),
                "-vf", f"select=eq(n\\,{ls})",
                "-vframes", "1",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1",
            ]
            proc = subprocess.run(cmd, capture_output=True, check=False)
            if proc.returncode != 0:
                fail(f"Ep {ep_idx} ({label}) step {ls}: ffmpeg failed to extract frame from {video_path}")
            else:
                raw_frame = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(native_h, native_w, 3)
                # Resize to match H5
                ref_frame = np.array(Image.fromarray(raw_frame).resize((image_size, image_size), Image.BILINEAR), dtype=np.uint8)
                if not np.array_equal(h5_pixels, ref_frame):
                    # Compute PSNR
                    mse = np.mean((h5_pixels.astype(np.float64) - ref_frame.astype(np.float64)) ** 2)
                    psnr = 20 * np.log10(255.0) - 10 * np.log10(mse) if mse > 0 else float("inf")
                    if mse > 1.0:  # More than 1 pixel diff on average
                        fail(f"Ep {ep_idx} ({label}) step {ls}: pixel mismatch, MSE={mse:.2f}, PSNR={psnr:.1f}dB")
                    else:
                        ok(f"Ep {ep_idx} ({label}) step {ls}: pixels ok (PSNR={psnr:.1f}dB)")
                else:
                    ok(f"Ep {ep_idx} ({label}) step {ls}: pixels exactly match (h5_row={h5_row})")


# ─── check 4 & 5: full action/state comparison ─────────────────────────────

def check_action_state_full(h5f: h5py.File, source_dir: Path, episodes_meta: list[dict], ep_offsets: np.ndarray):
    section("4. Full Action Element-wise Check")
    info("Comparing all 52970 actions (this may take a minute)...")

    total_bad = 0
    max_diff = 0.0
    first_bad = None

    for ep_idx, meta in enumerate(episodes_meta):
        lerobot_ep = int(meta["episode_index"])
        ep_len = int(meta["length"])
        h5_start = int(ep_offsets[ep_idx])
        h5_end = h5_start + ep_len

        pq_path = source_dir / "data" / "chunk-000" / f"episode_{lerobot_ep:06d}.parquet"
        pq_df = pd.read_parquet(pq_path)

        if pq_df["action"].dtype == object:
            pq_actions = np.stack(pq_df["action"].tolist()).astype(np.float32)
        else:
            pq_actions = np.asarray(pq_df["action"].values.tolist(), dtype=np.float32)

        h5_actions = np.asarray(h5f["action"][h5_start:h5_end], dtype=np.float32)

        close = np.isclose(pq_actions, h5_actions, atol=1e-6)
        if not np.all(close):
            bad_rows = np.flatnonzero(~np.all(close, axis=1))
            total_bad += len(bad_rows)
            diff = np.abs(pq_actions.astype(np.float64) - h5_actions.astype(np.float64))
            ep_max = float(diff.max())
            max_diff = max(max_diff, ep_max)
            if first_bad is None:
                bad_local = int(bad_rows[0])
                first_bad = (ep_idx, bad_local, pq_actions[bad_local], h5_actions[bad_local])

        if (ep_idx + 1) % 100 == 0:
            info(f"  Checked {ep_idx+1}/{len(episodes_meta)} episodes...")

    if total_bad == 0:
        ok(f"All 52970 actions match exactly (atol=1e-6)")
    else:
        ep, ls, pq, h5 = first_bad
        fail(f"{total_bad} action mismatches. First: ep={ep} step={ls}, max_diff={max_diff:.2e}")
        info(f"  parquet: {pq}")
        info(f"  h5:      {h5}")

    section("5. Full State Element-wise Check")
    info("Comparing all 52970 states...")

    total_bad = 0
    max_diff = 0.0
    first_bad = None

    for ep_idx, meta in enumerate(episodes_meta):
        lerobot_ep = int(meta["episode_index"])
        ep_len = int(meta["length"])
        h5_start = int(ep_offsets[ep_idx])
        h5_end = h5_start + ep_len

        pq_path = source_dir / "data" / "chunk-000" / f"episode_{lerobot_ep:06d}.parquet"
        pq_df = pd.read_parquet(pq_path)

        if pq_df["observation.state"].dtype == object:
            pq_states = np.stack(pq_df["observation.state"].tolist()).astype(np.float32)
        else:
            pq_states = np.asarray(pq_df["observation.state"].values.tolist(), dtype=np.float32)

        h5_states = np.asarray(h5f["state"][h5_start:h5_end], dtype=np.float32)

        close = np.isclose(pq_states, h5_states, atol=1e-6)
        if not np.all(close):
            bad_rows = np.flatnonzero(~np.all(close, axis=1))
            total_bad += len(bad_rows)
            diff = np.abs(pq_states.astype(np.float64) - h5_states.astype(np.float64))
            ep_max = float(diff.max())
            max_diff = max(max_diff, ep_max)
            if first_bad is None:
                bad_local = int(bad_rows[0])
                first_bad = (ep_idx, bad_local, pq_states[bad_local], h5_states[bad_local])

        if (ep_idx + 1) % 100 == 0:
            info(f"  Checked {ep_idx+1}/{len(episodes_meta)} episodes...")

    if total_bad == 0:
        ok(f"All 52970 states match exactly (atol=1e-6)")
    else:
        ep, ls, pq, h5 = first_bad
        fail(f"{total_bad} state mismatches. First: ep={ep} step={ls}, max_diff={max_diff:.2e}")
        info(f"  parquet: {pq}")
        info(f"  h5:      {h5}")


# ─── check 6: image visual consistency ─────────────────────────────────────

def check_image_consistency(
    h5f: h5py.File,
    source_dir: Path,
    episodes_meta: list[dict],
    ep_offsets: np.ndarray,
    image_size: int,
    num_spot_checks: int = 15,
):
    section("6. Image Visual Consistency (Spot Checks)")

    np.random.seed(42)
    total_frames = int(h5f["action"].shape[0])
    check_rows = np.random.choice(total_frames, size=min(num_spot_checks, total_frames), replace=False)
    check_rows = sorted(check_rows)

    # Build episode index lookup for fast episode finding
    ep_boundaries = np.concatenate([ep_offsets, np.array([total_frames])])

    all_ok = True
    min_psnr = float("inf")

    for row in check_rows:
        # Find which episode this row belongs to
        ep_idx = int(np.searchsorted(ep_boundaries, row, side="right") - 1)
        lerobot_ep = int(episodes_meta[ep_idx]["episode_index"])
        local_step = row - int(ep_offsets[ep_idx])

        h5_pixels = np.asarray(h5f["pixels"][row])
        video_path = source_dir / "videos" / "chunk-000" / "observation.images.image" / f"episode_{lerobot_ep:06d}.mp4"

        native_h, native_w = 256, 256
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{local_step})",
            "-vframes", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            fail(f"Row {row} (ep={ep_idx}, ls={local_step}): ffmpeg failed")
            all_ok = False
            continue

        raw_frame = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(native_h, native_w, 3)
        ref_frame = np.array(Image.fromarray(raw_frame).resize((image_size, image_size), Image.BILINEAR), dtype=np.uint8)

        mse = np.mean((h5_pixels.astype(np.float64) - ref_frame.astype(np.float64)) ** 2)
        if mse > 0:
            psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
            min_psnr = min(min_psnr, psnr)
            if mse > 2.0:
                fail(f"Row {row} (ep={ep_idx}, ls={local_step}): MSE={mse:.2f}, PSNR={psnr:.1f}dB")
                all_ok = False
            else:
                ok(f"Row {row} (ep={ep_idx}, ls={local_step}): PSNR={psnr:.1f}dB")

    if all_ok:
        ok(f"All {len(check_rows)} spot checks pass (min PSNR={min_psnr:.1f}dB)")
    else:
        info(f"Min PSNR across all checks: {min_psnr:.1f}dB")


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    global PASS_COUNT, FAIL_COUNT

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mapping-dir", type=Path, default=DEFAULT_MAPPING_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-image-checks", type=int, default=15)
    args = parser.parse_args()

    for p in [args.h5_path, args.source_dir]:
        if not p.exists():
            print(f"ERROR: not found: {p}")
            sys.exit(1)

    # Load metadata
    info_json = json.loads((args.source_dir / "meta" / "info.json").read_text())
    episodes_meta = []
    with open(args.source_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes_meta.append(json.loads(line))

    # Load H5
    h5f = h5py.File(args.h5_path, "r")

    print("=" * 60)
    print("  LIBERO-Spatial Conversion Verification")
    print("=" * 60)
    print(f"  H5:       {args.h5_path}")
    print(f"  Source:   {args.source_dir}")
    print(f"  Suite:    {info_json.get('codebase_version', 'unknown')}")
    print(f"  Image sz: {args.image_size}")

    # Build offsets
    ep_lengths = np.asarray([int(e["length"]) for e in episodes_meta], dtype=np.int64)
    ep_offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(ep_lengths[:-1])])

    # Run checks
    n_ep, n_frames = check_counts(h5f, info_json, episodes_meta)
    check_traceability(h5f, args.source_dir, episodes_meta, ep_offsets, ep_lengths, args.image_size)
    check_action_state_full(h5f, args.source_dir, episodes_meta, ep_offsets)
    check_image_consistency(h5f, args.source_dir, episodes_meta, ep_offsets, args.image_size, args.num_image_checks)

    h5f.close()

    # Final report
    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"{'='*60}")
    if FAIL_COUNT == 0:
        print("\n  [FINAL VERDICT] ALL CHECKS PASSED")
        print("  The LeRobot -> H5 chain is fully verified.")
    else:
        print(f"\n  [FINAL VERDICT] {FAIL_COUNT} CHECK(S) FAILED")
        print("  Do NOT proceed to training until all issues are resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
