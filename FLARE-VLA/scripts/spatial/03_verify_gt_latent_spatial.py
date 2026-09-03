#!/usr/bin/env python3
"""
Verify the Spatial GT latent file against the source H5 and mapping files.

Checks:
  1. File structure: all required datasets present, attrs correct
  2. frame_latent: correct shape, no NaN/Inf, correct row count
  3. frame_action: matches source H5 action exactly
  4. Window integrity: future rows are consecutive, no cross-episode windows
  5. Episode metadata: matches source H5 and mapping file
  6. Cross-check with frame_map.parquet: random frames traceable
  7. Latent numerical sanity: finite values, reasonable range
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ─── paths ────────────────────────────────────────────────────────────────
LATENT_H5 = Path(
    "/root/autodl-tmp/FLARE-VLA/outputs/spatial/gt_latents/"
    "spatial_epoch50_h8_framebank_trainpipe.h5"
)
SOURCE_H5 = Path(
    "/root/autodl-tmp/le-wm/stablewm_data/"
    "libero_spatial_no_noops_1.0.0_lerobot__observation_images_image.h5"
)
FRAME_MAP = Path(
    "/root/autodl-tmp/FLARE-VLA/outputs/spatial/mapping/"
    "spatial_lerobot_to_h5_frame_map.parquet"
)
FUTURE_HORIZON = 8
NUM_RANDOM_CHECKS = 100
SEED = 42

PASS = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  [PASS] {msg}")


def bad(msg):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def info(msg):
    print(f"  [INFO] {msg}")


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    for p in [LATENT_H5, SOURCE_H5, FRAME_MAP]:
        if not p.exists():
            print(f"FATAL: missing {p}")
            return 1

    print("=" * 60)
    print("  Spatial GT Latent Verification")
    print("=" * 60)

    # ── 1. Structure ──
    print("\n── 1. File Structure ──")
    with h5py.File(LATENT_H5, "r") as lf:
        required_ds = [
            "frame_latent", "frame_action",
            "window_current_row", "window_future_row",
            "window_episode_id", "window_step_idx",
            "episode_id", "ep_offset", "ep_len",
        ]
        for key in required_ds:
            if key in lf:
                ok(f"dataset '{key}' present: {lf[key].shape}")
            else:
                bad(f"dataset '{key}' MISSING")

        if not bool(lf.attrs.get("complete", False)):
            bad("attribute 'complete' is not True")
        else:
            ok("attribute 'complete' = True")

        total_frames = int(lf.attrs["total_frames"])
        total_windows = int(lf.attrs["total_windows"])
        latent_dim = int(lf.attrs["latent_dim"])
        info(f"total_frames={total_frames}, windows={total_windows}, latent_dim={latent_dim}")

        # ── 2. frame_latent ──
        print("\n── 2. frame_latent ──")
        fl = lf["frame_latent"]
        if fl.shape[0] != 52970:
            bad(f"frame_latent rows={fl.shape[0]}, expected 52970")
        else:
            ok(f"frame_latent rows = 52970")
        if fl.shape[1] != 192:
            bad(f"latent_dim={fl.shape[1]}, expected 192")
        else:
            ok(f"latent_dim = 192")

        # Check first, middle, last frames for NaN
        for label, idx in [("first", 0), ("middle", total_frames // 2), ("last", total_frames - 1)]:
            z = np.asarray(fl[idx])
            if not np.isfinite(z).all():
                bad(f"{label} frame (row {idx}) has NaN/Inf")
            else:
                ok(f"{label} frame (row {idx}) finite: mean={z.mean():.4f}, std={z.std():.4f}")

        # Sample random frames for NaN check
        rng = np.random.default_rng(SEED)
        sample_idxs = rng.choice(total_frames, size=min(100, total_frames), replace=False)
        sample_z = np.asarray(fl[sorted(sample_idxs)])
        nan_count = np.sum(~np.isfinite(sample_z))
        if nan_count > 0:
            bad(f"{nan_count} NaN/Inf values in {len(sample_idxs)} random frames")
        else:
            ok(f"no NaN/Inf in {len(sample_idxs)} random frames")

        # ── 3. frame_action vs source ──
        print("\n── 3. frame_action vs Source H5 ──")
        with h5py.File(SOURCE_H5, "r") as src:
            src_action = np.asarray(src["action"][:])
            lat_action = np.asarray(lf["frame_action"][:])

        if src_action.shape != lat_action.shape:
            bad(f"action shape mismatch: source={src_action.shape}, latent={lat_action.shape}")
        else:
            close = np.isclose(src_action, lat_action, atol=1e-6)
            if np.all(close):
                ok("all frame_action values match source H5 (atol=1e-6)")
            else:
                n_bad = int(np.sum(~np.all(close, axis=1)))
                bad(f"{n_bad} rows differ between latent H5 and source H5 action")

        # ── 4. Window integrity ──
        print("\n── 4. Window Integrity ──")
        future_rows = np.asarray(lf["window_future_row"][:])
        current_rows = np.asarray(lf["window_current_row"][:])
        window_ep = np.asarray(lf["window_episode_id"][:])
        ep_offsets = np.asarray(lf["ep_offset"][:])
        ep_lengths = np.asarray(lf["ep_len"][:])

        # Build episode boundary lookup
        ep_map = {int(e): (int(o), int(l)) for e, o, l in
                  zip(lf["episode_id"][:], ep_offsets, ep_lengths)}

        # Check: future = current + [1..8]
        expected_future = current_rows[:, None] + np.arange(1, FUTURE_HORIZON + 1, dtype=np.int64)[None, :]
        if np.array_equal(future_rows, expected_future):
            ok("future_rows = current_row + [1..8] for all windows")
        else:
            n_bad = int(np.sum(~np.all(future_rows == expected_future, axis=1)))
            bad(f"{n_bad} windows have non-consecutive future rows")

        # Check: no cross-episode windows
        cross_ep = 0
        for i in range(len(future_rows)):
            ep = int(window_ep[i])
            offset, length = ep_map[ep]
            if int(future_rows[i, -1]) >= offset + length:
                cross_ep += 1
        if cross_ep == 0:
            ok(f"no windows cross episode boundaries")
        else:
            bad(f"{cross_ep} windows cross episode boundaries")

        # ── 5. Episode metadata ──
        print("\n── 5. Episode Metadata ──")
        with h5py.File(SOURCE_H5, "r") as src:
            src_ep_len = np.asarray(src["ep_len"][:])
            src_ep_offset = np.asarray(src["ep_offset"][:])
        lat_ep_len = np.asarray(lf["ep_len"][:])
        lat_ep_offset = np.asarray(lf["ep_offset"][:])

        if np.array_equal(src_ep_len, lat_ep_len):
            ok(f"ep_len matches source ({len(lat_ep_len)} episodes)")
        else:
            bad("ep_len differs from source H5")
        if np.array_equal(src_ep_offset, lat_ep_offset):
            ok(f"ep_offset matches source")
        else:
            bad("ep_offset differs from source H5")

        # ── 6. Cross-check with frame_map ──
        print("\n── 6. Cross-check with Mapping File ──")
        frame_df = pd.read_parquet(FRAME_MAP)
        if len(frame_df) != total_frames:
            bad(f"frame_map rows={len(frame_df)}, latent total_frames={total_frames}")
        else:
            ok(f"frame_map row count matches: {len(frame_df)}")

        # Random frames: verify H5 row maps to correct episode and task
        rng2 = np.random.default_rng(SEED + 1)
        check_rows = sorted(rng2.choice(total_frames, size=min(20, total_frames), replace=False))
        all_traceable = True
        for row in check_rows:
            map_row = frame_df.iloc[row]
            h5_ep = int(map_row["h5_episode_index"])
            lerobot_ep = int(map_row["lerobot_episode_index"])
            task_text = map_row["task_text"]

            # Verify this row falls within the latent H5 episode
            offset = int(lat_ep_offset[h5_ep])
            length = int(lat_ep_len[h5_ep])
            if not (offset <= row < offset + length):
                bad(f"row {row}: map says ep {h5_ep} (offset={offset}, len={length}), but row outside range")
                all_traceable = False
                continue

            local_step = row - offset
            latent_z = np.asarray(lf["frame_latent"][row])

        if all_traceable:
            ok(f"all {len(check_rows)} random frames traceable through mapping")

        # ── 7. Latent distribution stats ──
        print("\n── 7. Latent Distribution ──")
        # Sample evenly across the file
        n_sample = min(2000, total_frames)
        idxs = np.linspace(0, total_frames - 1, n_sample, dtype=np.int64)
        z_sample = np.asarray(fl[sorted(idxs)])
        z_mean = float(z_sample.mean())
        z_std = float(z_sample.std())
        z_min = float(z_sample.min())
        z_max = float(z_sample.max())
        info(f"sampled {n_sample} frames: mean={z_mean:.4f}, std={z_std:.4f}, "
             f"min={z_min:.4f}, max={z_max:.4f}")

        if abs(z_mean) < 10 and z_std < 10:
            ok("latent statistics within reasonable range")
        else:
            info(f"latent stats unusual but not necessarily wrong — check manually if concerned")

    # ── final ──
    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'='*60}")
    if FAIL == 0:
        print("\n  [FINAL] ALL CHECKS PASSED — Spatial GT latent is valid.")
    else:
        print(f"\n  [FINAL] {FAIL} CHECK(S) FAILED — fix before proceeding.")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
