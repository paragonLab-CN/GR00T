#!/usr/bin/env python3
"""
Extract frame-level GT latents for LIBERO-Spatial no-noops.

Uses the training HDF5Dataset pipeline for image preprocessing consistency
(v3 approach), but builds window metadata from scratch (v1 approach) —
no dependency on old latent files.

Output H5 structure:
    frame_latent       [N, D]     float32   per-frame encoder embeddings
    frame_action       [N, 7]     float32   copied from source H5
    window_current_row [W]        int64     t for each valid window
    window_future_row  [W, 8]     int64     [t+1, ..., t+8]
    window_episode_id  [W]        int64
    window_step_idx    [W]        int64
    episode_id         [E]        int64
    ep_offset          [E]        int64
    ep_len             [E]        int64

Spatial-specific — does NOT read or modify any LIBERO-90 outputs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/le-wm")
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from torch.utils.data import DataLoader, Dataset

from utils import get_img_preprocessor

# ─── Libero-plus-specific paths ───────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "stablewm_data"
DATASET_NAME = "libero_plus_lerobot__observation_images_front"
SOURCE_H5 = DATA_ROOT / f"{DATASET_NAME}.h5"

CHECKPOINT_PATH = DATA_ROOT / "checkpoints/libero_plus_lewm_full/weights_epoch_9.pt"

OUTPUT_DIR = Path("/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/gt_latents")
OUTPUT_PATH = OUTPUT_DIR / "libero_plus_epoch9_h8_framebank_trainpipe.h5"

FUTURE_HORIZON = 8
FRAMESKIP = 1
BATCH_SIZE = 128
NUM_WORKERS = 0
PRINT_EVERY = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DTYPE = np.float32
OVERWRITE = False


# ─── helpers ──────────────────────────────────────────────────────────────

def check_paths():
    required = [
        PROJECT_ROOT / "config/train/lewm.yaml",
        PROJECT_ROOT / "config/train/data/libero_spatial.yaml",
        SOURCE_H5,
        CHECKPOINT_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files:\n" + "\n".join(f"  - {p}" for p in missing)
        )


def load_config():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "config/train"),
    ):
        return compose(
            config_name="lewm",
            overrides=["data=libero_plus"],
        )


def load_episode_layout(source):
    """Read episode boundaries from source H5."""
    if "ep_offset" in source and "ep_len" in source:
        ep_offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
        ep_lengths = np.asarray(source["ep_len"][:], dtype=np.int64)
        if "episode_idx" in source:
            frame_episode = np.asarray(source["episode_idx"][:], dtype=np.int64)
            episode_ids = frame_episode[ep_offsets]
        else:
            episode_ids = np.arange(len(ep_lengths), dtype=np.int64)
    else:
        raise KeyError("H5 missing ep_offset/ep_len")

    # Validate contiguous layout
    expected_next = ep_offsets[:-1] + ep_lengths[:-1]
    if not np.array_equal(expected_next, ep_offsets[1:]):
        raise RuntimeError("Episodes are not contiguous in H5 — stopping to avoid misalignment.")
    if int(ep_offsets[-1] + ep_lengths[-1]) != int(source["pixels"].shape[0]):
        raise RuntimeError("Last episode end != total frames.")

    return episode_ids, ep_offsets, ep_lengths


def make_transform(cfg):
    return spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=int(cfg.img_size))
    )


def build_frame_dataset(cfg):
    """Single-frame HDF5Dataset with training transform — ensures pixel preprocessing
    is IDENTICAL to what the model saw during training."""
    ds = swm.data.HDF5Dataset(
        name=DATASET_NAME,
        num_steps=1,
        frameskip=FRAMESKIP,
        keys_to_load=["pixels"],
        keys_to_cache=[],
        transform=None,
        cache_dir=str(DATA_ROOT),
    )
    ds.transform = make_transform(cfg)
    return ds


def close_dataset(ds):
    h5_file = getattr(ds, "h5_file", None)
    if h5_file is not None:
        try:
            h5_file.close()
        finally:
            ds.h5_file = None


def build_model(cfg, action_dim):
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = int(FRAMESKIP) * int(action_dim)

    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing model_state_dict. Keys: {list(ckpt.keys())}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()
    model.requires_grad_(False)
    return model, ckpt


# ─── window metadata (v1 approach — built from scratch) ───────────────────

def create_window_metadata(output, episode_ids, ep_offsets, ep_lengths):
    counts = np.maximum(ep_lengths - FUTURE_HORIZON, 0)
    total_windows = int(counts.sum())
    if total_windows <= 0:
        raise RuntimeError(f"No episodes longer than FUTURE_HORIZON={FUTURE_HORIZON}")

    output.create_dataset("episode_id", data=episode_ids, dtype="i8")
    output.create_dataset("ep_offset", data=ep_offsets, dtype="i8")
    output.create_dataset("ep_len", data=ep_lengths, dtype="i8")

    idx_chunk = max(1, min(total_windows, 65536))
    fut_chunk = max(1, min(total_windows, 8192))

    d_current = output.create_dataset(
        "window_current_row", shape=(total_windows,), dtype="i8",
        chunks=(idx_chunk,), compression="gzip", compression_opts=1)
    d_future = output.create_dataset(
        "window_future_row", shape=(total_windows, FUTURE_HORIZON), dtype="i8",
        chunks=(fut_chunk, FUTURE_HORIZON), compression="gzip", compression_opts=1)
    d_episode = output.create_dataset(
        "window_episode_id", shape=(total_windows,), dtype="i8",
        chunks=(idx_chunk,), compression="gzip", compression_opts=1)
    d_step = output.create_dataset(
        "window_step_idx", shape=(total_windows,), dtype="i8",
        chunks=(idx_chunk,), compression="gzip", compression_opts=1)

    cursor = 0
    future_delta = np.arange(1, FUTURE_HORIZON + 1, dtype=np.int64)

    for ep_id, offset, length, count in zip(episode_ids, ep_offsets, ep_lengths, counts):
        count = int(count)
        if count == 0:
            continue
        current_rows = int(offset) + np.arange(count, dtype=np.int64)
        future_rows = current_rows[:, None] + future_delta[None, :]
        sl = slice(cursor, cursor + count)
        d_current[sl] = current_rows
        d_future[sl] = future_rows
        d_episode[sl] = int(ep_id)
        d_step[sl] = np.arange(count, dtype=np.int64)
        cursor += count

    assert cursor == total_windows
    output.attrs["total_windows"] = total_windows
    output.attrs["window_metadata_ready"] = True
    output.flush()
    return total_windows


def copy_frame_actions(source, output, total_frames, action_dim):
    if "frame_action" in output:
        return
    d = output.create_dataset(
        "frame_action", shape=(total_frames, action_dim),
        dtype=source["action"].dtype,
        chunks=(min(total_frames, 65536), action_dim),
        compression="gzip", compression_opts=1)
    copy_batch = 65536
    for start in range(0, total_frames, copy_batch):
        end = min(start + copy_batch, total_frames)
        d[start:end] = source["action"][start:end]
    output.attrs["frame_action_ready"] = True
    output.flush()


def open_or_create_output(source, checkpoint, total_frames, action_dim,
                          episode_ids, ep_offsets, ep_lengths):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists() and OVERWRITE:
        OUTPUT_PATH.unlink()

    is_new = not OUTPUT_PATH.exists()
    output = h5py.File(OUTPUT_PATH, "a")

    if is_new:
        output.attrs["format"] = "lewm_gt_latent_framebank_spatial_v1"
        output.attrs["source_h5"] = str(SOURCE_H5)
        output.attrs["dataset_name"] = DATASET_NAME
        output.attrs["checkpoint_path"] = str(CHECKPOINT_PATH)
        output.attrs["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
        output.attrs["frameskip"] = int(FRAMESKIP)
        output.attrs["future_horizon"] = int(FUTURE_HORIZON)
        output.attrs["latent_layer"] = "encoder_cls_then_projector"
        output.attrs["total_frames"] = int(total_frames)
        output.attrs["action_dim"] = int(action_dim)
        output.attrs["frames_written"] = 0
        output.attrs["complete"] = False

        create_window_metadata(output, episode_ids, ep_offsets, ep_lengths)
        copy_frame_actions(source, output, total_frames, action_dim)
    else:
        expected = {
            "source_h5": str(SOURCE_H5),
            "checkpoint_path": str(CHECKPOINT_PATH),
            "future_horizon": int(FUTURE_HORIZON),
            "total_frames": int(total_frames),
        }
        for key, value in expected.items():
            saved = output.attrs.get(key)
            if saved != value:
                output.close()
                raise RuntimeError(
                    f"Existing output attr mismatch: {key}={saved!r} vs {value!r}. "
                    "Use OVERWRITE=True or change OUTPUT_PATH."
                )
    return output


# ─── main extraction ──────────────────────────────────────────────────────

class FrameDatasetWithRow(Dataset):
    """Wraps HDF5Dataset for sequential frame encoding with row tracking."""

    def __init__(self, base_dataset, row_map, start_item):
        self.base = base_dataset
        self.row_map = row_map
        self.start_item = int(start_item)

    def __len__(self):
        return len(self.base) - self.start_item

    def __getitem__(self, local_idx):
        item_idx = self.start_item + int(local_idx)
        sample = self.base[item_idx]
        pixels = torch.as_tensor(sample["pixels"]).float()
        if pixels.ndim != 4 or pixels.shape[0] != 1:
            raise RuntimeError(
                f"Expected [1,C,H,W], got {tuple(pixels.shape)} at item {item_idx}")
        return pixels, int(self.row_map[item_idx]), item_idx


def build_row_map(frame_dataset, ep_offsets, total_frames):
    clips = np.asarray(frame_dataset.clip_indices, dtype=np.int64)
    if clips.ndim != 2 or clips.shape[1] != 2:
        raise RuntimeError(f"clip_indices shape: {clips.shape}")
    episode_positions = clips[:, 0]
    starts = clips[:, 1]
    row_map = ep_offsets[episode_positions] + starts
    expected = np.arange(total_frames, dtype=np.int64)
    if len(row_map) != total_frames:
        raise RuntimeError(f"Dataset length {len(row_map)} != total_frames {total_frames}")
    if not np.array_equal(row_map, expected):
        first_bad = int(np.flatnonzero(row_map != expected)[0])
        raise RuntimeError(
            f"Row map mismatch at index {first_bad}: "
            f"row_map={row_map[first_bad]}, expected={first_bad}")
    return row_map


def encode_all(model, frame_dataset, row_map, output, total_frames):
    start_item = int(output.attrs.get("frames_written", 0))
    if bool(output.attrs.get("complete", False)):
        print("Output already marked complete. Skipping.")
        return
    if start_item >= total_frames:
        output.attrs["complete"] = True
        output.flush()
        return

    wrapped = FrameDatasetWithRow(frame_dataset, row_map, start_item)
    loader = DataLoader(
        wrapped, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
        drop_last=False, persistent_workers=False)

    print(f"\n{'='*60}")
    print("  Encoding frames")
    print(f"{'='*60}")
    print(f"  device:       {DEVICE}")
    print(f"  batch size:   {BATCH_SIZE}")
    print(f"  total frames: {total_frames}")
    print(f"  resume from:  {start_item}")
    print(f"  remaining:    {total_frames - start_item}")
    print(f"  output:       {OUTPUT_PATH}")

    begin = time.time()
    total_batches = len(loader)

    with torch.inference_mode():
        for batch_id, (pixels, rows, item_indices) in enumerate(loader):
            pixels = pixels.to(DEVICE, non_blocking=True)
            z = model.encode({"pixels": pixels})["emb"][:, 0].detach().cpu()

            if not torch.isfinite(z).all():
                raise RuntimeError("NaN/Inf in latent!")

            # Verify row continuity
            items_np = item_indices.numpy().astype(np.int64)
            rows_np = rows.numpy().astype(np.int64)

            if "frame_latent" not in output:
                latent_dim = int(z.shape[1])
                output.create_dataset(
                    "frame_latent", shape=(total_frames, latent_dim),
                    dtype=OUTPUT_DTYPE, chunks=(min(total_frames, 8192), latent_dim))
                output.attrs["latent_dim"] = latent_dim
                output.flush()

            write_start = int(rows_np[0])
            write_end = int(rows_np[-1]) + 1
            output["frame_latent"][write_start:write_end] = z.numpy().astype(OUTPUT_DTYPE, copy=False)

            output.attrs["frames_written"] = write_end
            output.flush()

            if batch_id % PRINT_EVERY == 0 or batch_id == total_batches - 1:
                elapsed = max(time.time() - begin, 1e-6)
                processed = int(items_np[-1]) + 1 - start_item
                speed = processed / elapsed
                pct = 100.0 * (int(items_np[-1]) + 1) / total_frames
                print(f"  [{batch_id+1:>5}/{total_batches}] "
                      f"frame {int(items_np[-1])+1:>7}/{total_frames} "
                      f"({pct:6.2f}%) | {speed:7.1f} frame/s")

    if int(output.attrs["frames_written"]) != total_frames:
        raise RuntimeError("frames_written != total_frames after loop")

    output.attrs["complete"] = True
    output.flush()

    elapsed = time.time() - begin
    print(f"\n  Done! {total_frames} frames in {elapsed:.1f}s "
          f"({total_frames/elapsed:.1f} fps)")
    print(f"  frame_latent: {tuple(output['frame_latent'].shape)}")
    print(f"  total windows: {int(output.attrs['total_windows'])}")


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    global CHECKPOINT_PATH, OUTPUT_PATH

    parser = argparse.ArgumentParser(
        description="Extract frame-level GT latents for LIBERO-Plus."
    )
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    CHECKPOINT_PATH = args.checkpoint_path
    OUTPUT_PATH = args.output_path

    check_paths()
    cfg = load_config()

    with h5py.File(SOURCE_H5, "r") as source:
        total_frames = int(source["pixels"].shape[0])
        action_dim = int(source["action"].shape[-1])

        if source["action"].shape[0] != total_frames:
            raise RuntimeError("pixels and action row counts differ")

        episode_ids, ep_offsets, ep_lengths = load_episode_layout(source)
        total_windows = int(np.maximum(ep_lengths - FUTURE_HORIZON, 0).sum())

        print("=" * 60)
        print("  LIBERO-Plus GT Latent Extraction")
        print("=" * 60)
        print(f"  source H5:     {SOURCE_H5}")
        print(f"  checkpoint:    {CHECKPOINT_PATH}")
        print(f"  episodes:      {len(ep_lengths)}")
        print(f"  total frames:  {total_frames}")
        print(f"  valid windows: {total_windows}")
        print(f"  horizon:       {FUTURE_HORIZON}")
        print(f"  output:        {OUTPUT_PATH}")

        model, ckpt = build_model(cfg, action_dim)

        output = open_or_create_output(
            source=source, checkpoint=ckpt, total_frames=total_frames,
            action_dim=action_dim, episode_ids=episode_ids,
            ep_offsets=ep_offsets, ep_lengths=ep_lengths)

    # Build frame dataset OUTSIDE the `with h5py.File` block to avoid SWMR conflicts
    frame_dataset = build_frame_dataset(cfg)
    try:
        row_map = build_row_map(frame_dataset, ep_offsets, total_frames)
        print(f"  row map:       [{int(row_map[0])}, ..., {int(row_map[-1])}]")
        print(f"  row map check: {'PASS' if np.array_equal(row_map, np.arange(total_frames, dtype=np.int64)) else 'FAIL'}")

        encode_all(model, frame_dataset, row_map, output, total_frames)
    finally:
        close_dataset(frame_dataset)
        output.close()

    print(f"\n  Output: {OUTPUT_PATH}")
    print(f"  Size:   {OUTPUT_PATH.stat().st_size / (1024**2):.1f} MiB")


if __name__ == "__main__":
    main()
