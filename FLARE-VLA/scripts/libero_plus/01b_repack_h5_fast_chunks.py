#!/usr/bin/env python3
"""
Repack the libero_plus front H5 into the fast-read chunk layout.

The initial conversion wrote pixels with chunks=(64, 224, 224, 3) + lzf.
LeWM's __getitem__ reads 4 consecutive rows per sample, so each read forces
h5py to decompress a full 9.4MB chunk — measured 0.4 it/s in training. The
libero90 H5 (2.7 it/s) uses chunks=(4, 224, 224, 3) + gzip, so this script
re-chunks to that proven layout.

Streaming copy: reads the source in row slices, writes the target with the
new chunking. No video re-decode needed (~1-2h for 180GB).

Usage:
    python FLARE-VLA/scripts/libero_plus/01b_repack_h5_fast_chunks.py \
        --src /root/autodl-tmp/le-wm/stablewm_data/libero_plus_lerobot__observation_images_front.h5 \
        --dst /root/autodl-tmp/le-wm/stablewm_data/libero_plus_lerobot__observation_images_front.fastchunk.h5 \
        --pixel-chunks 4 \
        --compression gzip
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

ROWS_PER_WRITE = 2048


def copy_dataset(src: h5py.Dataset, dst: h5py.File, name: str, **kwargs) -> None:
    n = src.shape[0]
    if n == 0:
        return
    dset = dst.create_dataset(name, shape=src.shape, dtype=src.dtype, **kwargs)
    for start in range(0, n, ROWS_PER_WRITE):
        end = min(start + ROWS_PER_WRITE, n)
        dset[start:end] = src[start:end]
    print(f"  copied {name}: {src.shape} {src.dtype}")


def repack(
    src_path: Path,
    dst_path: Path,
    pixel_chunks: int,
    compression: str,
) -> None:
    if dst_path.exists():
        raise FileExistsError(f"{dst_path} exists; remove it first")

    t0 = time.perf_counter()
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # attributes
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["repacked_by"] = "01b_repack_h5_fast_chunks.py"
        dst.attrs["pixel_chunk_rows"] = pixel_chunks

        # pixels: the only dataset whose chunking matters for training speed
        pixels = src["pixels"]
        n, h, w, c = pixels.shape
        pd = dst.create_dataset(
            "pixels",
            shape=(n, h, w, c),
            dtype=np.uint8,
            chunks=(pixel_chunks, h, w, c),
            compression=compression if compression != "none" else None,
        )
        for start in range(0, n, ROWS_PER_WRITE):
            end = min(start + ROWS_PER_WRITE, n)
            pd[start:end] = pixels[start:end]
            if (start // ROWS_PER_WRITE) % 50 == 0:
                elapsed = time.perf_counter() - t0
                pct = 100.0 * end / n
                rate = end / elapsed
                eta = (n - end) / rate
                print(
                    f"  pixels {end}/{n} ({pct:.1f}%) | "
                    f"{rate/1e6:.2f} Mrows/s | ETA {eta:.0f}s"
                )

        # scalar/tabular datasets: same chunking strategy as the original
        # (small data, chunk size irrelevant)
        for name in src.keys():
            if name == "pixels":
                continue
            ds = src[name]
            comp = compression if name not in ("ep_len", "ep_offset") else None
            copy_dataset(ds, dst, name, compression=comp)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.0f}s -> {dst_path} ({dst_path.stat().st_size/2**30:.1f} GiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/root/autodl-tmp/le-wm/stablewm_data/"
            "libero_plus_lerobot__observation_images_front.h5"
        ),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "/root/autodl-tmp/le-wm/stablewm_data/"
            "libero_plus_lerobot__observation_images_front.fastchunk.h5"
        ),
    )
    parser.add_argument("--pixel-chunks", type=int, default=4)
    parser.add_argument("--compression", default="gzip", choices=["none", "lzf", "gzip"])
    parser.add_argument(
        "--replace-original",
        action="store_true",
        help="After repacking, delete src and move dst to src path",
    )
    args = parser.parse_args()

    repack(args.src, args.dst, args.pixel_chunks, args.compression)

    if args.replace_original:
        args.src.unlink()
        shutil.move(str(args.dst), str(args.src))
        print(f"Replaced {args.src}")


if __name__ == "__main__":
    main()
