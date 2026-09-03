
#!/usr/bin/env python3
"""
Convert LIBERO-Spatial no-noops LeRobot v2.1 -> LeWM HDF5 + mapping files.
Spatial-specific script — does NOT modify any existing LIBERO-90 code or data.
"""

from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd
from PIL import Image

SPATIAL_SOURCE = Path('/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot')
SPATIAL_OUTPUT_DIR = Path('/root/autodl-tmp/FLARE-VLA/outputs/spatial')
DEFAULT_IMAGE_SIZE = 224
CAMERA_KEYS = ['observation.images.image', 'observation.images.wrist_image']

def _parse_episode_range(spec, total_episodes):
    if isinstance(spec, list): return [int(x) for x in spec]
    if isinstance(spec, str) and ':' in spec:
        start, end = spec.split(':')
        return list(range(int(start), int(end)))
    raise ValueError(f'Unsupported split spec: {spec!r}')

def _read_parquet_files(source_dir, episodes):
    data_dir = source_dir / 'data' / 'chunk-000'
    dfs = []
    pq_paths = []
    for ep in episodes:
        pq_path = data_dir / f'episode_{ep:06d}.parquet'
        if not pq_path.exists():
            raise FileNotFoundError(f'Missing parquet: {pq_path}')
        dfs.append(pd.read_parquet(pq_path))
        pq_paths.append(str(pq_path.relative_to(source_dir)))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['episode_index', 'frame_index']).reset_index(drop=True)
    return df, pq_paths

def _decode_video_ffmpeg(video_path, native_h, native_w):
    frame_bytes = native_h * native_w * 3
    cmd = ['ffmpeg', '-i', str(video_path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-v', 'error', 'pipe:1']
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr_msg = proc.stderr.decode(errors='replace')[:500]
        raise RuntimeError(f'ffmpeg failed on {video_path} (rc={proc.returncode}): {stderr_msg}')
    raw = proc.stdout
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(f'ffmpeg output size {len(raw)} not divisible by frame size {frame_bytes}')
    n_frames = len(raw) // frame_bytes
    if n_frames == 0:
        raise RuntimeError(f'No frames decoded from {video_path}')
    return np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, native_h, native_w, 3)

def _resize_frames_pil(frames, size):
    T = frames.shape[0]
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    resized = np.empty((T, size, size, 3), dtype=np.uint8)
    for t in range(T):
        img = Image.fromarray(frames[t], mode='RGB')
        img = img.resize((size, size), resample=Image.BILINEAR)
        resized[t] = np.array(img, dtype=np.uint8)
    return resized

def _column_to_numpy(df, col_name):
    vals = df[col_name].values
    if vals.dtype == object:
        return np.stack(vals.tolist()).astype(np.float32)
    return np.asarray(vals)

def _build_episode_metadata(episode_idx_arr, total_frames):
    changes = np.flatnonzero(episode_idx_arr[1:] != episode_idx_arr[:-1]) + 1
    ep_offsets = np.concatenate([np.array([0], dtype=np.int64), changes.astype(np.int64)])
    ep_ends = np.concatenate([changes.astype(np.int64), np.array([total_frames], dtype=np.int64)])
    ep_lengths = ep_ends - ep_offsets
    return ep_lengths, ep_offsets

def _create_h5_datasets(h5f, total_rows, state_dim, action_dim, pixel_shape, ep_len, ep_offset, chunk_rows, compression):
    dsets = {}
    dsets['pixels'] = h5f.create_dataset('pixels', shape=(total_rows,) + pixel_shape, dtype=np.uint8, chunks=(chunk_rows,) + pixel_shape, compression=compression)
    dsets['action'] = h5f.create_dataset('action', shape=(total_rows, action_dim), dtype=np.float32, chunks=(chunk_rows, action_dim), compression=compression)
    dsets['state'] = h5f.create_dataset('state', shape=(total_rows, state_dim), dtype=np.float32, chunks=(chunk_rows, state_dim), compression=compression)
    dsets['proprio'] = h5f.create_dataset('proprio', shape=(total_rows, state_dim), dtype=np.float32, chunks=(chunk_rows, state_dim), compression=compression)
    dsets['episode_idx'] = h5f.create_dataset('episode_idx', shape=(total_rows,), dtype=np.int64, chunks=(chunk_rows,))
    dsets['step_idx'] = h5f.create_dataset('step_idx', shape=(total_rows,), dtype=np.int64, chunks=(chunk_rows,))
    dsets['done'] = h5f.create_dataset('done', shape=(total_rows,), dtype=np.bool_, chunks=(chunk_rows,))
    dsets['timestamp'] = h5f.create_dataset('timestamp', shape=(total_rows,), dtype=np.float32, chunks=(chunk_rows,))
    dsets['index'] = h5f.create_dataset('index', shape=(total_rows,), dtype=np.int64, chunks=(chunk_rows,))
    dsets['task_index'] = h5f.create_dataset('task_index', shape=(total_rows,), dtype=np.int64, chunks=(chunk_rows,))
    dsets['ep_len'] = h5f.create_dataset('ep_len', data=ep_len, dtype=np.int64)
    dsets['ep_offset'] = h5f.create_dataset('ep_offset', data=ep_offset, dtype=np.int64)
    return dsets

print('Script skeleton loaded successfully')
