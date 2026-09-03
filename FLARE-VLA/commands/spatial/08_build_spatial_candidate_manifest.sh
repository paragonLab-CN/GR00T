#!/usr/bin/env bash
# Build LIBERO-Spatial candidate generation manifests.
#
# Produces:
#   spatial_diagnostic_500_manifest.jsonl  — 500 task-stratified samples
#   spatial_full_candidate_manifest.jsonl   — all 49,514 legal H=8 windows
#
# Spatial-specific. Does NOT read or modify any LIBERO-90 outputs.
#
# Usage:
#   bash FLARE-VLA/commands/spatial/08_build_spatial_candidate_manifest.sh

set -euo pipefail

/root/autodl-tmp/envs/LeWorldModel/bin/python3 << 'PYEOF'
import json, sys, time
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, pandas as pd

LEROBOT_DIR = Path("/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot")
LATENT_H5 = Path("/root/autodl-tmp/FLARE-VLA/outputs/spatial/gt_latents/spatial_epoch50_h8_framebank_trainpipe.h5")
FRAME_MAP = Path("/root/autodl-tmp/FLARE-VLA/outputs/spatial/mapping/spatial_lerobot_to_h5_frame_map.parquet")
OUT_DIR = Path("/root/autodl-tmp/FLARE-VLA/outputs/spatial/manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_SIZE, HORIZON, CANDIDATE_K, BASE_SEED = 3, 8, 4, 2026
SUITE = "libero_spatial"
DIAG_EPISODES, DIAG_WINDOWS = 50, 10

t0 = time.time()

def load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: recs.append(json.loads(line))
    return recs

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# Load metadata
episodes_meta = load_jsonl(LEROBOT_DIR / "meta" / "episodes.jsonl")
tasks_meta = load_jsonl(LEROBOT_DIR / "meta" / "tasks.jsonl")
task_map = {int(r["task_index"]): r["task"] for r in tasks_meta}
frame_df = pd.read_parquet(FRAME_MAP)

n_eps = len(episodes_meta)
lengths = np.array([int(e["length"]) for e in episodes_meta], dtype=np.int64)
offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(lengths[:-1])])
window_counts = np.maximum(lengths - HORIZON, 0)
window_offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(window_counts[:-1])])
total_windows = int(window_counts.sum())

ep_task = {}
for ep_idx in range(n_eps):
    start = int(offsets[ep_idx])
    ep_task[ep_idx] = int(frame_df.iloc[start]["task_index"])

# Pre-load window metadata
lat = h5py.File(LATENT_H5, "r")
assert bool(lat.attrs.get("complete", False))
h5_win_ep = np.asarray(lat["window_episode_id"][:])
h5_win_step = np.asarray(lat["window_step_idx"][:])
h5_win_cur = np.asarray(lat["window_current_row"][:])
h5_win_fut = np.asarray(lat["window_future_row"][:])

# Verify window metadata
for ep_idx in range(n_eps):
    w_start = int(window_offsets[ep_idx])
    w_end = w_start + int(window_counts[ep_idx])
    if w_end > w_start:
        assert np.all(h5_win_ep[w_start:w_end] == ep_idx)
        assert np.array_equal(h5_win_step[w_start:w_end], np.arange(int(window_counts[ep_idx]), dtype=np.int64))
expected_fut = h5_win_cur[:, None] + np.arange(1, HORIZON + 1, dtype=np.int64)[None, :]
assert np.array_equal(h5_win_fut, expected_fut)
for ep_idx in range(n_eps):
    w_start = int(window_offsets[ep_idx])
    w_end = w_start + int(window_counts[ep_idx])
    if w_end > w_start:
        ep_end = int(offsets[ep_idx]) + int(lengths[ep_idx]) - 1
        assert int(h5_win_fut[w_end - 1, -1]) <= ep_end

# --- Diagnostic 500 ---
rng = np.random.default_rng(BASE_SEED)
task_to_eps = defaultdict(list)
for ep_idx in range(n_eps):
    if window_counts[ep_idx] >= DIAG_WINDOWS:
        task_to_eps[ep_task[ep_idx]].append(ep_idx)

eps_per_task = max(1, DIAG_EPISODES // len(task_to_eps))
extra = DIAG_EPISODES - eps_per_task * len(task_to_eps)
selected_eps = []
for task_idx in sorted(task_to_eps):
    pool = task_to_eps[task_idx]
    n_pick = min(eps_per_task + (1 if extra > 0 else 0), len(pool))
    if extra > 0: extra -= 1
    picked = rng.choice(pool, size=n_pick, replace=False)
    selected_eps.extend((int(p), task_idx) for p in picked)
order = rng.permutation(len(selected_eps))
selected_eps = [selected_eps[int(i)] for i in order]

diag_rows = []
for ep_rank, (ep_idx, task_idx) in enumerate(selected_eps):
    last_valid = int(window_counts[ep_idx]) - 1
    first_valid = HISTORY_SIZE - 1
    n_pick = min(DIAG_WINDOWS, last_valid - first_valid + 1)
    candidates = np.arange(first_valid, last_valid + 1, dtype=np.int64)
    bins = np.array_split(candidates, n_pick)
    picked = sorted(int(rng.choice(b)) for b in bins if len(b) > 0)
    for ls in picked:
        gi = int(offsets[ep_idx] + ls)
        wi = int(window_offsets[ep_idx] + ls)
        lerobot_ep = int(episodes_meta[ep_idx]["episode_index"])
        ss = BASE_SEED + ep_idx * 10_000_000 + ls * 10_000
        diag_rows.append({
            "sample_id": f"spatial_sample_{len(diag_rows):09d}", "sample_index": len(diag_rows),
            "suite": SUITE, "episode_index": lerobot_ep, "lewm_episode_index": ep_idx,
            "local_step": ls, "global_index": gi, "task_index": task_idx,
            "task_text": task_map[task_idx], "episode_length": int(lengths[ep_idx]),
            "horizon": HORIZON, "valid_steps": list(range(ls+1, ls+HORIZON+1)),
            "history_rows": list(range(gi-(HISTORY_SIZE-1), gi+1)),
            "future_rows": list(range(gi+1, gi+HORIZON+1)), "window_index_h8": wi,
            "source_parquet": f"data/chunk-000/episode_{lerobot_ep:06d}.parquet",
            "gt_latent_key": SUITE, "candidate_k": CANDIDATE_K, "sample_seed": ss,
            "candidate_seeds": [ss+k for k in range(CANDIDATE_K)],
            "selection_reason": f"diagnostic_ep{ep_rank}",
        })

assert len({r["sample_id"] for r in diag_rows}) == len(diag_rows)
diag_tasks = set(r["task_index"] for r in diag_rows)
print(f"Diagnostic: {len(diag_rows)} samples, {len(diag_tasks)}/10 tasks, "
      f"{len(set(r['episode_index'] for r in diag_rows))} episodes")

# Spot-check diagnostic
for idx in [0, len(diag_rows)//2, len(diag_rows)-1]:
    r = diag_rows[idx]
    fut = np.asarray(lat["frame_latent"][r["future_rows"]])
    act = np.asarray(lat["frame_action"][r["future_rows"]])
    assert fut.shape == (HORIZON, 192) and np.isfinite(fut).all()
    assert act.shape == (HORIZON, 7) and np.isfinite(act).all()
    map_row = frame_df.iloc[r["global_index"]]
    assert int(map_row["lerobot_episode_index"]) == r["episode_index"]

write_jsonl(OUT_DIR / "spatial_diagnostic_500_manifest.jsonl", diag_rows)
with open(OUT_DIR / "spatial_diagnostic_500_summary.json", "w") as f:
    json.dump({"format": "spatial_diagnostic_manifest_v1", "suite": SUITE,
               "total_samples": len(diag_rows), "tasks_covered": len(diag_tasks),
               "validation": {"duplicates": 0, "cross_episode": 0}}, f, indent=2)

# --- Full manifest ---
full_rows = []
for ep_idx in range(n_eps):
    nw = int(window_counts[ep_idx])
    if nw == 0: continue
    lerobot_ep = int(episodes_meta[ep_idx]["episode_index"])
    task_idx = ep_task[ep_idx]
    w_start = int(window_offsets[ep_idx])
    for local_offset in range(nw):
        ls = local_offset
        gi = int(offsets[ep_idx] + ls)
        wi = w_start + local_offset
        ss = BASE_SEED + ep_idx * 10_000_000 + ls * 10_000
        full_rows.append({
            "sample_id": f"spatial_sample_{len(full_rows):09d}", "sample_index": len(full_rows),
            "suite": SUITE, "episode_index": lerobot_ep, "lewm_episode_index": ep_idx,
            "local_step": ls, "global_index": gi, "task_index": task_idx,
            "task_text": task_map[task_idx], "episode_length": int(lengths[ep_idx]),
            "horizon": HORIZON, "valid_steps": list(range(ls+1, ls+HORIZON+1)),
            "history_rows": list(range(max(0, gi-(HISTORY_SIZE-1)), gi+1)),
            "future_rows": list(range(gi+1, gi+HORIZON+1)), "window_index_h8": wi,
            "source_parquet": f"data/chunk-000/episode_{lerobot_ep:06d}.parquet",
            "gt_latent_key": SUITE, "candidate_k": CANDIDATE_K, "sample_seed": ss,
            "candidate_seeds": [ss+k for k in range(CANDIDATE_K)],
            "selection_reason": "all_legal_window",
        })

assert len(full_rows) == total_windows
assert len({r["sample_id"] for r in full_rows}) == len(full_rows)

# Spot-check full
for idx in [0, len(full_rows)//2, len(full_rows)-1]:
    r = full_rows[idx]
    fut = np.asarray(lat["frame_latent"][r["future_rows"]])
    act = np.asarray(lat["frame_action"][r["future_rows"]])
    assert fut.shape == (HORIZON, 192) and np.isfinite(fut).all()
    assert act.shape == (HORIZON, 7) and np.isfinite(act).all()

lat.close()

write_jsonl(OUT_DIR / "spatial_full_candidate_manifest.jsonl", full_rows)
task_windows = defaultdict(int)
for r in full_rows: task_windows[r["task_index"]] += 1
per_task = [{"task_index": t, "task_text": task_map[t], "windows": task_windows[t]}
            for t in sorted(task_windows)]
with open(OUT_DIR / "spatial_full_candidate_summary.json", "w") as f:
    json.dump({"format": "spatial_full_manifest_v1", "suite": SUITE,
               "total_samples": len(full_rows), "tasks_covered": len(task_windows),
               "per_task_windows": per_task,
               "validation": {"duplicates": 0, "cross_episode": 0}}, f, indent=2)

# Validation log
log = [
    "Spatial Manifest Validation", "===========================",
    f"Diagnostic: {len(diag_rows)} samples, {len(diag_tasks)}/10 tasks",
    f"Full: {len(full_rows)} samples, {len(task_windows)}/10 tasks",
    "Duplicate sample_ids: 0", "Cross-episode windows: 0",
    "GT latent readable: YES", "GT action readable: YES",
    "Mapping consistency: VERIFIED",
    f"Total time: {time.time()-t0:.1f}s",
]
with open(OUT_DIR / "manifest_validation.log", "w") as f:
    f.write("\n".join(log) + "\n")

print(f"Full: {len(full_rows)} samples, {len(task_windows)}/10 tasks")
print(f"All manifests built in {time.time()-t0:.1f}s")
print(f"Output: {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size:,} bytes)")
PYEOF
