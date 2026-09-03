#!/usr/bin/env python3
"""Build a deterministic, task-stratified FLARE-VLA validation manifest.

Default protocol:
  * 100 LIBERO-90 episodes
  * 5 valid H=8 windows per episode
  * 500 total validation samples
  * at least one episode per task when num_episodes >= num_tasks
  * local_step >= history_size - 1, so LeWM has 3 history latents
  * one local step sampled from each temporal bin to cover the whole episode

This stage loads metadata only. It does not load GR00T, LeWM, videos, or weights.
Optionally, it verifies every selected row against the reordered GT-latent HDF5.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/root/autodl-tmp/lerobot_data/libero_90_lerobot"
)
DEFAULT_GT_H5 = Path(
    "/root/autodl-tmp/le-wm/outputs/gt_latents/"
    "libero90_epoch50_h8_framebank_trainpipe_lerobot_order.h5"
)
DEFAULT_OUTPUT_DIR = Path(
    "/root/autodl-tmp/Isaac-GR00T/outputs/flare_validation_500"
)
EXPECTED_CODEBASE_VERSIONS = {"v2", "v2.0", "v2.1"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
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


def cumulative_offsets(lengths: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )


def task_text_from_episode(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], str):
        raise RuntimeError(
            "This validation builder expects exactly one task string per episode; "
            f"episode={episode.get('episode_index')}, tasks={tasks!r}"
        )
    return tasks[0]


def allocate_task_quotas(
    *,
    task_to_episodes: dict[int, list[int]],
    num_episodes: int,
    rng: np.random.Generator,
) -> dict[int, int]:
    """Allocate episode quotas as evenly across tasks as availability allows."""
    task_ids = sorted(task_to_episodes)
    total_available = sum(len(task_to_episodes[t]) for t in task_ids)
    if num_episodes < 1:
        raise ValueError("num_episodes must be >= 1")
    if num_episodes > total_available:
        raise ValueError(
            f"Requested {num_episodes} episodes but only {total_available} are available"
        )

    quotas = {task_id: 0 for task_id in task_ids}
    remaining = num_episodes

    # Seeded task order prevents low task indices from always receiving extras.
    task_order = [int(v) for v in rng.permutation(task_ids)]

    # Repeated round-robin yields near-equal task coverage while respecting
    # per-task availability. With the default 100 episodes and 74 tasks, every
    # task gets one episode and 26 tasks get a second episode.
    while remaining > 0:
        progressed = False
        for task_id in task_order:
            if remaining == 0:
                break
            if quotas[task_id] < len(task_to_episodes[task_id]):
                quotas[task_id] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("Could not allocate all requested episodes")

    return quotas


def choose_episode_ids(
    *,
    task_to_episodes: dict[int, list[int]],
    quotas: dict[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Return (episode_index, task_index) pairs in deterministic shuffled order."""
    selected: list[tuple[int, int]] = []
    for task_id in sorted(task_to_episodes):
        pool = np.asarray(task_to_episodes[task_id], dtype=np.int64)
        quota = quotas[task_id]
        chosen = rng.choice(pool, size=quota, replace=False)
        selected.extend((int(ep), int(task_id)) for ep in chosen)

    order = rng.permutation(len(selected))
    return [selected[int(i)] for i in order]


def choose_temporally_spread_steps(
    *,
    first_step: int,
    last_step: int,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample one distinct step from each temporal bin.

    Returns (local_step, temporal_bin_index).
    """
    if first_step > last_step:
        raise ValueError(f"Empty valid interval [{first_step}, {last_step}]")
    available = last_step - first_step + 1
    if count > available:
        raise ValueError(
            f"Need {count} windows but only {available} valid starts are available"
        )

    values = np.arange(first_step, last_step + 1, dtype=np.int64)
    bins = np.array_split(values, count)
    chosen: list[tuple[int, int]] = []
    for bin_index, bin_values in enumerate(bins):
        if bin_values.size == 0:
            raise RuntimeError("Temporal bin unexpectedly empty")
        local_step = int(rng.choice(bin_values))
        chosen.append((local_step, bin_index))

    # Output chronological windows within each episode, while retaining bin id.
    chosen.sort(key=lambda item: item[0])
    return chosen


def validate_against_h5(
    *,
    gt_h5: Path,
    episodes: list[dict[str, Any]],
    lengths: np.ndarray,
    offsets: np.ndarray,
    window_offsets: np.ndarray,
    manifest: list[dict[str, Any]],
    history_size: int,
    horizon: int,
) -> dict[str, Any]:
    if not gt_h5.is_file():
        raise FileNotFoundError(gt_h5)

    with h5py.File(gt_h5, "r") as h5:
        if not bool(h5.attrs.get("complete", False)):
            raise RuntimeError("GT-latent H5 is not marked complete")
        if str(h5.attrs.get("episode_order")) != "lerobot_v2.1_episode_index":
            raise RuntimeError(
                f"Unexpected H5 episode_order={h5.attrs.get('episode_order')!r}"
            )
        if int(h5.attrs.get("future_horizon", -1)) != horizon:
            raise RuntimeError(
                f"H5 future_horizon={h5.attrs.get('future_horizon')} != {horizon}"
            )

        h5_lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
        h5_offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        if not np.array_equal(h5_lengths, lengths):
            raise RuntimeError("H5 ep_len does not match episodes.jsonl")
        if not np.array_equal(h5_offsets, offsets):
            raise RuntimeError("H5 ep_offset does not match cumulative episode lengths")

        for sample in manifest:
            episode_index = int(sample["episode_index"])
            local_step = int(sample["local_step"])
            global_index = int(sample["global_index"])
            window_index = int(sample["window_index_h8"])

            if int(h5["window_episode_id"][window_index]) != episode_index:
                raise RuntimeError(f"H5 episode mismatch for sample {sample['sample_id']}")
            if int(h5["window_step_idx"][window_index]) != local_step:
                raise RuntimeError(f"H5 local-step mismatch for sample {sample['sample_id']}")
            if int(h5["window_current_row"][window_index]) != global_index:
                raise RuntimeError(f"H5 global-row mismatch for sample {sample['sample_id']}")

            expected_future = global_index + np.arange(1, horizon + 1, dtype=np.int64)
            actual_future = np.asarray(h5["window_future_row"][window_index], dtype=np.int64)
            if not np.array_equal(actual_future, expected_future):
                raise RuntimeError(f"H5 future-row mismatch for sample {sample['sample_id']}")

            history_rows = np.arange(
                global_index - (history_size - 1), global_index + 1, dtype=np.int64
            )
            if not np.all(np.asarray(h5["episode_id"][history_rows]) == episode_index):
                raise RuntimeError(f"History crosses episode for sample {sample['sample_id']}")
            if not np.all(np.asarray(h5["episode_id"][actual_future]) == episode_index):
                raise RuntimeError(f"Future crosses episode for sample {sample['sample_id']}")

    return {
        "status": "PASS",
        "samples_checked": len(manifest),
        "gt_h5": str(gt_h5),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--gt-h5", type=Path, default=DEFAULT_GT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--windows-per-episode", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument(
        "--skip-h5-validation",
        action="store_true",
        help="Build from metadata only; not recommended for the first run.",
    )
    args = parser.parse_args()

    if args.history_size < 1:
        raise ValueError("history_size must be >= 1")
    if args.horizon < 1:
        raise ValueError("horizon must be >= 1")
    if args.windows_per_episode < 1:
        raise ValueError("windows_per_episode must be >= 1")
    if args.candidate_k < 2:
        raise ValueError("candidate_k must be >= 2 for candidate ranking")

    dataset_root = args.dataset_root.resolve()
    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    modality_path = meta_dir / "modality.json"
    for path in (info_path, episodes_path, tasks_path, modality_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    info = load_json(info_path)
    episodes = load_jsonl(episodes_path)
    tasks = load_jsonl(tasks_path)
    modality = load_json(modality_path)

    codebase_version = str(info.get("codebase_version"))
    if codebase_version not in EXPECTED_CODEBASE_VERSIONS:
        raise RuntimeError(f"Expected LeRobot v2.x, got {codebase_version!r}")
    if int(info["total_episodes"]) != len(episodes):
        raise RuntimeError("info.total_episodes does not match episodes.jsonl")
    if int(info["total_tasks"]) != len(tasks):
        raise RuntimeError("info.total_tasks does not match tasks.jsonl")
    if len(modality.get("action", {})) != 7:
        raise RuntimeError("Expected 7 action fields in modality.json")

    episode_ids = np.asarray(
        [int(ep.get("episode_index", -1)) for ep in episodes], dtype=np.int64
    )
    if not np.array_equal(episode_ids, np.arange(len(episodes), dtype=np.int64)):
        raise RuntimeError("episodes.jsonl must be ordered by contiguous episode_index")

    lengths = np.asarray([int(ep["length"]) for ep in episodes], dtype=np.int64)
    if int(lengths.sum()) != int(info["total_frames"]):
        raise RuntimeError("Sum of episode lengths does not match info.total_frames")

    task_text_to_index = {str(row["task"]): int(row["task_index"]) for row in tasks}
    if len(task_text_to_index) != len(tasks):
        raise RuntimeError("tasks.jsonl contains duplicate task strings")

    task_to_episodes: dict[int, list[int]] = defaultdict(list)
    episode_task_index: dict[int, int] = {}
    for ep in episodes:
        episode_index = int(ep["episode_index"])
        task_text = task_text_from_episode(ep)
        if task_text not in task_text_to_index:
            raise RuntimeError(
                f"Episode {episode_index} task is absent from tasks.jsonl: {task_text!r}"
            )
        task_index = task_text_to_index[task_text]
        task_to_episodes[task_index].append(episode_index)
        episode_task_index[episode_index] = task_index

    offsets = cumulative_offsets(lengths)
    window_counts = np.maximum(lengths - int(args.horizon), 0)
    window_offsets = cumulative_offsets(window_counts)

    min_valid_count = int(args.windows_per_episode)
    first_valid_step = int(args.history_size - 1)
    eligible_task_to_episodes: dict[int, list[int]] = {}
    excluded_short_episodes: list[int] = []
    for task_index, episode_pool in task_to_episodes.items():
        eligible: list[int] = []
        for episode_index in episode_pool:
            last_valid_step = int(lengths[episode_index] - args.horizon - 1)
            valid_count = last_valid_step - first_valid_step + 1
            if valid_count >= min_valid_count:
                eligible.append(episode_index)
            else:
                excluded_short_episodes.append(episode_index)
        if eligible:
            eligible_task_to_episodes[task_index] = eligible

    missing_tasks = sorted(set(task_to_episodes) - set(eligible_task_to_episodes))
    if missing_tasks:
        raise RuntimeError(
            "Some tasks have no episode long enough for the requested protocol: "
            f"{missing_tasks}"
        )

    rng = np.random.default_rng(int(args.base_seed))
    quotas = allocate_task_quotas(
        task_to_episodes=eligible_task_to_episodes,
        num_episodes=int(args.num_episodes),
        rng=rng,
    )
    selected_episodes = choose_episode_ids(
        task_to_episodes=eligible_task_to_episodes,
        quotas=quotas,
        rng=rng,
    )

    manifest: list[dict[str, Any]] = []
    for episode_rank, (episode_index, task_index) in enumerate(selected_episodes):
        episode = episodes[episode_index]
        episode_length = int(lengths[episode_index])
        task_text = task_text_from_episode(episode)
        last_valid_step = episode_length - int(args.horizon) - 1

        selected_steps = choose_temporally_spread_steps(
            first_step=first_valid_step,
            last_step=last_valid_step,
            count=int(args.windows_per_episode),
            rng=rng,
        )

        for within_episode_rank, (local_step, temporal_bin) in enumerate(selected_steps):
            sample_index = len(manifest)
            sample_id = f"sample_{sample_index:06d}"
            global_index = int(offsets[episode_index] + local_step)
            window_index = int(window_offsets[episode_index] + local_step)
            sample_seed = (
                int(args.base_seed)
                + int(episode_index) * 10_000_000
                + int(local_step) * 10_000
            )
            candidate_seeds = [sample_seed + k for k in range(int(args.candidate_k))]

            manifest.append(
                {
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "episode_rank": episode_rank,
                    "within_episode_rank": within_episode_rank,
                    "temporal_bin": int(temporal_bin),
                    "episode_index": int(episode_index),
                    "local_step": int(local_step),
                    "frame_index": int(local_step),
                    "global_index": global_index,
                    "episode_offset": int(offsets[episode_index]),
                    "episode_length": episode_length,
                    "window_horizon": int(args.horizon),
                    "window_offset_h8": int(window_offsets[episode_index]),
                    "window_index_h8": window_index,
                    "history_size": int(args.history_size),
                    "history_rows": list(
                        range(global_index - (args.history_size - 1), global_index + 1)
                    ),
                    "future_rows": list(
                        range(global_index + 1, global_index + args.horizon + 1)
                    ),
                    "task_index": int(task_index),
                    "task_text": task_text,
                    "candidate_k": int(args.candidate_k),
                    "noise_mode": "independent",
                    "noise_scale": 1.0,
                    "noise_base_seed": int(args.base_seed),
                    "sample_seed": sample_seed,
                    "candidate_seeds": candidate_seeds,
                }
            )

    expected_samples = int(args.num_episodes) * int(args.windows_per_episode)
    if len(manifest) != expected_samples:
        raise RuntimeError(
            f"Manifest size mismatch: actual={len(manifest)}, expected={expected_samples}"
        )
    if len({row["sample_id"] for row in manifest}) != len(manifest):
        raise RuntimeError("Duplicate sample_id detected")
    if len({(row["episode_index"], row["local_step"]) for row in manifest}) != len(manifest):
        raise RuntimeError("Duplicate episode/local_step window detected")

    h5_check: dict[str, Any] | None = None
    if not args.skip_h5_validation:
        h5_check = validate_against_h5(
            gt_h5=args.gt_h5.resolve(),
            episodes=episodes,
            lengths=lengths,
            offsets=offsets,
            window_offsets=window_offsets,
            manifest=manifest,
            history_size=int(args.history_size),
            horizon=int(args.horizon),
        )

    selected_episode_counts = Counter(row["task_index"] for row in manifest[:: args.windows_per_episode])
    sample_task_counts = Counter(row["task_index"] for row in manifest)
    selected_episode_ids = sorted({row["episode_index"] for row in manifest})
    local_steps = np.asarray([row["local_step"] for row in manifest], dtype=np.int64)

    summary: dict[str, Any] = {
        "format": "flare_validation_manifest_v1",
        "status": "PASS",
        "dataset_root": str(dataset_root),
        "gt_h5": None if args.skip_h5_validation else str(args.gt_h5.resolve()),
        "codebase_version": codebase_version,
        "dataset_total_episodes": len(episodes),
        "dataset_total_frames": int(lengths.sum()),
        "dataset_total_tasks": len(tasks),
        "protocol": {
            "num_episodes": int(args.num_episodes),
            "windows_per_episode": int(args.windows_per_episode),
            "total_samples": len(manifest),
            "history_size": int(args.history_size),
            "horizon": int(args.horizon),
            "candidate_k": int(args.candidate_k),
            "base_seed": int(args.base_seed),
            "episode_sampling": "near-uniform task-stratified round-robin",
            "window_sampling": "one seeded random step per temporal bin",
        },
        "coverage": {
            "selected_episodes": len(selected_episode_ids),
            "selected_tasks": len(selected_episode_counts),
            "all_tasks_covered": len(selected_episode_counts) == len(tasks),
            "episode_quota_min": min(selected_episode_counts.values()),
            "episode_quota_max": max(selected_episode_counts.values()),
            "sample_count_per_task_min": min(sample_task_counts.values()),
            "sample_count_per_task_max": max(sample_task_counts.values()),
            "excluded_short_episodes": len(excluded_short_episodes),
            "local_step_min": int(local_steps.min()),
            "local_step_max": int(local_steps.max()),
        },
        "task_episode_quotas": {
            str(task_id): int(quotas.get(task_id, 0)) for task_id in sorted(task_to_episodes)
        },
        "selected_episode_ids": selected_episode_ids,
        "h5_validation": h5_check,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "validation_manifest.jsonl"
    summary_path = args.output_dir / "validation_manifest_summary.json"
    write_jsonl(manifest_path, manifest)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("========== FLARE validation manifest ==========")
    print("dataset root          :", dataset_root)
    print("dataset episodes      :", len(episodes))
    print("dataset tasks         :", len(tasks))
    print("selected episodes     :", len(selected_episode_ids))
    print("selected tasks        :", len(selected_episode_counts))
    print("windows per episode   :", args.windows_per_episode)
    print("total samples         :", len(manifest))
    print("history size          :", args.history_size)
    print("future horizon        :", args.horizon)
    print("candidate K           :", args.candidate_k)
    print("local step range      :", (int(local_steps.min()), int(local_steps.max())))
    print("episode quota/task    :", (min(selected_episode_counts.values()), max(selected_episode_counts.values())))
    print("all 74 tasks covered  :", len(selected_episode_counts) == len(tasks))
    print("H5 validation         :", "SKIPPED" if h5_check is None else "PASS")
    print("manifest              :", manifest_path)
    print("summary               :", summary_path)

    print("\nFirst 5 samples:")
    for row in manifest[:5]:
        print(
            row["sample_id"],
            f"ep={row['episode_index']}",
            f"t={row['local_step']}",
            f"global={row['global_index']}",
            f"window={row['window_index_h8']}",
            f"task={row['task_index']}",
        )


if __name__ == "__main__":
    main()
