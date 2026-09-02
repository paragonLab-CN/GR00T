#!/usr/bin/env python3
"""Build a deterministic full-dataset FLARE candidate-window manifest.

Protocols implemented by this script:

* select episodes either from the dataset prefix or from an existing manifest;
* default sparse mode: create base H=8 windows with stride 8, then densify around
  gripper transitions and large normalized continuous-action changes;
* exhaustive mode: enumerate every legal window start for the selected episodes;
* deduplicate all starts and preserve the reason for selecting each window;
* optionally validate every row against the reordered LeWM GT-latent frame bank.

This stage reads metadata plus the ``action`` column of each episode parquet. It
never decodes videos and never loads GR00T or LeWM weights.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


DEFAULT_DATASET_ROOT = Path("/root/autodl-tmp/lerobot_data/libero_90_lerobot")
DEFAULT_GT_H5 = Path(
    "/root/autodl-tmp/le-wm/outputs/gt_latents/"
    "libero90_epoch50_h8_framebank_trainpipe_lerobot_order.h5"
)
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/Isaac-GR00T/outputs/flare_full_h8_s8")
EXPECTED_CODEBASE_VERSIONS = {"v2", "v2.0", "v2.1"}
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(value).__name__}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected JSON object at {path}:{line_no}, "
                    f"got {type(value).__name__}"
                )
            rows.append(value)
    return rows


def selected_episode_ids_from_manifest(
    path: Path,
    *,
    dataset_episode_count: int,
) -> list[int]:
    rows = load_jsonl(path)
    if not rows:
        raise RuntimeError(f"Episode-selection manifest is empty: {path}")

    episode_ids: set[int] = set()
    for line_no, row in enumerate(rows, start=1):
        if "episode_index" not in row:
            raise RuntimeError(
                f"Episode-selection manifest row {line_no} has no episode_index: {path}"
            )
        episode_index = int(row["episode_index"])
        if episode_index < 0 or episode_index >= dataset_episode_count:
            raise RuntimeError(
                f"episode_index={episode_index} at row {line_no} is outside "
                f"[0, {dataset_episode_count - 1}]"
            )
        episode_ids.add(episode_index)

    return sorted(episode_ids)


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
            "Expected exactly one task string per episode; "
            f"episode={episode.get('episode_index')}, tasks={tasks!r}"
        )
    return tasks[0]


def episode_parquet_path(
    dataset_root: Path,
    info: dict[str, Any],
    episode_index: int,
) -> Path:
    chunk_size = int(info["chunks_size"])
    episode_chunk = episode_index // chunk_size
    relative = str(info["data_path"]).format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    return dataset_root / relative


def load_episode_actions(path: Path, expected_length: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["action"])
    values = [np.asarray(value, dtype=np.float32).reshape(-1) for value in frame["action"]]
    if not values:
        raise RuntimeError(f"Episode parquet contains no actions: {path}")
    actions = np.stack(values).astype(np.float32, copy=False)
    if actions.shape != (expected_length, 7):
        raise RuntimeError(
            f"Unexpected action shape in {path}: {actions.shape}, "
            f"expected {(expected_length, 7)}"
        )
    if not np.isfinite(actions).all():
        raise RuntimeError(f"Actions contain NaN or Inf: {path}")
    return actions


def deterministic_phase(episode_index: int, base_seed: int, stride: int) -> int:
    if stride == 1:
        return 0
    rng = np.random.default_rng(int(base_seed) + int(episode_index) * 1_000_003)
    return int(rng.integers(0, stride))


def normalized_action_change_scores(
    actions: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    """Return one action-change score per episode frame.

    Frame zero has no predecessor and therefore receives ``-inf``. The gripper
    dimension is excluded because gripper transitions are handled explicitly.
    """

    scale = np.asarray(action_std[:6], dtype=np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0)
    diffs = np.diff(actions[:, :6], axis=0) / scale[None]
    scores = np.full((len(actions),), -np.inf, dtype=np.float32)
    scores[1:] = np.linalg.norm(diffs, axis=1)
    return scores


def build_episode_window_starts(
    *,
    episode_index: int,
    actions: np.ndarray,
    action_std: np.ndarray,
    first_valid_step: int,
    last_valid_step: int,
    stride: int,
    base_seed: int,
    gripper_radius: int,
    action_change_topk: int,
    action_change_radius: int,
    max_extra_windows: int,
) -> tuple[list[int], dict[int, set[str]], dict[int, float], int, dict[str, int]]:
    if first_valid_step > last_valid_step:
        raise ValueError(
            f"Empty valid interval for episode {episode_index}: "
            f"[{first_valid_step}, {last_valid_step}]"
        )

    reasons: dict[int, set[str]] = defaultdict(set)
    priority: dict[int, float] = {}
    action_scores = normalized_action_change_scores(actions, action_std)

    def add(step: int, reason: str, score: float) -> None:
        if step < first_valid_step or step > last_valid_step:
            return
        reasons[int(step)].add(reason)
        priority[int(step)] = max(float(score), priority.get(int(step), -np.inf))

    phase = deterministic_phase(episode_index, base_seed, stride)
    first_stride_step = first_valid_step + phase
    if first_stride_step > last_valid_step:
        first_stride_step = first_valid_step
    for step in range(first_stride_step, last_valid_step + 1, stride):
        add(step, "base_stride", 1_000.0)

    # Always retain the earliest and latest legal windows so the complete episode
    # is represented even when the random phase shifts the stride grid inward.
    add(first_valid_step, "boundary_start", 1_100.0)
    add(last_valid_step, "boundary_end", 1_100.0)

    gripper_open = actions[:, 6] >= 0.5
    gripper_events = (np.flatnonzero(gripper_open[1:] != gripper_open[:-1]) + 1).tolist()
    for event_step in gripper_events:
        for offset in range(-gripper_radius, gripper_radius + 1):
            add(
                event_step + offset,
                "gripper_transition",
                900.0 - abs(offset),
            )

    valid_change_steps = np.arange(first_valid_step, last_valid_step + 1, dtype=np.int64)
    finite_mask = np.isfinite(action_scores[valid_change_steps])
    valid_change_steps = valid_change_steps[finite_mask]
    if action_change_topk > 0 and valid_change_steps.size:
        ranked = sorted(
            (int(step) for step in valid_change_steps),
            key=lambda step: (-float(action_scores[step]), step),
        )
        selected_events = ranked[: min(action_change_topk, len(ranked))]
        max_score = max(float(action_scores[step]) for step in selected_events)
        normalizer = max(max_score, 1e-6)
        for event_step in selected_events:
            normalized_score = float(action_scores[event_step]) / normalizer
            for offset in range(-action_change_radius, action_change_radius + 1):
                add(
                    event_step + offset,
                    "action_change",
                    700.0 + normalized_score - 0.01 * abs(offset),
                )

    fixed_steps = {
        step
        for step, step_reasons in reasons.items()
        if "base_stride" in step_reasons
        or "boundary_start" in step_reasons
        or "boundary_end" in step_reasons
    }
    extra_steps = [step for step in reasons if step not in fixed_steps]
    dropped_extra = 0
    if max_extra_windows >= 0 and len(extra_steps) > max_extra_windows:
        ranked_extra = sorted(extra_steps, key=lambda step: (-priority[step], step))
        keep_extra = set(ranked_extra[:max_extra_windows])
        for step in extra_steps:
            if step not in keep_extra:
                reasons.pop(step, None)
                priority.pop(step, None)
                dropped_extra += 1

    starts = sorted(reasons)
    diagnostics = {
        "base_windows": sum("base_stride" in reasons[step] for step in starts),
        "gripper_events": len(gripper_events),
        "gripper_windows": sum("gripper_transition" in reasons[step] for step in starts),
        "action_change_windows": sum("action_change" in reasons[step] for step in starts),
        "extra_windows_dropped_by_cap": dropped_extra,
    }
    return starts, reasons, action_scores, phase, diagnostics


def validate_against_h5(
    *,
    gt_h5: Path,
    lengths: np.ndarray,
    offsets: np.ndarray,
    manifest: list[dict[str, Any]],
    horizon: int,
    chunk_size: int = 8192,
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

        if not np.array_equal(np.asarray(h5["ep_len"][:], dtype=np.int64), lengths):
            raise RuntimeError("H5 ep_len does not match episodes.jsonl")
        if not np.array_equal(np.asarray(h5["ep_offset"][:], dtype=np.int64), offsets):
            raise RuntimeError("H5 ep_offset does not match cumulative episode lengths")

        for start in range(0, len(manifest), chunk_size):
            rows = manifest[start : start + chunk_size]
            indices = np.asarray([int(row["window_index_h8"]) for row in rows], dtype=np.int64)
            if indices.size > 1 and not bool(np.all(indices[1:] > indices[:-1])):
                raise RuntimeError("Manifest window indices must be strictly increasing")

            expected_episode = np.asarray(
                [int(row["episode_index"]) for row in rows], dtype=np.int64
            )
            expected_step = np.asarray([int(row["local_step"]) for row in rows], dtype=np.int64)
            expected_current = np.asarray(
                [int(row["global_index"]) for row in rows], dtype=np.int64
            )
            expected_future = expected_current[:, None] + np.arange(
                1, horizon + 1, dtype=np.int64
            )[None]

            if not np.array_equal(
                np.asarray(h5["window_episode_id"][indices], dtype=np.int64),
                expected_episode,
            ):
                raise RuntimeError(f"H5 episode mismatch in manifest rows {start}:{start + len(rows)}")
            if not np.array_equal(
                np.asarray(h5["window_step_idx"][indices], dtype=np.int64),
                expected_step,
            ):
                raise RuntimeError(f"H5 step mismatch in manifest rows {start}:{start + len(rows)}")
            if not np.array_equal(
                np.asarray(h5["window_current_row"][indices], dtype=np.int64),
                expected_current,
            ):
                raise RuntimeError(
                    f"H5 current-row mismatch in manifest rows {start}:{start + len(rows)}"
                )
            if not np.array_equal(
                np.asarray(h5["window_future_row"][indices], dtype=np.int64),
                expected_future,
            ):
                raise RuntimeError(
                    f"H5 future-row mismatch in manifest rows {start}:{start + len(rows)}"
                )

    return {
        "status": "PASS",
        "samples_checked": len(manifest),
        "gt_h5": str(gt_h5),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--gt-h5", type=Path, default=DEFAULT_GT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--gripper-radius", type=int, default=2)
    parser.add_argument("--action-change-topk", type=int, default=3)
    parser.add_argument("--action-change-radius", type=int, default=1)
    parser.add_argument(
        "--max-extra-windows-per-episode",
        type=int,
        default=20,
        help="Maximum non-stride key windows per episode; -1 disables the cap.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Use the first N episodes for a smoke manifest; default uses all episodes.",
    )
    parser.add_argument(
        "--episode-selection-manifest",
        type=Path,
        default=None,
        help=(
            "Select the unique episode_index values referenced by an existing JSONL manifest. "
            "This is mutually exclusive with --max-episodes."
        ),
    )
    parser.add_argument(
        "--all-legal-windows",
        action="store_true",
        help=(
            "Enumerate every legal local_step in each selected episode. In this mode stride "
            "and key-event densification settings are ignored."
        ),
    )
    parser.add_argument("--skip-h5-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.history_size < 1:
        raise ValueError("history-size must be >= 1")
    if args.horizon < 1:
        raise ValueError("horizon must be >= 1")
    if args.stride < 1:
        raise ValueError("stride must be >= 1")
    if args.candidate_k < 2:
        raise ValueError("candidate-k must be >= 2")
    if args.gripper_radius < 0 or args.action_change_radius < 0:
        raise ValueError("densification radii must be >= 0")
    if args.action_change_topk < 0:
        raise ValueError("action-change-topk must be >= 0")
    if args.max_extra_windows_per_episode < -1:
        raise ValueError("max-extra-windows-per-episode must be >= -1")
    if args.max_episodes is not None and args.episode_selection_manifest is not None:
        raise ValueError(
            "--max-episodes and --episode-selection-manifest are mutually exclusive"
        )

    dataset_root = args.dataset_root.expanduser().resolve()
    meta_dir = dataset_root / "meta"
    info = load_json(meta_dir / "info.json")
    episodes = load_jsonl(meta_dir / "episodes.jsonl")
    tasks = load_jsonl(meta_dir / "tasks.jsonl")
    modality = load_json(meta_dir / "modality.json")
    stats = load_json(meta_dir / "stats.json")

    if str(info.get("codebase_version")) not in EXPECTED_CODEBASE_VERSIONS:
        raise RuntimeError(f"Expected LeRobot v2.x, got {info.get('codebase_version')!r}")
    if int(info["total_episodes"]) != len(episodes):
        raise RuntimeError("info.total_episodes does not match episodes.jsonl")
    if int(info["total_tasks"]) != len(tasks):
        raise RuntimeError("info.total_tasks does not match tasks.jsonl")
    if list(modality.get("action", {})) != ACTION_KEYS:
        raise RuntimeError(
            f"Unexpected action key order: {list(modality.get('action', {}))}; "
            f"expected {ACTION_KEYS}"
        )

    episode_ids = np.asarray(
        [int(episode.get("episode_index", -1)) for episode in episodes], dtype=np.int64
    )
    if not np.array_equal(episode_ids, np.arange(len(episodes), dtype=np.int64)):
        raise RuntimeError("episodes.jsonl must be ordered by contiguous episode_index")

    selection_manifest: Path | None = None
    selection_source_pairs: set[tuple[int, int]] = set()
    if args.episode_selection_manifest is not None:
        selection_manifest = args.episode_selection_manifest.expanduser().resolve()
        if not selection_manifest.is_file():
            raise FileNotFoundError(selection_manifest)
        selection_rows = load_jsonl(selection_manifest)
        selected_episode_ids = selected_episode_ids_from_manifest(
            selection_manifest,
            dataset_episode_count=len(episodes),
        )
        selection_source_pairs = {
            (int(row["episode_index"]), int(row["local_step"]))
            for row in selection_rows
            if "local_step" in row
        }
        episode_selection_mode = "episode_index values from existing manifest"
    elif args.max_episodes is not None:
        if args.max_episodes < 1 or args.max_episodes > len(episodes):
            raise ValueError(
                f"max-episodes must lie in [1, {len(episodes)}], got {args.max_episodes}"
            )
        selected_episode_ids = list(range(int(args.max_episodes)))
        episode_selection_mode = "dataset prefix"
    else:
        selected_episode_ids = list(range(len(episodes)))
        episode_selection_mode = "all dataset episodes"
    selected_episode_count = len(selected_episode_ids)

    lengths = np.asarray([int(episode["length"]) for episode in episodes], dtype=np.int64)
    if int(lengths.sum()) != int(info["total_frames"]):
        raise RuntimeError("Sum of episode lengths does not match info.total_frames")
    offsets = cumulative_offsets(lengths)
    window_counts = np.maximum(lengths - int(args.horizon), 0)
    window_offsets = cumulative_offsets(window_counts)

    task_text_to_index = {str(row["task"]): int(row["task_index"]) for row in tasks}
    if len(task_text_to_index) != len(tasks):
        raise RuntimeError("tasks.jsonl contains duplicate task strings")

    action_std = np.asarray(stats["action"]["std"], dtype=np.float32)
    if action_std.shape != (7,) or not np.isfinite(action_std).all():
        raise RuntimeError(f"Invalid action std in stats.json: {action_std}")

    first_valid_step = int(args.history_size - 1)
    manifest: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    phase_counts: Counter[int] = Counter()

    for processed_episode_rank, episode_index in enumerate(selected_episode_ids):
        episode = episodes[episode_index]
        episode_length = int(lengths[episode_index])
        last_valid_step = episode_length - int(args.horizon) - 1
        if first_valid_step > last_valid_step:
            raise RuntimeError(
                f"Episode {episode_index} is too short for history={args.history_size}, "
                f"horizon={args.horizon}, length={episode_length}"
            )

        task_text = task_text_from_episode(episode)
        if task_text not in task_text_to_index:
            raise RuntimeError(f"Episode {episode_index} has unknown task {task_text!r}")
        task_index = int(task_text_to_index[task_text])

        if args.all_legal_windows:
            starts = list(range(first_valid_step, last_valid_step + 1))
            reasons = {step: {"all_legal_window"} for step in starts}
            change_scores = np.full((episode_length,), np.nan, dtype=np.float32)
            phase = 0
            diagnostics = {
                "base_windows": len(starts),
                "gripper_events": 0,
                "gripper_windows": 0,
                "action_change_windows": 0,
                "extra_windows_dropped_by_cap": 0,
            }
        else:
            parquet_path = episode_parquet_path(dataset_root, info, episode_index)
            actions = load_episode_actions(parquet_path, episode_length)
            starts, reasons, change_scores, phase, diagnostics = build_episode_window_starts(
                episode_index=episode_index,
                actions=actions,
                action_std=action_std,
                first_valid_step=first_valid_step,
                last_valid_step=last_valid_step,
                stride=int(args.stride),
                base_seed=int(args.base_seed),
                gripper_radius=int(args.gripper_radius),
                action_change_topk=int(args.action_change_topk),
                action_change_radius=int(args.action_change_radius),
                max_extra_windows=int(args.max_extra_windows_per_episode),
            )
        phase_counts[phase] += 1

        for local_step in starts:
            sample_index = len(manifest)
            global_index = int(offsets[episode_index] + local_step)
            window_index = int(window_offsets[episode_index] + local_step)
            sample_seed = (
                int(args.base_seed)
                + int(episode_index) * 10_000_000
                + int(local_step) * 10_000
            )
            selection_reasons = sorted(reasons[local_step])
            reason_counts.update(selection_reasons)

            manifest.append(
                {
                    "sample_id": f"full_{sample_index:09d}",
                    "sample_index": sample_index,
                    "episode_index": episode_index,
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
                    "task_index": task_index,
                    "task_text": task_text,
                    "candidate_k": int(args.candidate_k),
                    "noise_mode": "independent",
                    "noise_scale": 1.0,
                    "noise_base_seed": int(args.base_seed),
                    "sample_seed": sample_seed,
                    "candidate_seeds": [sample_seed + k for k in range(args.candidate_k)],
                    "window_stride": 1 if args.all_legal_windows else int(args.stride),
                    "episode_phase": int(phase),
                    "selection_reasons": selection_reasons,
                    "normalized_action_change_score": (
                        None
                        if not np.isfinite(change_scores[local_step])
                        else float(change_scores[local_step])
                    ),
                }
            )

        episode_summaries.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "task_text": task_text,
                "episode_length": episode_length,
                "phase": phase,
                "windows": len(starts),
                **diagnostics,
            }
        )

        processed_episode_count = processed_episode_rank + 1
        if processed_episode_count % 100 == 0 or processed_episode_count == selected_episode_count:
            print(
                f"processed episodes {processed_episode_count}/{selected_episode_count}; "
                f"manifest windows={len(manifest)}"
            )

    pairs = [(int(row["episode_index"]), int(row["local_step"])) for row in manifest]
    pair_set = set(pairs)
    if len(pairs) != len(pair_set):
        raise RuntimeError("Duplicate episode/local_step window detected")
    missing_source_pairs = sorted(selection_source_pairs - pair_set)
    if missing_source_pairs:
        raise RuntimeError(
            "The rebuilt manifest does not contain every source-manifest window; "
            f"first missing pairs={missing_source_pairs[:10]}"
        )
    sample_seeds = [int(row["sample_seed"]) for row in manifest]
    if len(sample_seeds) != len(set(sample_seeds)):
        raise RuntimeError("Duplicate sample_seed detected")

    h5_check: dict[str, Any] | None = None
    if not args.skip_h5_validation:
        h5_check = validate_against_h5(
            gt_h5=args.gt_h5.expanduser().resolve(),
            lengths=lengths,
            offsets=offsets,
            manifest=manifest,
            horizon=int(args.horizon),
        )

    episode_df = pd.DataFrame(episode_summaries)
    task_df = episode_df.groupby(["task_index", "task_text"], as_index=False).agg(
        episodes=("episode_index", "count"),
        windows=("windows", "sum"),
        windows_min=("windows", "min"),
        windows_max=("windows", "max"),
        base_windows=("base_windows", "sum"),
        gripper_events=("gripper_events", "sum"),
        gripper_windows=("gripper_windows", "sum"),
        action_change_windows=("action_change_windows", "sum"),
    )

    windows_per_episode = episode_df["windows"].to_numpy(dtype=np.int64)
    summary = {
        "format": "flare_full_manifest_v1",
        "status": "PASS",
        "dataset_root": str(dataset_root),
        "gt_h5": None if args.skip_h5_validation else str(args.gt_h5.expanduser().resolve()),
        "dataset_total_episodes": len(episodes),
        "selected_episodes": selected_episode_count,
        "selected_episode_ids": selected_episode_ids,
        "selected_tasks": int(episode_df["task_index"].nunique()),
        "selected_windows": len(manifest),
        "episode_selection": {
            "mode": episode_selection_mode,
            "source_manifest": None if selection_manifest is None else str(selection_manifest),
        },
        "protocol": {
            "history_size": int(args.history_size),
            "horizon": int(args.horizon),
            "coverage_mode": (
                "all legal local_step values" if args.all_legal_windows else "stride plus key events"
            ),
            "stride": 1 if args.all_legal_windows else int(args.stride),
            "candidate_k": int(args.candidate_k),
            "base_seed": int(args.base_seed),
            "phase": (
                "not applicable"
                if args.all_legal_windows
                else "deterministic per-episode uniform integer in [0, stride)"
            ),
            "gripper_radius": 0 if args.all_legal_windows else int(args.gripper_radius),
            "action_change_topk": 0 if args.all_legal_windows else int(args.action_change_topk),
            "action_change_radius": (
                0 if args.all_legal_windows else int(args.action_change_radius)
            ),
            "max_extra_windows_per_episode": (
                0
                if args.all_legal_windows
                else int(args.max_extra_windows_per_episode)
            ),
        },
        "coverage": {
            "all_selected_episodes_covered": len(episode_df) == selected_episode_count,
            "source_manifest_windows": len(selection_source_pairs),
            "all_source_manifest_windows_included": not missing_source_pairs,
            "windows_per_episode_min": int(windows_per_episode.min()),
            "windows_per_episode_mean": float(windows_per_episode.mean()),
            "windows_per_episode_median": float(np.median(windows_per_episode)),
            "windows_per_episode_max": int(windows_per_episode.max()),
            "reason_counts": dict(sorted(reason_counts.items())),
            "phase_counts": {str(k): int(v) for k, v in sorted(phase_counts.items())},
        },
        "h5_validation": h5_check,
        "outputs": {
            "manifest": str(args.output_dir / "full_manifest.jsonl"),
            "episode_summary": str(args.output_dir / "full_manifest_episode_summary.csv"),
            "task_summary": str(args.output_dir / "full_manifest_task_summary.csv"),
        },
    }

    print("\n========== FLARE full manifest ==========")
    print("episode selection    :", episode_selection_mode)
    print("selection manifest   :", selection_manifest)
    print("coverage mode        :", "all legal windows" if args.all_legal_windows else "sparse")
    print("episodes             :", selected_episode_count)
    print("tasks                :", int(episode_df["task_index"].nunique()))
    print("windows              :", len(manifest))
    print(
        "windows/episode      :",
        (
            int(windows_per_episode.min()),
            float(windows_per_episode.mean()),
            int(windows_per_episode.max()),
        ),
    )
    print("selection reasons    :", dict(sorted(reason_counts.items())))
    print("phase counts         :", dict(sorted(phase_counts.items())))
    print("H5 validation        :", "SKIPPED" if h5_check is None else "PASS")

    if args.dry_run:
        print("dry-run: no files written")
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "full_manifest.jsonl"
    summary_path = output_dir / "full_manifest_summary.json"
    episode_csv = output_dir / "full_manifest_episode_summary.csv"
    task_csv = output_dir / "full_manifest_task_summary.csv"

    write_jsonl(manifest_path, manifest)
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    episode_df.to_csv(episode_csv, index=False)
    task_df.to_csv(task_csv, index=False)

    print("manifest             :", manifest_path)
    print("summary              :", summary_path)
    print("episode summary      :", episode_csv)
    print("task summary         :", task_csv)


if __name__ == "__main__":
    main()
