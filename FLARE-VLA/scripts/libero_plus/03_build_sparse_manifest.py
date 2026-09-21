#!/usr/bin/env python3
"""Build a sparse LIBERO-Plus candidate-window manifest.

Implements the same sparse + densification strategy as the spatial pipeline:

* Base stride sampling with a deterministic per-episode phase
* Always retain the first and last legal windows
* Densify around gripper transitions
* Densify around the largest normalized action changes
* Cap non-stride windows per episode

LIBERO-Plus task indices are interleaved across four official source suites.
The builder resolves each task against the canonical BDDL files and writes the
correct suite and gt_latent_key into the full manifest. The resulting manifest
can then be split by 03_split_manifest_by_suite.py for suite-specific teacher
candidate generation.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and LIBERO-Plus constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAAC_ROOT = PROJECT_ROOT.parent
LIBERO_PLUS_DATASET = (
    ISAAC_ROOT / "examples" / "LIBERO" / "libero_plus_lerobot"
)
LIBERO_PLUS_GT_H5 = (
    PROJECT_ROOT
    / "outputs"
    / "libero_plus"
    / "gt_latents"
    / "libero_plus_epoch9_h8_framebank_trainpipe.h5"
)
LIBERO_PLUS_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "libero_plus" / "manifests"
)
LIBERO_PLUS_BDDL_ROOT = (
    ISAAC_ROOT
    / "external_dependencies"
    / "LIBERO-plus"
    / "libero"
    / "libero"
    / "bddl_files"
)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_TASK_COUNT = 40
EXPECTED_TASKS_PER_SUITE = 10
OUTPUT_STEM = "libero_plus_sparse_manifest"


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
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
                raise TypeError(f"Expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def normalize_task_name(value: str) -> str:
    """Make task text comparable with a BDDL filename stem."""
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def derive_official_task_suites(
    tasks: list[dict[str, Any]],
    bddl_root: Path,
) -> dict[str, str]:
    """Return task text -> official suite using canonical BDDL filenames."""
    task_indices = [int(row["task_index"]) for row in tasks]
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} LIBERO-Plus tasks, "
            f"found {len(tasks)}"
        )
    if set(task_indices) != set(range(EXPECTED_TASK_COUNT)):
        raise RuntimeError(
            "LIBERO-Plus task indices must be contiguous 0..39"
        )

    files_by_suite: dict[str, list[Path]] = {}
    for suite in SUITES:
        suite_dir = bddl_root / suite
        if not suite_dir.is_dir():
            raise FileNotFoundError(suite_dir)
        files = sorted(suite_dir.glob("*.bddl"))
        if not files:
            raise RuntimeError(f"No BDDL files found in {suite_dir}")
        files_by_suite[suite] = files

    task_to_suite: dict[str, str] = {}
    for row in tasks:
        task_index = int(row["task_index"])
        task_text = str(row["task"])
        task_name = normalize_task_name(task_text)
        matches: list[tuple[str, Path]] = []
        for suite, paths in files_by_suite.items():
            for bddl_path in paths:
                bddl_stem = normalize_task_name(bddl_path.stem)
                if (
                    bddl_stem == task_name
                    or bddl_stem.endswith(f"_{task_name}")
                ):
                    matches.append((suite, bddl_path))

        if len(matches) != 1:
            rendered = [
                f"{suite}:{path.name}" for suite, path in matches
            ]
            raise RuntimeError(
                f"task_index {task_index} ({task_text!r}) must match "
                f"exactly one canonical BDDL file; found "
                f"{len(matches)}: {rendered}"
            )
        task_to_suite[task_text] = matches[0][0]

    suite_counts = Counter(task_to_suite.values())
    invalid_counts = {
        suite: suite_counts[suite]
        for suite in SUITES
        if suite_counts[suite] != EXPECTED_TASKS_PER_SUITE
    }
    if invalid_counts:
        raise RuntimeError(
            f"Expected {EXPECTED_TASKS_PER_SUITE} tasks per suite, "
            f"got {invalid_counts}"
        )
    return task_to_suite


def cumulative_offsets(lengths: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )

# ---------------------------------------------------------------------------
# Sparse + densification window selection (from 03_build_full_manifest.py)
# ---------------------------------------------------------------------------

def deterministic_phase(episode_index: int, base_seed: int, stride: int) -> int:
    """Deterministic per-episode random offset in [0, stride)."""
    if stride == 1:
        return 0
    rng = np.random.default_rng(int(base_seed) + int(episode_index) * 1_000_003)
    return int(rng.integers(0, stride))


def normalized_action_change_scores(
    actions: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    """Return one action-change score per episode frame.

    Frame zero has no predecessor and therefore receives -inf. The gripper
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
    """Build sparse + densified window starts for one episode.

    Returns:
        starts: sorted list of selected local_step values
        reasons: {step: {reason strings}}
        priority: {step: priority score} (higher = more important)
        phase: the deterministic phase used for this episode
        diagnostics: counts by category
    """
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

    # 1. Base stride sampling with deterministic phase
    phase = deterministic_phase(episode_index, base_seed, stride)
    first_stride_step = first_valid_step + phase
    if first_stride_step > last_valid_step:
        first_stride_step = first_valid_step
    for step in range(first_stride_step, last_valid_step + 1, stride):
        add(step, "base_stride", 1_000.0)

    # 2. Always retain the earliest and latest legal windows
    add(first_valid_step, "boundary_start", 1_100.0)
    add(last_valid_step, "boundary_end", 1_100.0)

    # 3. Gripper transition densification
    gripper_open = actions[:, 6] >= 0.5
    gripper_events = (
        np.flatnonzero(gripper_open[1:] != gripper_open[:-1]) + 1
    ).tolist()
    for event_step in gripper_events:
        for offset in range(-gripper_radius, gripper_radius + 1):
            add(event_step + offset, "gripper_transition", 900.0 - abs(offset))

    # 4. Action change densification
    valid_change_steps = np.arange(
        first_valid_step, last_valid_step + 1, dtype=np.int64
    )
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
            for offset in range(
                -action_change_radius, action_change_radius + 1
            ):
                add(
                    event_step + offset,
                    "action_change",
                    700.0 + normalized_score - 0.01 * abs(offset),
                )

    # 5. Cap extra windows
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
        ranked_extra = sorted(
            extra_steps, key=lambda step: (-priority[step], step)
        )
        keep_extra = set(ranked_extra[:max_extra_windows])
        for step in extra_steps:
            if step not in keep_extra:
                reasons.pop(step, None)
                priority.pop(step, None)
                dropped_extra += 1

    starts = sorted(reasons)
    diagnostics = {
        "base_windows": sum(
            "base_stride" in reasons[step] for step in starts
        ),
        "gripper_events": len(gripper_events),
        "gripper_windows": sum(
            "gripper_transition" in reasons[step] for step in starts
        ),
        "action_change_windows": sum(
            "action_change" in reasons[step] for step in starts
        ),
        "extra_windows_dropped_by_cap": dropped_extra,
    }
    return starts, reasons, action_scores, phase, diagnostics

# ---------------------------------------------------------------------------
# H5 validation
# ---------------------------------------------------------------------------

def validate_against_libero_plus_h5(
    *,
    gt_h5: Path,
    lengths: np.ndarray,
    offsets: np.ndarray,
    manifest: list[dict[str, Any]],
    horizon: int,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Validate manifest against the LIBERO-Plus GT-latent H5."""
    if not gt_h5.is_file():
        raise FileNotFoundError(gt_h5)

    with h5py.File(gt_h5, "r") as h5:
        if not bool(h5.attrs.get("complete", False)):
            raise RuntimeError("GT-latent H5 is not marked complete")
        if int(h5.attrs.get("future_horizon", -1)) != horizon:
            raise RuntimeError(
                f"H5 future_horizon={h5.attrs.get('future_horizon')} != {horizon}"
            )

        # Verify ep_len and ep_offset match
        if not np.array_equal(
            np.asarray(h5["ep_len"][:], dtype=np.int64), lengths
        ):
            raise RuntimeError("H5 ep_len does not match episodes.jsonl")
        if not np.array_equal(
            np.asarray(h5["ep_offset"][:], dtype=np.int64), offsets
        ):
            raise RuntimeError(
                "H5 ep_offset does not match cumulative episode lengths"
            )

        # Chunked window metadata validation
        for start in range(0, len(manifest), chunk_size):
            rows = manifest[start : start + chunk_size]
            indices = np.asarray(
                [int(row["window_index_h8"]) for row in rows], dtype=np.int64
            )
            if indices.size > 1 and not bool(
                np.all(indices[1:] > indices[:-1])
            ):
                raise RuntimeError(
                    "Manifest window indices must be strictly increasing"
                )

            expected_episode = np.asarray(
                [int(row["episode_index"]) for row in rows], dtype=np.int64
            )
            expected_step = np.asarray(
                [int(row["local_step"]) for row in rows], dtype=np.int64
            )
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
                raise RuntimeError(
                    f"H5 episode mismatch at rows {start}:{start + len(rows)}"
                )
            if not np.array_equal(
                np.asarray(h5["window_step_idx"][indices], dtype=np.int64),
                expected_step,
            ):
                raise RuntimeError(
                    f"H5 step mismatch at rows {start}:{start + len(rows)}"
                )
            if not np.array_equal(
                np.asarray(h5["window_current_row"][indices], dtype=np.int64),
                expected_current,
            ):
                raise RuntimeError(
                    f"H5 current-row mismatch at rows {start}:{start + len(rows)}"
                )
            if not np.array_equal(
                np.asarray(h5["window_future_row"][indices], dtype=np.int64),
                expected_future,
            ):
                raise RuntimeError(
                    f"H5 future-row mismatch at rows {start}:{start + len(rows)}"
                )

    return {
        "status": "PASS",
        "samples_checked": len(manifest),
        "gt_h5": str(gt_h5),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sparse LIBERO-Plus candidate-window manifest"
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=LIBERO_PLUS_DATASET
    )
    parser.add_argument("--gt-h5", type=Path, default=LIBERO_PLUS_GT_H5)
    parser.add_argument(
        "--output-dir", type=Path, default=LIBERO_PLUS_OUTPUT_DIR
    )
    parser.add_argument(
        "--bddl-root", type=Path, default=LIBERO_PLUS_BDDL_ROOT
    )
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
    parser.add_argument("--skip-h5-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Validate arguments
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

    dataset_root = args.dataset_root.expanduser().resolve()
    meta_dir = dataset_root / "meta"
    info = load_json(meta_dir / "info.json")
    episodes = load_jsonl(meta_dir / "episodes.jsonl")
    tasks = load_jsonl(meta_dir / "tasks.jsonl")
    stats = load_json(meta_dir / "stats.json")

    if int(info["total_episodes"]) != len(episodes):
        raise RuntimeError(
            "info.total_episodes does not match episodes.jsonl"
        )

    episode_ids = np.asarray(
        [int(ep.get("episode_index", -1)) for ep in episodes], dtype=np.int64
    )
    if not np.array_equal(
        episode_ids, np.arange(len(episodes), dtype=np.int64)
    ):
        raise RuntimeError(
            "episodes.jsonl must be ordered by contiguous episode_index"
        )

    lengths = np.asarray(
        [int(ep["length"]) for ep in episodes], dtype=np.int64
    )
    if int(lengths.sum()) != int(info["total_frames"]):
        raise RuntimeError(
            "Sum of episode lengths does not match info.total_frames"
        )
    offsets = cumulative_offsets(lengths)
    window_counts = np.maximum(lengths - int(args.horizon), 0)
    window_offsets = cumulative_offsets(window_counts)

    task_text_to_index = {
        str(row["task"]): int(row["task_index"]) for row in tasks
    }
    task_text_to_suite = derive_official_task_suites(
        tasks,
        args.bddl_root.expanduser().resolve(),
    )
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

    t0 = time.time()
    selected_episode_ids = list(range(len(episodes)))

    for processed_rank, episode_index in enumerate(selected_episode_ids):
        episode = episodes[episode_index]
        episode_length = int(lengths[episode_index])
        last_valid_step = episode_length - int(args.horizon) - 1
        if first_valid_step > last_valid_step:
            # Episode too short for history + horizon, skip
            continue

        # Get task info
        ep_tasks = episode.get("tasks", [])
        if not isinstance(ep_tasks, list) or len(ep_tasks) != 1:
            raise RuntimeError(
                f"Expected exactly one task per episode {episode_index}"
            )
        task_text = str(ep_tasks[0])
        if task_text not in task_text_to_index:
            raise RuntimeError(
                f"Unknown task {task_text!r} for episode {episode_index}"
            )
        task_index = int(task_text_to_index[task_text])
        task_suite = task_text_to_suite[task_text]

        # Load episode actions
        chunk_size = int(info["chunks_size"])
        episode_chunk = episode_index // chunk_size
        relative = str(info["data_path"]).format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        parquet_path = dataset_root / relative
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        frame = pd.read_parquet(parquet_path, columns=["action"])
        values = [
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in frame["action"]
        ]
        actions = np.stack(values).astype(np.float32, copy=False)
        if actions.shape != (episode_length, 7):
            raise RuntimeError(
                f"Unexpected action shape in {parquet_path}: "
                f"{actions.shape}, expected {(episode_length, 7)}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError(
                f"Actions contain NaN or Inf: {parquet_path}"
            )

        # Build sparse windows for this episode
        starts, reasons, change_scores, phase, diagnostics = \
            build_episode_window_starts(
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

            manifest.append({
                "sample_id": f"libero_plus_sparse_{sample_index:09d}",
                "sample_index": sample_index,
                "suite": task_suite,
                "episode_index": episode_index,
                "lewm_episode_index": episode_index,
                "local_step": int(local_step),
                "global_index": global_index,
                "task_index": task_index,
                "task_text": task_text,
                "episode_length": episode_length,
                "horizon": int(args.horizon),
                "window_horizon": int(args.horizon),
                "history_size": int(args.history_size),
                "window_index_h8": window_index,
                "history_rows": list(
                    range(
                        max(0, global_index - (args.history_size - 1)),
                        global_index + 1,
                    )
                ),
                "future_rows": list(
                    range(
                        global_index + 1,
                        global_index + args.horizon + 1,
                    )
                ),
                "valid_steps": list(
                    range(local_step + 1, local_step + args.horizon + 1)
                ),
                "source_parquet": relative,
                "gt_latent_key": task_suite,
                "candidate_k": int(args.candidate_k),
                "sample_seed": sample_seed,
                "candidate_seeds": [
                    sample_seed + k for k in range(args.candidate_k)
                ],
                "selection_reason": selection_reasons,
                "window_stride": int(args.stride),
                "episode_phase": int(phase),
                "noise_mode": "independent",
                "noise_scale": 1.0,
                "noise_base_seed": int(args.base_seed),
            })

        episode_summaries.append({
            "episode_index": episode_index,
            "suite": task_suite,
            "task_index": task_index,
            "task_text": task_text,
            "episode_length": episode_length,
            "phase": phase,
            "windows": len(starts),
            **diagnostics,
        })

        if (processed_rank + 1) % 100 == 0 or \
           (processed_rank + 1) == len(selected_episode_ids):
            print(
                f"processed episodes {processed_rank + 1}/"
                f"{len(selected_episode_ids)}; "
                f"manifest windows={len(manifest)}"
            )

    # Validation
    pairs = [
        (int(row["episode_index"]), int(row["local_step"]))
        for row in manifest
    ]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("Duplicate episode/local_step window detected")
    sample_seeds = [int(row["sample_seed"]) for row in manifest]
    if len(sample_seeds) != len(set(sample_seeds)):
        raise RuntimeError("Duplicate sample_seed detected")

    # H5 validation
    h5_check: dict[str, Any] | None = None
    if not args.skip_h5_validation:
        h5_check = validate_against_libero_plus_h5(
            gt_h5=args.gt_h5.expanduser().resolve(),
            lengths=lengths,
            offsets=offsets,
            manifest=manifest,
            horizon=int(args.horizon),
        )

    # Build summary
    episode_df = pd.DataFrame(episode_summaries)
    task_df = episode_df.groupby(
        ["suite", "task_index", "task_text"], as_index=False
    ).agg(
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
    total_windows_h5 = int(
        window_counts[selected_episode_ids].sum()
    )
    suite_window_counts = Counter(
        str(row["suite"]) for row in manifest
    )
    summary = {
        "format": "libero_plus_sparse_manifest_v1",
        "status": "PASS",
        "suite_mapping": {
            "source": "exact canonical BDDL filename match",
            "bddl_root": str(args.bddl_root.expanduser().resolve()),
            "task_counts": dict(
                sorted(Counter(task_text_to_suite.values()).items())
            ),
            "window_counts": dict(sorted(suite_window_counts.items())),
        },
        "dataset_root": str(dataset_root),
        "gt_h5": str(args.gt_h5.expanduser().resolve())
        if not args.skip_h5_validation
        else None,
        "dataset_total_episodes": len(episodes),
        "selected_episodes": len(selected_episode_ids),
        "selected_tasks": int(episode_df["task_index"].nunique()),
        "selected_windows": len(manifest),
        "total_possible_windows": int(total_windows_h5),
        "coverage_ratio": (
            len(manifest) / total_windows_h5
            if total_windows_h5 > 0
            else 0.0
        ),
        "protocol": {
            "history_size": int(args.history_size),
            "horizon": int(args.horizon),
            "coverage_mode": "stride plus key events",
            "stride": int(args.stride),
            "candidate_k": int(args.candidate_k),
            "base_seed": int(args.base_seed),
            "gripper_radius": int(args.gripper_radius),
            "action_change_topk": int(args.action_change_topk),
            "action_change_radius": int(args.action_change_radius),
            "max_extra_windows_per_episode": int(
                args.max_extra_windows_per_episode
            ),
        },
        "coverage": {
            "all_episodes_covered": len(episode_df) == len(
                selected_episode_ids
            ),
            "windows_per_episode_min": int(windows_per_episode.min()),
            "windows_per_episode_mean": float(windows_per_episode.mean()),
            "windows_per_episode_median": float(
                np.median(windows_per_episode)
            ),
            "windows_per_episode_max": int(windows_per_episode.max()),
            "reason_counts": dict(sorted(reason_counts.items())),
            "phase_counts": {
                str(k): int(v) for k, v in sorted(phase_counts.items())
            },
        },
        "h5_validation": h5_check,
        "outputs": {
            "manifest": str(
                args.output_dir / f"{OUTPUT_STEM}.jsonl"
            ),
            "episode_summary": str(
                args.output_dir
                / f"{OUTPUT_STEM}_episode_summary.csv"
            ),
            "task_summary": str(
                args.output_dir
                / f"{OUTPUT_STEM}_task_summary.csv"
            ),
        },
        "elapsed_seconds": time.time() - t0,
    }

    # Print summary
    print(f"\n{'='*50}")
    print("  LIBERO-Plus Sparse Manifest Summary")
    print(f"{'='*50}")
    print(f"  episodes             : {len(selected_episode_ids)}")
    print(f"  tasks                : {int(episode_df['task_index'].nunique())}")
    print(f"  windows              : {len(manifest)}")
    print(f"  total possible       : {total_windows_h5}")
    print(f"  coverage ratio       : {len(manifest)/total_windows_h5:.3f}")
    print(f"  windows/episode      : "
          f"min={int(windows_per_episode.min())}, "
          f"mean={windows_per_episode.mean():.1f}, "
          f"max={int(windows_per_episode.max())}")
    print(f"  selection reasons    : "
          f"{dict(sorted(reason_counts.items()))}")
    print(f"  phase counts         : "
          f"{dict(sorted(phase_counts.items()))}")
    print(f"  H5 validation        : "
          f"{'SKIPPED' if h5_check is None else 'PASS'}")
    print(f"  elapsed              : {time.time() - t0:.1f}s")

    if args.dry_run:
        print("\n  dry-run: no files written")
        return

    # Write outputs
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / f"{OUTPUT_STEM}.jsonl"
    summary_path = output_dir / f"{OUTPUT_STEM}_summary.json"
    episode_csv = (
        output_dir / f"{OUTPUT_STEM}_episode_summary.csv"
    )
    task_csv = output_dir / f"{OUTPUT_STEM}_task_summary.csv"

    write_jsonl(manifest_path, manifest)
    atomic_write_json(summary_path, summary)
    episode_df.to_csv(episode_csv, index=False)
    task_df.to_csv(task_csv, index=False)

    print(f"\n  manifest             : {manifest_path}")
    print(f"  summary              : {summary_path}")
    print(f"  episode summary      : {episode_csv}")
    print(f"  task summary         : {task_csv}")


if __name__ == "__main__":
    main()
