#!/usr/bin/env python3
"""Generate GR00T K-candidate action chunks for a fixed FLARE validation manifest.

Run this script inside the Isaac-GR00T repository/environment. It loads the
policy and LeRobot dataset once, then processes only the exact episode/frame
pairs listed in ``validation_manifest.jsonl``.

Key properties:
- one cached-backbone candidate call per sample;
- deterministic explicit initial noises from manifest seeds;
- one trajectory load per selected episode;
- atomic per-sample saves;
- resume support with validation of existing files;
- separate failure and summary reports.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

# These must be set before importing torch/GR00T.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

# Import the already-tested helpers from the sibling candidate script.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_k_candidates_openloop import (  # noqa: E402
    _extract_gt_action_chunk,
    _get_internal_noise_shape,
    _infer_batch_size,
    _to_saveable_cpu,
    build_explicit_initial_noises,
    prepare_observation_data,
    set_seed,
)

FORMAT = "flare_groot_manifest_candidates_v1"
REQUIRED_MANIFEST_KEYS = {
    "sample_id",
    "sample_index",
    "episode_index",
    "local_step",
    "global_index",
    "window_index_h8",
    "episode_length",
    "task_index",
    "task_text",
    "candidate_k",
    "noise_mode",
    "noise_scale",
    "sample_seed",
    "candidate_seeds",
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            missing = REQUIRED_MANIFEST_KEYS - set(row)
            if missing:
                raise RuntimeError(
                    f"Manifest line {line_no} is missing keys: {sorted(missing)}"
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    return rows


def validate_manifest(rows: list[dict[str, Any]]) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Manifest contains duplicate sample_id values")

    sample_indices = [int(row["sample_index"]) for row in rows]
    if sample_indices != list(range(len(rows))):
        raise RuntimeError(
            "Manifest sample_index must be contiguous and ordered 0..N-1"
        )

    all_candidate_seeds: list[int] = []
    for row in rows:
        local_step = int(row["local_step"])
        episode_length = int(row["episode_length"])
        if local_step < 2 or local_step > episode_length - 9:
            raise RuntimeError(
                f"Illegal temporal window in {row['sample_id']}: "
                f"local_step={local_step}, episode_length={episode_length}"
            )
        candidate_k = int(row["candidate_k"])
        candidate_seeds = [int(v) for v in row["candidate_seeds"]]
        if len(candidate_seeds) != candidate_k:
            raise RuntimeError(
                f"candidate_seeds length mismatch in {row['sample_id']}"
            )
        expected = [int(row["sample_seed"]) + i for i in range(candidate_k)]
        if candidate_seeds != expected:
            raise RuntimeError(
                f"candidate_seeds do not match sample_seed in {row['sample_id']}"
            )
        all_candidate_seeds.extend(candidate_seeds)

    if len(all_candidate_seeds) != len(set(all_candidate_seeds)):
        raise RuntimeError("Manifest candidate seeds are not globally unique")


def get_embodiment_tag(name: str) -> EmbodimentTag:
    try:
        return EmbodimentTag[name]
    except KeyError as exc:
        valid = ", ".join(tag.name for tag in EmbodimentTag)
        raise ValueError(f"Unknown embodiment tag {name!r}; valid names: {valid}") from exc


def candidate_file(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "candidates" / f"{sample_id}.pt"


def validate_existing_candidate(
    path: Path,
    manifest_row: dict[str, Any],
    action_horizon: int,
) -> tuple[bool, str]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return False, f"load failed: {exc}"

    try:
        if payload.get("format") != FORMAT:
            return False, f"format={payload.get('format')!r}"
        if payload.get("sample_id") != manifest_row["sample_id"]:
            return False, "sample_id mismatch"
        if int(payload.get("episode_index")) != int(manifest_row["episode_index"]):
            return False, "episode_index mismatch"
        if int(payload.get("local_step")) != int(manifest_row["local_step"]):
            return False, "local_step mismatch"
        if int(payload.get("global_index")) != int(manifest_row["global_index"]):
            return False, "global_index mismatch"
        if int(payload.get("window_index_h8")) != int(manifest_row["window_index_h8"]):
            return False, "window_index_h8 mismatch"

        actions = torch.as_tensor(payload["candidate_actions_concat"])
        expected_shape = (
            1,
            int(manifest_row["candidate_k"]),
            action_horizon,
            7,
        )
        if tuple(actions.shape) != expected_shape:
            return False, f"candidate shape={tuple(actions.shape)}"
        if not torch.isfinite(actions).all():
            return False, "candidate actions contain NaN/Inf"

        gt = torch.as_tensor(payload["gt_actions_concat"])
        if tuple(gt.shape) != (action_horizon, 7):
            return False, f"GT shape={tuple(gt.shape)}"
        if not torch.isfinite(gt).all():
            return False, "GT actions contain NaN/Inf"

        seeds = [int(v) for v in payload["candidate_seeds"]]
        if seeds != [int(v) for v in manifest_row["candidate_seeds"]]:
            return False, "candidate_seeds mismatch"
    except Exception as exc:
        return False, f"validation failed: {exc}"

    return True, "ok"


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    parser.add_argument(
        "--video-backend",
        choices=["decord", "torchvision_av", "torchcodec"],
        default="torchcodec",
    )
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-initial-noises", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-observations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--global-seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    for path in (args.manifest, args.dataset_path, args.model_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.action_horizon != 8:
        raise ValueError("Current FLARE validation manifest and GT framebank require H=8")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")

    manifest_rows = load_jsonl(args.manifest)
    validate_manifest(manifest_rows)
    manifest_sha256 = sha256_file(args.manifest)

    selected_rows = [
        row for row in manifest_rows if int(row["sample_index"]) >= args.start_index
    ]
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be >= 1")
        selected_rows = selected_rows[: args.max_samples]
    if not selected_rows:
        raise RuntimeError("No manifest samples selected")

    output_dir = args.output_dir.resolve()
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "candidate_generation_failures.jsonl"
    events_path = output_dir / "candidate_generation_events.jsonl"

    logging.info("Manifest: %s", args.manifest)
    logging.info("Manifest SHA-256: %s", manifest_sha256)
    logging.info("Selected samples: %d / %d", len(selected_rows), len(manifest_rows))
    logging.info("Dataset: %s", args.dataset_path)
    logging.info("Model: %s", args.model_path)
    logging.info("Output: %s", output_dir)

    set_seed(args.global_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GR00T candidate generation")

    embodiment_tag = get_embodiment_tag(args.embodiment_tag)

    load_start = time.time()
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=str(args.model_path),
        device="cuda",
    )
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(
        dataset_path=str(args.dataset_path),
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )
    logging.info("Policy and loader initialized in %.2f s", time.time() - load_start)

    action_keys = list(loader.modality_configs["action"].modality_keys)
    if action_keys != ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]:
        raise RuntimeError(f"Unexpected LIBERO action key order: {action_keys}")
    observation_modality = deepcopy(loader.modality_configs)
    observation_modality.pop("action")

    rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        rows_by_episode[int(row["episode_index"])].append(row)
    for episode_rows in rows_by_episode.values():
        episode_rows.sort(key=lambda item: int(item["local_step"]))

    counters = {
        "selected": len(selected_rows),
        "generated": 0,
        "skipped_valid": 0,
        "regenerated_invalid": 0,
        "failed": 0,
    }
    start_time = time.time()

    for episode_position, episode_index in enumerate(sorted(rows_by_episode), start=1):
        episode_rows = rows_by_episode[episode_index]
        logging.info(
            "Episode %d (%d/%d): %d manifest samples",
            episode_index,
            episode_position,
            len(rows_by_episode),
            len(episode_rows),
        )

        try:
            traj = loader[episode_index]
        except Exception as exc:
            for row in episode_rows:
                counters["failed"] += 1
                append_jsonl(
                    failures_path,
                    {
                        "sample_id": row["sample_id"],
                        "sample_index": row["sample_index"],
                        "episode_index": episode_index,
                        "local_step": row["local_step"],
                        "stage": "episode_load",
                        "error": repr(exc),
                        "time": time.time(),
                    },
                )
            logging.exception("Failed to load episode %d", episode_index)
            continue

        expected_length = int(episode_rows[0]["episode_length"])
        if len(traj) != expected_length:
            raise RuntimeError(
                f"Episode {episode_index} length mismatch: loader={len(traj)}, "
                f"manifest={expected_length}"
            )

        for row in episode_rows:
            sample_id = str(row["sample_id"])
            sample_index = int(row["sample_index"])
            local_step = int(row["local_step"])
            output_path = candidate_file(output_dir, sample_id)

            if output_path.exists() and not args.overwrite:
                valid, reason = validate_existing_candidate(
                    output_path, row, args.action_horizon
                )
                if valid:
                    counters["skipped_valid"] += 1
                    logging.info("[%s] skip valid existing", sample_id)
                    continue
                counters["regenerated_invalid"] += 1
                logging.warning("[%s] existing file invalid (%s); regenerating", sample_id, reason)

            sample_start = time.time()
            try:
                parsed_obs = prepare_observation_data(
                    traj=traj,
                    step_count=local_step,
                    modality_configs=observation_modality,
                    embodiment_tag=embodiment_tag,
                    loader=loader,
                )
                batch_size = _infer_batch_size(parsed_obs)
                if batch_size != 1:
                    raise RuntimeError(f"Expected batch size 1, got {batch_size}")

                noise_shape = _get_internal_noise_shape(policy, batch_size)
                initial_noises, candidate_seeds, shared_base_seed = (
                    build_explicit_initial_noises(
                        shape=noise_shape,
                        candidate_k=int(row["candidate_k"]),
                        sample_seed=int(row["sample_seed"]),
                        noise_mode=str(row["noise_mode"]),
                        noise_scale=float(row["noise_scale"]),
                    )
                )
                expected_seeds = [int(v) for v in row["candidate_seeds"]]
                if candidate_seeds != expected_seeds:
                    raise RuntimeError(
                        f"Generated seeds {candidate_seeds} != manifest {expected_seeds}"
                    )

                candidate_batch, candidate_info = policy.get_action_candidates(
                    parsed_obs,
                    initial_noises=initial_noises,
                )

                stacked_fields: dict[str, np.ndarray] = {}
                concat_fields: list[np.ndarray] = []
                candidate_k = int(row["candidate_k"])
                for key in action_keys:
                    if key not in candidate_batch:
                        raise KeyError(f"Policy output missing action key {key!r}")
                    field = np.asarray(candidate_batch[key], dtype=np.float32)
                    if field.ndim != 4:
                        raise RuntimeError(
                            f"Expected policy action {key} [B,K,T,D], got {field.shape}"
                        )
                    field = field[:, :candidate_k, : args.action_horizon].copy()
                    stacked_fields[key] = field
                    concat_fields.append(field)
                candidate_actions_concat = np.concatenate(concat_fields, axis=-1)

                expected_candidate_shape = (
                    1,
                    candidate_k,
                    args.action_horizon,
                    7,
                )
                if candidate_actions_concat.shape != expected_candidate_shape:
                    raise RuntimeError(
                        f"Candidate shape {candidate_actions_concat.shape} != "
                        f"{expected_candidate_shape}"
                    )
                if not np.isfinite(candidate_actions_concat).all():
                    raise RuntimeError("Candidate actions contain NaN/Inf")

                gt_fields, gt_action_concat = _extract_gt_action_chunk(
                    traj=traj,
                    step_count=local_step,
                    action_keys=action_keys,
                    action_horizon=args.action_horizon,
                )
                if gt_action_concat.shape != (args.action_horizon, 7):
                    raise RuntimeError(f"Unexpected GT shape: {gt_action_concat.shape}")
                if not np.isfinite(gt_action_concat).all():
                    raise RuntimeError("GT actions contain NaN/Inf")

                pairwise_diff = None
                if candidate_k > 1:
                    pairwise_diff = float(
                        np.max(
                            np.abs(
                                candidate_actions_concat[:, 0]
                                - candidate_actions_concat[:, 1]
                            )
                        )
                    )

                payload: dict[str, Any] = {
                    "format": FORMAT,
                    "manifest_format": "flare_validation_manifest_v1",
                    "manifest_sha256": manifest_sha256,
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "episode_index": int(row["episode_index"]),
                    "local_step": local_step,
                    "global_index": int(row["global_index"]),
                    "window_index_h8": int(row["window_index_h8"]),
                    "task_index": int(row["task_index"]),
                    "task_text": str(row["task_text"]),
                    "history_rows": [int(v) for v in row["history_rows"]],
                    "future_rows": [int(v) for v in row["future_rows"]],
                    "candidate_k": candidate_k,
                    "decoded_action_horizon": int(args.action_horizon),
                    "noise_mode": str(row["noise_mode"]),
                    "noise_scale": float(row["noise_scale"]),
                    "sample_seed": int(row["sample_seed"]),
                    "candidate_seeds": candidate_seeds,
                    "shared_base_seed": shared_base_seed,
                    "internal_noise_shape_per_candidate": list(noise_shape),
                    "action_keys": action_keys,
                    "candidate_actions": _to_saveable_cpu(stacked_fields),
                    "candidate_actions_concat": torch.from_numpy(
                        candidate_actions_concat.copy()
                    ),
                    "gt_actions": _to_saveable_cpu(gt_fields),
                    "gt_actions_concat": torch.from_numpy(gt_action_concat.copy()),
                    "candidate_0_vs_1_max_abs_diff": pairwise_diff,
                    "candidate_generation_mode": (
                        "cached_backbone_vectorized_action_head"
                    ),
                    "backbone_evaluations": int(
                        candidate_info.get("backbone_evaluations", 1)
                    ),
                    "generation_seconds": time.time() - sample_start,
                }
                if args.save_initial_noises:
                    payload["initial_noises"] = initial_noises.cpu()
                if args.save_observations:
                    payload["observations"] = _to_saveable_cpu(parsed_obs)

                atomic_torch_save(payload, output_path)
                valid, reason = validate_existing_candidate(
                    output_path, row, args.action_horizon
                )
                if not valid:
                    raise RuntimeError(f"Post-save validation failed: {reason}")

                counters["generated"] += 1
                append_jsonl(
                    events_path,
                    {
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "episode_index": episode_index,
                        "local_step": local_step,
                        "status": "generated",
                        "seconds": time.time() - sample_start,
                        "output": str(output_path),
                        "time": time.time(),
                    },
                )
                logging.info(
                    "[%s] generated actions=%s diff01=%s time=%.2fs",
                    sample_id,
                    candidate_actions_concat.shape,
                    pairwise_diff,
                    time.time() - sample_start,
                )

            except Exception as exc:
                counters["failed"] += 1
                append_jsonl(
                    failures_path,
                    {
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "episode_index": episode_index,
                        "local_step": local_step,
                        "stage": "sample_generation",
                        "error": repr(exc),
                        "time": time.time(),
                    },
                )
                logging.exception("[%s] generation failed", sample_id)
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    summary = {
        "format": "flare_candidate_generation_summary_v1",
        "status": "PASS" if counters["failed"] == 0 else "PARTIAL",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "dataset_path": str(args.dataset_path.resolve()),
        "model_path": str(args.model_path.resolve()),
        "output_dir": str(output_dir),
        "embodiment_tag": args.embodiment_tag,
        "action_horizon": args.action_horizon,
        "save_initial_noises": bool(args.save_initial_noises),
        "save_observations": bool(args.save_observations),
        "counters": counters,
        "elapsed_seconds": elapsed,
        "average_seconds_per_selected_sample": elapsed / len(selected_rows),
    }
    summary_path = output_dir / "candidate_generation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== Candidate generation summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("summary:", summary_path)
    if counters["failed"]:
        print("failures:", failures_path)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
