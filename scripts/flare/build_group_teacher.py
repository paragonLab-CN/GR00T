#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add trusted-prefix group targets to existing FLARE sidecars, without WM inference.

Legacy score parts do not record the behavior-policy checkpoint hash. Therefore
an explicit provenance manifest must bind their candidate file hashes to the
frozen policy snapshot used during generation. A mutable checkpoint path alone
is not sufficient evidence, and the input dataset is never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
from itertools import islice
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.data.dataset.flare_group_teacher import (  # noqa: E402
    GROUP_TEACHER_FORMAT,
    POLICY_PROVENANCE_FORMAT,
    GroupTeacherSelectionConfig,
    require_sha256,
    select_group_teacher,
    validate_group_teacher,
)
from gr00t.data.dataset.flare_sidecars import (  # noqa: E402
    DISTILLATION_INDEX_FILENAME,
    DISTILLATION_SUMMARY_FILENAME,
    TEACHER_RECORDS_FILENAME,
    resolve_flare_distillation_sidecars,
)


SCORE_PART_FORMAT = "flare_lewm_libero_plus_score_part_v2"
TEACHER_RECORDS_FORMAT = "flare_distillation_teacher_records_v1"
_IDENTITY_FIELDS = {
    "sample_id": "sample_ids",
    "sample_index": "sample_indices",
    "episode_index": "episode_indices",
    "local_step": "local_steps",
}
_SELECTED_PART_FIELDS = {
    *_IDENTITY_FIELDS,
    "suite",
    "gt_latent_key",
    "manifest_sha256",
    "candidate_k",
    "future_horizon",
    "candidate_actions_lerobot",
    "mse_per_step",
    "oracle_mse_per_step",
    "cosine_per_step",
    "cosine_score",
    "mse_score",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dict in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _load_provenance(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    value = _load_json(path)
    if value.get("format") != POLICY_PROVENANCE_FORMAT:
        raise ValueError(f"Unsupported policy provenance format in {path}")
    require_sha256(value.get("policy_checkpoint_sha256"), "policy_checkpoint_sha256")
    round_id = value.get("round_id")
    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
        raise ValueError("Policy provenance round_id must be a nonnegative integer")
    if not isinstance(value.get("candidates"), list) or not value["candidates"]:
        raise ValueError("Policy provenance must contain nonempty candidates")
    mapping: dict[str, str] = {}
    expected_policy = value["policy_checkpoint_sha256"].lower()
    for row in value["candidates"]:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            raise ValueError("Each provenance candidate needs a string sample_id")
        sample_id = row["sample_id"]
        if sample_id in mapping:
            raise ValueError(f"Duplicate provenance sample_id: {sample_id}")
        mapping[sample_id] = require_sha256(row.get("candidate_sha256"), "candidate_sha256")
        if "policy_checkpoint_sha256" in row:
            if (
                require_sha256(row["policy_checkpoint_sha256"], "policy_checkpoint_sha256")
                != expected_policy
            ):
                raise ValueError(f"Cross-policy candidate provenance for {sample_id}")
        if "round_id" in row and row["round_id"] != round_id:
            raise ValueError(f"Cross-round candidate provenance for {sample_id}")
    # The potentially full-dataset candidate list is no longer needed after
    # building its digest lookup; do not keep a second copy of those records.
    return {name: field for name, field in value.items() if name != "candidates"}, mapping


def _load_score_parts(
    directory: Path,
    provenance: dict[str, Any],
    candidate_hashes: dict[str, str],
    *,
    keep_sample_ids: set[str] | None = None,
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], str]:
    """Validate every part while retaining only compact selected teacher fields."""

    if not directory.is_dir():
        raise FileNotFoundError(f"score_parts directory does not exist: {directory}")
    parts: dict[str, tuple[Path, dict[str, Any]]] = {}
    seen: set[str] = set()
    wm_hashes: set[str] = set()
    candidate_model_paths: set[str] = set()
    expected_policy = provenance["policy_checkpoint_sha256"].lower()
    # Final teacher ordering is dictated by the index, not filesystem order.
    # Avoid materializing a sorted list of hundreds of thousands of paths.
    for path in directory.rglob("*.pt"):
        part = _load_torch(path)
        if part.get("format") != SCORE_PART_FORMAT:
            raise ValueError(f"Unsupported score part format in {path}")
        sample_id = str(part.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"Missing sample_id in score part {path}")
        if sample_id in seen:
            raise ValueError(f"Duplicate score-part sample_id: {sample_id}")
        if sample_id not in candidate_hashes:
            raise ValueError(f"Missing policy provenance for score part {sample_id}")
        candidate_hash = require_sha256(part.get("candidate_sha256"), "candidate_sha256")
        if candidate_hash != candidate_hashes[sample_id]:
            raise ValueError(f"Candidate hash mismatch for {sample_id}")
        seen.add(sample_id)
        wm_hashes.add(require_sha256(part.get("checkpoint_sha256"), "checkpoint_sha256"))
        if len(wm_hashes) > 1:
            raise ValueError("Cross-world-model score parts are not allowed in one round")
        if "policy_checkpoint_sha256" in part:
            if (
                require_sha256(part["policy_checkpoint_sha256"], "policy_checkpoint_sha256")
                != expected_policy
            ):
                raise ValueError(f"Cross-policy score part for {sample_id}")
        if "round_id" in part and part["round_id"] != provenance["round_id"]:
            raise ValueError(f"Cross-round score part for {sample_id}")
        candidate_path = Path(str(part.get("candidate_path", "")))
        if candidate_path.is_file():
            if sha256_file(candidate_path) != candidate_hash:
                raise ValueError(f"On-disk candidate hash mismatch for {sample_id}")
            candidate = _load_torch(candidate_path)
            if "candidate_actions_concat" in candidate:
                source_actions = torch.as_tensor(
                    candidate["candidate_actions_concat"], dtype=torch.float32
                )
                if source_actions.ndim == 4 and source_actions.shape[0] == 1:
                    source_actions = source_actions[0]
                scored_actions = torch.as_tensor(
                    part["candidate_actions_lerobot"], dtype=torch.float32
                )
                horizon = int(part["future_horizon"])
                if (
                    source_actions.ndim != 3
                    or scored_actions.ndim != 3
                    or scored_actions.shape[1] != horizon
                    or source_actions.shape[0] != scored_actions.shape[0]
                    or source_actions.shape[2] != scored_actions.shape[2]
                    or source_actions.shape[1] < horizon
                    or not torch.equal(source_actions[:, :horizon], scored_actions)
                ):
                    raise ValueError(
                        f"All-K candidate actions do not match the hashed source file for {sample_id}"
                    )
            if "policy_checkpoint_sha256" in candidate:
                if (
                    require_sha256(
                        candidate["policy_checkpoint_sha256"], "policy_checkpoint_sha256"
                    )
                    != expected_policy
                ):
                    raise ValueError(f"Cross-policy on-disk candidate for {sample_id}")
            if "model_path" in candidate:
                model_path = str(Path(str(candidate["model_path"])).resolve())
                candidate_model_paths.add(model_path)
                if len(candidate_model_paths) > 1:
                    raise ValueError(
                        "Cross-policy candidate model paths are not allowed in one round"
                    )
                if "model_path" in provenance and model_path != str(
                    Path(provenance["model_path"]).resolve()
                ):
                    raise ValueError(
                        f"Candidate model_path does not match provenance for {sample_id}"
                    )
            del candidate
        if keep_sample_ids is None or sample_id in keep_sample_ids:
            parts[sample_id] = (
                path,
                {name: field for name, field in part.items() if name in _SELECTED_PART_FIELDS},
            )
        # Optional prediction/GT latent tensors are diagnostic-only and can be
        # much larger than action groups. Release them before loading the next part.
        del part
    if not seen:
        raise ValueError(f"No score parts found in {directory}")
    if len(seen) != len(candidate_hashes):
        missing = list(
            islice((sample_id for sample_id in candidate_hashes if sample_id not in seen), 10)
        )
        raise ValueError(f"Missing score parts from provenance: {missing}")
    if len(wm_hashes) != 1:
        raise ValueError("Cross-world-model score parts are not allowed in one round")
    if len(candidate_model_paths) > 1:
        raise ValueError("Cross-policy candidate model paths are not allowed in one round")
    return parts, wm_hashes.pop()


def _check_teacher_identity(
    teacher: dict[str, Any], teacher_row: int, row: dict[str, Any], context: str
) -> None:
    for scalar_name, plural_name in _IDENTITY_FIELDS.items():
        if scalar_name not in row or plural_name not in teacher:
            raise KeyError(f"Missing teacher identity {scalar_name} in {context}")
        reference = teacher[plural_name][teacher_row]
        if torch.is_tensor(reference):
            reference = reference.item()
        actual = row[scalar_name]
        if scalar_name == "sample_id":
            matches = str(actual) == str(reference)
        else:
            matches = int(actual) == int(reference)
        if not matches:
            raise ValueError(f"Teacher identity mismatch for {scalar_name} in {context}")


def build_group_teacher(
    *,
    source_dir: Path,
    score_parts_dir: Path,
    output_dir: Path,
    policy_provenance_path: Path,
    selection_config: GroupTeacherSelectionConfig,
    score_metric: str = "cosine",
    score_scope: str = "shared-prefix",
    expected_round_id: int | None = None,
    expected_world_model_sha256: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Produce v1-compatible sidecars, exactly aligned with original teacher rows."""

    source_dir = source_dir.expanduser().resolve()
    score_parts_dir = score_parts_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )
    for input_dir in (source_dir, score_parts_dir):
        if output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
            raise ValueError(
                f"Output and input directories must not be nested: {output_dir}, {input_dir}"
            )
    sidecars = resolve_flare_distillation_sidecars(
        distillation_dataset_dir=source_dir,
        distillation_index_path=None,
        teacher_records_path=None,
    )
    assert sidecars is not None
    teacher = _load_torch(sidecars.teacher_records_path)
    if teacher.get("format") != TEACHER_RECORDS_FORMAT:
        raise ValueError("Expected v1 enriched teacher records")
    if "group_teacher" in teacher:
        raise ValueError("Source already contains group_teacher; use original top1 sidecars")
    sample_ids = [str(value) for value in teacher["sample_ids"]]
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Teacher sample_ids must be nonempty and unique")
    old_actions = torch.as_tensor(teacher["teacher_actions_lerobot"], dtype=torch.float32)
    n, horizon, action_dim = (
        len(sample_ids),
        int(teacher["teacher_horizon"]),
        int(teacher["action_dim"]),
    )
    if (
        tuple(old_actions.shape) != (n, horizon, action_dim)
        or not torch.isfinite(old_actions).all()
    ):
        raise ValueError("Original top1 teacher actions do not match finite [N, H, D] metadata")
    seen_teacher_rows: set[int] = set()
    with sidecars.index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["source"] != "teacher":
                continue
            teacher_row = int(row["teacher_row"])
            if not 0 <= teacher_row < n or teacher_row in seen_teacher_rows:
                raise ValueError(f"Invalid or duplicate teacher_row in index: {teacher_row}")
            _check_teacher_identity(teacher, teacher_row, row, "distillation index")
            seen_teacher_rows.add(teacher_row)
    if seen_teacher_rows != set(range(n)):
        raise ValueError("Index must reference every original teacher row exactly once")

    provenance, candidate_hashes = _load_provenance(policy_provenance_path)
    if expected_round_id is not None and provenance["round_id"] != expected_round_id:
        raise ValueError("Policy provenance round_id mismatch")
    parts, wm_hash = _load_score_parts(
        score_parts_dir, provenance, candidate_hashes, keep_sample_ids=set(sample_ids)
    )
    if expected_world_model_sha256 is not None:
        if wm_hash != require_sha256(expected_world_model_sha256, "expected_world_model_sha256"):
            raise ValueError("World-model checkpoint SHA-256 mismatch")
    if score_metric not in {"cosine", "mse"}:
        raise ValueError("score_metric must be cosine or mse")
    if score_scope not in {"full", "shared-prefix"}:
        raise ValueError("score_scope must be full or shared-prefix")

    action_groups, probabilities, prefixes, score_groups = [], [], [], []
    part_paths, part_hashes, matched_candidate_hashes = [], [], []
    candidate_k: int | None = None
    for teacher_row, sample_id in enumerate(sample_ids):
        if sample_id not in parts:
            raise ValueError(f"Missing score part for teacher sample_id: {sample_id}")
        path, part = parts.pop(sample_id)
        _check_teacher_identity(teacher, teacher_row, part, str(path))
        if (
            "manifest_sha256" in teacher
            and part.get("manifest_sha256") != teacher["manifest_sha256"]
        ):
            raise ValueError(f"Manifest hash mismatch for {sample_id}")
        for scalar_name, plural_name in (("suite", "suites"), ("gt_latent_key", "gt_latent_keys")):
            if plural_name in teacher and str(part.get(scalar_name)) != str(
                teacher[plural_name][teacher_row]
            ):
                raise ValueError(f"Teacher {scalar_name} mismatch for {sample_id}")
        actions = torch.as_tensor(part["candidate_actions_lerobot"], dtype=torch.float32)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (horizon, action_dim):
            raise ValueError(f"Candidate action shape mismatch for {sample_id}")
        k = actions.shape[0]
        if candidate_k is None:
            candidate_k = k
        if k != candidate_k or int(part.get("candidate_k", -1)) != k:
            raise ValueError(f"Candidate K mismatch for {sample_id}")
        if int(part.get("future_horizon", -1)) != horizon:
            raise ValueError(f"Candidate future_horizon mismatch for {sample_id}")
        if "teacher_idx_primary" in teacher:
            selected = int(teacher["teacher_idx_primary"][teacher_row])
            if not 0 <= selected < k or not torch.equal(
                actions[selected], old_actions[teacher_row]
            ):
                raise ValueError(f"Original top1 teacher/candidate action mismatch for {sample_id}")
        elif not any(torch.equal(action, old_actions[teacher_row]) for action in actions):
            raise ValueError(f"Original top1 teacher is not a candidate for {sample_id}")
        scores = torch.as_tensor(part[f"{score_metric}_score"], dtype=torch.float32)
        step_scores = None
        if score_scope == "shared-prefix":
            step_scores = torch.as_tensor(part[f"{score_metric}_per_step"], dtype=torch.float32)
        if score_metric == "mse":
            scores = -scores
            if step_scores is not None:
                step_scores = -step_scores
        q, lengths = select_group_teacher(
            actions,
            part["mse_per_step"],
            part["oracle_mse_per_step"],
            scores,
            selection_config,
            score_per_step=step_scores,
            score_scope=score_scope,
        )
        # Unsupervised invalid tails must not reach the official action processor.
        clean_actions = torch.where(torch.isfinite(actions), actions, 0.0)
        action_groups.append(clean_actions)
        probabilities.append(q)
        prefixes.append(lengths)
        score_groups.append(scores)
        part_paths.append(str(path.resolve()))
        part_hashes.append(sha256_file(path))
        matched_candidate_hashes.append(candidate_hashes[sample_id])

    group = {
        "format": GROUP_TEACHER_FORMAT,
        "candidate_actions_lerobot": torch.stack(action_groups),
        "candidate_probabilities": torch.stack(probabilities),
        "prefix_lengths": torch.stack(prefixes),
        "round_id": provenance["round_id"],
        "policy_checkpoint_sha256": provenance["policy_checkpoint_sha256"].lower(),
        "world_model_checkpoint_sha256": wm_hash,
        "selection_config": {
            **selection_config.to_dict(),
            "score_metric": score_metric,
            "score_scope": score_scope,
        },
        "candidate_scores_full": torch.stack(score_groups),
        "candidate_sha256": matched_candidate_hashes,
        "score_part_paths": part_paths,
        "score_part_sha256": part_hashes,
        **{name: copy.deepcopy(teacher[name]) for name in _IDENTITY_FIELDS.values()},
    }
    group = validate_group_teacher(group, teacher)
    assert group is not None
    enriched = {**teacher, "group_teacher": group}
    teacher_group_count = int((group["candidate_probabilities"].sum(dim=1) > 0).sum())
    prefix_counts = Counter(int(value) for value in group["prefix_lengths"].flatten() if value > 0)
    summary = (
        _load_json(sidecars.summary_path)
        if sidecars.summary_path is not None
        else {
            "format": "flare_distillation_summary_v1",
            "status": "PASS",
            "protocol": {
                "total_samples": sidecars.total_records,
                "gt_samples": sidecars.source_counts.get("gt", 0),
                "teacher_samples": sidecars.source_counts.get("teacher", 0),
            },
        }
    )
    summary["group_teacher"] = {
        "format": GROUP_TEACHER_FORMAT,
        "round_id": group["round_id"],
        "policy_checkpoint_sha256": group["policy_checkpoint_sha256"],
        "world_model_checkpoint_sha256": wm_hash,
        "selection_config": group["selection_config"],
        "groups": n,
        "candidate_k": candidate_k,
        "trusted_teacher_groups": teacher_group_count,
        "gt_fallback_groups": n - teacher_group_count,
        "positive_prefix_counts": dict(sorted(prefix_counts.items())),
        "index_counts_unchanged": True,
        "action_representation": "raw_lerobot_before_gr00t_normalization",
        "uninformative_group_policy": "GT fallback for fewer than two eligible candidates or insufficient score span",
        "legacy_top1_fields_preserved": True,
    }
    summary["group_teacher_inputs"] = {
        "source_dir": str(source_dir),
        "source_teacher_sha256": sha256_file(sidecars.teacher_records_path),
        "source_index_sha256": sha256_file(sidecars.index_path),
        "score_parts_dir": str(score_parts_dir.resolve()),
        "policy_provenance": str(policy_provenance_path.resolve()),
        "policy_provenance_sha256": sha256_file(policy_provenance_path),
    }
    summary["outputs"] = {
        "distillation_index": str(output_dir / DISTILLATION_INDEX_FILENAME),
        "teacher_index": str(output_dir / "teacher_index.jsonl"),
        "teacher_records_enriched": str(output_dir / TEACHER_RECORDS_FILENAME),
        "distillation_summary": str(output_dir / DISTILLATION_SUMMARY_FILENAME),
    }
    if dry_run:
        return summary

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent
    ) as tmp:
        staging = Path(tmp) / "sidecars"
        staging.mkdir()
        shutil.copy2(sidecars.index_path, staging / DISTILLATION_INDEX_FILENAME)
        original_teacher_index = source_dir / "teacher_index.jsonl"
        if original_teacher_index.is_file():
            shutil.copy2(original_teacher_index, staging / "teacher_index.jsonl")
        else:
            with (
                sidecars.index_path.open("r", encoding="utf-8") as source_index,
                (staging / "teacher_index.jsonl").open("w", encoding="utf-8") as teacher_index,
            ):
                teacher_index_position = 0
                for line in source_index:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["source"] == "teacher":
                        row["distillation_index"] = teacher_index_position
                        teacher_index.write(json.dumps(row, ensure_ascii=True) + "\n")
                        teacher_index_position += 1
        torch.save(enriched, staging / TEACHER_RECORDS_FILENAME)
        (staging / DISTILLATION_SUMMARY_FILENAME).write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--source-dir", "--distillation-dataset-dir", dest="source_dir", type=Path, required=True
    )
    parser.add_argument(
        "--score-parts-dir", "--score-parts", dest="score_parts_dir", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy-provenance",
        type=Path,
        required=True,
        help="Hash-bound generation manifest; legacy caches require an externally justified attestation.",
    )
    parser.add_argument(
        "--round-id", type=int, default=None, help="Optional expected provenance round_id."
    )
    parser.add_argument("--world-model-checkpoint-sha256", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--oracle-max-mse",
        type=float,
        default=None,
        help="Training-calibrated oracle error threshold; required unless --no-oracle-anchor.",
    )
    parser.add_argument(
        "--max-paired-gap",
        type=float,
        default=None,
        help="Training-calibrated candidate-minus-oracle MSE tolerance; equality is accepted.",
    )
    parser.add_argument("--min-prefix", type=int, default=1)
    parser.add_argument("--min-score-span", type=float, default=0.0)
    parser.add_argument("--score-std-floor", type=float, default=1e-6)
    parser.add_argument(
        "--no-oracle-anchor",
        action="store_true",
        help="Ablation: disable oracle reliability and paired-gap gates.",
    )
    parser.add_argument(
        "--fixed-prefix",
        type=int,
        default=None,
        help="Ablation: supervise this length only when the whole prefix passes enabled gates.",
    )
    parser.add_argument("--score-metric", choices=("cosine", "mse"), default="cosine")
    parser.add_argument(
        "--score-scope",
        choices=("full", "shared-prefix"),
        default="shared-prefix",
        help="Rank all eligible candidates over the same trusted prefix; full is an ablation.",
    )
    parser.add_argument("--action-min", type=float, default=-1.0)
    parser.add_argument("--action-max", type=float, default=1.0)
    parser.add_argument(
        "--disable-action-bounds",
        action="store_true",
        help="Use for verified non-LIBERO raw action conventions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GroupTeacherSelectionConfig(
        temperature=args.temperature,
        oracle_max_mse=args.oracle_max_mse,
        max_paired_gap=args.max_paired_gap,
        min_prefix=args.min_prefix,
        min_score_span=args.min_score_span,
        score_std_floor=args.score_std_floor,
        oracle_anchor=not args.no_oracle_anchor,
        fixed_prefix=args.fixed_prefix,
        action_min=None if args.disable_action_bounds else args.action_min,
        action_max=None if args.disable_action_bounds else args.action_max,
    )
    summary = build_group_teacher(
        source_dir=args.source_dir,
        score_parts_dir=args.score_parts_dir,
        output_dir=args.output_dir,
        policy_provenance_path=args.policy_provenance,
        selection_config=config,
        score_metric=args.score_metric,
        score_scope=args.score_scope,
        expected_round_id=args.round_id,
        expected_world_model_sha256=args.world_model_checkpoint_sha256,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary["group_teacher"], indent=2, ensure_ascii=True))
    print("PASS (dry-run; no files written)" if args.dry_run else f"PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
