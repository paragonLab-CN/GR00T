#!/usr/bin/env python3
"""Build the FLARE-VLA Spatial Teacher Distillation Dataset index.

This script joins:

1. the sparse spatial manifest (all 12,599 GT windows), and
2. the quality-filtered teacher records (8,998 selected by
   06_audit_teacher_quality.py).

It produces a GR00T-compatible distillation index and enriched teacher
records in the format expected by ``FlareDistillationDataset`` (v1).

The spatial teacher records are stored in v3 format
(``flare_teacher_quality_filtered_records_v3_direct_augmentation``) which
lacks some fields that ``FlareDistillationDataset`` needs.  This script
derives those fields from the teacher_selected JSONL and the checkpoint
processor config, then writes the enriched v1 format.

Outputs:
  * distillation_index.jsonl        mixed GT + teacher window index
  * teacher_index.jsonl             teacher-only subset (for weighted mixing)
  * teacher_records_enriched.pt     v1 format with all required fields
  * distillation_summary.json       provenance, counts, interface contract
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


FORMAT_VERSION = "flare_distillation_index_v1"
TEACHER_FORMAT_VERSION = "flare_distillation_teacher_records_v1"
SUMMARY_FORMAT_VERSION = "flare_distillation_summary_v1"


SPATIAL_MANIFEST_COUNT = 12599


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a mixed GT + quality-filtered FLARE teacher index for "
            "spatial data without modifying the original LeRobot dataset."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, required=True)
    parser.add_argument("--teacher-selected", type=Path, required=True)
    parser.add_argument("--teacher-records", type=Path, required=True)
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--processor-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--embodiment-key",
        default="libero_sim",
        help="Processor modality-config key for GR00T training.",
    )
    parser.add_argument(
        "--index-mode",
        choices=["mixed", "teacher_only"],
        default="teacher_only",
        help=(
            "mixed writes manifest GT rows plus selected teacher rows. "
            "teacher_only writes only teacher rows for use with "
            "--keep-full-gt-dataset."
        ),
    )
    parser.add_argument(
        "--mix-order",
        choices=["gt_then_teacher", "interleave"],
        default="gt_then_teacher",
    )
    parser.add_argument(
        "--expected-gt-count",
        type=int,
        default=SPATIAL_MANIFEST_COUNT,
        help=f"Expected manifest row count (default {SPATIAL_MANIFEST_COUNT}).",
    )
    parser.add_argument(
        "--expected-teacher-count",
        type=int,
        default=0,
        help="Set to a positive value to enforce an exact selected-teacher count.",
    )
    parser.add_argument(
        "--quality-levels",
        nargs="+",
        choices=["high", "medium"],
        default=["high", "medium"],
        help="Teacher quality levels to include (default: high medium).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and print the summary without writing files.",
    )
    return parser.parse_args()


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
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"Expected JSON object at {path}:{line_no}, "
                    f"got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def load_torch(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict in {path}, got {type(value).__name__}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(to_jsonable(value), ensure_ascii=False, indent=2) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(to_jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def require_fields(row: dict[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise KeyError(f"{context} is missing fields: {missing}")


def tensor_1d(value: Any, *, name: str, length: int, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous()
    if tensor.shape != (length,):
        raise RuntimeError(f"{name} shape={tuple(tensor.shape)}, expected {(length,)}")
    return tensor


def validate_manifest(
    rows: list[dict[str, Any]], expected_count: int
) -> tuple[dict[str, dict[str, Any]], int]:
    if not rows:
        raise RuntimeError("Manifest is empty")
    if expected_count > 0 and len(rows) != expected_count:
        raise RuntimeError(
            f"Manifest contains {len(rows)} rows, expected {expected_count}"
        )

    required = [
        "sample_id",
        "sample_index",
        "episode_index",
        "local_step",
        "task_index",
        "task_text",
        "window_horizon",
        "history_size",
    ]
    by_id: dict[str, dict[str, Any]] = {}
    horizons: set[int] = set()
    for row_index, row in enumerate(rows):
        require_fields(row, required, f"manifest row {row_index}")
        sample_id = str(row["sample_id"])
        sample_index = int(row["sample_index"])
        if sample_id in by_id:
            raise RuntimeError(f"Duplicate manifest sample_id={sample_id!r}")
        if sample_index != row_index:
            raise RuntimeError(
                f"Manifest row {row_index} has sample_index={sample_index}; "
                "the current pipeline requires contiguous manifest order"
            )
        if int(row["episode_index"]) < 0 or int(row["local_step"]) < 0:
            raise RuntimeError(f"Negative episode/local step in manifest sample {sample_id}")
        horizon = int(row["window_horizon"])
        if horizon < 1:
            raise RuntimeError(f"Invalid window_horizon={horizon} for {sample_id}")
        if "future_rows" in row and len(row["future_rows"]) != horizon:
            raise RuntimeError(
                f"{sample_id} future_rows length={len(row['future_rows'])}, "
                f"expected {horizon}"
            )
        horizons.add(horizon)
        by_id[sample_id] = row

    if len(horizons) != 1:
        raise RuntimeError(f"Manifest contains multiple horizons: {sorted(horizons)}")
    return by_id, horizons.pop()


def validate_checkpoint_interface(
    checkpoint_config: dict[str, Any],
    processor_config: dict[str, Any],
    embodiment_key: str,
    teacher_horizon: int,
    action_dim: int,
) -> dict[str, Any]:
    model_horizon = int(checkpoint_config.get("action_horizon", -1))
    if model_horizon < teacher_horizon:
        raise RuntimeError(
            f"Checkpoint action_horizon={model_horizon} is shorter than "
            f"teacher_horizon={teacher_horizon}"
        )

    try:
        embodiment_cfg = processor_config["processor_kwargs"]["modality_configs"][
            embodiment_key
        ]
        action_cfg = embodiment_cfg["action"]
    except KeyError as exc:
        raise KeyError(
            f"processor_config does not contain modality_configs[{embodiment_key!r}]"
        ) from exc

    delta_indices = [int(v) for v in action_cfg["delta_indices"]]
    action_keys = [str(v) for v in action_cfg["modality_keys"]]
    if not delta_indices:
        raise RuntimeError(f"Processor {embodiment_key} action delta_indices is empty")
    if delta_indices != list(range(len(delta_indices))):
        raise RuntimeError(
            f"Processor {embodiment_key} action delta_indices must be contiguous "
            f"from zero, got {delta_indices}"
        )
    processor_horizon = len(delta_indices)
    if processor_horizon < teacher_horizon:
        raise RuntimeError(
            f"Processor action horizon={processor_horizon} is shorter than "
            f"teacher_horizon={teacher_horizon}"
        )
    if len(action_keys) != action_dim:
        raise RuntimeError(
            f"Processor action key count={len(action_keys)}, teacher action_dim={action_dim}; "
            f"keys={action_keys}"
        )

    max_processor_horizon = int(
        processor_config.get("processor_kwargs", {}).get(
            "max_action_horizon", model_horizon
        )
    )
    if max_processor_horizon != model_horizon:
        raise RuntimeError(
            f"processor max_action_horizon={max_processor_horizon} does not match "
            f"checkpoint action_horizon={model_horizon}"
        )

    return {
        "embodiment_key": embodiment_key,
        "action_keys": action_keys,
        "teacher_horizon": teacher_horizon,
        "processor_action_horizon": processor_horizon,
        "model_action_horizon": model_horizon,
        "action_dim": action_dim,
    }


def validate_teacher_records(
    selected_rows: list[dict[str, Any]],
    records: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
    manifest_horizon: int,
    expected_count: int,
) -> dict[str, Any]:
    if expected_count > 0 and len(selected_rows) != expected_count:
        raise RuntimeError(
            f"Selected JSONL contains {len(selected_rows)} rows, expected {expected_count}"
        )
    if not selected_rows:
        raise RuntimeError("Selected teacher JSONL is empty")

    required_record_keys = [
        "sample_ids",
        "sample_indices",
        "teacher_idx_primary",
        "confidence_weight",
        "teacher_actions_lerobot",
        "gt_actions_lerobot",
        "relative_mse_margin",
        "teacher_gt_action_mse",
        "terminal_mse",
    ]
    missing = [key for key in required_record_keys if key not in records]
    if missing:
        raise KeyError(f"Filtered teacher records are missing keys: {missing}")

    sample_ids = [str(v) for v in records["sample_ids"]]
    n = len(sample_ids)
    if n != len(selected_rows):
        raise RuntimeError(
            f"Teacher PT has {n} sample_ids but selected JSONL has "
            f"{len(selected_rows)} rows"
        )
    if expected_count > 0 and n != expected_count:
        raise RuntimeError(f"Teacher PT contains {n} rows, expected {expected_count}")
    if len(set(sample_ids)) != n:
        raise RuntimeError("Filtered teacher records contain duplicate sample_ids")

    teacher_actions = torch.as_tensor(
        records["teacher_actions_lerobot"], dtype=torch.float32
    ).detach().cpu().contiguous()
    gt_actions = torch.as_tensor(
        records["gt_actions_lerobot"], dtype=torch.float32
    ).detach().cpu().contiguous()
    if teacher_actions.ndim != 3:
        raise RuntimeError(
            f"teacher_actions_lerobot shape={tuple(teacher_actions.shape)}, "
            "expected [N,H,D]"
        )
    if gt_actions.shape != teacher_actions.shape:
        raise RuntimeError(
            f"gt_actions_lerobot shape={tuple(gt_actions.shape)} does not match "
            f"teacher_actions_lerobot shape={tuple(teacher_actions.shape)}"
        )
    if teacher_actions.shape[0] != n:
        raise RuntimeError(
            f"teacher_actions first dimension={teacher_actions.shape[0]}, expected {n}"
        )
    if not torch.isfinite(teacher_actions).all():
        raise RuntimeError("teacher_actions_lerobot contains NaN or Inf")
    if not torch.isfinite(gt_actions).all():
        raise RuntimeError("gt_actions_lerobot contains NaN or Inf")

    teacher_horizon = int(teacher_actions.shape[1])
    action_dim = int(teacher_actions.shape[2])
    if teacher_horizon != manifest_horizon:
        raise RuntimeError(
            f"Teacher horizon={teacher_horizon} does not match manifest "
            f"window_horizon={manifest_horizon}"
        )

    sample_indices = tensor_1d(
        records["sample_indices"], name="sample_indices", length=n, dtype=torch.long
    )
    teacher_idx = tensor_1d(
        records["teacher_idx_primary"],
        name="teacher_idx_primary",
        length=n,
        dtype=torch.long,
    )
    confidence_weight = tensor_1d(
        records["confidence_weight"],
        name="confidence_weight",
        length=n,
        dtype=torch.float32,
    )
    relative_margin = tensor_1d(
        records["relative_mse_margin"],
        name="relative_mse_margin",
        length=n,
        dtype=torch.float32,
    )
    action_mse = tensor_1d(
        records["teacher_gt_action_mse"],
        name="teacher_gt_action_mse",
        length=n,
        dtype=torch.float32,
    )
    terminal_mse = tensor_1d(
        records["terminal_mse"],
        name="terminal_mse",
        length=n,
        dtype=torch.float32,
    )

    for name, tensor in [
        ("confidence_weight", confidence_weight),
        ("relative_mse_margin", relative_margin),
        ("teacher_gt_action_mse", action_mse),
        ("terminal_mse", terminal_mse),
    ]:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN or Inf")
    if bool(((confidence_weight < 0) | (confidence_weight > 1)).any()):
        raise RuntimeError("confidence_weight must lie in [0,1]")

    selected_required = [
        "sample_id",
        "sample_index",
        "episode_index",
        "local_step",
        "task_index",
        "teacher_idx_primary",
        "confidence_level",
        "confidence_weight",
    ]
    episode_indices: list[int] = []
    local_steps: list[int] = []
    task_indices: list[int] = []
    global_indices: list[int] = []
    confidence_levels: list[str] = []

    for teacher_row, selected in enumerate(selected_rows):
        require_fields(selected, selected_required, f"selected teacher row {teacher_row}")
        selected_id = str(selected["sample_id"])
        if selected_id != sample_ids[teacher_row]:
            raise RuntimeError(
                f"Teacher order mismatch at row {teacher_row}: PT={sample_ids[teacher_row]!r}, "
                f"JSONL={selected_id!r}"
            )
        if selected_id not in manifest_by_id:
            raise RuntimeError(f"Selected teacher {selected_id} is absent from manifest")

        manifest = manifest_by_id[selected_id]
        comparisons = {
            "sample_index": int(selected["sample_index"]),
            "episode_index": int(selected["episode_index"]),
            "local_step": int(selected["local_step"]),
            "task_index": int(selected["task_index"]),
        }
        for field, selected_value in comparisons.items():
            manifest_value = int(manifest[field])
            if selected_value != manifest_value:
                raise RuntimeError(
                    f"{selected_id} {field} mismatch: selected={selected_value}, "
                    f"manifest={manifest_value}"
                )

        if int(sample_indices[teacher_row]) != int(selected["sample_index"]):
            raise RuntimeError(f"{selected_id} sample_index mismatch between PT and JSONL")
        if int(teacher_idx[teacher_row]) != int(selected["teacher_idx_primary"]):
            raise RuntimeError(f"{selected_id} teacher_idx mismatch between PT and JSONL")
        if not math.isclose(
            float(confidence_weight[teacher_row]),
            float(selected["confidence_weight"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"{selected_id} confidence_weight mismatch between PT and JSONL"
            )

        episode_indices.append(int(manifest["episode_index"]))
        local_steps.append(int(manifest["local_step"]))
        task_indices.append(int(manifest["task_index"]))
        global_indices.append(int(manifest.get("global_index", -1)))
        confidence_levels.append(str(selected["confidence_level"]))

    return {
        "source_format": str(records.get("format", "unknown")),
        "source_primary_metric": str(records.get("primary_metric", "unknown")),
        "thresholds": records.get("thresholds", {}),
        "sample_ids": sample_ids,
        "sample_indices": sample_indices,
        "episode_indices": torch.tensor(episode_indices, dtype=torch.long),
        "local_steps": torch.tensor(local_steps, dtype=torch.long),
        "task_indices": torch.tensor(task_indices, dtype=torch.long),
        "global_indices": torch.tensor(global_indices, dtype=torch.long),
        "teacher_idx_primary": teacher_idx,
        "confidence_weight": confidence_weight,
        "confidence_levels": confidence_levels,
        "teacher_actions_lerobot": teacher_actions,
        "gt_actions_lerobot": gt_actions,
        "relative_mse_margin": relative_margin,
        "teacher_gt_action_mse": action_mse,
        "terminal_mse": terminal_mse,
        "teacher_horizon": teacher_horizon,
        "action_dim": action_dim,
    }


def common_index_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "sample_id": str(manifest["sample_id"]),
        "sample_index": int(manifest["sample_index"]),
        "episode_index": int(manifest["episode_index"]),
        "local_step": int(manifest["local_step"]),
        "task_index": int(manifest["task_index"]),
        "task_text": str(manifest["task_text"]),
        "window_horizon": int(manifest["window_horizon"]),
        "history_size": int(manifest["history_size"]),
    }
    for key in [
        "global_index",
        "episode_length",
        "episode_offset",
        "window_index_h8",
    ]:
        if key in manifest:
            fields[key] = int(manifest[key])
    return fields


def build_index(
    manifest_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    teacher_data: dict[str, Any],
    interface: dict[str, Any],
    index_mode: str,
    mix_order: str,
) -> list[dict[str, Any]]:
    teacher_row_by_id = {
        sample_id: row for row, sample_id in enumerate(teacher_data["sample_ids"])
    }
    selected_by_id = {str(row["sample_id"]): row for row in selected_rows}

    gt_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        sample_id = str(manifest["sample_id"])
        base = common_index_fields(manifest)
        gt_rows.append(
            {
                **base,
                "format": FORMAT_VERSION,
                "record_id": f"{sample_id}:gt",
                "source": "gt",
                "target_source": "lerobot_dataset",
                "teacher_row": None,
                "confidence_weight": 1.0,
                "supervision_policy": "official_gr00t_action_mask",
            }
        )

        if sample_id in teacher_row_by_id:
            teacher_row = teacher_row_by_id[sample_id]
            selected = selected_by_id[sample_id]
            teacher_rows.append(
                {
                    **base,
                    "format": FORMAT_VERSION,
                    "record_id": f"{sample_id}:teacher",
                    "source": "teacher",
                    "target_source": "flare_teacher_records",
                    "teacher_row": int(teacher_row),
                    "teacher_idx_primary": int(selected["teacher_idx_primary"]),
                    "confidence_level": str(selected["confidence_level"]),
                    "confidence_weight": float(
                        teacher_data["confidence_weight"][teacher_row]
                    ),
                    "supervised_prefix_horizon": int(
                        teacher_data["teacher_horizon"]
                    ),
                    "processor_action_horizon": int(
                        interface["processor_action_horizon"]
                    ),
                    "model_action_horizon": int(interface["model_action_horizon"]),
                    "supervision_policy": "replace_prefix_and_zero_tail_mask",
                }
            )

    if len(teacher_rows) != len(selected_rows):
        raise RuntimeError(
            f"Built {len(teacher_rows)} teacher index rows, expected {len(selected_rows)}"
        )

    if index_mode == "teacher_only":
        mixed = teacher_rows
    elif index_mode == "mixed":
        if mix_order == "gt_then_teacher":
            mixed = gt_rows + teacher_rows
        elif mix_order == "interleave":
            teacher_by_sample = {str(row["sample_id"]): row for row in teacher_rows}
            mixed = []
            for gt_row in gt_rows:
                mixed.append(gt_row)
                teacher = teacher_by_sample.get(str(gt_row["sample_id"]))
                if teacher is not None:
                    mixed.append(teacher)
        else:
            raise ValueError(f"Unsupported mix_order={mix_order!r}")
    else:
        raise ValueError(f"Unsupported index_mode={index_mode!r}")

    for index, row in enumerate(mixed):
        row["distillation_index"] = index
    return mixed


def main() -> None:
    args = parse_args()
    required_paths = [
        args.manifest,
        args.manifest_summary,
        args.teacher_selected,
        args.teacher_records,
        args.teacher_summary,
        args.checkpoint_config,
        args.processor_config,
    ]
    require_files(required_paths)

    manifest_rows = load_jsonl(args.manifest)
    manifest_summary = load_json(args.manifest_summary)
    selected_rows = load_jsonl(args.teacher_selected)
    teacher_records = load_torch(args.teacher_records)
    teacher_summary = load_json(args.teacher_summary)
    checkpoint_config = load_json(args.checkpoint_config)
    processor_config = load_json(args.processor_config)

    manifest_by_id, manifest_horizon = validate_manifest(
        manifest_rows, args.expected_gt_count
    )
    teacher_data = validate_teacher_records(
        selected_rows=selected_rows,
        records=teacher_records,
        manifest_by_id=manifest_by_id,
        manifest_horizon=manifest_horizon,
        expected_count=args.expected_teacher_count,
    )
    interface = validate_checkpoint_interface(
        checkpoint_config=checkpoint_config,
        processor_config=processor_config,
        embodiment_key=args.embodiment_key,
        teacher_horizon=int(teacher_data["teacher_horizon"]),
        action_dim=int(teacher_data["action_dim"]),
    )

    # Spatial manifest summary uses "selected_windows" not "protocol.total_samples"
    manifest_total = manifest_summary.get("selected_windows", len(manifest_rows))
    if int(manifest_total) != len(manifest_rows):
        raise RuntimeError(
            f"Manifest summary selected_windows={manifest_total} != manifest rows={len(manifest_rows)}"
        )
    # Spatial teacher summary uses "selected_for_direct_augmentation" or "selected_for_distillation"
    selected_key = (
        "selected_for_direct_augmentation"
        if "selected_for_direct_augmentation" in teacher_summary
        else "selected_for_distillation"
    )
    selected_summary = teacher_summary.get(selected_key, {})
    selected_summary_count = int(selected_summary.get("count", len(selected_rows)))
    if selected_summary_count != len(selected_rows):
        raise RuntimeError(
            f"Teacher summary {selected_key}.count={selected_summary_count} "
            f"!= selected rows={len(selected_rows)}"
        )

    # ---- Quality-level filter ----
    if args.quality_levels:
        allowed = set(args.quality_levels)
        pre_count = len(selected_rows)
        keep = [
            i for i, row in enumerate(selected_rows)
            if str(row.get("confidence_level", row.get("teacher_quality_level", ""))) in allowed
        ]
        selected_rows = [selected_rows[i] for i in keep]
        # Rebuild teacher_data list-fields to match filtered selected_rows
        list_keys = ["sample_ids", "confidence_levels"]
        tensor_keys_1d = [
            "sample_indices", "teacher_idx_primary", "confidence_weight",
            "relative_mse_margin", "teacher_gt_action_mse", "terminal_mse",
            "episode_indices", "local_steps", "task_indices", "global_indices",
        ]
        tensor_keys_N = ["teacher_actions_lerobot", "gt_actions_lerobot"]
        keep_tensor = torch.tensor(keep, dtype=torch.long)
        for key in list_keys:
            if key in teacher_data:
                old = teacher_data[key]
                teacher_data[key] = [old[i] for i in keep]
        for key in tensor_keys_1d:
            if key in teacher_data:
                teacher_data[key] = teacher_data[key][keep_tensor]
        for key in tensor_keys_N:
            if key in teacher_data:
                teacher_data[key] = teacher_data[key][keep_tensor]
        post_count = len(selected_rows)
        print(f"Quality filter ({sorted(allowed)}): {pre_count} -> {post_count} teachers")
        if post_count == 0:
            raise RuntimeError(
                f"No teachers remain after --quality-levels {sorted(allowed)} filter"
            )

    mixed_index = build_index(
        manifest_rows=manifest_rows,
        selected_rows=selected_rows,
        teacher_data=teacher_data,
        interface=interface,
        index_mode=args.index_mode,
        mix_order=args.mix_order,
    )

    gt_count = sum(row["source"] == "gt" for row in mixed_index)
    teacher_count = sum(row["source"] == "teacher" for row in mixed_index)
    total_count = len(mixed_index)
    expected_gt_count = len(manifest_rows) if args.index_mode == "mixed" else 0
    if gt_count != expected_gt_count or teacher_count != len(selected_rows):
        raise RuntimeError(
            "Final distillation-index counts are inconsistent: "
            f"gt={gt_count}/{expected_gt_count}, "
            f"teacher={teacher_count}/{len(selected_rows)}"
        )

    output_dir = args.output_dir.resolve()
    index_path = output_dir / "distillation_index.jsonl"
    teacher_index_path = output_dir / "teacher_index.jsonl"
    teacher_path = output_dir / "teacher_records_enriched.pt"
    summary_path = output_dir / "distillation_summary.json"

    input_hashes = {str(path): sha256_file(path) for path in required_paths}
    wrapper_contract = {
        "action_representation": "raw_lerobot_before_gr00t_normalization",
        "teacher_action_shape": [
            teacher_count,
            int(teacher_data["teacher_horizon"]),
            int(teacher_data["action_dim"]),
        ],
        "teacher_sample_steps": [
            "load the same observation using episode_index + local_step",
            "obtain the normal official GR00T action chunk and action mask",
            "replace only action[0:teacher_horizon] with the selected teacher action",
            "apply the same GR00T action normalization used for GT targets",
            "set action_mask[0:teacher_horizon] to one",
            "set action_mask[teacher_horizon:] to zero",
        ],
        "important": (
            "Do not concatenate GT and teacher along action_dim; do not change the "
            "checkpoint action_horizon; do not invent extra teacher steps."
        ),
    }

    enriched_teacher_records = {
        "format": TEACHER_FORMAT_VERSION,
        "source_format": teacher_data["source_format"],
        "source_primary_metric": teacher_data["source_primary_metric"],
        "thresholds": teacher_data["thresholds"],
        "manifest_sha256": input_hashes[str(args.manifest)],
        "teacher_selected_sha256": input_hashes[str(args.teacher_selected)],
        "teacher_records_sha256": input_hashes[str(args.teacher_records)],
        "sample_ids": teacher_data["sample_ids"],
        "sample_indices": teacher_data["sample_indices"],
        "episode_indices": teacher_data["episode_indices"],
        "local_steps": teacher_data["local_steps"],
        "task_indices": teacher_data["task_indices"],
        "global_indices": teacher_data["global_indices"],
        "teacher_idx_primary": teacher_data["teacher_idx_primary"],
        "confidence_weight": teacher_data["confidence_weight"],
        "confidence_levels": teacher_data["confidence_levels"],
        "teacher_actions_lerobot": teacher_data["teacher_actions_lerobot"],
        "gt_actions_lerobot_h8_for_audit": teacher_data["gt_actions_lerobot"],
        "relative_mse_margin": teacher_data["relative_mse_margin"],
        "teacher_gt_action_mse": teacher_data["teacher_gt_action_mse"],
        "terminal_mse": teacher_data["terminal_mse"],
        "teacher_horizon": int(teacher_data["teacher_horizon"]),
        "processor_action_horizon": int(interface["processor_action_horizon"]),
        "model_action_horizon": int(interface["model_action_horizon"]),
        "action_dim": int(teacher_data["action_dim"]),
        "action_keys": interface["action_keys"],
        "wrapper_contract": wrapper_contract,
    }

    confidence_counts = Counter(teacher_data["confidence_levels"])
    task_count = len({int(row["task_index"]) for row in manifest_rows})
    episode_count = len({int(row["episode_index"]) for row in manifest_rows})
    summary = {
        "format": SUMMARY_FORMAT_VERSION,
        "status": "PASS",
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_summary": str(args.manifest_summary.resolve()),
            "teacher_selected": str(args.teacher_selected.resolve()),
            "teacher_records": str(args.teacher_records.resolve()),
            "teacher_summary": str(args.teacher_summary.resolve()),
            "checkpoint_config": str(args.checkpoint_config.resolve()),
            "processor_config": str(args.processor_config.resolve()),
            "sha256": input_hashes,
        },
        "protocol": {
            "index_mode": args.index_mode,
            "mix_order": args.mix_order if args.index_mode == "mixed" else None,
            "gt_samples": gt_count,
            "teacher_samples": teacher_count,
            "total_samples": total_count,
            "teacher_fraction": teacher_count / total_count,
            "gt_fraction": gt_count / total_count,
            "tasks": task_count,
            "episodes": episode_count,
            "confidence_counts": dict(sorted(confidence_counts.items())),
        },
        "gr00t_interface": interface,
        "wrapper_contract": wrapper_contract,
        "outputs": {
            "distillation_index": str(index_path),
            "teacher_index": str(teacher_index_path),
            "teacher_records_enriched": str(teacher_path),
            "distillation_summary": str(summary_path),
        },
    }

    print("========== FLARE distillation dataset ==========")
    print("index mode                :", args.index_mode)
    print("index GT samples          :", gt_count)
    print("selected teacher samples  :", teacher_count)
    print("mixed training samples    :", total_count)
    print("teacher fraction          :", f"{teacher_count / total_count:.4f}")
    print("teacher action shape      :", tuple(teacher_data["teacher_actions_lerobot"].shape))
    print("processor action horizon  :", interface["processor_action_horizon"])
    print("model action horizon      :", interface["model_action_horizon"])
    print("mix order                 :", args.mix_order)

    # Always extract a teacher-only index for weighted-mixing configs
    teacher_only_rows = [row for row in mixed_index if row["source"] == "teacher"]

    if args.dry_run:
        print("teacher-only index rows   :", len(teacher_only_rows))
        print("status                    : PASS (dry-run; no files written)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(index_path, mixed_index)
    atomic_write_jsonl(teacher_index_path, teacher_only_rows)
    atomic_torch_save(teacher_path, enriched_teacher_records)
    atomic_write_json(summary_path, summary)

    print("distillation index        :", index_path)
    print("teacher-only index        :", teacher_index_path)
    print("enriched teacher records  :", teacher_path)
    print("summary                   :", summary_path)
    print("status                    : PASS")


if __name__ == "__main__":
    main()
