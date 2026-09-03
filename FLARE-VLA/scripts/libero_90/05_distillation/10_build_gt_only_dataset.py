#!/usr/bin/env python3
"""Build the exact 500-record GT-only control dataset from FLARE sidecars.

The control group must use the same GT windows as the FLARE mixed experiment.
This script therefore filters ``source == "gt"`` rows from an existing
``distillation_index.jsonl`` instead of rebuilding or resampling a manifest.

The GR00T FLARE dataset interface requires a teacher sidecar even when there are
no teacher rows. A valid empty ``teacher_records_enriched.pt`` is emitted so the
same Dataset implementation, processor, and training entry point can be reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


INDEX_FILENAME = "distillation_index.jsonl"
TEACHER_FILENAME = "teacher_records_enriched.pt"
SUMMARY_FILENAME = "distillation_summary.json"
INDEX_FORMAT = "flare_distillation_index_v1"
TEACHER_FORMAT = "flare_distillation_teacher_records_v1"
SUMMARY_FORMAT = "flare_distillation_summary_v1"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_source_dir() -> Path:
    server = Path(
        "/root/autodl-tmp/Isaac-GR00T/outputs/flare_distillation_filtered_direct"
    )
    if server.is_dir():
        return server
    return project_root() / "Isaac-GR00T" / "outputs" / "flare_distillation_filtered_direct"


def default_output_dir() -> Path:
    server_root = Path("/root/autodl-tmp/Isaac-GR00T")
    if server_root.is_dir():
        return server_root / "outputs" / "flare_distillation_gt_only_500"
    return project_root() / "Isaac-GR00T" / "outputs" / "flare_distillation_gt_only_500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=default_source_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--expected-gt-count", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
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
                    f"Expected JSON object at {path}:{line_no}, got {type(row).__name__}"
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Index is empty: {path}")
    return rows


def load_torch_dict(path: Path) -> dict[str, Any]:
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
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def require_fields(value: dict[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise KeyError(f"{context} is missing required fields: {missing}")


def build_gt_rows(rows: list[dict[str, Any]], expected_count: int) -> list[dict[str, Any]]:
    required = [
        "format",
        "distillation_index",
        "record_id",
        "sample_id",
        "sample_index",
        "episode_index",
        "local_step",
        "source",
        "teacher_row",
    ]
    gt_rows: list[dict[str, Any]] = []
    record_ids: set[str] = set()

    for source_index, row in enumerate(rows):
        require_fields(row, required, f"source index row {source_index}")
        if str(row["format"]) != INDEX_FORMAT:
            raise RuntimeError(
                f"Unsupported index format at source row {source_index}: {row['format']!r}"
            )
        if int(row["distillation_index"]) != source_index:
            raise RuntimeError(
                f"Source distillation_index is not contiguous at row {source_index}"
            )
        if str(row["source"]) != "gt":
            continue
        if row["teacher_row"] is not None:
            raise RuntimeError(f"GT source row {source_index} must have teacher_row=null")

        copied = dict(row)
        copied["distillation_index"] = len(gt_rows)
        copied["control_group"] = "gt_only"
        copied["control_source_index"] = source_index

        record_id = str(copied["record_id"])
        if record_id in record_ids:
            raise RuntimeError(f"Duplicate GT record_id: {record_id}")
        record_ids.add(record_id)
        gt_rows.append(copied)

    if expected_count > 0 and len(gt_rows) != expected_count:
        raise RuntimeError(
            f"Filtered {len(gt_rows)} GT rows, expected exactly {expected_count}"
        )
    if not gt_rows:
        raise RuntimeError("No GT rows were found in the source index")
    return gt_rows


def build_empty_teacher_records(source: dict[str, Any]) -> dict[str, Any]:
    required = [
        "format",
        "teacher_horizon",
        "processor_action_horizon",
        "model_action_horizon",
        "action_dim",
        "action_keys",
        "teacher_actions_lerobot",
    ]
    require_fields(source, required, "source teacher records")
    if str(source["format"]) != TEACHER_FORMAT:
        raise RuntimeError(
            f"Unsupported teacher format {source['format']!r}; expected {TEACHER_FORMAT!r}"
        )

    teacher_horizon = int(source["teacher_horizon"])
    action_dim = int(source["action_dim"])
    source_actions = torch.as_tensor(source["teacher_actions_lerobot"])
    if source_actions.ndim != 3:
        raise RuntimeError(
            "source teacher_actions_lerobot must have shape [N, H, D], got "
            f"{tuple(source_actions.shape)}"
        )
    if tuple(source_actions.shape[1:]) != (teacher_horizon, action_dim):
        raise RuntimeError(
            "Teacher metadata/tensor shape mismatch: "
            f"tensor={tuple(source_actions.shape)}, horizon={teacher_horizon}, dim={action_dim}"
        )

    empty_float = torch.empty((0,), dtype=torch.float32)
    empty_long = torch.empty((0,), dtype=torch.long)
    empty_actions = source_actions.detach().cpu().new_empty(
        (0, teacher_horizon, action_dim)
    )

    return {
        "format": TEACHER_FORMAT,
        "source_format": source.get("source_format"),
        "source_primary_metric": source.get("source_primary_metric"),
        "thresholds": source.get("thresholds", {}),
        "control_group": "gt_only",
        "sample_ids": [],
        "sample_indices": empty_long.clone(),
        "episode_indices": empty_long.clone(),
        "local_steps": empty_long.clone(),
        "task_indices": empty_long.clone(),
        "global_indices": empty_long.clone(),
        "teacher_idx_primary": empty_long.clone(),
        "confidence_weight": empty_float.clone(),
        "confidence_levels": [],
        "teacher_actions_lerobot": empty_actions,
        "gt_actions_lerobot_h8_for_audit": empty_actions.clone(),
        "relative_mse_margin": empty_float.clone(),
        "teacher_gt_action_mse": empty_float.clone(),
        "terminal_mse": empty_float.clone(),
        "teacher_horizon": teacher_horizon,
        "processor_action_horizon": int(source["processor_action_horizon"]),
        "model_action_horizon": int(source["model_action_horizon"]),
        "action_dim": action_dim,
        "action_keys": [str(key) for key in source["action_keys"]],
        "wrapper_contract": {
            "control_group": "gt_only",
            "teacher_action_shape": [0, teacher_horizon, action_dim],
            "supervision_policy": "official_gr00t_action_mask_for_all_records",
            "important": "No teacher action is referenced or injected in this control group.",
        },
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_index = source_dir / INDEX_FILENAME
    source_teacher = source_dir / TEACHER_FILENAME
    source_summary = source_dir / SUMMARY_FILENAME

    for path in (source_index, source_teacher, source_summary):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_rows = load_jsonl(source_index)
    source_teacher_records = load_torch_dict(source_teacher)
    source_summary_data = load_json(source_summary)
    if str(source_summary_data.get("status")) != "PASS":
        raise RuntimeError(f"Source summary status is not PASS: {source_summary}")

    gt_rows = build_gt_rows(source_rows, args.expected_gt_count)
    empty_teacher_records = build_empty_teacher_records(source_teacher_records)

    index_path = output_dir / INDEX_FILENAME
    teacher_path = output_dir / TEACHER_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    tasks = len({int(row["task_index"]) for row in gt_rows if "task_index" in row})
    episodes = len(
        {int(row["episode_index"]) for row in gt_rows if "episode_index" in row}
    )
    gr00t_interface = dict(source_summary_data.get("gr00t_interface", {}))
    teacher_horizon = int(empty_teacher_records["teacher_horizon"])
    action_dim = int(empty_teacher_records["action_dim"])

    summary = {
        "format": SUMMARY_FORMAT,
        "status": "PASS",
        "control_group": "gt_only",
        "inputs": {
            "source_distillation_dir": str(source_dir),
            "source_distillation_index": str(source_index),
            "source_teacher_records": str(source_teacher),
            "source_distillation_summary": str(source_summary),
            "sha256": {
                str(source_index): sha256_file(source_index),
                str(source_teacher): sha256_file(source_teacher),
                str(source_summary): sha256_file(source_summary),
            },
        },
        "protocol": {
            "mix_order": "gt_only",
            "gt_samples": len(gt_rows),
            "teacher_samples": 0,
            "total_samples": len(gt_rows),
            "teacher_fraction": 0.0,
            "gt_fraction": 1.0,
            "tasks": tasks,
            "episodes": episodes,
            "source_total_samples": len(source_rows),
            "source_teacher_samples_removed": len(source_rows) - len(gt_rows),
        },
        "gr00t_interface": gr00t_interface,
        "wrapper_contract": {
            "action_representation": "official_lerobot_gt_action",
            "teacher_action_shape": [0, teacher_horizon, action_dim],
            "supervision_policy": "official_gr00t_action_mask_for_all_records",
            "important": (
                "This control uses exactly the GT rows from the FLARE mixed index. "
                "No teacher action is referenced or injected."
            ),
        },
        "outputs": {
            "distillation_index": str(index_path),
            "teacher_records_enriched": str(teacher_path),
            "distillation_summary": str(summary_path),
        },
    }

    print("========== GR00T GT-only control dataset ==========")
    print("source mixed records       :", len(source_rows))
    print("retained GT records        :", len(gt_rows))
    print("teacher records            :", 0)
    print("teacher action shape       :", tuple(empty_teacher_records["teacher_actions_lerobot"].shape))
    print("tasks                      :", tasks)
    print("episodes                   :", episodes)

    if args.dry_run:
        print("status                     : PASS (dry-run; no files written)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(index_path, gt_rows)
    atomic_torch_save(teacher_path, empty_teacher_records)
    atomic_write_json(summary_path, summary)

    print("GT-only index              :", index_path)
    print("empty teacher sidecar      :", teacher_path)
    print("summary                    :", summary_path)
    print("status                     : PASS")


if __name__ == "__main__":
    main()
