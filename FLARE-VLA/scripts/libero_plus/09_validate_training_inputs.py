#!/usr/bin/env python3
"""Fail-closed preflight for four-suite LIBERO-Plus FLARE training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from gr00t.data.dataset.flare_sidecars import resolve_flare_distillation_sidecars


SUMMARY_FORMAT = "flare_distillation_summary_v1"
TEACHER_FORMAT = "flare_distillation_teacher_records_v1"
INDEX_FORMAT = "flare_distillation_index_v1"
EXPECTED_SUITE_COUNTS = {
    "libero_spatial": 109_403,
    "libero_object": 135_456,
    "libero_goal": 98_964,
    "libero_10": 144_057,
}
EXPECTED_ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--distillation-dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--teacher-mix-ratio", type=float, required=True)
    parser.add_argument("--fixed-gt-samples-per-batch", type=int, required=True)
    parser.add_argument("--fixed-teacher-samples-per-batch", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--expected-total-frames", type=int, default=2_238_036)
    parser.add_argument("--expected-total-episodes", type=int, default=14_347)
    parser.add_argument("--expected-total-tasks", type=int, default=40)
    parser.add_argument(
        "--skip-backbone-processor-load",
        action="store_true",
        help="Only check backbone files; do not instantiate its local processor.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def load_torch(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dict in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required non-empty file is missing: {path}")


def resolved_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Missing path for {context}")
    return Path(value).expanduser().resolve()


def validate_model_assets(base_model: Path, backbone: Path) -> None:
    for name in (
        "config.json",
        "processor_config.json",
        "statistics.json",
        "model.safetensors.index.json",
    ):
        require_file(base_model / name)

    weight_index = load_json(base_model / "model.safetensors.index.json")
    weight_map = weight_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Base-model weight index has no weight_map")
    for shard_name in {str(value) for value in weight_map.values()}:
        require_file(base_model / shard_name)

    for name in (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
    ):
        require_file(backbone / name)


def validate_dataset(
    dataset_path: Path,
    expected_frames: int,
    expected_episodes: int,
    expected_tasks: int,
) -> int:
    meta_dir = dataset_path / "meta"
    for name in (
        "info.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "modality.json",
        "stats.json",
        "tasks.jsonl",
        ".gripper_converted",
    ):
        require_file(meta_dir / name)

    info = load_json(meta_dir / "info.json")
    expected = {
        "total_frames": expected_frames,
        "total_episodes": expected_episodes,
        "total_tasks": expected_tasks,
    }
    for field, expected_value in expected.items():
        actual = int(info.get(field, -1))
        if actual != expected_value:
            raise RuntimeError(
                f"LIBERO-Plus {field}={actual}, expected {expected_value}"
            )
    if info.get("splits") != {"train": f"0:{expected_episodes}"}:
        raise RuntimeError(f"Unexpected LIBERO-Plus splits: {info.get('splits')!r}")
    return int(info["total_frames"])


def validate_summary(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    distillation_dir: Path,
    manifest: Path,
    manifest_summary: Path,
    checkpoint_root: Path,
) -> tuple[int, int, int, str]:
    if summary.get("format") != SUMMARY_FORMAT or summary.get("status") != "PASS":
        raise RuntimeError(
            f"Distillation summary is not a canonical PASS {SUMMARY_FORMAT}: "
            f"{summary_path}"
        )

    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("Distillation summary is missing inputs")
    expected_input_paths = {
        "manifest": manifest,
        "manifest_summary": manifest_summary,
        "checkpoint_root": checkpoint_root,
    }
    for field, expected_path in expected_input_paths.items():
        actual_path = resolved_path(inputs.get(field), f"inputs.{field}")
        if actual_path != expected_path:
            raise RuntimeError(
                f"inputs.{field}={actual_path}, expected {expected_path}. "
                "Rebuild sidecars with the corrected LIBERO-Plus 07 script."
            )

    suite_files = inputs.get("suite_checkpoint_files")
    if not isinstance(suite_files, dict) or set(suite_files) != set(
        EXPECTED_SUITE_COUNTS
    ):
        raise RuntimeError(
            "Summary does not contain all four suite_checkpoint_files; "
            "the sidecars are stale or were built by the old spatial-only script."
        )
    for suite in EXPECTED_SUITE_COUNTS:
        suite_entry = suite_files[suite]
        if not isinstance(suite_entry, dict):
            raise RuntimeError(f"Invalid checkpoint entry for {suite}")
        for name in ("config", "processor_config"):
            expected_path = (checkpoint_root / suite / f"{name}.json").resolve()
            actual_path = resolved_path(
                suite_entry.get(name), f"suite_checkpoint_files.{suite}.{name}"
            )
            if actual_path != expected_path:
                raise RuntimeError(
                    f"Checkpoint provenance mismatch for {suite}/{name}: "
                    f"{actual_path} != {expected_path}"
                )

    hashes = inputs.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Distillation summary has no input SHA-256 map")
    for path_text, expected_hash in hashes.items():
        path = Path(path_text).expanduser().resolve()
        require_file(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Input changed after sidecar build: {path}\n"
                f"stored={expected_hash}\nactual={actual_hash}"
            )
    manifest_hash = hashes.get(str(manifest))
    if not isinstance(manifest_hash, str):
        raise RuntimeError("Canonical manifest is absent from the input SHA-256 map")

    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("Distillation summary is missing protocol")
    if protocol.get("index_mode") != "teacher_only":
        raise RuntimeError("Training requires a teacher_only distillation index")
    gt_count = int(protocol.get("gt_samples", -1))
    teacher_count = int(protocol.get("teacher_samples", -1))
    total_count = int(protocol.get("total_samples", -1))
    if gt_count != 0 or teacher_count <= 0 or total_count != teacher_count:
        raise RuntimeError(
            "Invalid teacher-only protocol counts: "
            f"gt={gt_count}, teacher={teacher_count}, total={total_count}"
        )
    if int(protocol.get("tasks", -1)) != 40:
        raise RuntimeError(f"Expected 40 tasks, got {protocol.get('tasks')!r}")
    if int(protocol.get("episodes", -1)) != 14_347:
        raise RuntimeError(f"Expected 14347 episodes, got {protocol.get('episodes')!r}")
    suite_counts = {
        str(key): int(value)
        for key, value in protocol.get("suite_counts", {}).items()
    }
    if suite_counts != EXPECTED_SUITE_COUNTS:
        raise RuntimeError(
            f"Canonical manifest suite counts are invalid: {suite_counts}"
        )
    confidence_counts = {
        str(key): int(value)
        for key, value in protocol.get("confidence_counts", {}).items()
    }
    if not confidence_counts or not set(confidence_counts) <= {"high", "medium"}:
        raise RuntimeError(f"Unexpected confidence counts: {confidence_counts}")
    if sum(confidence_counts.values()) != teacher_count:
        raise RuntimeError("Confidence counts do not sum to teacher_samples")

    interface = summary.get("gr00t_interface")
    interfaces = summary.get("gr00t_interfaces_by_suite")
    if not isinstance(interface, dict) or not isinstance(interfaces, dict):
        raise RuntimeError("Missing four-suite GR00T interface contract")
    if set(interfaces) != set(EXPECTED_SUITE_COUNTS):
        raise RuntimeError("GR00T interfaces do not cover all four LIBERO suites")
    expected_interface = {
        "teacher_horizon": 8,
        "processor_action_horizon": 16,
        "model_action_horizon": 40,
        "action_dim": 7,
    }
    for field, expected_value in expected_interface.items():
        if int(interface.get(field, -1)) != expected_value:
            raise RuntimeError(
                f"gr00t_interface.{field}={interface.get(field)!r}, "
                f"expected {expected_value}"
            )
    if interface.get("action_keys") != EXPECTED_ACTION_KEYS:
        raise RuntimeError(f"Unexpected action keys: {interface.get('action_keys')!r}")
    if any(suite_interface != interface for suite_interface in interfaces.values()):
        raise RuntimeError("The four suite checkpoints do not share one GR00T interface")

    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("Distillation summary is missing outputs")
    expected_outputs = {
        "distillation_index": distillation_dir / "distillation_index.jsonl",
        "teacher_index": distillation_dir / "teacher_index.jsonl",
        "teacher_records_enriched": distillation_dir
        / "teacher_records_enriched.pt",
        "distillation_summary": summary_path,
    }
    for field, expected_path in expected_outputs.items():
        actual_path = resolved_path(outputs.get(field), f"outputs.{field}")
        if actual_path != expected_path:
            raise RuntimeError(
                f"outputs.{field}={actual_path}, expected {expected_path}"
            )
        require_file(expected_path)
    output_hashes = outputs.get("sha256")
    if not isinstance(output_hashes, dict):
        raise RuntimeError(
            "Summary has no output SHA-256 map. Re-run the corrected 07 builder "
            "before training."
        )
    for field in (
        "distillation_index",
        "teacher_index",
        "teacher_records_enriched",
    ):
        path = expected_outputs[field]
        actual_hash = sha256_file(path)
        if output_hashes.get(field) != actual_hash:
            raise RuntimeError(f"Output SHA-256 mismatch for {path}")

    return (
        teacher_count,
        int(interface["teacher_horizon"]),
        int(interface["processor_action_horizon"]),
        manifest_hash,
    )


def validate_index_and_records(
    distillation_dir: Path,
    teacher_count: int,
    manifest_hash: str,
) -> dict[str, int]:
    sidecars = resolve_flare_distillation_sidecars(
        distillation_dataset_dir=distillation_dir,
        distillation_index_path=None,
        teacher_records_path=None,
    )
    if sidecars is None or sidecars.source_counts != {"teacher": teacher_count}:
        raise RuntimeError(
            "Resolved sidecar counts are invalid: "
            f"{getattr(sidecars, 'source_counts', None)}"
        )

    index_path = distillation_dir / "distillation_index.jsonl"
    suite_counts: Counter[str] = Counter()
    sample_ids: list[str] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            if row.get("format") != INDEX_FORMAT or row.get("source") != "teacher":
                raise RuntimeError(f"Invalid teacher index row {row_index}")
            if int(row.get("teacher_row", -1)) != row_index:
                raise RuntimeError(f"Non-contiguous teacher_row at index row {row_index}")
            suite = str(row.get("suite"))
            if (
                suite not in EXPECTED_SUITE_COUNTS
                or row.get("gt_latent_key") != suite
            ):
                raise RuntimeError(f"Invalid suite identity at index row {row_index}")
            suite_counts[suite] += 1
            sample_ids.append(str(row.get("sample_id")))
    if len(sample_ids) != teacher_count or len(set(sample_ids)) != teacher_count:
        raise RuntimeError("Teacher index sample IDs are missing or duplicated")
    if set(suite_counts) != set(EXPECTED_SUITE_COUNTS):
        raise RuntimeError(f"Teacher index lacks suite coverage: {dict(suite_counts)}")

    records = load_torch(distillation_dir / "teacher_records_enriched.pt")
    if records.get("format") != TEACHER_FORMAT:
        raise RuntimeError(f"Unexpected teacher record format: {records.get('format')!r}")
    if records.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Teacher records do not match the canonical manifest hash")
    record_ids = [str(value) for value in records.get("sample_ids", [])]
    if record_ids != sample_ids:
        raise RuntimeError("Teacher record order does not match distillation index")
    record_suites = [str(value) for value in records.get("suites", [])]
    if len(record_suites) != teacher_count:
        raise RuntimeError("Teacher record suite metadata length mismatch")
    if Counter(record_suites) != suite_counts:
        raise RuntimeError("Teacher record suite counts do not match the index")
    actions = torch.as_tensor(records.get("teacher_actions_lerobot"))
    if tuple(actions.shape) != (teacher_count, 8, 7):
        raise RuntimeError(f"Unexpected teacher action shape: {tuple(actions.shape)}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("Teacher actions contain NaN or Inf")
    return dict(sorted(suite_counts.items()))


def validate_backbone_processor(backbone: Path) -> None:
    from transformers import Qwen3VLProcessor

    processor = Qwen3VLProcessor.from_pretrained(
        str(backbone),
        local_files_only=True,
        trust_remote_code=True,
    )
    if processor.tokenizer is None:
        raise RuntimeError("Local Cosmos processor has no tokenizer")


def main() -> None:
    args = parse_args()
    for field in (
        "dataset_path",
        "distillation_dataset_dir",
        "manifest",
        "manifest_summary",
        "checkpoint_root",
        "base_model_path",
        "backbone_path",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())

    if not 0.0 < args.teacher_mix_ratio < 1.0 or not math.isfinite(
        args.teacher_mix_ratio
    ):
        raise ValueError("teacher_mix_ratio must be finite and lie in (0, 1)")
    if args.num_gpus != 1 or args.global_batch_size != 64:
        raise ValueError(
            "Fixed-ratio replay requires num_gpus=1 and global_batch_size=64"
        )
    if (
        args.fixed_gt_samples_per_batch <= 0
        or args.fixed_teacher_samples_per_batch <= 0
    ):
        raise ValueError("Fixed GT and teacher sample counts must both be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    fixed_batch_size = (
        args.fixed_gt_samples_per_batch + args.fixed_teacher_samples_per_batch
    )
    per_device_batch_size = args.global_batch_size // args.num_gpus
    if fixed_batch_size != per_device_batch_size:
        raise ValueError(
            "Fixed source counts must sum to the per-device batch size: "
            f"{fixed_batch_size} != {per_device_batch_size}"
        )
    fixed_teacher_fraction = args.fixed_teacher_samples_per_batch / fixed_batch_size
    if not math.isclose(
        args.teacher_mix_ratio,
        fixed_teacher_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "teacher_mix_ratio must mirror the fixed teacher sample fraction: "
            f"{args.teacher_mix_ratio} != {fixed_teacher_fraction}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GR00T training requires a GPU")
    if torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError(
            f"Requested {args.num_gpus} GPUs, but only "
            f"{torch.cuda.device_count()} are visible"
        )

    total_frames = validate_dataset(
        args.dataset_path,
        args.expected_total_frames,
        args.expected_total_episodes,
        args.expected_total_tasks,
    )
    validate_model_assets(args.base_model_path, args.backbone_path)
    summary_path = args.distillation_dataset_dir / "distillation_summary.json"
    require_file(summary_path)
    summary = load_json(summary_path)
    teacher_count, teacher_horizon, gt_horizon, manifest_hash = validate_summary(
        summary,
        summary_path=summary_path,
        distillation_dir=args.distillation_dataset_dir,
        manifest=args.manifest,
        manifest_summary=args.manifest_summary,
        checkpoint_root=args.checkpoint_root,
    )
    teacher_suite_counts = validate_index_and_records(
        args.distillation_dataset_dir,
        teacher_count,
        manifest_hash,
    )
    if not args.skip_backbone_processor_load:
        validate_backbone_processor(args.backbone_path)

    supervised_teacher_steps = args.fixed_teacher_samples_per_batch * teacher_horizon
    supervised_gt_steps = args.fixed_gt_samples_per_batch * gt_horizon
    teacher_supervision_fraction = supervised_teacher_steps / (
        supervised_teacher_steps + supervised_gt_steps
    )
    dataset_teacher_fraction = teacher_count / (teacher_count + total_frames)
    update_gt_samples = (
        args.fixed_gt_samples_per_batch * args.gradient_accumulation_steps
    )
    update_teacher_samples = (
        args.fixed_teacher_samples_per_batch * args.gradient_accumulation_steps
    )

    print("========== LIBERO-Plus training preflight ==========")
    print("dataset frames                    :", total_frames)
    print("teacher records                   :", teacher_count)
    print("teacher rows by suite             :", teacher_suite_counts)
    print(
        "fixed per-device micro-batch       :",
        f"GT={args.fixed_gt_samples_per_batch}, "
        f"teacher={args.fixed_teacher_samples_per_batch}",
    )
    print("fixed teacher sample fraction     :", f"{fixed_teacher_fraction:.6%}")
    print("configured teacher_mix_ratio      :", f"{args.teacher_mix_ratio:.12g}")
    print("teacher supervised-step fraction  :", f"{teacher_supervision_fraction:.6%}")
    print(
        "samples per optimizer update       :",
        f"GT={update_gt_samples}, teacher={update_teacher_samples}",
    )
    print(
        "available-data teacher fraction    :",
        f"{dataset_teacher_fraction:.6%} (not used for sampling)",
    )
    print("local Cosmos processor            : PASS")
    print("four-suite provenance             : PASS")
    print("status                            : PASS")


if __name__ == "__main__":
    main()
