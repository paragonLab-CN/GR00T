#!/usr/bin/env python3
"""Preflight real fixed-ratio GT-replay batches before formal training."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import os
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from check_augmentation_pipeline import COLOR_JITTER, EXTRA_AUG, build_processor
from gr00t.configs.base_config import get_default_config
from gr00t.data.dataset.flare_distillation_dataset import FlareDistillationDataset
from gr00t.data.dataset.sharded_mixture_dataset import (
    FIXED_BATCH_SOURCE_KEY,
    FixedRatioShardedMixtureDataset,
)
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.types import EmbodimentTag
from gr00t.experiment.trainer import FixedSourceCountCollator
import torch
from torch.utils.data import DataLoader


GR00T_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=GR00T_ROOT / "examples/LIBERO/libero_plus_lerobot",
    )
    parser.add_argument("--distillation-dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=GR00T_ROOT / "checkpoints/GR00T-N1.7-3B",
    )
    parser.add_argument(
        "--backbone-path",
        type=Path,
        default=GR00T_ROOT / "checkpoints/Cosmos-Reason2-2B",
    )
    parser.add_argument("--fixed-gt-samples-per-batch", type=int, default=54)
    parser.add_argument("--fixed-teacher-samples-per-batch", type=int, default=10)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--num-shards-per-epoch", type=int, default=1000)
    return parser.parse_args()


def _assert_finite(value, path: str = "batch") -> None:
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite tensor at {path}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")


class AuditCollator:
    def __init__(self, collator, expected_counts: dict[str, int]):
        self.expected_counts = expected_counts
        self.validating_collator = FixedSourceCountCollator(
            collator,
            expected_counts,
        )

    def __call__(self, features):
        sources = [feature[FIXED_BATCH_SOURCE_KEY] for feature in features]
        batch = self.validating_collator(features)
        batch["_preflight_sources"] = sources
        return batch


def main() -> None:
    args = parse_args()
    for field in (
        "dataset_path",
        "distillation_dataset_dir",
        "base_model_path",
        "backbone_path",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())

    expected_counts = {
        "gt": args.fixed_gt_samples_per_batch,
        "teacher": args.fixed_teacher_samples_per_batch,
    }
    if any(count <= 0 for count in expected_counts.values()):
        raise ValueError("Fixed GT and teacher sample counts must both be positive")
    for name in ("num_batches", "shard_size", "num_shards_per_epoch"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < args.episode_sampling_rate <= 1.0:
        raise ValueError("episode_sampling_rate must lie in (0, 1]")

    batch_size = sum(expected_counts.values())
    index_path = args.distillation_dataset_dir / "distillation_index.jsonl"
    records_path = args.distillation_dataset_dir / "teacher_records_enriched.pt"
    required_paths = (
        args.dataset_path / "meta/info.json",
        args.base_model_path / "processor_config.json",
        args.backbone_path / "preprocessor_config.json",
        index_path,
        records_path,
    )
    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(f"Required preflight input is missing: {required_path}")

    processor = build_processor(
        extra_augmentation_config=EXTRA_AUG,
        base_checkpoint=args.base_model_path,
        backbone_path=args.backbone_path,
    )
    processor.train()
    if processor.color_jitter_params != COLOR_JITTER:
        raise RuntimeError(
            f"ColorJitter mismatch: {processor.color_jitter_params} != {COLOR_JITTER}"
        )
    if processor.extra_augmentation_config != EXTRA_AUG:
        raise RuntimeError(
            "Extra augmentation mismatch: "
            f"{processor.extra_augmentation_config} != {EXTRA_AUG}"
        )
    transform_names = [
        type(transform).__name__
        for transform in processor.train_image_transform.transforms
    ]
    for expected_transform in ("ColorJitter", "GaussianBlur", "MotionBlur"):
        if expected_transform not in transform_names:
            raise RuntimeError(
                f"{expected_transform} missing from training transforms: "
                f"{transform_names}"
            )

    modality = get_default_config().data.modality_configs["libero_sim"]
    common = {
        "dataset_path": args.dataset_path,
        "embodiment_tag": EmbodimentTag("libero_sim"),
        "modality_configs": modality,
        "video_backend": "torchcodec",
        "shard_size": args.shard_size,
        "seed": 42,
        "allow_padding": True,
    }
    gt_dataset = ShardedSingleStepDataset(
        **common,
        episode_sampling_rate=args.episode_sampling_rate,
    )
    teacher_dataset = FlareDistillationDataset(
        **common,
        distillation_index_path=index_path,
        teacher_records_path=records_path,
    )
    if teacher_dataset.teacher_horizon != 8:
        raise RuntimeError(
            f"Teacher horizon must remain 8, got {teacher_dataset.teacher_horizon}"
        )
    if teacher_dataset.processor_action_horizon != 16:
        raise RuntimeError(
            "GT processor horizon must remain 16, got "
            f"{teacher_dataset.processor_action_horizon}"
        )

    replay_dataset = FixedRatioShardedMixtureDataset(
        datasets=[gt_dataset, teacher_dataset],
        source_labels=["gt", "teacher"],
        source_counts=[
            args.fixed_gt_samples_per_batch,
            args.fixed_teacher_samples_per_batch,
        ],
        processor=processor,
        seed=42,
        num_shards_per_epoch=args.num_shards_per_epoch,
        override_pretraining_statistics=True,
        prefetch_shards=False,
    )
    loader = DataLoader(
        replay_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=AuditCollator(processor.collator, expected_counts),
    )

    checked_batches = 0
    for batch_index, batch in enumerate(loader, start=1):
        sources = batch.pop("_preflight_sources")
        actual_counts = Counter(sources)
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Batch {batch_index} source mismatch: {actual_counts}"
            )
        _assert_finite(batch)

        model_inputs = batch["inputs"]
        action = model_inputs["action"]
        action_mask = model_inputs["action_mask"]
        expected_shape = (batch_size, 40, 132)
        if action.shape != action_mask.shape or tuple(action.shape) != expected_shape:
            raise RuntimeError(
                f"Batch {batch_index} action shapes mismatch: "
                f"action={tuple(action.shape)}, mask={tuple(action_mask.shape)}"
            )
        teacher_indices = [
            index for index, source in enumerate(sources) if source == "teacher"
        ]
        gt_indices = [
            index for index, source in enumerate(sources) if source == "gt"
        ]
        if action_mask[teacher_indices, 8:].count_nonzero().item() != 0:
            raise RuntimeError(f"Batch {batch_index} teacher mask leaks beyond step 8")
        if action[teacher_indices, 8:].count_nonzero().item() != 0:
            raise RuntimeError(
                f"Batch {batch_index} teacher action leaks beyond step 8"
            )
        if not torch.all(action_mask[teacher_indices, :8].sum(dim=(1, 2)) > 0):
            raise RuntimeError(f"Batch {batch_index} teacher prefix is unsupervised")
        if not torch.all(action_mask[gt_indices, 8:16].sum(dim=(1, 2)) > 0):
            raise RuntimeError(
                f"Batch {batch_index} GT samples lack valid step 9-16 supervision"
            )
        if action_mask[gt_indices, 16:].count_nonzero().item() != 0:
            raise RuntimeError(f"Batch {batch_index} GT mask leaks beyond step 16")

        checked_batches += 1
        print(
            f"[PASS] real batch {batch_index}/{args.num_batches}: "
            f"GT={actual_counts['gt']} teacher={actual_counts['teacher']}, "
            f"action={tuple(action.shape)}, finite tensors, masks 16/8"
        )
        if checked_batches == args.num_batches:
            break

    if checked_batches != args.num_batches:
        raise RuntimeError(
            f"Only checked {checked_batches}/{args.num_batches} requested batches"
        )
    print(
        "[PASS] augmentation: ColorJitter 0.5/0.6/0.7/0.12, "
        "GaussianBlur p=0.5, MotionBlur p=0.3"
    )
    print("========== ALL FIXED GT-REPLAY PREFLIGHT CHECKS PASSED ==========")


if __name__ == "__main__":
    main()
