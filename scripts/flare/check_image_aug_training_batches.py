#!/usr/bin/env python3
"""Load and collate one real GT batch and one real FLARE-teacher batch."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from check_augmentation_pipeline import build_processor
from gr00t.configs.base_config import get_default_config
from gr00t.data.dataset.flare_distillation_dataset import FlareDistillationDataset
from gr00t.data.types import EmbodimentTag, MessageType


GR00T_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = GR00T_ROOT / "examples/LIBERO/libero_plus_lerobot"
SIDECAR_DIR = Path(
    "/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/distillation_dataset"
)


def _tensor_summary(value, prefix=""):
    summaries = []
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite tensor in collated batch at {prefix}")
        summaries.append(f"{prefix}:{tuple(value.shape)}/{value.dtype}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            summaries.extend(_tensor_summary(item, child))
    return summaries


def main() -> None:
    print("========== Real GT + teacher training-batch smoke test ==========")
    processor = build_processor(
        extra_augmentation_config={
            "gaussian_blur_params": {"blur_limit": [3, 7], "p": 0.5},
            "motion_blur_params": {"blur_limit": [3, 7], "p": 0.3},
        }
    )
    processor.train()

    modality = get_default_config().data.modality_configs["libero_sim"]
    dataset = FlareDistillationDataset(
        dataset_path=DATASET_PATH,
        embodiment_tag=EmbodimentTag("libero_sim"),
        modality_configs=modality,
        distillation_index_path=SIDECAR_DIR / "distillation_index.jsonl",
        teacher_records_path=SIDECAR_DIR / "teacher_records_enriched.pt",
        video_backend="torchcodec",
        shard_size=1024,
        seed=42,
        allow_padding=True,
    )
    processor.set_statistics(
        {"libero_sim": dataset.get_dataset_statistics()}, override=True
    )
    dataset.processor = processor

    teacher_index = next(
        index
        for index, record in enumerate(dataset.index_rows)
        if record["source"] == "teacher"
    )
    raw_gt = dataset.get_raw_record(teacher_index, apply_teacher=False)
    gt_sample = processor(
        [{"type": MessageType.EPISODE_STEP.value, "content": raw_gt}]
    )
    teacher_sample = dataset.get_processed_record(teacher_index)

    for name, sample in (("GT", gt_sample), ("teacher", teacher_sample)):
        action = sample["action"]
        action_mask = sample["action_mask"]
        if action.shape != action_mask.shape or action.shape != (40, 132):
            raise RuntimeError(
                f"{name} action encoding has unexpected shapes: "
                f"action={tuple(action.shape)}, mask={tuple(action_mask.shape)}"
            )
        batch = processor.collator([sample])
        summaries = _tensor_summary(batch, name)
        if not summaries:
            raise RuntimeError(f"{name} collator produced no tensor fields")
        print(f"[PASS] {name} processor + collator: " + ", ".join(summaries))

    teacher_horizon = dataset.teacher_horizon
    processor_horizon = dataset.processor_action_horizon
    if teacher_sample["action_mask"][teacher_horizon:].count_nonzero().item() != 0:
        raise RuntimeError("teacher action mask has supervision beyond teacher horizon")
    if teacher_sample["action"][teacher_horizon:].count_nonzero().item() != 0:
        raise RuntimeError("teacher action values leak into the masked tail")
    if gt_sample["action_mask"][teacher_horizon:processor_horizon].count_nonzero().item() == 0:
        raise RuntimeError("GT sample unexpectedly lacks horizon-16 supervision")
    print(
        "[PASS] teacher supervision is limited to 8 steps while paired GT retains "
        "the 16-step processor horizon"
    )
    print("========== ALL REAL-BATCH CHECKS PASSED ==========")


if __name__ == "__main__":
    main()
