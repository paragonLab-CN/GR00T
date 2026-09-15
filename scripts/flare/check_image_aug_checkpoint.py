#!/usr/bin/env python3
"""Fail closed unless a completed checkpoint records the planned augmentation."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import yaml


EXPECTED_COLOR = {
    "brightness": 0.5,
    "contrast": 0.6,
    "saturation": 0.7,
    "hue": 0.12,
}
EXPECTED_EXTRA = {
    "gaussian_blur_params": {"blur_limit": [3, 7], "p": 0.5},
    "motion_blur_params": {"blur_limit": [3, 7], "p": 0.3},
}


def _parse_value(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int, default=10000)
    parser.add_argument("--fixed-gt-samples-per-batch", type=int)
    parser.add_argument("--fixed-teacher-samples-per-batch", type=int)
    parser.add_argument("--require-full-training-state", action="store_true")
    args = parser.parse_args()

    processor_path = args.checkpoint / "processor_config.json"
    final_path = args.checkpoint / "experiment_cfg" / "final_processor_config.json"
    trainer_state_path = args.checkpoint / "trainer_state.json"
    for path in (processor_path, final_path, trainer_state_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint file is missing: {path}")

    processor = json.loads(processor_path.read_text(encoding="utf-8"))["processor_kwargs"]
    final = json.loads(final_path.read_text(encoding="utf-8"))
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))

    for label, config in (
        ("processor_config.json", processor),
        ("final_processor_config.json", final),
    ):
        color = _parse_value(config.get("color_jitter_params"))
        extra = _parse_value(config.get("extra_augmentation_config"))
        if color != EXPECTED_COLOR:
            raise RuntimeError(f"{label} has invalid ColorJitter: {color!r}")
        if extra != EXPECTED_EXTRA:
            raise RuntimeError(f"{label} has invalid blur config: {extra!r}")
        print(f"[PASS] {label} records the complete ColorJitter + blur config")

    actual_step = int(trainer_state["global_step"])
    if actual_step != args.expected_step:
        raise RuntimeError(
            f"checkpoint is not step {args.expected_step}: "
            f"global_step={actual_step}"
        )
    print(f"[PASS] trainer_state global_step = {args.expected_step}")
    logged_metrics = [
        entry
        for entry in trainer_state.get("log_history", [])
        if "loss" in entry and "grad_norm" in entry
    ]
    if not logged_metrics:
        raise RuntimeError("trainer_state contains no loss/grad_norm metrics")
    latest_metrics = logged_metrics[-1]
    invalid_metrics = {
        key: latest_metrics.get(key)
        for key in ("loss", "grad_norm")
        if not isinstance(latest_metrics.get(key), (int, float))
        or not math.isfinite(float(latest_metrics[key]))
    }
    if invalid_metrics:
        raise RuntimeError(f"Non-finite checkpoint metrics: {invalid_metrics}")
    print(
        "[PASS] latest loss/grad_norm are finite: "
        f"loss={latest_metrics['loss']}, grad_norm={latest_metrics['grad_norm']}"
    )
    if args.require_full_training_state:
        required_state_files = ("optimizer.pt", "scheduler.pt", "rng_state.pth")
        missing = [
            name
            for name in required_state_files
            if not (args.checkpoint / name).is_file()
        ]
        has_model = (args.checkpoint / "model.safetensors").is_file() or (
            args.checkpoint / "model.safetensors.index.json"
        ).is_file()
        if missing or not has_model:
            details = missing + ([] if has_model else ["model weights"])
            raise RuntimeError(
                "Checkpoint is not resumable; missing full training state: "
                + ", ".join(details)
            )
        print("[PASS] model, optimizer, scheduler, and RNG state are present")
    if (args.fixed_gt_samples_per_batch is None) != (
        args.fixed_teacher_samples_per_batch is None
    ):
        raise ValueError("Both fixed replay counts must be supplied together")
    if args.fixed_gt_samples_per_batch is not None:
        resolved_path = args.checkpoint / "experiment_cfg" / "conf.yaml"
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Resolved checkpoint config is missing: {resolved_path}"
            )
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        data_config = resolved["data"]
        training_config = resolved["training"]
        expected_counts = {
            "fixed_gt_samples_per_batch": args.fixed_gt_samples_per_batch,
            "fixed_teacher_samples_per_batch": args.fixed_teacher_samples_per_batch,
        }
        actual_counts = {
            key: data_config.get(key) for key in expected_counts
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Embedded fixed replay counts mismatch: {actual_counts} != {expected_counts}"
            )
        if training_config["global_batch_size"] != sum(expected_counts.values()):
            raise RuntimeError("Embedded global batch size does not match fixed counts")
        if training_config["gradient_accumulation_steps"] != 10:
            raise RuntimeError("Expected gradient accumulation 10")
        dataset_types = [
            dataset["dataset_type"] for dataset in data_config["datasets"]
        ]
        if sorted(dataset_types) != ["flare_distillation", "physical_embodiment"]:
            raise RuntimeError(
                f"Embedded replay data sources are invalid: {dataset_types}"
            )
        gt_horizon = len(
            data_config["modality_configs"]["libero_sim"]["action"]["delta_indices"]
        )
        teacher_dataset = next(
            dataset
            for dataset in data_config["datasets"]
            if dataset["dataset_type"] == "flare_distillation"
        )
        summary_path = (
            Path(teacher_dataset["distillation_index_path"]).parent
            / "distillation_summary.json"
        )
        sidecar_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        teacher_horizon = int(
            sidecar_summary["gr00t_interface"]["teacher_horizon"]
        )
        if (gt_horizon, teacher_horizon) != (16, 8):
            raise RuntimeError(
                f"Embedded GT/teacher horizons are {gt_horizon}/{teacher_horizon}, expected 16/8"
            )
        print(
            "[PASS] embedded fixed replay config: "
            f"GT={args.fixed_gt_samples_per_batch}, "
            f"teacher={args.fixed_teacher_samples_per_batch}, "
            "batch=64, accumulation=10 (540 GT + 100 teacher/update)"
        )
    print(f"Valid image-augmentation checkpoint: {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
