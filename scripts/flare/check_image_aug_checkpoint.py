#!/usr/bin/env python3
"""Fail closed unless a completed checkpoint records the planned augmentation."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


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

    if int(trainer_state["global_step"]) != 10000:
        raise RuntimeError(
            f"checkpoint is not step 10000: global_step={trainer_state['global_step']}"
        )
    print("[PASS] trainer_state global_step = 10000")
    print(f"Valid image-augmentation checkpoint: {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
