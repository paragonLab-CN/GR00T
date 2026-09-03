#!/usr/bin/env python3
"""Pixel-level smoke test: prove that extra_augmentation_config (GaussianBlur +
MotionBlur) really reaches the training image pipeline of Gr00tN1d7Processor.

Background: Gr00tN1d7Processor.from_pretrained() used to silently drop the
``extra_augmentation_config`` kwarg (it was missing from the override allowlist),
so experiments that configured blur only got ColorJitter. This script fails fast
if that regression comes back.

Checks (any failure -> non-zero exit):
  1. The processor built exactly like setup.py keeps extra_augmentation_config.
  2. The training transform (ReplayCompose) contains ColorJitter, GaussianBlur
     and MotionBlur.
  3. Deterministic blur (p=1.0, jitter off): on a REAL training video frame,
     Laplacian-variance sharpness drops sharply vs a matched-geometry control.
  4. Experiment config (jitter on, blur p=0.5/0.3): over 100 paired samples with
     identical random seeds (=> same crop + same jitter draws), the blur variant
     is measurably softer than the no-blur control.
  5. The eval transform stays deterministic and free of jitter/blur.
  6. save_pretrained -> from_pretrained round-trip preserves the config.

Usage: .venv/bin/python scripts/flare/check_augmentation_pipeline.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import albumentations as A
import av
import cv2
import numpy as np

GR00T_ROOT = Path(__file__).resolve().parents[2]
BASE_CHECKPOINT = GR00T_ROOT / "checkpoints" / "GR00T-N1.7-3B"
BACKBONE_PATH = GR00T_ROOT / "checkpoints" / "Cosmos-Reason2-2B"
VIDEO_PATH = (
    GR00T_ROOT
    / "examples/LIBERO/libero_plus_lerobot/videos/chunk-000/"
    "observation.images.front/episode_000000.mp4"
)

COLOR_JITTER = {"brightness": 0.5, "contrast": 0.6, "saturation": 0.7, "hue": 0.12}
EXTRA_AUG = {
    "gaussian_blur_params": {"blur_limit": [3, 7], "p": 0.5},
    "motion_blur_params": {"blur_limit": [3, 7], "p": 0.3},
}

IMAGE_CROP_SIZE = [230, 230]
IMAGE_TARGET_SIZE = [256, 256]


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def load_real_frame() -> "PIL.Image.Image":
    if not VIDEO_PATH.exists():
        fail(f"training video not found: {VIDEO_PATH}")
    with av.open(str(VIDEO_PATH)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            return frame.to_image()
    fail("no frame decoded")


def build_processor(extra_augmentation_config):
    """Mirror setup.py's start_from_checkpoint path (the training code path)."""
    from transformers import AutoProcessor

    from gr00t.configs.base_config import get_default_config
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor  # noqa: F401

    modality_configs = get_default_config().data.modality_configs
    if "libero_sim" not in modality_configs:
        fail("libero_sim modality config not registered")

    return AutoProcessor.from_pretrained(
        str(BASE_CHECKPOINT),
        # Overrides (same as gr00t/model/gr00t_n1d7/setup.py::_create_dataset)
        modality_configs=modality_configs,
        use_percentiles=True,
        image_crop_size=IMAGE_CROP_SIZE,
        image_target_size=IMAGE_TARGET_SIZE,
        random_rotation_angle=None,
        color_jitter_params=COLOR_JITTER,
        model_name=str(BACKBONE_PATH),
        model_type="qwen",
        formalize_language=True,
        apply_sincos_state_encoding=False,
        max_action_horizon=40,
        use_albumentations=True,
        extra_augmentation_config=extra_augmentation_config,
        shortest_image_edge=None,
        crop_fraction=None,
        use_alternate_vl_dit=True,
        use_relative_action=True,
        exclude_state=False,
        state_dropout_prob=0.2,
        use_mean_std=False,
    )


def sharpness(img_tensor: np.ndarray) -> float:
    """Laplacian variance of a (C,H,W) uint8 image tensor."""
    arr = img_tensor.transpose(1, 2, 0)  # HWC
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_raw_transform(extra_augmentation_config, color_jitter_params):
    """Build the same albumentations train transform the processor uses."""
    from gr00t.model.gr00t_n1d7.image_augmentations import (
        build_image_transformations_albumentations,
    )

    train_transform, _ = build_image_transformations_albumentations(
        image_target_size=IMAGE_TARGET_SIZE,
        image_crop_size=IMAGE_CROP_SIZE,
        random_rotation_angle=None,
        color_jitter_params=color_jitter_params,
        shortest_image_edge=None,
        crop_fraction=None,
        extra_augmentation_config=extra_augmentation_config,
    )
    return train_transform


def apply_once(transform, pil_img, seed: int):
    """Apply the transform to one PIL image with a fixed random seed so two
    different transforms see identical crop/jitter draws (blur draws happen
    after jitter draws, so pairs stay aligned)."""
    from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay

    random.seed(seed)
    np.random.seed(seed)
    tensors, _ = apply_with_replay(transform, [pil_img])
    return tensors[0].numpy()


def check_transforms_of(transform, label):
    if isinstance(transform, A.ReplayCompose):
        transforms = transform.transforms
    elif isinstance(transform, A.Compose):
        transforms = transform.transforms
    else:
        fail(f"{label}: unexpected transform type {type(transform)}")
    names = [type(t).__name__ for t in transforms]
    print(f"[INFO] {label} pipeline: {names}")
    return names


def main() -> None:
    print("========== GR00T augmentation pipeline smoke test ==========")
    frame = load_real_frame()
    print(f"[INFO] real training frame: {VIDEO_PATH} ({frame.size[0]}x{frame.size[1]})")

    # ---- Check 1: config survives from_pretrained (the fixed path) ----
    processor = build_processor(extra_augmentation_config=EXTRA_AUG)
    if processor.extra_augmentation_config != EXTRA_AUG:
        fail(
            "extra_augmentation_config was dropped by from_pretrained: "
            f"{processor.extra_augmentation_config!r}"
        )
    print("[PASS] 1. extra_augmentation_config survives from_pretrained "
          f"(= {EXTRA_AUG})")
    print(f"[INFO] effective processor image pipeline: "
          f"crop={processor.image_crop_size} target={processor.image_target_size} "
          f"albumentations={processor.use_albumentations}")

    # ---- Check 2: transform structure ----
    train_names = check_transforms_of(processor.train_image_transform, "train")
    for cls in ("ColorJitter", "GaussianBlur", "MotionBlur"):
        if cls not in train_names:
            fail(f"2. {cls} missing from train transform pipeline")
    print("[PASS] 2. train transform contains ColorJitter + GaussianBlur + MotionBlur")

    # ---- Check 3: deterministic blur on real pixels (jitter off, blur p=1.0) ----
    control_tf = build_raw_transform(extra_augmentation_config={}, color_jitter_params=None)
    for blur_kind, cfg in [
        (
            "gaussian p=1.0",
            {"gaussian_blur_params": {"blur_limit": [3, 7], "p": 1.0}},
        ),
        (
            "motion p=1.0",
            {"motion_blur_params": {"blur_limit": [3, 7], "p": 1.0}},
        ),
    ]:
        blur_tf = build_raw_transform(extra_augmentation_config=cfg, color_jitter_params=None)
        sharper_pairs = 0
        ratios = []
        for i in range(10):
            ctrl_sharp = sharpness(apply_once(control_tf, frame, seed=i))
            blur_sharp = sharpness(apply_once(blur_tf, frame, seed=i))
            ratios.append(blur_sharp / max(ctrl_sharp, 1e-9))
            if blur_sharp < ctrl_sharp:
                sharper_pairs += 1
        mean_ratio = float(np.mean(ratios))
        print(f"[INFO] 3. {blur_kind}: mean sharpness ratio blur/control = {mean_ratio:.4f} "
              f"({sharper_pairs}/10 pairs softer)")
        if mean_ratio > 0.8 or sharper_pairs < 8:
            fail(f"3. {blur_kind} did not measurably soften real pixels")
    print("[PASS] 3. deterministic blur softens real training pixels (p=1.0)")

    # ---- Check 4: experiment config on real pixels (jitter on, blur p=0.5/0.3) ----
    ctrl_proc_tf = build_raw_transform(extra_augmentation_config={}, color_jitter_params=COLOR_JITTER)
    exp_tf = build_raw_transform(extra_augmentation_config=EXTRA_AUG, color_jitter_params=COLOR_JITTER)
    softer_pairs = 0
    ratios = []
    for i in range(100):
        ctrl_sharp = sharpness(apply_once(ctrl_proc_tf, frame, seed=i))
        exp_sharp = sharpness(apply_once(exp_tf, frame, seed=i))
        ratios.append(exp_sharp / max(ctrl_sharp, 1e-9))
        if exp_sharp < ctrl_sharp:
            softer_pairs += 1
    mean_ratio = float(np.mean(ratios))
    print(f"[INFO] 4. experiment config: mean sharpness ratio blur/control = {mean_ratio:.4f} "
          f"({softer_pairs}/100 pairs softer)")
    if mean_ratio > 0.9 or softer_pairs < 45:
        fail("4. experiment blur config does not measurably soften real pixels")
    print("[PASS] 4. experiment config (jitter + gaussian p=0.5 + motion p=0.3) "
          "softens real pixels vs matched no-blur control")

    # ---- Check 5: eval transform stays deterministic ----
    eval_names = check_transforms_of(processor.eval_image_transform, "eval")
    if any(c in eval_names for c in ("ColorJitter", "GaussianBlur", "MotionBlur", "Blur")):
        fail("5. eval transform contains augmentation transforms")
    out1 = apply_once(processor.eval_image_transform, frame, seed=0)
    out2 = apply_once(processor.eval_image_transform, frame, seed=123)
    if not np.array_equal(out1, out2):
        fail("5. eval transform is not deterministic")
    print("[PASS] 5. eval transform deterministic, no jitter/blur")

    # ---- Check 6: save_pretrained -> from_pretrained round-trip ----
    with tempfile.TemporaryDirectory() as tmp:
        saved = processor.save_pretrained(tmp)
        print(f"[INFO] 6. saved processor to {saved}")
        saved_cfg = json.loads(Path(tmp, "processor_config.json").read_text())
        if saved_cfg["processor_kwargs"].get("extra_augmentation_config") != EXTRA_AUG:
            fail("6. save_pretrained did not serialize extra_augmentation_config")
        from transformers import AutoProcessor

        reloaded = AutoProcessor.from_pretrained(tmp)
        if reloaded.extra_augmentation_config != EXTRA_AUG:
            fail("6. round-trip lost extra_augmentation_config: "
                 f"{reloaded.extra_augmentation_config!r}")
    print("[PASS] 6. save_pretrained -> from_pretrained round-trip preserves config")

    print("========== ALL CHECKS PASSED ==========")


if __name__ == "__main__":
    main()
