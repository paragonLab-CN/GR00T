# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content identities for local, frozen policy checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


_INFERENCE_CONFIG_FILES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "tokenizer.model",
    "spiece.model",
    "merges.txt",
    "vocab.txt",
    "chat_template.jinja",
}
_TRAINING_METADATA_DIRS = {"experiment_cfg", "wandb", "logs", "tensorboard"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_snapshot_sha256(path: str | Path) -> str:
    """Hash weights and inference configuration, not training metadata.

    Directory identities use a sorted relative-path/content-digest manifest;
    moving an identical checkpoint does not change its identity. This requires
    a local checkpoint, not a mutable Hugging Face repository name.
    """
    return _checkpoint_sha256(path, weights_only=False)


def checkpoint_weights_sha256(path: str | Path) -> str:
    """Hash only model weight artifacts so configuration changes are not updates."""
    return _checkpoint_sha256(path, weights_only=True)


def _checkpoint_sha256(path: str | Path, *, weights_only: bool) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {path}")

    files: list[Path] = []
    weights: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path)
        if any(part in _TRAINING_METADATA_DIRS for part in relative.parts[:-1]):
            continue
        name = candidate.name
        is_weight = (
            candidate.suffix == ".safetensors"
            or (name.startswith("pytorch_model") and candidate.suffix == ".bin")
            or name in {"model.pt", "weights.pt"}
        )
        is_config = name in _INFERENCE_CONFIG_FILES or (
            relative.parts[0] == "processor" and candidate.suffix == ".json"
        )
        if is_weight or (is_config and not weights_only):
            files.append(candidate)
        if is_weight:
            weights.append(candidate)
    if not weights:
        raise ValueError(f"Policy checkpoint contains no supported model weights: {path}")
    manifest = [
        {"path": str(candidate.relative_to(path)), "sha256": file_sha256(candidate)}
        for candidate in files
    ]
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
