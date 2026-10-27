#!/usr/bin/env python3
"""Score the full sparse LIBERO-Plus GR00T candidate set with LeWM.

Run this script in the ``LeWorldModel`` environment. It loads LeWM once, reads
manifest-selected GR00T candidate files, predicts K candidate future latent
trajectories plus one demonstrated-action oracle trajectory, writes resumable
per-sample score parts, and aggregates full-dataset reports.

The LIBERO-Plus source dataset was converted before LeWM training, so both
candidate files and the GT latent frame bank use the LeRobot gripper convention:
1=open and 0=close.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable


# Avoid invalid inherited OpenMP settings before importing torch/OpenMP users.
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp.isdigit() or int(_omp) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"

import h5py
import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import open_dict
import pandas as pd
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
FLARE_ROOT = SCRIPT_PATH.parents[2]
ISAAC_ROOT = FLARE_ROOT.parent
WORKSPACE_ROOT = ISAAC_ROOT.parent
PROJECT_ROOT = Path(os.environ.get("LEWM_ROOT", WORKSPACE_ROOT / "le-wm")).expanduser().resolve()
sys.path.insert(0, str(PROJECT_ROOT))
if str(ISAAC_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAAC_ROOT))

from gr00t.utils.checkpoint_snapshot import checkpoint_snapshot_sha256  # noqa: E402


DEFAULT_EXPECTED_CKPT_SHA256 = "ceb52c6b525ca34b5bbdc395e20f6e2b0663bb9d86995e9c00a7ae443ba1d3da"
FULL_MANIFEST_FORMAT = "libero_plus_sparse_manifest_v1"
CANDIDATE_FORMAT = "flare_groot_manifest_candidates_v1"
CANDIDATE_SUMMARY_FORMAT = "flare_candidate_generation_summary_v1"
SCORE_PART_FORMAT = "flare_lewm_libero_plus_score_part_v2"
EXPECTED_DATASET_NAME = "libero_plus_lerobot__observation_images_front"
EXPECTED_H5_FORMATS = {
    "lewm_gt_latent_framebank_spatial_v1",
    "lewm_gt_latent_framebank_libero_plus_v2",
}
SUITES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
}

DEFAULT_MANIFEST = FLARE_ROOT / "outputs/libero_plus/manifests/libero_plus_sparse_manifest.jsonl"
DEFAULT_CANDIDATE_ROOT = ISAAC_ROOT / "outputs/libero_plus/sparse_candidates"
DEFAULT_GT_H5 = (
    FLARE_ROOT / "outputs/libero_plus/gt_latents/libero_plus_epoch9_h8_framebank_trainpipe.h5"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "stablewm_data/checkpoints/libero_plus_lewm_full/weights_epoch_9.pt"
)
DEFAULT_OUTPUT_DIR = FLARE_ROOT / "outputs/libero_plus/teacher_scoring"

# stable_pretraining registers the OmegaConf ``eval`` resolver at import time.
# Do not register a second resolver in this script.
import stable_pretraining  # noqa: E402,F401


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), ensure_ascii=False) + "\n")


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def ensure_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        bad = int((~torch.isfinite(tensor)).sum())
        raise RuntimeError(f"{name} contains {bad} NaN/Inf values")


def is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _validate_single_policy_options(
    single_policy_checkpoint: Path | None, policy_checkpoint_sha256: str | None
) -> None:
    if (single_policy_checkpoint is None) != (policy_checkpoint_sha256 is None):
        raise ValueError("Use --single-policy-checkpoint and --policy-checkpoint-sha256 together")
    if policy_checkpoint_sha256 is not None and not is_sha256(policy_checkpoint_sha256):
        raise ValueError("--policy-checkpoint-sha256 must be 64 lowercase hex characters")


def verify_single_policy_checkpoint(
    single_policy_checkpoint: Path | None, policy_checkpoint_sha256: str | None
) -> tuple[Path | None, str | None]:
    """Verify a frozen policy once per run, never once per scored sample."""

    _validate_single_policy_options(single_policy_checkpoint, policy_checkpoint_sha256)
    if single_policy_checkpoint is None:
        return None, None
    checkpoint = single_policy_checkpoint.expanduser().resolve()
    actual = checkpoint_snapshot_sha256(checkpoint)
    if actual != policy_checkpoint_sha256:
        raise RuntimeError(
            f"Policy checkpoint SHA-256 mismatch: {actual} != {policy_checkpoint_sha256}"
        )
    return checkpoint, actual


def validate_manifest(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Manifest is empty")

    seen_sample_ids: set[str] = set()
    for expected_index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError(f"Manifest row {expected_index} has invalid sample_id")
        if sample_id in seen_sample_ids:
            raise RuntimeError(f"Manifest contains duplicate sample_id {sample_id}")
        seen_sample_ids.add(sample_id)

        if row.get("sample_index") != expected_index:
            raise RuntimeError(
                f"Manifest row {expected_index} has sample_index={row.get('sample_index')!r}"
            )
        suite = row.get("suite")
        if suite not in SUITES:
            raise RuntimeError(f"Manifest sample {sample_id} has invalid suite={suite!r}")
        if row.get("gt_latent_key") != suite:
            raise RuntimeError(
                f"Manifest sample {sample_id} has gt_latent_key="
                f"{row.get('gt_latent_key')!r}, expected {suite!r}"
            )
        if int(row.get("lewm_episode_index", -1)) != int(row.get("episode_index", -2)):
            raise RuntimeError(f"Manifest sample {sample_id} is not in LeRobot episode order")

        global_index = int(row["global_index"])
        history_rows = [int(value) for value in row.get("history_rows", [])]
        future_rows = [int(value) for value in row.get("future_rows", [])]
        if history_rows != [global_index - 2, global_index - 1, global_index]:
            raise RuntimeError(f"Manifest sample {sample_id} has invalid history_rows")
        if future_rows != list(range(global_index + 1, global_index + 9)):
            raise RuntimeError(f"Manifest sample {sample_id} has invalid future_rows")
        if (
            int(row.get("history_size", -1)) != 3
            or int(row.get("window_horizon", row.get("horizon", -1))) != 8
            or int(row.get("candidate_k", -1)) != 4
        ):
            raise RuntimeError(f"Manifest sample {sample_id} does not satisfy history=3,H=8,K=4")


def validate_h5_contract(
    h5: h5py.File,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    checkpoint_sha256: str,
    horizon: int,
) -> np.ndarray:
    attrs = h5.attrs
    if not bool(attrs.get("complete", False)):
        raise RuntimeError("GT latent H5 is not marked complete")
    if not bool(attrs.get("frame_action_ready", False)):
        raise RuntimeError("GT latent H5 frame_action is not marked ready")
    if not bool(attrs.get("window_metadata_ready", False)):
        raise RuntimeError("GT latent H5 window metadata is not marked ready")
    if attrs.get("format") not in EXPECTED_H5_FORMATS:
        raise RuntimeError(f"Unexpected GT latent H5 format: {attrs.get('format')!r}")
    if attrs.get("dataset_name") != EXPECTED_DATASET_NAME:
        raise RuntimeError(
            f"GT latent H5 dataset_name={attrs.get('dataset_name')!r}, "
            f"expected {EXPECTED_DATASET_NAME!r}"
        )

    expected_scalars = {
        "frameskip": 1,
        "future_horizon": horizon,
        "action_dim": 7,
        "latent_dim": 192,
    }
    for key, expected in expected_scalars.items():
        if int(attrs.get(key, -1)) != expected:
            raise RuntimeError(f"GT latent H5 {key}={attrs.get(key)!r}, expected {expected}")

    required_datasets = {
        "frame_latent",
        "frame_action",
        "episode_id",
        "ep_offset",
        "ep_len",
        "window_current_row",
        "window_episode_id",
        "window_step_idx",
        "window_future_row",
    }
    missing = required_datasets - set(h5.keys())
    if missing:
        raise RuntimeError(f"GT latent H5 is missing datasets: {sorted(missing)}")

    total_frames = int(attrs.get("total_frames", -1))
    total_windows = int(attrs.get("total_windows", -1))
    if int(attrs.get("frames_written", -1)) != total_frames:
        raise RuntimeError("GT latent H5 frames_written/total_frames mismatch")
    if h5["frame_latent"].shape != (total_frames, 192):
        raise RuntimeError(f"Unexpected frame_latent shape: {h5['frame_latent'].shape}")
    if h5["frame_action"].shape != (total_frames, 7):
        raise RuntimeError(f"Unexpected frame_action shape: {h5['frame_action'].shape}")
    if h5["window_future_row"].shape != (total_windows, horizon):
        raise RuntimeError(f"Unexpected window_future_row shape: {h5['window_future_row'].shape}")

    episode_ids = np.asarray(h5["episode_id"][:], dtype=np.int64)
    ep_offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
    ep_lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
    if not (
        len(episode_ids) == len(ep_offsets) == len(ep_lengths)
        and np.array_equal(episode_ids, np.arange(len(episode_ids), dtype=np.int64))
    ):
        raise RuntimeError("GT latent H5 is not in contiguous LeRobot episode order")
    expected_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(ep_lengths[:-1], dtype=np.int64)]
    )
    if not np.array_equal(ep_offsets, expected_offsets):
        raise RuntimeError("GT latent H5 episode offsets are inconsistent")
    if int(ep_lengths.sum()) != total_frames:
        raise RuntimeError("GT latent H5 episode lengths do not sum to total_frames")

    h5_checkpoint_path = Path(str(attrs.get("checkpoint_path", ""))).expanduser()
    if h5_checkpoint_path.resolve() != checkpoint_path.resolve():
        raise RuntimeError(
            f"GT latent H5 was generated with a different checkpoint path: {h5_checkpoint_path}"
        )
    if int(attrs.get("checkpoint_epoch", -1)) != int(checkpoint.get("epoch", -2)):
        raise RuntimeError("GT latent H5/checkpoint epoch mismatch")
    recorded_sha = attrs.get("checkpoint_sha256")
    if recorded_sha is not None and str(recorded_sha) != checkpoint_sha256:
        raise RuntimeError("GT latent H5/checkpoint SHA-256 mismatch")
    return ep_offsets


# -----------------------------------------------------------------------------
# LeWM model/action helpers
# -----------------------------------------------------------------------------


def load_cfg():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "config/train"),
    ):
        return compose(config_name="lewm", overrides=["data=libero_plus"])


def build_model(cfg, checkpoint_path: Path, device: torch.device):
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = 7
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = load_torch(checkpoint_path)
    if checkpoint.get("format") != "lewm_libero_state_dict_v1":
        raise RuntimeError(f"Unexpected checkpoint format: {checkpoint.get('format')}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint


def action_stats(frame_action_ds: h5py.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce utils.get_column_normalizer (PyTorch sample standard deviation)."""
    values = torch.from_numpy(np.asarray(frame_action_ds[:], dtype=np.float32))
    values = values[torch.isfinite(values).all(dim=1)]
    if values.numel() == 0:
        raise RuntimeError("No finite actions in GT latent H5")
    gripper = values[:, 6]
    if torch.any(gripper < -1e-5) or torch.any(gripper > 1.0 + 1e-5):
        raise RuntimeError("GT latent H5 does not use the expected LeRobot gripper convention")
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, keepdim=True)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError("Non-finite action normalization statistics")
    if torch.any(std <= 0):
        raise RuntimeError(f"Non-positive action std: {std}")
    return mean, std


def normalize_actions(
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return torch.nan_to_num((actions.float() - mean) / std, nan=0.0)


def lewm_action_to_lerobot(actions: torch.Tensor) -> torch.Tensor:
    """Map raw LIBERO/LeWM gripper -1=open,+1=close to LeRobot 1=open,0=close."""
    converted = actions.clone().float()
    converted[..., 6] = 1.0 - torch.clamp(converted[..., 6], 0.0, 1.0)
    return converted


def lerobot_action_to_lewm(actions: torch.Tensor) -> torch.Tensor:
    """Map LeRobot gripper values in [0,1] to LeWM/raw LIBERO [-1,+1]."""
    converted = actions.clone().float()
    gripper = converted[..., 6]
    if torch.any(gripper < -1e-5) or torch.any(gripper > 1.0 + 1e-5):
        raise ValueError(
            "LeRobot gripper values must lie in [0,1]; "
            f"min={float(gripper.min())}, max={float(gripper.max())}"
        )
    converted[..., 6] = 1.0 - 2.0 * torch.clamp(gripper, 0.0, 1.0)
    return converted


@torch.inference_mode()
def rollout_batch_from_latents(
    model,
    history_latents: torch.Tensor,  # [B,3,192]
    action_sequences: torch.Tensor,  # [B,S,10,7]
    history_size: int,
    future_horizon: int,
) -> torch.Tensor:
    """Vectorized latent rollout for B samples and S candidate/oracle plans."""
    if history_latents.ndim != 3:
        raise ValueError(f"history_latents must be [B,H,D], got {history_latents.shape}")
    if action_sequences.ndim != 4:
        raise ValueError(f"action_sequences must be [B,S,T,7], got {action_sequences.shape}")

    batch_size, sample_count, total_actions, action_dim = action_sequences.shape
    if history_latents.shape != (batch_size, history_size, 192):
        raise ValueError(
            f"history_latents expected {(batch_size, history_size, 192)}, "
            f"got {tuple(history_latents.shape)}"
        )
    expected_actions = history_size - 1 + future_horizon
    if total_actions != expected_actions or action_dim != 7:
        raise ValueError(
            f"action_sequences expected [B,S,{expected_actions},7], "
            f"got {tuple(action_sequences.shape)}"
        )

    # Flatten B and S so all K candidates and the oracle are evaluated together.
    emb = (
        history_latents[:, None]
        .expand(batch_size, sample_count, history_size, 192)
        .reshape(batch_size * sample_count, history_size, 192)
        .clone()
    )
    actions = action_sequences.reshape(batch_size * sample_count, total_actions, action_dim)
    act = actions[:, :history_size].clone()
    act_future = actions[:, history_size:]

    # Seven loop predictions followed by the final eighth prediction.
    for step in range(future_horizon - 1):
        act_emb = model.action_encoder(act)
        pred = model.predict(
            emb[:, -history_size:],
            act_emb[:, -history_size:],
        )[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        act = torch.cat([act, act_future[:, step : step + 1]], dim=1)

    act_emb = model.action_encoder(act)
    pred = model.predict(
        emb[:, -history_size:],
        act_emb[:, -history_size:],
    )[:, -1:]
    emb = torch.cat([emb, pred], dim=1)

    future = emb[:, history_size:]
    expected_shape = (batch_size * sample_count, future_horizon, 192)
    if tuple(future.shape) != expected_shape:
        raise RuntimeError(f"Unexpected rollout shape {future.shape}; expected {expected_shape}")
    return future.reshape(batch_size, sample_count, future_horizon, 192)


# -----------------------------------------------------------------------------
# Candidate/H5 validation and batching
# -----------------------------------------------------------------------------


def validate_candidate_summary(
    summary_path: Path,
    manifest_path: Path,
    candidate_dir: Path,
    manifest_sha256: str,
    expected_total: int,
    single_policy_checkpoint: Path | None = None,
    policy_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    _validate_single_policy_options(single_policy_checkpoint, policy_checkpoint_sha256)
    summary = load_json(summary_path)
    if summary.get("format") != CANDIDATE_SUMMARY_FORMAT:
        raise RuntimeError(f"Unexpected candidate summary format: {summary.get('format')!r}")
    if summary.get("status") != "PASS":
        raise RuntimeError(f"Candidate generation did not PASS: {summary}")
    if summary.get("manifest_format") != FULL_MANIFEST_FORMAT:
        raise RuntimeError(
            f"Candidate summary manifest_format={summary.get('manifest_format')!r}, "
            f"expected {FULL_MANIFEST_FORMAT!r}"
        )
    if summary.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("Candidate summary manifest SHA-256 mismatch")
    if Path(str(summary.get("manifest", ""))).resolve() != manifest_path.resolve():
        raise RuntimeError("Candidate summary points to a different manifest")
    if Path(str(summary.get("output_dir", ""))).resolve() != candidate_dir.resolve():
        raise RuntimeError("Candidate summary points to a different candidate directory")
    if single_policy_checkpoint is not None:
        if Path(str(summary.get("model_path", ""))).resolve() != single_policy_checkpoint.resolve():
            raise RuntimeError("Single-policy candidate summary model_path mismatch")
        if summary.get("policy_checkpoint_sha256") != policy_checkpoint_sha256:
            raise RuntimeError("Single-policy candidate summary policy_checkpoint_sha256 mismatch")

    counters = summary.get("counters", {})
    selected = int(counters.get("selected", -1))
    completed = int(counters.get("generated", 0)) + int(counters.get("skipped_valid", 0))
    if (
        selected != expected_total
        or int(counters.get("failed", -1)) != 0
        or completed != expected_total
    ):
        raise RuntimeError(
            f"Candidate generation incomplete: selected={selected}, "
            f"completed={completed}, expected={expected_total}, "
            f"failed={counters.get('failed')}"
        )
    return summary


def candidate_path(candidate_dir: Path, row: dict[str, Any]) -> Path:
    return candidate_dir / f"{row['sample_id']}.pt"


def score_part_path(parts_dir: Path, row: dict[str, Any]) -> Path:
    return parts_dir / f"{row['sample_id']}.pt"


def valid_score_part(
    path: Path,
    row: dict[str, Any],
    manifest_sha256: str,
    checkpoint_sha256: str,
    gt_h5_sha256: str,
    candidate_sha256: str,
    candidate_k: int,
    horizon: int,
    policy_checkpoint_sha256: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        part = load_torch(path)
        if part.get("format") != SCORE_PART_FORMAT:
            return False
        if part.get("sample_id") != row["sample_id"]:
            return False
        if int(part.get("sample_index", -1)) != int(row["sample_index"]):
            return False
        if part.get("suite") != row["suite"]:
            return False
        if part.get("gt_latent_key") != row["gt_latent_key"]:
            return False
        if part.get("manifest_sha256") != manifest_sha256:
            return False
        if part.get("checkpoint_sha256") != checkpoint_sha256:
            return False
        if part.get("gt_h5_sha256") != gt_h5_sha256:
            return False
        if part.get("candidate_sha256") != candidate_sha256:
            return False
        if (
            policy_checkpoint_sha256 is not None
            and part.get("policy_checkpoint_sha256") != policy_checkpoint_sha256
        ):
            return False
        checks = {
            "cosine_per_step": (candidate_k, horizon),
            "mse_per_step": (candidate_k, horizon),
            "cosine_score": (candidate_k,),
            "mse_score": (candidate_k,),
            "oracle_cosine_per_step": (horizon,),
            "oracle_mse_per_step": (horizon,),
            "teacher_action_cosine_lerobot": (horizon, 7),
            "teacher_action_mse_lerobot": (horizon, 7),
        }
        for key, shape in checks.items():
            value = part.get(key)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                return False
            if not torch.isfinite(value).all():
                return False
        return True
    except Exception:
        return False


def prepare_sample(
    row: dict[str, Any],
    candidate_file: Path,
    candidate_sha256: str,
    h5: h5py.File,
    ep_offsets: np.ndarray,
    candidate_k: int,
    history_size: int,
    horizon: int,
    manifest_sha256: str,
    single_policy_checkpoint: Path | None = None,
    policy_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    _validate_single_policy_options(single_policy_checkpoint, policy_checkpoint_sha256)
    candidate = load_torch(candidate_file)
    if candidate.get("format") != CANDIDATE_FORMAT:
        raise RuntimeError(f"Unexpected candidate format: {candidate.get('format')}")
    if candidate.get("manifest_format") != FULL_MANIFEST_FORMAT:
        raise RuntimeError(f"Candidate manifest_format={candidate.get('manifest_format')!r}")
    if candidate.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("Candidate manifest SHA-256 mismatch")
    for key in ("suite", "gt_latent_key"):
        if candidate.get(key) != row.get(key):
            raise RuntimeError(
                f"Candidate/manifest mismatch for {key}: "
                f"candidate={candidate.get(key)!r}, manifest={row.get(key)!r}"
            )
    if single_policy_checkpoint is None:
        if Path(str(candidate.get("model_path", ""))).name != row["suite"]:
            raise RuntimeError("Candidate was generated by the wrong suite checkpoint")
        for key in ("per_suite_manifest_sha256", "source_candidate_sha256"):
            if not is_sha256(candidate.get(key)):
                raise RuntimeError(f"Candidate has invalid {key}")
    else:
        if (
            Path(str(candidate.get("model_path", ""))).resolve()
            != single_policy_checkpoint.resolve()
        ):
            raise RuntimeError(
                "Candidate was generated by a different single-policy checkpoint path"
            )
        if candidate.get("policy_checkpoint_sha256") != policy_checkpoint_sha256:
            raise RuntimeError("Candidate single-policy checkpoint SHA-256 mismatch")

    scalar_fields = (
        "sample_id",
        "sample_index",
        "episode_index",
        "local_step",
        "global_index",
        "window_index_h8",
        "task_index",
    )
    for key in scalar_fields:
        if str(candidate.get(key)) != str(row.get(key)):
            raise RuntimeError(
                f"Candidate/manifest mismatch for {key}: "
                f"candidate={candidate.get(key)!r}, manifest={row.get(key)!r}"
            )

    if int(candidate.get("candidate_k", -1)) != candidate_k:
        raise RuntimeError("candidate_k mismatch")
    if int(candidate.get("decoded_action_horizon", -1)) != horizon:
        raise RuntimeError("action horizon mismatch")
    if list(candidate.get("history_rows", [])) != list(row["history_rows"]):
        raise RuntimeError("history_rows mismatch")
    if list(candidate.get("future_rows", [])) != list(row["future_rows"]):
        raise RuntimeError("future_rows mismatch")

    actions_lerobot = torch.as_tensor(candidate["candidate_actions_concat"]).float()
    if tuple(actions_lerobot.shape) != (1, candidate_k, horizon, 7):
        raise RuntimeError(f"Candidate action shape mismatch: {actions_lerobot.shape}")
    actions_lerobot = actions_lerobot[0]
    ensure_finite("candidate_actions_lerobot", actions_lerobot)
    # LIBERO-Plus was converted to the LeRobot 0/1 gripper convention before LeWM training.
    actions_lewm = actions_lerobot

    episode_index = int(row["episode_index"])
    global_row = int(row["global_index"])
    window_index = int(row["window_index_h8"])
    history_rows = np.asarray(row["history_rows"], dtype=np.int64)
    future_rows_expected = np.asarray(row["future_rows"], dtype=np.int64)

    if int(h5["window_current_row"][window_index]) != global_row:
        raise RuntimeError("window_current_row/global_index mismatch")
    if int(h5["window_episode_id"][window_index]) != episode_index:
        raise RuntimeError("window_episode_id mismatch")
    if int(h5["window_step_idx"][window_index]) != int(row["local_step"]):
        raise RuntimeError("window_step_idx mismatch")
    if int(np.searchsorted(ep_offsets, global_row, side="right") - 1) != episode_index:
        raise RuntimeError("episode_id mismatch")
    if not np.all(np.searchsorted(ep_offsets, history_rows, side="right") - 1 == episode_index):
        raise RuntimeError("History rows cross episode boundary")

    future_rows = np.asarray(h5["window_future_row"][window_index], dtype=np.int64)
    if not np.array_equal(future_rows, future_rows_expected):
        raise RuntimeError("future_rows mismatch against H5")

    history_latents = torch.from_numpy(
        np.asarray(h5["frame_latent"][history_rows], dtype=np.float32)
    )
    gt_latents = torch.from_numpy(np.asarray(h5["frame_latent"][future_rows], dtype=np.float32))
    historical_actions = torch.from_numpy(
        np.asarray(
            h5["frame_action"][global_row - (history_size - 1) : global_row],
            dtype=np.float32,
        )
    )
    h5_gt_actions = torch.from_numpy(
        np.asarray(h5["frame_action"][global_row : global_row + horizon], dtype=np.float32)
    )
    saved_gt_lerobot = torch.as_tensor(candidate["gt_actions_concat"]).float()
    converted_gt_lerobot = h5_gt_actions
    if not torch.equal(converted_gt_lerobot, saved_gt_lerobot):
        diff = float((converted_gt_lerobot - saved_gt_lerobot).abs().max())
        raise RuntimeError(f"Candidate/H5 GT action mismatch after conversion: {diff}")

    ensure_finite("history_latents", history_latents)
    ensure_finite("gt_latents", gt_latents)
    ensure_finite("historical_actions", historical_actions)
    ensure_finite("h5_gt_actions", h5_gt_actions)

    history_prefix = historical_actions.unsqueeze(0).expand(candidate_k, -1, -1)
    candidate_sequences = torch.cat([history_prefix, actions_lewm], dim=1)
    oracle_sequence = torch.cat([historical_actions, h5_gt_actions], dim=0).unsqueeze(0)
    all_sequences = torch.cat([candidate_sequences, oracle_sequence], dim=0)

    expected_sequence_shape = (candidate_k + 1, history_size - 1 + horizon, 7)
    if tuple(all_sequences.shape) != expected_sequence_shape:
        raise RuntimeError(
            f"Action sequence shape {all_sequences.shape}, expected {expected_sequence_shape}"
        )

    prepared = {
        "row": row,
        "candidate_path": candidate_file,
        "candidate_sha256": candidate_sha256,
        "candidate_actions_lerobot": actions_lerobot,
        "gt_actions_lerobot": saved_gt_lerobot,
        "history_latents": history_latents,
        "gt_latents": gt_latents,
        "all_action_sequences": all_sequences,
        "candidate_generation_seconds": float(candidate.get("generation_seconds", math.nan)),
        "candidate_0_vs_1_max_abs_diff": float(
            candidate.get("candidate_0_vs_1_max_abs_diff", math.nan)
        ),
    }
    if policy_checkpoint_sha256 is not None:
        prepared["policy_checkpoint_sha256"] = policy_checkpoint_sha256
    return prepared


# -----------------------------------------------------------------------------
# Per-sample score parts and aggregation
# -----------------------------------------------------------------------------


def second_largest(values: torch.Tensor) -> float:
    return float(torch.topk(values, k=2, largest=True).values[1])


def second_smallest(values: torch.Tensor) -> float:
    return float(torch.topk(values, k=2, largest=False).values[1])


def make_score_part(
    prepared: dict[str, Any],
    predicted: torch.Tensor,  # [K,H,192]
    oracle_predicted: torch.Tensor,  # [H,192]
    checkpoint_sha256: str,
    manifest_sha256: str,
    gt_h5_sha256: str,
    scoring_seconds: float,
    save_predicted_latents: bool,
) -> dict[str, Any]:
    row = prepared["row"]
    candidate_actions = prepared["candidate_actions_lerobot"]
    gt_latents = prepared["gt_latents"]
    candidate_k, horizon = predicted.shape[:2]

    target = gt_latents.unsqueeze(0).expand(candidate_k, -1, -1)
    cosine_per_step = F.cosine_similarity(predicted, target, dim=-1, eps=1e-12)
    mse_per_step = (predicted - target).pow(2).mean(dim=-1)
    cosine_score = cosine_per_step.mean(dim=-1)
    mse_score = mse_per_step.mean(dim=-1)

    oracle_cosine_per_step = F.cosine_similarity(oracle_predicted, gt_latents, dim=-1, eps=1e-12)
    oracle_mse_per_step = (oracle_predicted - gt_latents).pow(2).mean(dim=-1)
    oracle_cosine_score = oracle_cosine_per_step.mean()
    oracle_mse_score = oracle_mse_per_step.mean()

    ensure_finite("cosine_per_step", cosine_per_step)
    ensure_finite("mse_per_step", mse_per_step)
    ensure_finite("oracle_cosine_per_step", oracle_cosine_per_step)
    ensure_finite("oracle_mse_per_step", oracle_mse_per_step)

    teacher_cosine = int(torch.argmax(cosine_score))
    teacher_mse = int(torch.argmin(mse_score))
    best_cosine = float(cosine_score[teacher_cosine])
    best_mse = float(mse_score[teacher_mse])
    cosine_margin = best_cosine - second_largest(cosine_score)
    mse_margin = second_smallest(mse_score) - best_mse

    part: dict[str, Any] = {
        "format": SCORE_PART_FORMAT,
        "sample_id": row["sample_id"],
        "sample_index": int(row["sample_index"]),
        "suite": row["suite"],
        "gt_latent_key": row["gt_latent_key"],
        "episode_index": int(row["episode_index"]),
        "local_step": int(row["local_step"]),
        "global_index": int(row["global_index"]),
        "window_index_h8": int(row["window_index_h8"]),
        "task_index": int(row["task_index"]),
        "task_text": row["task_text"],
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "gt_h5_sha256": gt_h5_sha256,
        "candidate_path": str(prepared["candidate_path"]),
        "candidate_sha256": prepared["candidate_sha256"],
        "candidate_k": candidate_k,
        "future_horizon": horizon,
        "cosine_per_step": cosine_per_step.cpu(),
        "mse_per_step": mse_per_step.cpu(),
        "cosine_score": cosine_score.cpu(),
        "mse_score": mse_score.cpu(),
        "teacher_by_cosine": teacher_cosine,
        "teacher_by_mse": teacher_mse,
        "teacher_agreement": teacher_cosine == teacher_mse,
        "best_cosine_score": best_cosine,
        "best_mse_score": best_mse,
        "cosine_margin": cosine_margin,
        "mse_margin": mse_margin,
        "oracle_cosine_per_step": oracle_cosine_per_step.cpu(),
        "oracle_mse_per_step": oracle_mse_per_step.cpu(),
        "oracle_cosine_score": float(oracle_cosine_score),
        "oracle_mse_score": float(oracle_mse_score),
        "oracle_better_than_all_cosine": float(oracle_cosine_score) > best_cosine,
        "oracle_better_than_all_mse": float(oracle_mse_score) < best_mse,
        "oracle_cosine_gap": float(oracle_cosine_score) - best_cosine,
        "oracle_mse_gap": best_mse - float(oracle_mse_score),
        "candidate_actions_lerobot": candidate_actions.cpu(),
        "gt_actions_lerobot": prepared["gt_actions_lerobot"].cpu(),
        "teacher_action_cosine_lerobot": candidate_actions[teacher_cosine].cpu(),
        "teacher_action_mse_lerobot": candidate_actions[teacher_mse].cpu(),
        "candidate_generation_seconds": prepared["candidate_generation_seconds"],
        "candidate_0_vs_1_max_abs_diff": prepared["candidate_0_vs_1_max_abs_diff"],
        "scoring_seconds": scoring_seconds,
    }
    if save_predicted_latents:
        part["predicted_latents"] = predicted.cpu()
        part["oracle_predicted_latents"] = oracle_predicted.cpu()
        part["gt_latents"] = gt_latents.cpu()
    if "policy_checkpoint_sha256" in prepared:
        part["policy_checkpoint_sha256"] = prepared["policy_checkpoint_sha256"]
    return part


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        keys = ("mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max")
        return {key: math.nan for key in keys}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def write_parquet_best_effort(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    try:
        df.to_parquet(path, index=False, compression="snappy")
        return {"written": True, "path": str(path), "error": None}
    except Exception as exc:
        return {"written": False, "path": str(path), "error": repr(exc)}


def aggregate_results(
    selected_rows: list[dict[str, Any]],
    completed_mask: np.ndarray,
    parts_dir: Path,
    output_dir: Path,
    manifest_sha256: str,
    checkpoint_sha256: str,
    gt_h5_sha256: str,
    candidate_k: int,
    horizon: int,
    policy_checkpoint_sha256: str | None = None,
    single_policy_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if completed_mask.shape != (len(selected_rows),):
        raise RuntimeError("Internal completed-mask shape mismatch")

    part_paths: list[Path] = []
    missing: list[str] = []
    for row, is_complete in zip(selected_rows, completed_mask, strict=True):
        path = score_part_path(parts_dir, row)
        if not bool(is_complete) or not path.is_file():
            missing.append(row["sample_id"])
            continue
        part_paths.append(path)

    if not part_paths:
        raise RuntimeError("No valid score parts were available for aggregation")

    records: list[dict[str, Any]] = []
    cosine_per_step = []
    mse_per_step = []
    oracle_cosine_per_step = []
    oracle_mse_per_step = []
    teacher_actions_cosine = []
    teacher_actions_mse = []
    gt_actions = []

    # Load one score part at a time. Keeping every torch payload alive at once
    # is prohibitively expensive for the 487,880-window LIBERO-Plus manifest.
    for path in part_paths:
        part = load_torch(path)
        if (
            policy_checkpoint_sha256 is not None
            and part.get("policy_checkpoint_sha256") != policy_checkpoint_sha256
        ):
            raise RuntimeError(f"Policy checkpoint SHA-256 mismatch in aggregation: {path}")
        cosine = torch.as_tensor(part["cosine_score"]).float()
        mse = torch.as_tensor(part["mse_score"]).float()
        record: dict[str, Any] = {
            "sample_id": part["sample_id"],
            "sample_index": int(part["sample_index"]),
            "suite": part["suite"],
            "gt_latent_key": part["gt_latent_key"],
            "episode_index": int(part["episode_index"]),
            "local_step": int(part["local_step"]),
            "global_index": int(part["global_index"]),
            "window_index_h8": int(part["window_index_h8"]),
            "task_index": int(part["task_index"]),
            "task_text": part["task_text"],
            "teacher_by_cosine": int(part["teacher_by_cosine"]),
            "teacher_by_mse": int(part["teacher_by_mse"]),
            "teacher_agreement": bool(part["teacher_agreement"]),
            "best_cosine_score": float(part["best_cosine_score"]),
            "best_mse_score": float(part["best_mse_score"]),
            "cosine_margin": float(part["cosine_margin"]),
            "mse_margin": float(part["mse_margin"]),
            "oracle_cosine_score": float(part["oracle_cosine_score"]),
            "oracle_mse_score": float(part["oracle_mse_score"]),
            "oracle_better_than_all_cosine": bool(part["oracle_better_than_all_cosine"]),
            "oracle_better_than_all_mse": bool(part["oracle_better_than_all_mse"]),
            "oracle_cosine_gap": float(part["oracle_cosine_gap"]),
            "oracle_mse_gap": float(part["oracle_mse_gap"]),
            "candidate_cosine_mean": float(cosine.mean()),
            "candidate_cosine_std": float(cosine.std(unbiased=True)),
            "candidate_mse_mean": float(mse.mean()),
            "candidate_mse_std": float(mse.std(unbiased=True)),
            "candidate_generation_seconds": float(
                part.get("candidate_generation_seconds", math.nan)
            ),
            "scoring_seconds": float(part.get("scoring_seconds", math.nan)),
            "candidate_0_vs_1_max_abs_diff": float(
                part.get("candidate_0_vs_1_max_abs_diff", math.nan)
            ),
        }
        for candidate_index in range(candidate_k):
            record[f"candidate_{candidate_index}_cosine"] = float(cosine[candidate_index])
            record[f"candidate_{candidate_index}_mse"] = float(mse[candidate_index])
        records.append(record)

        cosine_per_step.append(torch.as_tensor(part["cosine_per_step"]).numpy())
        mse_per_step.append(torch.as_tensor(part["mse_per_step"]).numpy())
        oracle_cosine_per_step.append(torch.as_tensor(part["oracle_cosine_per_step"]).numpy())
        oracle_mse_per_step.append(torch.as_tensor(part["oracle_mse_per_step"]).numpy())
        teacher_actions_cosine.append(
            torch.as_tensor(part["teacher_action_cosine_lerobot"]).numpy()
        )
        teacher_actions_mse.append(torch.as_tensor(part["teacher_action_mse_lerobot"]).numpy())
        gt_actions.append(torch.as_tensor(part["gt_actions_lerobot"]).numpy())

    df = pd.DataFrame(records).sort_values("sample_index").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "validation_scores.csv"
    parquet_path = output_dir / "validation_scores.parquet"
    df.to_csv(csv_path, index=False)
    parquet_status = write_parquet_best_effort(df, parquet_path)

    task_agg = df.groupby(["suite", "task_index", "task_text"], as_index=False).agg(
        samples=("sample_id", "count"),
        episodes=("episode_index", "nunique"),
        teacher_agreement_rate=("teacher_agreement", "mean"),
        oracle_better_cosine_rate=("oracle_better_than_all_cosine", "mean"),
        oracle_better_mse_rate=("oracle_better_than_all_mse", "mean"),
        mean_best_cosine=("best_cosine_score", "mean"),
        mean_best_mse=("best_mse_score", "mean"),
        mean_oracle_cosine=("oracle_cosine_score", "mean"),
        mean_oracle_mse=("oracle_mse_score", "mean"),
        mean_cosine_margin=("cosine_margin", "mean"),
        mean_mse_margin=("mse_margin", "mean"),
    )
    task_csv = output_dir / "validation_task_summary.csv"
    task_parquet = output_dir / "validation_task_summary.parquet"
    task_agg.to_csv(task_csv, index=False)
    task_parquet_status = write_parquet_best_effort(task_agg, task_parquet)

    suite_agg = df.groupby("suite", as_index=False).agg(
        samples=("sample_id", "count"),
        tasks=("task_index", "nunique"),
        episodes=("episode_index", "nunique"),
        teacher_agreement_rate=("teacher_agreement", "mean"),
        oracle_better_cosine_rate=("oracle_better_than_all_cosine", "mean"),
        oracle_better_mse_rate=("oracle_better_than_all_mse", "mean"),
        mean_best_cosine=("best_cosine_score", "mean"),
        mean_best_mse=("best_mse_score", "mean"),
        mean_oracle_cosine=("oracle_cosine_score", "mean"),
        mean_oracle_mse=("oracle_mse_score", "mean"),
    )
    suite_csv = output_dir / "validation_suite_summary.csv"
    suite_parquet = output_dir / "validation_suite_summary.parquet"
    suite_agg.to_csv(suite_csv, index=False)
    suite_parquet_status = write_parquet_best_effort(suite_agg, suite_parquet)

    cosine_step = np.stack(cosine_per_step).astype(np.float32)
    mse_step = np.stack(mse_per_step).astype(np.float32)
    oracle_cos_step = np.stack(oracle_cosine_per_step).astype(np.float32)
    oracle_mse_step = np.stack(oracle_mse_per_step).astype(np.float32)
    teacher_cos_idx = df["teacher_by_cosine"].to_numpy(dtype=np.int64)
    teacher_mse_idx = df["teacher_by_mse"].to_numpy(dtype=np.int64)
    row_index = np.arange(len(df), dtype=np.int64)
    teacher_cosine_step = cosine_step[row_index, teacher_cos_idx]
    teacher_mse_step = mse_step[row_index, teacher_mse_idx]

    stepwise_path = output_dir / "validation_stepwise_metrics.npz"
    np.savez_compressed(
        stepwise_path,
        sample_ids=df["sample_id"].to_numpy(dtype="U32"),
        cosine_per_step=cosine_step,
        mse_per_step=mse_step,
        oracle_cosine_per_step=oracle_cos_step,
        oracle_mse_per_step=oracle_mse_step,
        teacher_cosine_per_step=teacher_cosine_step,
        teacher_mse_per_step=teacher_mse_step,
        teacher_by_cosine=teacher_cos_idx,
        teacher_by_mse=teacher_mse_idx,
    )

    teacher_records = {
        "format": "flare_validation_teacher_records_v1",
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "gt_h5_sha256": gt_h5_sha256,
        "sample_ids": df["sample_id"].tolist(),
        "sample_indices": torch.from_numpy(df["sample_index"].to_numpy(dtype=np.int64)),
        "suites": df["suite"].tolist(),
        "task_indices": torch.from_numpy(df["task_index"].to_numpy(dtype=np.int64)),
        "teacher_by_cosine": torch.from_numpy(teacher_cos_idx),
        "teacher_by_mse": torch.from_numpy(teacher_mse_idx),
        "teacher_actions_cosine_lerobot": torch.from_numpy(
            np.stack(teacher_actions_cosine).astype(np.float32)
        ),
        "teacher_actions_mse_lerobot": torch.from_numpy(
            np.stack(teacher_actions_mse).astype(np.float32)
        ),
        "gt_actions_lerobot": torch.from_numpy(np.stack(gt_actions).astype(np.float32)),
        "cosine_scores": torch.from_numpy(
            df[[f"candidate_{i}_cosine" for i in range(candidate_k)]].to_numpy(dtype=np.float32)
        ),
        "mse_scores": torch.from_numpy(
            df[[f"candidate_{i}_mse" for i in range(candidate_k)]].to_numpy(dtype=np.float32)
        ),
    }
    teacher_records_path = output_dir / "validation_teacher_records.pt"
    if policy_checkpoint_sha256 is not None:
        teacher_records["policy_checkpoint_sha256"] = policy_checkpoint_sha256
    atomic_torch_save(teacher_records_path, teacher_records)

    cosine_counts = Counter(int(v) for v in teacher_cos_idx)
    mse_counts = Counter(int(v) for v in teacher_mse_idx)
    n = len(df)
    uniform = 1.0 / candidate_k
    cosine_props = {str(i): cosine_counts.get(i, 0) / n for i in range(candidate_k)}
    mse_props = {str(i): mse_counts.get(i, 0) / n for i in range(candidate_k)}

    summary = {
        "format": "flare_validation_summary_v1",
        "status": "PASS" if not missing and len(part_paths) == len(selected_rows) else "PARTIAL",
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "gt_h5_sha256": gt_h5_sha256,
        "requested_samples": len(selected_rows),
        "scored_samples": len(part_paths),
        "missing_samples": missing,
        "tasks": int(df["task_index"].nunique()),
        "episodes": int(df["episode_index"].nunique()),
        "candidate_k": candidate_k,
        "future_horizon": horizon,
        "teacher_agreement": {
            "count": int(df["teacher_agreement"].sum()),
            "rate": float(df["teacher_agreement"].mean()),
        },
        "oracle_better_than_all_candidates": {
            "cosine_count": int(df["oracle_better_than_all_cosine"].sum()),
            "cosine_rate": float(df["oracle_better_than_all_cosine"].mean()),
            "mse_count": int(df["oracle_better_than_all_mse"].sum()),
            "mse_rate": float(df["oracle_better_than_all_mse"].mean()),
            "both_count": int(
                (df["oracle_better_than_all_cosine"] & df["oracle_better_than_all_mse"]).sum()
            ),
            "both_rate": float(
                (df["oracle_better_than_all_cosine"] & df["oracle_better_than_all_mse"]).mean()
            ),
        },
        "teacher_index_distribution": {
            "expected_uniform_proportion": uniform,
            "cosine_counts": {str(i): cosine_counts.get(i, 0) for i in range(candidate_k)},
            "cosine_proportions": cosine_props,
            "cosine_max_abs_deviation_from_uniform": max(
                abs(value - uniform) for value in cosine_props.values()
            ),
            "mse_counts": {str(i): mse_counts.get(i, 0) for i in range(candidate_k)},
            "mse_proportions": mse_props,
            "mse_max_abs_deviation_from_uniform": max(
                abs(value - uniform) for value in mse_props.values()
            ),
        },
        "metric_distributions": {
            "best_cosine_score": numeric_summary(df["best_cosine_score"].to_numpy()),
            "best_mse_score": numeric_summary(df["best_mse_score"].to_numpy()),
            "oracle_cosine_score": numeric_summary(df["oracle_cosine_score"].to_numpy()),
            "oracle_mse_score": numeric_summary(df["oracle_mse_score"].to_numpy()),
            "cosine_margin": numeric_summary(df["cosine_margin"].to_numpy()),
            "mse_margin": numeric_summary(df["mse_margin"].to_numpy()),
            "oracle_cosine_gap": numeric_summary(df["oracle_cosine_gap"].to_numpy()),
            "oracle_mse_gap": numeric_summary(df["oracle_mse_gap"].to_numpy()),
            "candidate_generation_seconds": numeric_summary(
                df["candidate_generation_seconds"].to_numpy()
            ),
            "scoring_seconds": numeric_summary(df["scoring_seconds"].to_numpy()),
        },
        "stepwise_means": {
            "all_candidates_cosine": cosine_step.mean(axis=(0, 1)).tolist(),
            "all_candidates_mse": mse_step.mean(axis=(0, 1)).tolist(),
            "cosine_teacher_cosine": teacher_cosine_step.mean(axis=0).tolist(),
            "mse_teacher_mse": teacher_mse_step.mean(axis=0).tolist(),
            "oracle_cosine": oracle_cos_step.mean(axis=0).tolist(),
            "oracle_mse": oracle_mse_step.mean(axis=0).tolist(),
        },
        "outputs": {
            "scores_csv": str(csv_path),
            "scores_parquet": parquet_status,
            "task_summary_csv": str(task_csv),
            "task_summary_parquet": task_parquet_status,
            "suite_summary_csv": str(suite_csv),
            "suite_summary_parquet": suite_parquet_status,
            "stepwise_metrics": str(stepwise_path),
            "teacher_records": str(teacher_records_path),
        },
    }
    if policy_checkpoint_sha256 is not None:
        summary["policy_checkpoint_sha256"] = policy_checkpoint_sha256
        summary["single_policy_checkpoint"] = str(single_policy_checkpoint)
    atomic_write_json(output_dir / "validation_summary.json", summary)
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT / "candidates",
    )
    parser.add_argument(
        "--candidate-summary",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT / "candidate_generation_summary.json",
    )
    parser.add_argument("--gt-h5", type=Path, default=DEFAULT_GT_H5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--single-policy-checkpoint",
        type=Path,
        default=None,
        help="Opt in to one frozen GR00T policy for the full manifest, without per-suite merger provenance.",
    )
    parser.add_argument(
        "--policy-checkpoint-sha256",
        default=None,
        help="Content identity of --single-policy-checkpoint; both options are required together.",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=DEFAULT_EXPECTED_CKPT_SHA256,
        help=(
            "Expected SHA-256 for the checkpoint used to build the GT frame bank. "
            "Override this together with --checkpoint and --gt-h5."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predicted-latents", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1")
    if not is_sha256(args.expected_checkpoint_sha256):
        raise ValueError("--expected-checkpoint-sha256 must be 64 lowercase hex characters")
    args.single_policy_checkpoint, args.policy_checkpoint_sha256 = verify_single_policy_checkpoint(
        args.single_policy_checkpoint, args.policy_checkpoint_sha256
    )

    # Reproducible inference settings. The script contains no intentional RNG,
    # but these settings reduce backend-dependent variation across reruns.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    for name in ("manifest", "candidate_dir", "candidate_summary", "gt_h5", "checkpoint"):
        path = getattr(args, name).expanduser().resolve()
        setattr(args, name, path)
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir = args.output_dir.expanduser().resolve()

    all_rows = load_jsonl(args.manifest)
    validate_manifest(all_rows)
    selected_rows = all_rows[: args.max_samples] if args.max_samples is not None else all_rows
    if not selected_rows:
        raise RuntimeError("No manifest samples selected")

    manifest_sha256 = sha256_file(args.manifest)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    gt_h5_sha256 = sha256_file(args.gt_h5)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: {checkpoint_sha256} != {args.expected_checkpoint_sha256}"
        )
    generation_summary = validate_candidate_summary(
        args.candidate_summary,
        args.manifest,
        args.candidate_dir,
        manifest_sha256,
        len(all_rows),
        args.single_policy_checkpoint,
        args.policy_checkpoint_sha256,
    )

    history_size = 3
    horizon = 8
    candidate_k = 4

    output_dir = args.output_dir.resolve()
    parts_dir = output_dir / "score_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "scoring_events.jsonl"
    failures_path = output_dir / "scoring_failures.jsonl"
    summary_path = output_dir / "scoring_run_summary.json"

    pending: list[tuple[int, dict[str, Any], str]] = []
    completed_mask = np.zeros(len(selected_rows), dtype=bool)
    skipped = 0
    regenerated = 0
    for selected_position, row in enumerate(selected_rows):
        cpath = candidate_path(args.candidate_dir, row)
        if not cpath.is_file():
            raise FileNotFoundError(cpath)
        csha = sha256_file(cpath)
        part_path = score_part_path(parts_dir, row)
        if args.overwrite:
            pending.append((selected_position, row, csha))
        elif valid_score_part(
            part_path,
            row,
            manifest_sha256,
            checkpoint_sha256,
            gt_h5_sha256,
            csha,
            candidate_k,
            horizon,
            policy_checkpoint_sha256=args.policy_checkpoint_sha256,
        ):
            skipped += 1
            completed_mask[selected_position] = True
        else:
            if part_path.exists():
                regenerated += 1
            pending.append((selected_position, row, csha))

    print("========== LIBERO-Plus FLARE batch scoring ==========")
    print("manifest samples       :", len(all_rows))
    print("selected samples       :", len(selected_rows))
    print("pending                :", len(pending))
    print("skipped valid          :", skipped)
    print("regenerate invalid     :", regenerated)
    print("batch size             :", args.batch_size)
    print("history/H/K            :", history_size, horizon, candidate_k)
    print("checkpoint SHA-256     :", checkpoint_sha256)
    print("GT H5 SHA-256          :", gt_h5_sha256)
    print("candidate summary      :", generation_summary.get("status"))

    started = time.time()
    generated = 0
    failed = 0

    # Avoid model construction when every selected score part already exists.
    if pending:
        device = torch.device(args.device)
        cfg = load_cfg()
        if int(cfg.history_size) != history_size or int(cfg.num_preds) != 1:
            raise RuntimeError(
                f"Unexpected LeWM config: history_size={cfg.history_size}, "
                f"num_preds={cfg.num_preds}"
            )
        model, checkpoint = build_model(cfg, args.checkpoint, device)

        with h5py.File(args.gt_h5, "r") as h5:
            ep_offsets = validate_h5_contract(
                h5=h5,
                checkpoint_path=args.checkpoint,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                horizon=horizon,
            )
            mean, std = action_stats(h5["frame_action"])

            total_batches = math.ceil(len(pending) / args.batch_size)
            for batch_number, batch_rows in enumerate(chunks(pending, args.batch_size), start=1):
                prepared_items: list[dict[str, Any]] = []
                for selected_position, row, candidate_sha256 in batch_rows:
                    try:
                        prepared = prepare_sample(
                            row=row,
                            candidate_file=candidate_path(args.candidate_dir, row),
                            candidate_sha256=candidate_sha256,
                            h5=h5,
                            ep_offsets=ep_offsets,
                            candidate_k=candidate_k,
                            history_size=history_size,
                            horizon=horizon,
                            manifest_sha256=manifest_sha256,
                            single_policy_checkpoint=args.single_policy_checkpoint,
                            policy_checkpoint_sha256=args.policy_checkpoint_sha256,
                        )
                        prepared["selected_position"] = selected_position
                        prepared_items.append(prepared)
                    except Exception as exc:
                        failed += 1
                        failure = {
                            "time": time.time(),
                            "stage": "prepare_sample",
                            "sample_id": row["sample_id"],
                            "sample_index": row["sample_index"],
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        append_jsonl(failures_path, failure)
                        print(f"[FAIL prepare] {row['sample_id']}: {exc}")

                if not prepared_items:
                    continue

                batch_start = time.time()
                history_latents = torch.stack(
                    [item["history_latents"] for item in prepared_items]
                ).to(device)
                action_sequences = torch.stack(
                    [item["all_action_sequences"] for item in prepared_items]
                )
                action_sequences = normalize_actions(action_sequences, mean, std).to(device)

                predicted_all = rollout_batch_from_latents(
                    model=model,
                    history_latents=history_latents,
                    action_sequences=action_sequences,
                    history_size=history_size,
                    future_horizon=horizon,
                ).cpu()
                batch_elapsed = time.time() - batch_start
                per_sample_seconds = batch_elapsed / len(prepared_items)

                for item_index, item in enumerate(prepared_items):
                    try:
                        predicted = predicted_all[item_index, :candidate_k]
                        oracle_predicted = predicted_all[item_index, candidate_k]
                        part = make_score_part(
                            prepared=item,
                            predicted=predicted,
                            oracle_predicted=oracle_predicted,
                            checkpoint_sha256=checkpoint_sha256,
                            manifest_sha256=manifest_sha256,
                            gt_h5_sha256=gt_h5_sha256,
                            scoring_seconds=per_sample_seconds,
                            save_predicted_latents=args.save_predicted_latents,
                        )
                        output_path = score_part_path(parts_dir, item["row"])
                        atomic_torch_save(output_path, part)
                        if not valid_score_part(
                            output_path,
                            item["row"],
                            manifest_sha256,
                            checkpoint_sha256,
                            gt_h5_sha256,
                            item["candidate_sha256"],
                            candidate_k,
                            horizon,
                            policy_checkpoint_sha256=args.policy_checkpoint_sha256,
                        ):
                            raise RuntimeError("Saved score part failed post-write validation")
                        completed_mask[item["selected_position"]] = True
                        generated += 1
                        append_jsonl(
                            events_path,
                            {
                                "time": time.time(),
                                "event": "scored",
                                "sample_id": item["row"]["sample_id"],
                                "sample_index": item["row"]["sample_index"],
                                "teacher_by_cosine": part["teacher_by_cosine"],
                                "teacher_by_mse": part["teacher_by_mse"],
                                "teacher_agreement": part["teacher_agreement"],
                                "best_cosine_score": part["best_cosine_score"],
                                "best_mse_score": part["best_mse_score"],
                                "oracle_cosine_score": part["oracle_cosine_score"],
                                "oracle_mse_score": part["oracle_mse_score"],
                            },
                        )
                    except Exception as exc:
                        failed += 1
                        failure = {
                            "time": time.time(),
                            "stage": "save_score",
                            "sample_id": item["row"]["sample_id"],
                            "sample_index": item["row"]["sample_index"],
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        append_jsonl(failures_path, failure)
                        print(f"[FAIL score] {item['row']['sample_id']}: {exc}")

                print(
                    f"[batch {batch_number:03d}/{total_batches:03d}] "
                    f"processed={len(prepared_items)}, generated_total={generated}, "
                    f"failed_total={failed}, batch_seconds={batch_elapsed:.3f}"
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    verify_single_policy_checkpoint(args.single_policy_checkpoint, args.policy_checkpoint_sha256)
    aggregation = aggregate_results(
        selected_rows=selected_rows,
        completed_mask=completed_mask,
        parts_dir=parts_dir,
        output_dir=output_dir,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        gt_h5_sha256=gt_h5_sha256,
        candidate_k=candidate_k,
        horizon=horizon,
        policy_checkpoint_sha256=args.policy_checkpoint_sha256,
        single_policy_checkpoint=args.single_policy_checkpoint,
    )

    elapsed = time.time() - started
    run_summary = {
        "format": "flare_lewm_scoring_run_summary_v1",
        "status": "PASS"
        if failed == 0 and aggregation.get("status") == "PASS"
        else "PARTIAL_OR_FAIL",
        "manifest": str(args.manifest),
        "manifest_format": FULL_MANIFEST_FORMAT,
        "manifest_sha256": manifest_sha256,
        "manifest_samples": len(all_rows),
        "max_samples": args.max_samples,
        "selection_is_full_manifest": len(selected_rows) == len(all_rows),
        "candidate_dir": str(args.candidate_dir),
        "candidate_summary": str(args.candidate_summary),
        "gt_h5": str(args.gt_h5),
        "gt_h5_sha256": gt_h5_sha256,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "output_dir": str(output_dir),
        "selected": len(selected_rows),
        "generated": generated,
        "skipped_valid": skipped,
        "regenerated_invalid": regenerated,
        "failed": failed,
        "scored_parts": aggregation.get("scored_samples"),
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "average_seconds_per_selected_sample": elapsed / len(selected_rows),
        "validation_summary": str(output_dir / "validation_summary.json"),
    }
    if args.policy_checkpoint_sha256 is not None:
        run_summary["policy_checkpoint_sha256"] = args.policy_checkpoint_sha256
        run_summary["single_policy_checkpoint"] = str(args.single_policy_checkpoint)
    atomic_write_json(summary_path, run_summary)

    print("\n========== FINAL ==========")
    print("run status             :", run_summary["status"])
    print("generated              :", generated)
    print("skipped valid          :", skipped)
    print("failed                 :", failed)
    print("scored parts           :", aggregation.get("scored_samples"))
    print("teacher agreement rate :", aggregation["teacher_agreement"]["rate"])
    print(
        "oracle>all cosine rate:",
        aggregation["oracle_better_than_all_candidates"]["cosine_rate"],
    )
    print(
        "oracle>all MSE rate   :",
        aggregation["oracle_better_than_all_candidates"]["mse_rate"],
    )
    print("summary                :", output_dir / "validation_summary.json")
    print("scores CSV             :", output_dir / "validation_scores.csv")
    print("teacher records        :", output_dir / "validation_teacher_records.pt")

    if run_summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
