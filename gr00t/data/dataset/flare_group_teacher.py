# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only selection and validation for optional FLARE group teachers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
import operator
import re
from typing import Any

import torch


GROUP_TEACHER_FORMAT = "flare_group_teacher_v1"
POLICY_PROVENANCE_FORMAT = "flare_group_policy_provenance_v1"


def require_sha256(value: Any, name: str) -> str:
    """Reject missing provenance rather than deriving it from a mutable path."""

    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest, got {value!r}")
    return value.lower()


@dataclass(frozen=True)
class GroupTeacherSelectionConfig:
    """Selection thresholds must be calibrated using training data only.

    LIBERO's raw seven-dimensional action space is [-1, 1], not the
    normalized GR00T action space. Set both bounds to None for other datasets.
    A fixed prefix still respects all enabled reliability and action gates.
    """

    temperature: float = 1.0
    oracle_max_mse: float | None = None
    max_paired_gap: float | None = None
    min_prefix: int = 1
    min_score_span: float = 0.0
    score_std_floor: float = 1e-6
    oracle_anchor: bool = True
    fixed_prefix: int | None = None
    action_min: float | None = -1.0
    action_max: float | None = 1.0

    def __post_init__(self) -> None:
        for name in ("temperature", "min_score_span", "score_std_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.temperature <= 0 or self.score_std_floor <= 0:
            raise ValueError("temperature and score_std_floor must be positive")
        if self.min_score_span < 0:
            raise ValueError("min_score_span must be nonnegative")
        if isinstance(self.min_prefix, bool) or operator.index(self.min_prefix) < 1:
            raise ValueError("min_prefix must be a positive integer")
        if self.fixed_prefix is not None:
            if isinstance(self.fixed_prefix, bool) or operator.index(self.fixed_prefix) < 1:
                raise ValueError("fixed_prefix must be a positive integer")
            if self.fixed_prefix < self.min_prefix:
                raise ValueError("fixed_prefix must be at least min_prefix")
        if self.oracle_anchor and (self.oracle_max_mse is None or self.max_paired_gap is None):
            raise ValueError("oracle anchoring requires oracle_max_mse and max_paired_gap")
        for name in ("oracle_max_mse", "max_paired_gap", "action_min", "action_max"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when provided")
        if self.oracle_max_mse is not None and self.oracle_max_mse < 0:
            raise ValueError("oracle_max_mse must be nonnegative")
        if (self.action_min is None) != (self.action_max is None):
            raise ValueError("Specify both action_min and action_max, or disable both")
        if self.action_min is not None and self.action_min >= self.action_max:
            raise ValueError("action_min must be smaller than action_max")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_group_teacher(
    candidate_actions_lerobot: Any,
    mse_per_step: Any,
    oracle_mse_per_step: Any,
    scores: Any,
    config: GroupTeacherSelectionConfig,
    *,
    score_per_step: Any | None = None,
    score_scope: str = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nonnegative candidate probabilities and continuous prefix lengths.

    Scores are higher-is-better and never used as signed FM loss weights.
    With shared-prefix scoring, every eligible candidate is ranked over the
    same shortest trusted prefix; its own longer prefix can still be taught.
    Uninformative groups return all-zero probabilities for GT replay. The
    oracle determines eligibility/length, not a common softmax offset.
    """

    actions = torch.as_tensor(candidate_actions_lerobot, dtype=torch.float64, device="cpu")
    errors = torch.as_tensor(mse_per_step, dtype=torch.float64, device="cpu")
    oracle = torch.as_tensor(oracle_mse_per_step, dtype=torch.float64, device="cpu")
    values = torch.as_tensor(scores, dtype=torch.float64, device="cpu")
    if actions.ndim != 3 or min(actions.shape) <= 0:
        raise ValueError("candidate_actions_lerobot must have nonempty shape [K, H, D]")
    k, horizon, _ = actions.shape
    if tuple(errors.shape) != (k, horizon):
        raise ValueError(f"mse_per_step must have shape {(k, horizon)}")
    if tuple(oracle.shape) != (horizon,) or tuple(values.shape) != (k,):
        raise ValueError("oracle_mse_per_step or scores shape does not match candidates")
    if score_scope not in {"full", "shared-prefix"}:
        raise ValueError("score_scope must be full or shared-prefix")
    step_scores = None
    if score_scope == "shared-prefix":
        if score_per_step is None:
            raise ValueError("shared-prefix scoring requires score_per_step")
        step_scores = torch.as_tensor(score_per_step, dtype=torch.float64, device="cpu")
        if tuple(step_scores.shape) != (k, horizon):
            raise ValueError(f"score_per_step must have shape {(k, horizon)}")
    if config.min_prefix > horizon:
        raise ValueError("min_prefix exceeds candidate horizon")
    if config.fixed_prefix is not None and config.fixed_prefix > horizon:
        raise ValueError("fixed_prefix exceeds candidate horizon")

    step_ok = torch.isfinite(actions).all(dim=-1) & torch.isfinite(errors) & (errors >= 0)
    if step_scores is not None:
        step_ok &= torch.isfinite(step_scores)
    if config.action_min is not None:
        step_ok &= (actions >= config.action_min).all(dim=-1)
        step_ok &= (actions <= config.action_max).all(dim=-1)
    if config.oracle_anchor:
        oracle_ok = torch.isfinite(oracle) & (oracle >= 0) & (oracle <= config.oracle_max_mse)
        step_ok &= oracle_ok.unsqueeze(0)
        step_ok &= errors - oracle.unsqueeze(0) <= config.max_paired_gap

    # cumprod prevents a later good prediction from restarting an invalid prefix.
    lengths = step_ok.to(torch.long).cumprod(dim=1).sum(dim=1)
    if config.fixed_prefix is not None:
        lengths = torch.where(lengths >= config.fixed_prefix, config.fixed_prefix, 0)
    eligible = lengths >= config.min_prefix
    if score_scope == "full":
        eligible &= torch.isfinite(values)
    lengths = torch.where(eligible, lengths, 0)
    probabilities = torch.zeros(k, dtype=torch.float64)
    if int(eligible.sum()) < 2:
        return probabilities.float(), torch.zeros_like(lengths)
    if step_scores is not None:
        shared_prefix = int(lengths[eligible].min())
        values = step_scores[:, :shared_prefix].mean(dim=1)
    usable_scores = values[eligible]
    span = float(usable_scores.max() - usable_scores.min())
    if span <= config.min_score_span:
        return probabilities.float(), torch.zeros_like(lengths)

    scale = max(float(usable_scores.std(unbiased=False)), config.score_std_floor)
    logits = (usable_scores - usable_scores.max()) / scale / config.temperature
    probabilities[eligible] = torch.softmax(logits, dim=0)
    # Very small probabilities may underflow at the serialized float32 boundary.
    probabilities = probabilities.float()
    probabilities /= probabilities.sum()
    lengths = torch.where(probabilities > 0, lengths, 0)
    return probabilities, lengths


def _cpu_tensor(value: Any, name: str, dtype: torch.dtype) -> torch.Tensor:
    result = torch.as_tensor(value, device="cpu")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if (
        dtype == torch.long
        and result.is_floating_point()
        and not torch.equal(result, result.round())
    ):
        raise ValueError(f"{name} must contain integers")
    return result.to(dtype=dtype).contiguous()


def validate_group_teacher(
    group: Mapping[str, Any] | None,
    teacher_records: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate and canonicalize the optional group sidecar, preserving old files.

    Candidate rows have exactly the ordering and action convention of the
    original top1 teacher rows. A zero-probability row is a GT fallback, and
    every zero-probability candidate has a zero supervised prefix.
    """

    if group is None:
        return None
    if not isinstance(group, Mapping):
        raise TypeError("group_teacher must be a mapping")
    required = (
        "format",
        "candidate_actions_lerobot",
        "candidate_probabilities",
        "prefix_lengths",
        "round_id",
        "policy_checkpoint_sha256",
        "world_model_checkpoint_sha256",
        "selection_config",
    )
    missing = [name for name in required if name not in group]
    if missing:
        raise KeyError(f"group_teacher is missing required fields: {missing}")
    if group["format"] != GROUP_TEACHER_FORMAT:
        raise ValueError(f"Unsupported group_teacher format: {group['format']!r}")
    actions = _cpu_tensor(
        group["candidate_actions_lerobot"], "candidate_actions_lerobot", torch.float32
    )
    probabilities = _cpu_tensor(
        group["candidate_probabilities"], "candidate_probabilities", torch.float32
    )
    lengths = _cpu_tensor(group["prefix_lengths"], "prefix_lengths", torch.long)
    if actions.ndim != 4:
        raise ValueError("candidate_actions_lerobot must have shape [N, K, H, D]")
    n, k, horizon, action_dim = actions.shape
    expected = (
        len(teacher_records["sample_ids"]),
        int(teacher_records["teacher_horizon"]),
        int(teacher_records["action_dim"]),
    )
    if k <= 0 or (n, horizon, action_dim) != expected:
        raise ValueError(
            f"Group candidate shape does not match teacher rows: {tuple(actions.shape)}, expected N/H/D={expected}"
        )
    if tuple(probabilities.shape) != (n, k) or tuple(lengths.shape) != (n, k):
        raise ValueError("Group probabilities and prefix lengths must have shape [N, K]")
    if (probabilities < 0).any():
        raise ValueError("candidate_probabilities must be nonnegative")
    sums = probabilities.sum(dim=1)
    normalized = torch.isclose(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)
    if not ((sums == 0) | normalized).all():
        raise ValueError("Each candidate probability row must sum to zero or one")
    if ((lengths < 0) | (lengths > horizon)).any():
        raise ValueError("prefix_lengths must be between zero and the candidate horizon")
    if ((probabilities > 0) & (lengths == 0)).any() or (
        (probabilities == 0) & (lengths != 0)
    ).any():
        raise ValueError(
            "Positive probabilities require positive prefixes; zero probabilities require zero prefixes"
        )

    if isinstance(group["round_id"], bool):
        raise ValueError("round_id must be a nonnegative integer")
    round_id = operator.index(group["round_id"])
    if round_id < 0:
        raise ValueError("round_id must be a nonnegative integer")
    policy_hash = require_sha256(group["policy_checkpoint_sha256"], "policy_checkpoint_sha256")
    wm_hash = require_sha256(
        group["world_model_checkpoint_sha256"], "world_model_checkpoint_sha256"
    )
    if not isinstance(group["selection_config"], dict):
        raise TypeError("selection_config must be a dict")
    for name in ("sample_ids", "sample_indices", "episode_indices", "local_steps"):
        if name not in group:
            continue
        if name == "sample_ids":
            actual = [str(value) for value in group[name]]
            reference = [str(value) for value in teacher_records[name]]
            matches = actual == reference
        else:
            actual = _cpu_tensor(group[name], name, torch.long).reshape(-1)
            reference = _cpu_tensor(teacher_records[name], name, torch.long).reshape(-1)
            matches = torch.equal(actual, reference)
        if not matches:
            raise ValueError(f"group_teacher {name} ordering does not match top1 teacher rows")

    return {
        **group,
        "candidate_actions_lerobot": actions,
        "candidate_probabilities": probabilities,
        "prefix_lengths": lengths,
        "round_id": round_id,
        "policy_checkpoint_sha256": policy_hash,
        "world_model_checkpoint_sha256": wm_hash,
    }


def sample_group_candidate(
    probabilities: Any,
    generator: torch.Generator | None = None,
) -> int | None:
    """Sample one whole candidate, or return None for the GT fallback."""

    values = _cpu_tensor(probabilities, "candidate_probabilities", torch.float32)
    if values.ndim != 1 or values.numel() == 0 or (values < 0).any():
        raise ValueError("candidate_probabilities must be a nonempty nonnegative vector")
    total = values.sum()
    if total == 0:
        return None
    if not torch.isclose(total, torch.ones_like(total), rtol=1e-5, atol=1e-6):
        raise ValueError("candidate_probabilities must sum to zero or one")
    return int(torch.multinomial(values, 1, generator=generator).item())
