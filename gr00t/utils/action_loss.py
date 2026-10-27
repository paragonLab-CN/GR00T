# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-testable, backward-compatible reduction of masked action losses."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist


def _global_weight_mass(weight_mass: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Share only normalization constants; DDP still synchronizes gradients."""
    if not dist.is_available() or not dist.is_initialized():
        return weight_mass, 1
    weight_mass = weight_mass.detach().clone()
    dist.all_reduce(weight_mass, op=dist.ReduceOp.SUM)
    return weight_mass, dist.get_world_size()


def reduce_action_loss(
    masked_error: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    sample_sources: torch.Tensor | None = None,
    teacher_loss_weight: float = 0.3,
) -> torch.Tensor:
    """Reduce coordinate errors, optionally balancing GT and group teachers.

    ``masked_error`` must already include ``action_mask``. Without metadata the
    original batch-global coordinate mean is returned unchanged. Group mode
    first averages each sample over its valid coordinates. Source IDs are 0
    for GT (including abstention fallback), and 1 for a sampled group teacher.
    When both sources exist their means are mixed with ``teacher_loss_weight``;
    when one source is absent, the remaining source receives the full weight.
    """
    if masked_error.shape != action_mask.shape or masked_error.ndim < 2:
        raise ValueError("Action errors and masks must have the same [B, ...] shape")
    if sample_weights is None and sample_sources is None:
        return masked_error.sum() / (action_mask.sum() + 1e-6)
    if not math.isfinite(teacher_loss_weight) or not 0.0 <= teacher_loss_weight <= 1.0:
        raise ValueError("teacher_loss_weight must be finite and lie in [0, 1]")

    batch_size = masked_error.shape[0]
    device = masked_error.device
    coordinate_mass = action_mask.float().flatten(1).sum(1)
    sample_error = masked_error.float().flatten(1).sum(1)
    sample_loss = sample_error / coordinate_mass.clamp_min(1.0)
    valid = coordinate_mass > 0

    weights = (
        torch.ones(batch_size, device=device)
        if sample_weights is None
        else torch.as_tensor(sample_weights, device=device, dtype=torch.float32).detach()
    )
    if weights.shape != (batch_size,):
        raise ValueError(f"sample_weights must have shape [{batch_size}]")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("sample_weights must be finite and nonnegative")
    weights = weights * valid

    if sample_sources is None:
        mass, world_size = _global_weight_mass(weights.sum().reshape(1))
        return world_size * (sample_loss * weights).sum() / mass[0].clamp_min(1e-6)

    sources = torch.as_tensor(sample_sources, device=device).detach()
    if sources.shape != (batch_size,):
        raise ValueError(f"sample_sources must have shape [{batch_size}]")
    if not bool(((sources == 0) | (sources == 1)).all()):
        raise ValueError("sample_sources must contain only 0 (GT) or 1 (teacher)")

    gt_weights = weights * (sources == 0)
    teacher_weights = weights * (sources == 1)
    masses, world_size = _global_weight_mass(torch.stack((gt_weights.sum(), teacher_weights.sum())))
    gt_loss = world_size * (sample_loss * gt_weights).sum() / masses[0].clamp_min(1e-6)
    teacher_loss = world_size * (sample_loss * teacher_weights).sum() / masses[1].clamp_min(1e-6)
    has_gt = bool(masses[0] > 0)
    has_teacher = bool(masses[1] > 0)
    if has_gt and has_teacher:
        return (1.0 - teacher_loss_weight) * gt_loss + teacher_loss_weight * teacher_loss
    return gt_loss + teacher_loss
