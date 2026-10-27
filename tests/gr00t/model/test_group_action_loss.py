# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for legacy and grouped masked flow-matching reductions."""

from __future__ import annotations

from gr00t.utils import action_loss
import pytest
import torch


def _coordinate_losses(values, prefix_lengths, *, horizon=4, action_dim=2):
    mask = torch.zeros(len(values), horizon, action_dim)
    for index, length in enumerate(prefix_lengths):
        mask[index, :length] = 1
    errors = torch.tensor(values, dtype=torch.float32)[:, None, None].expand_as(mask)
    return errors * mask, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_metadata_preserves_exact_legacy_coordinate_reduction(dtype):
    errors = torch.arange(24, dtype=dtype).reshape(2, 4, 3).square()
    mask = torch.ones_like(errors)
    mask[0, 1:] = 0
    masked_error = errors * mask

    expected = masked_error.sum() / (mask.sum() + 1e-6)
    actual = action_loss.reduce_action_loss(masked_error, mask)

    assert torch.equal(actual, expected)
    assert actual.dtype == dtype


def test_group_samples_have_equal_mass_despite_different_prefix_lengths():
    masked_error, mask = _coordinate_losses([1.0, 9.0], [1, 4])

    actual = action_loss.reduce_action_loss(masked_error, mask, sample_weights=torch.ones(2))

    torch.testing.assert_close(actual, torch.tensor(5.0))
    assert not torch.isclose(actual, action_loss.reduce_action_loss(masked_error, mask))


@pytest.mark.parametrize("teacher_weight", [0.0, 0.25, 0.3, 1.0])
def test_source_means_use_explicit_teacher_coefficient(teacher_weight):
    masked_error, mask = _coordinate_losses([1.0, 3.0, 9.0, 21.0], [1, 4, 2, 3])
    weights = torch.tensor([3.0, 1.0, 1.0, 3.0])

    actual = action_loss.reduce_action_loss(
        masked_error,
        mask,
        sample_weights=weights,
        sample_sources=torch.tensor([0, 0, 1, 1]),
        teacher_loss_weight=teacher_weight,
    )

    expected = (1 - teacher_weight) * 1.5 + teacher_weight * 18.0
    torch.testing.assert_close(actual, torch.tensor(expected))


@pytest.mark.parametrize("source", [0, 1])
def test_single_source_gets_full_mass_including_gt_abstention_fallback(source):
    masked_error, mask = _coordinate_losses([2.0, 10.0], [4, 1])

    actual = action_loss.reduce_action_loss(
        masked_error,
        mask,
        sample_sources=torch.full((2,), source),
        teacher_loss_weight=0.2,
    )

    torch.testing.assert_close(actual, torch.tensor(6.0))


@pytest.mark.parametrize("zero_teacher_by", ["mask", "weight"])
def test_zero_mass_teacher_does_not_shrink_remaining_gt_loss(zero_teacher_by):
    masked_error, mask = _coordinate_losses(
        [7.0, 100.0], [2, 0 if zero_teacher_by == "mask" else 4]
    )
    weights = torch.tensor([1.0, 0.0 if zero_teacher_by == "weight" else 1.0])

    actual = action_loss.reduce_action_loss(
        masked_error,
        mask,
        sample_weights=weights,
        sample_sources=torch.tensor([0, 1]),
        teacher_loss_weight=0.9,
    )

    torch.testing.assert_close(actual, torch.tensor(7.0))


@pytest.mark.parametrize("with_sources", [False, True])
@pytest.mark.parametrize("zero_by", ["mask", "weight"])
def test_no_effective_samples_return_differentiable_zero(with_sources, zero_by):
    predictions = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.zeros_like(predictions) if zero_by == "mask" else torch.ones_like(predictions)
    weights = torch.ones(2) if zero_by == "mask" else torch.zeros(2)

    loss = action_loss.reduce_action_loss(
        predictions.square() * mask,
        mask,
        sample_weights=weights,
        sample_sources=torch.tensor([0, 1]) if with_sources else None,
    )
    loss.backward()

    assert loss.item() == 0
    assert loss.requires_grad
    assert torch.equal(predictions.grad, torch.zeros_like(predictions))


@pytest.mark.parametrize("weights", [[-1.0, 1.0], [float("nan"), 1.0], [1.0, float("inf")]])
def test_rejects_negative_or_nonfinite_sample_weights(weights):
    errors, mask = _coordinate_losses([1.0, 2.0], [1, 2])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        action_loss.reduce_action_loss(errors, mask, sample_weights=torch.tensor(weights))


@pytest.mark.parametrize("sources", [[-1, 0], [0, 2], [0.5, 1], [0, float("nan")]])
def test_rejects_invalid_source_ids(sources):
    errors, mask = _coordinate_losses([1.0, 2.0], [1, 2])
    with pytest.raises(ValueError, match="only 0.*or 1"):
        action_loss.reduce_action_loss(errors, mask, sample_sources=torch.tensor(sources))


@pytest.mark.parametrize("teacher_weight", [-0.01, 1.01, float("nan"), float("inf")])
def test_rejects_invalid_teacher_coefficient(teacher_weight):
    errors, mask = _coordinate_losses([1.0, 2.0], [1, 2])
    with pytest.raises(ValueError, match="teacher_loss_weight must be finite"):
        action_loss.reduce_action_loss(
            errors,
            mask,
            sample_sources=torch.tensor([0, 1]),
            teacher_loss_weight=teacher_weight,
        )


@pytest.mark.parametrize("field", ["sample_weights", "sample_sources"])
@pytest.mark.parametrize("bad_shape", [(2, 1), (1,), ()])
def test_metadata_requires_one_value_per_batch_sample(field, bad_shape):
    errors, mask = _coordinate_losses([1.0, 2.0], [1, 2])
    with pytest.raises(ValueError, match=f"{field} must have shape"):
        action_loss.reduce_action_loss(errors, mask, **{field: torch.ones(bad_shape)})


@pytest.mark.parametrize("error_shape,mask_shape", [((2,), (2,)), ((2, 3), (2, 4))])
def test_rejects_invalid_action_shapes(error_shape, mask_shape):
    with pytest.raises(ValueError, match="same.*shape"):
        action_loss.reduce_action_loss(torch.zeros(error_shape), torch.zeros(mask_shape))


def test_backpropagates_only_valid_coordinates_and_detaches_sample_weights():
    predictions = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    mask = torch.tensor([[[1.0, 1.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 0.0]]])
    weights = torch.tensor([1.0, 2.0], requires_grad=True)

    loss = action_loss.reduce_action_loss(
        predictions.square() * mask,
        mask,
        sample_weights=weights,
        sample_sources=torch.tensor([0, 1]),
        teacher_loss_weight=0.25,
    )
    loss.backward()

    scale = torch.tensor([0.75 / 2, 0.25 / 3])[:, None, None]
    torch.testing.assert_close(predictions.grad, 2 * predictions.detach() * mask * scale)
    assert weights.grad is None


def test_sampling_one_teacher_from_q_estimates_group_loss_without_double_weighting():
    errors, mask = _coordinate_losses([1.0, 9.0], [4, 1])
    probabilities = torch.tensor([0.25, 0.75])
    candidate_losses = torch.stack(
        [
            action_loss.reduce_action_loss(
                errors[index : index + 1],
                mask[index : index + 1],
                sample_sources=torch.ones(1, dtype=torch.long),
                sample_weights=torch.ones(1),
            )
            for index in range(2)
        ]
    )
    group_objective = action_loss.reduce_action_loss(
        errors, mask, sample_weights=probabilities, sample_sources=torch.ones(2, dtype=torch.long)
    )

    torch.testing.assert_close((candidate_losses * probabilities).sum(), group_objective)
    torch.testing.assert_close(group_objective, torch.tensor(7.0))


@pytest.mark.parametrize("with_sources", [False, True])
def test_ddp_global_mass_matches_combined_batch_loss_and_average_gradient(
    monkeypatch, with_sources
):
    targets = torch.tensor([0.0, 2.0, 1.0, 5.0])[:, None, None]
    _, mask = _coordinate_losses([1.0] * 4, [1, 4, 2, 1])
    weights = torch.tensor([1.0, 3.0, 2.0, 1.0])
    sources = torch.tensor([0, 0, 1, 1]) if with_sources else None
    reference_parameter = torch.tensor(0.25, requires_grad=True)
    reference_loss = action_loss.reduce_action_loss(
        (reference_parameter - targets).square() * mask,
        mask,
        sample_weights=weights,
        sample_sources=sources,
    )
    reference_loss.backward()
    global_mass = torch.tensor([4.0, 3.0]) if with_sources else torch.tensor([7.0])
    reduced_local_masses = []

    def all_reduce(tensor, op):
        assert op == action_loss.dist.ReduceOp.SUM
        assert not tensor.requires_grad
        reduced_local_masses.append(tensor.clone())
        tensor.copy_(global_mass)

    monkeypatch.setattr(action_loss.dist, "is_available", lambda: True)
    monkeypatch.setattr(action_loss.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(action_loss.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(action_loss.dist, "all_reduce", all_reduce)
    rank_losses = []
    rank_gradients = []
    for rank in range(2):
        index = slice(rank * 2, rank * 2 + 2)
        parameter = torch.tensor(0.25, requires_grad=True)
        loss = action_loss.reduce_action_loss(
            (parameter - targets[index]).square() * mask[index],
            mask[index],
            sample_weights=weights[index],
            sample_sources=sources[index] if with_sources else None,
        )
        loss.backward()
        rank_losses.append(loss.detach())
        rank_gradients.append(parameter.grad)

    torch.testing.assert_close(torch.stack(rank_losses).mean(), reference_loss.detach())
    torch.testing.assert_close(torch.stack(rank_gradients).mean(), reference_parameter.grad)
    torch.testing.assert_close(torch.stack(reduced_local_masses).sum(0), global_mass)
