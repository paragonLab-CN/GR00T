# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only FLARE target, fallback, and group-sampling regressions."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gr00t.data.dataset.flare_distillation_dataset import FlareDistillationDataset
from gr00t.data.types import EmbodimentTag, ModalityConfig
import numpy as np
import pandas as pd
import pytest
import torch


_TEACHER_HORIZON = 3
_PROCESSOR_HORIZON = 5
_MODEL_HORIZON = 7
_ACTION_DIM = 3
_PADDED_ACTION_DIM = 5


class _FakeEpisodeLoader:
    def __init__(self, **kwargs: Any) -> None:
        self.episodes_metadata = [{"episode_index": 7, "length": 24}]
        self.data = pd.DataFrame(
            {
                "state.pose": [
                    np.array([step / 100, step / 50 + 0.1, -0.2], dtype=np.float32)
                    for step in range(24)
                ],
                "action.arm": [
                    np.array([step / 10 + 0.25, step / 20 - 0.45], dtype=np.float32)
                    for step in range(24)
                ],
                "action.gripper": [np.array([0.7], dtype=np.float32) for _ in range(24)],
                "video.camera": [np.full((4, 4, 3), step, dtype=np.uint8) for step in range(24)],
                "language.task": ["pick up the object"] * 24,
            }
        )

    def get_episode_length(self, idx: int) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> pd.DataFrame:
        return self.data

    def get_dataset_statistics(self) -> dict[str, Any]:
        return {"fake_statistics": True}

    def get_initial_actions(self) -> list:
        return []


class _FakeRelativeProcessor:
    """Apply a state-dependent transformation before horizon/dimension padding."""

    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raw = messages[0]["content"]
        actions = np.concatenate([raw.actions["arm"], raw.actions["gripper"]], axis=1)
        state = torch.from_numpy(raw.states["pose"][0].copy())
        transformed = (torch.from_numpy(actions.copy()) - state) * 2.0
        action = torch.zeros(_MODEL_HORIZON, _PADDED_ACTION_DIM)
        action[:_PROCESSOR_HORIZON, :_ACTION_DIM] = transformed
        mask = torch.zeros_like(action)
        mask[:_PROCESSOR_HORIZON, :_ACTION_DIM] = 1.0
        return {
            "action": action,
            "action_mask": mask,
            "state": state,
            "debug_local_step": round(float(state[0]) * 100),
        }


def _group_teacher(
    probabilities: tuple[float, float] = (0.25, 0.75),
    prefix_lengths: tuple[int, int] = (3, 2),
    round_id: int = 0,
) -> dict[str, Any]:
    return {
        "format": "flare_group_teacher_v1",
        "candidate_actions_lerobot": torch.tensor(
            [
                [
                    [[0.1, -0.2, 0.3], [0.2, -0.1, 0.2], [0.3, 0.0, 0.1]],
                    [[-0.6, 0.5, -0.4], [-0.5, 0.4, -0.3], [-0.4, 0.3, -0.2]],
                ]
            ],
            dtype=torch.float32,
        ),
        "candidate_probabilities": torch.tensor([probabilities], dtype=torch.float32),
        "prefix_lengths": torch.tensor([prefix_lengths], dtype=torch.long),
        "round_id": round_id,
        "policy_checkpoint_sha256": "a" * 64,
        "world_model_checkpoint_sha256": "b" * 64,
        "selection_config": {"method": "test"},
        "sample_ids": ["group-0"],
        "sample_indices": torch.tensor([11]),
        "episode_indices": torch.tensor([7]),
        "local_steps": torch.tensor([1]),
    }


@pytest.fixture
def make_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "gr00t.data.dataset.flare_distillation_dataset.LeRobotEpisodeLoader",
        _FakeEpisodeLoader,
    )
    creations = 0

    def create(
        *,
        group: dict[str, Any] | None = None,
        seed: int = 42,
        shard_size: int = 8,
    ) -> FlareDistillationDataset:
        nonlocal creations
        root = tmp_path / str(creations)
        root.mkdir()
        creations += 1
        rows = [
            {
                "format": "flare_distillation_index_v1",
                "distillation_index": 0,
                "record_id": "gt-0",
                "sample_id": "gt-0",
                "sample_index": 10,
                "episode_index": 7,
                "local_step": 0,
                "source": "gt",
                "teacher_row": None,
            },
            {
                "format": "flare_distillation_index_v1",
                "distillation_index": 1,
                "record_id": "teacher-0",
                "sample_id": "group-0",
                "sample_index": 11,
                "episode_index": 7,
                "local_step": 1,
                "source": "teacher",
                "teacher_row": 0,
            },
        ]
        index_path = root / "distillation_index.jsonl"
        index_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        teacher_records = {
            "format": "flare_distillation_teacher_records_v1",
            "sample_ids": ["group-0"],
            "sample_indices": torch.tensor([11]),
            "episode_indices": torch.tensor([7]),
            "local_steps": torch.tensor([1]),
            "teacher_actions_lerobot": torch.full((1, _TEACHER_HORIZON, _ACTION_DIM), 0.9),
            "teacher_horizon": _TEACHER_HORIZON,
            "processor_action_horizon": _PROCESSOR_HORIZON,
            "model_action_horizon": _MODEL_HORIZON,
            "action_dim": _ACTION_DIM,
            "action_keys": ["arm", "gripper"],
        }
        if group is not None:
            teacher_records["group_teacher"] = deepcopy(group)
        teacher_path = root / "teacher_records_enriched.pt"
        torch.save(teacher_records, teacher_path)
        dataset = FlareDistillationDataset(
            dataset_path=root,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_configs={
                "state": ModalityConfig(delta_indices=[0], modality_keys=["pose"]),
                "action": ModalityConfig(
                    delta_indices=list(range(_PROCESSOR_HORIZON)),
                    modality_keys=["arm", "gripper"],
                ),
                "video": ModalityConfig(delta_indices=[0], modality_keys=["camera"]),
                "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
            },
            distillation_index_path=index_path,
            teacher_records_path=teacher_path,
            seed=seed,
            shard_size=shard_size,
        )
        dataset.set_processor(_FakeRelativeProcessor())
        return dataset

    return create


def _assert_prefix_mask(sample: dict[str, Any], prefix_length: int) -> None:
    assert sample["action"].shape == (_MODEL_HORIZON, _PADDED_ACTION_DIM)
    assert bool((sample["action_mask"][:prefix_length, :_ACTION_DIM] == 1).all())
    assert bool((sample["action_mask"][prefix_length:] == 0).all())
    assert bool((sample["action"][prefix_length:] == 0).all())
    assert bool((sample["action_mask"][:, _ACTION_DIM:] == 0).all())


def test_legacy_teacher_tail_is_zero_without_new_loss_metadata(make_dataset):
    dataset = make_dataset()
    sample = dataset.get_processed_record(1)
    raw = dataset.get_raw_record(1)
    gt = dataset.get_raw_record(1, apply_teacher=False)
    expected = (dataset.teacher_actions[0] - sample["state"]) * 2
    torch.testing.assert_close(sample["action"][:_TEACHER_HORIZON, :_ACTION_DIM], expected)
    np.testing.assert_array_equal(
        raw.actions["arm"][:_TEACHER_HORIZON], dataset.teacher_actions[0, :, :2]
    )
    np.testing.assert_array_equal(
        raw.actions["arm"][_TEACHER_HORIZON:], gt.actions["arm"][_TEACHER_HORIZON:]
    )
    _assert_prefix_mask(sample, _TEACHER_HORIZON)
    assert "action_loss_weight" not in sample
    assert "action_loss_source" not in sample
    assert "action_loss_source" not in dataset.get_processed_record(0)


def test_group_uses_one_whole_candidate_before_relative_processing(make_dataset):
    group = _group_teacher(probabilities=(0.0, 1.0), prefix_lengths=(0, 2))
    dataset = make_dataset(group=group)
    raw = dataset.get_raw_record(1)
    gt = dataset.get_raw_record(1, apply_teacher=False)
    selected = group["candidate_actions_lerobot"][0, 1]
    np.testing.assert_array_equal(raw.actions["arm"][:_TEACHER_HORIZON], selected[:, :2])
    np.testing.assert_array_equal(raw.actions["gripper"][:_TEACHER_HORIZON], selected[:, 2:])
    np.testing.assert_array_equal(raw.states["pose"], gt.states["pose"])
    np.testing.assert_array_equal(raw.images["camera"][0], gt.images["camera"][0])
    assert raw.text == gt.text
    assert raw.metadata["flare_group_candidate_index"] == 1
    assert raw.metadata["flare_group_prefix_length"] == 2

    sample = dataset.get_processed_record(1)
    expected = (selected[:2] - sample["state"]) * 2
    torch.testing.assert_close(sample["action"][:2, :_ACTION_DIM], expected)
    _assert_prefix_mask(sample, 2)
    assert sample["action_loss_weight"].shape == torch.Size([])
    assert sample["action_loss_weight"].dtype == torch.float32
    assert sample["action_loss_weight"].item() == 1
    assert sample["action_loss_source"].shape == torch.Size([])
    assert sample["action_loss_source"].dtype == torch.long
    assert sample["action_loss_source"].item() == 1
    assert dataset.get_processed_record(0)["action_loss_source"].item() == 0


def test_zero_probability_group_falls_back_to_full_gt_without_changing_index(make_dataset):
    dataset = make_dataset(group=_group_teacher(probabilities=(0.0, 0.0), prefix_lengths=(0, 0)))
    raw = dataset.get_raw_record(1)
    gt = dataset.get_raw_record(1, apply_teacher=False)
    for key in dataset.action_keys:
        np.testing.assert_array_equal(raw.actions[key], gt.actions[key])
    assert raw.metadata["flare_group_fallback"] is True
    sample = dataset.get_processed_record(1)
    expected = _FakeRelativeProcessor()([{"content": gt}])
    torch.testing.assert_close(sample["action"], expected["action"])
    torch.testing.assert_close(sample["action_mask"], expected["action_mask"])
    assert sample["action_loss_source"].item() == 0
    assert sample["action_loss_weight"].item() == 1
    assert dataset.source_counts == {"gt": 1, "teacher": 1}
    assert dataset.get_index_record(1)["teacher_row"] == 0


def _sample_sequence(dataset: FlareDistillationDataset, count: int, diagnostics: bool = False):
    indices = []
    for _ in range(count):
        if diagnostics:
            dataset.get_raw_record(1)
            dataset.get_raw_record(1)
            dataset.get_processed_record(1)
        dataset.get_shard(0)
        raw = dataset.get_raw_record(1)
        indices.append(raw.metadata["flare_group_candidate_index"])
    return indices


def test_shard_reloads_resample_q_without_changing_teacher_loss_weight(make_dataset):
    dataset = make_dataset(group=_group_teacher())
    sequence = _sample_sequence(dataset, 128)
    assert set(sequence) == {0, 1}
    assert sequence.count(1) > sequence.count(0)
    assert dataset.get_processed_record(1)["action_loss_weight"].item() == 1.0
    assert dataset._group_shard_load_counts[(0, 0, 0)] == 128


def test_diagnostics_reuse_preview_and_do_not_advance_sampling(make_dataset):
    dataset = make_dataset(group=_group_teacher())
    preview = dataset.get_raw_record(1).metadata["flare_group_candidate_index"]
    for _ in range(5):
        assert dataset.get_raw_record(1).metadata["flare_group_candidate_index"] == preview
        dataset.get_processed_record(1)
    assert not dataset._group_shard_load_counts
    assert not dataset._group_candidate_by_record
    dataset.get_shard(0)
    assert dataset.get_raw_record(1).metadata["flare_group_candidate_index"] == preview

    with_diagnostics = make_dataset(group=_group_teacher())
    without_diagnostics = make_dataset(group=_group_teacher())
    assert _sample_sequence(with_diagnostics, 24, diagnostics=True) == _sample_sequence(
        without_diagnostics, 24
    )


def test_loading_another_shard_does_not_advance_the_teacher_shard_stream(make_dataset):
    dataset = make_dataset(group=_group_teacher(), shard_size=1)
    teacher_shard = int(dataset._record_shard_indices[1])
    gt_shard = int(dataset._record_shard_indices[0])
    assert teacher_shard != gt_shard
    preview = dataset.get_raw_record(1).metadata["flare_group_candidate_index"]
    for _ in range(5):
        dataset.get_shard(gt_shard)
    dataset.get_shard(teacher_shard)
    assert dataset.get_raw_record(1).metadata["flare_group_candidate_index"] == preview
    assert dataset._group_shard_load_counts[(0, 0, teacher_shard)] == 1
    assert dataset._group_shard_load_counts[(0, 0, gt_shard)] == 5


def test_seed_and_round_id_control_reproducible_refreshes(make_dataset):
    first = _sample_sequence(make_dataset(group=_group_teacher(), seed=42), 32)
    repeated = _sample_sequence(make_dataset(group=_group_teacher(), seed=42), 32)
    new_seed = _sample_sequence(make_dataset(group=_group_teacher(), seed=43), 32)
    new_round = _sample_sequence(make_dataset(group=_group_teacher(round_id=1), seed=42), 32)
    assert first == repeated
    assert first != new_seed
    assert first != new_round


def test_worker_and_rank_sampling_streams_are_distinct_and_reproducible(make_dataset, monkeypatch):
    def sequence_for_context(rank: int, worker_id: int):
        dataset = make_dataset(group=_group_teacher())
        monkeypatch.setattr(dataset, "_group_sampling_context", lambda: (rank, worker_id))
        return _sample_sequence(dataset, 32)

    baseline = sequence_for_context(0, 0)
    assert baseline == sequence_for_context(0, 0)
    assert baseline != sequence_for_context(0, 1)
    assert baseline != sequence_for_context(1, 0)


def test_group_context_uses_distributed_rank_and_dataloader_worker(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 4)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=2))
    assert FlareDistillationDataset._group_sampling_context() == (4, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format", "unsupported_group_format"),
        ("candidate_probabilities", torch.tensor([[-0.1, 1.1]])),
        ("candidate_probabilities", torch.tensor([[0.2, 0.2]])),
        ("candidate_probabilities", torch.ones(1, 3) / 3),
        ("prefix_lengths", torch.tensor([[0, 2]])),
        ("prefix_lengths", torch.tensor([[1.5, 2.0]])),
        ("round_id", -1),
        ("policy_checkpoint_sha256", "not-a-sha256"),
        ("sample_ids", ["different-group"]),
    ],
)
def test_dataset_rejects_malformed_group_teacher(make_dataset, field, value):
    group = _group_teacher()
    group[field] = value
    with pytest.raises((ValueError, RuntimeError, TypeError, KeyError)):
        make_dataset(group=group)


def test_dataset_rejects_nonfinite_candidate_even_in_unsupervised_tail(make_dataset):
    group = _group_teacher()
    group["candidate_actions_lerobot"][0, 1, 2, 0] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        make_dataset(group=group)
