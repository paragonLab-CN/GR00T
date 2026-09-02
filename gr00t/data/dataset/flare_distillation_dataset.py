# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexed FLARE-VLA self-distillation dataset.

This dataset keeps the original LeRobot dataset unchanged and reads an explicit
mixed index produced by ``08_build_distillation_dataset.py``.  Each index row is
one training window:

* ``source == "gt"`` uses the normal GR00T action chunk from LeRobot.
* ``source == "teacher"`` loads the same observation/state/language, replaces
  the raw LeRobot action prefix with the selected FLARE teacher action, then
  applies the official GR00T processor and masks the unsupervised action tail.

The teacher replacement intentionally happens *before* GR00T relative-action
conversion and normalization.  This is required because the saved teacher
records use raw LeRobot action convention.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig, VLAStepData

from .lerobot_episode_loader import LeRobotEpisodeLoader
from .sharded_single_step_dataset import extract_step_data


DISTILLATION_INDEX_FORMAT = "flare_distillation_index_v1"
TEACHER_RECORDS_FORMAT = "flare_distillation_teacher_records_v1"
_SUPPORTED_SOURCES = {"gt", "teacher"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"Expected a JSON object at {path}:{line_no}, "
                    f"got {type(row).__name__}"
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Distillation index is empty: {path}")
    return rows


def _load_torch_dict(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(value).__name__}")
    return value


def _require_fields(row: dict[str, Any], fields: list[str], context: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise KeyError(f"{context} is missing required fields: {missing}")


def _as_cpu_float_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu").contiguous()
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    return tensor


class FlareDistillationDataset(ShardedDataset):
    """Exact indexed GT + FLARE-teacher dataset for GR00T training.

    ``len(dataset)`` follows the GR00T sharded-dataset interface and therefore
    returns the number of shards, not the number of training records.  Use
    :attr:`num_records` for the exact mixed-index size (631 in the current
    pilot).

    Args:
        dataset_path: Root of the original LeRobot dataset.
        embodiment_tag: GR00T embodiment tag for this dataset.
        modality_configs: Official GR00T modality configuration.
        distillation_index_path: ``distillation_index.jsonl``.
        teacher_records_path: ``teacher_records_enriched.pt``.
        video_backend: Video decoding backend used by the official loader.
        video_backend_kwargs: Optional backend-specific arguments.
        shard_size: Number of indexed records per shard.
        seed: Stored for reproducibility and future shard policies.
        allow_padding: Clamp temporal indices to the episode boundary.  The
            current FLARE manifest was constructed for horizon 8 while the
            GR00T processor requests horizon 16, so this normally needs to be
            enabled for the 500+131 pilot.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        distillation_index_path: str | Path,
        teacher_records_path: str | Path,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        shard_size: int = 2**10,
        seed: int = 42,
        allow_padding: bool = False,
    ) -> None:
        super().__init__(dataset_path)

        self.dataset_path = Path(dataset_path)
        self.distillation_index_path = Path(distillation_index_path)
        self.teacher_records_path = Path(teacher_records_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.shard_size = int(shard_size)
        self.seed = int(seed)
        self.allow_padding = bool(allow_padding)
        self.processor = None

        if self.shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {self.shard_size}")
        if "action" not in self.modality_configs:
            raise KeyError("FLARE distillation requires an action modality config")
        if not self.distillation_index_path.is_file():
            raise FileNotFoundError(
                f"Distillation index does not exist: {self.distillation_index_path}"
            )
        if not self.teacher_records_path.is_file():
            raise FileNotFoundError(
                f"Teacher records do not exist: {self.teacher_records_path}"
            )

        self.index_rows = _load_jsonl(self.distillation_index_path)
        self.teacher_records = _load_torch_dict(self.teacher_records_path)

        action_delta_indices = [
            int(value) for value in self.modality_configs["action"].delta_indices
        ]
        if not action_delta_indices:
            raise ValueError("Action delta_indices must not be empty")
        if action_delta_indices != list(range(len(action_delta_indices))):
            raise ValueError(
                "FLARE action delta_indices must be contiguous from zero, got "
                f"{action_delta_indices}"
            )
        self.processor_action_horizon = len(action_delta_indices)
        self.action_keys = list(self.modality_configs["action"].modality_keys)

        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_configs,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )
        self._episode_position_by_id = self._build_episode_position_map()

        self._validate_teacher_records()
        self._validate_index()
        self._build_shards()

    @property
    def num_records(self) -> int:
        """Return the exact number of mixed GT/teacher training records."""

        return len(self.index_rows)

    @property
    def source_counts(self) -> dict[str, int]:
        """Return counts of GT and teacher records."""

        return dict(Counter(str(row["source"]) for row in self.index_rows))

    def _build_episode_position_map(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for position, metadata in enumerate(self.episode_loader.episodes_metadata):
            episode_id = int(metadata["episode_index"])
            if episode_id in mapping:
                raise RuntimeError(f"Duplicate episode_index in metadata: {episode_id}")
            mapping[episode_id] = position
        return mapping

    def _validate_teacher_records(self) -> None:
        required = [
            "format",
            "sample_ids",
            "sample_indices",
            "episode_indices",
            "local_steps",
            "teacher_actions_lerobot",
            "teacher_horizon",
            "processor_action_horizon",
            "model_action_horizon",
            "action_dim",
            "action_keys",
        ]
        _require_fields(self.teacher_records, required, "teacher records")

        actual_format = str(self.teacher_records["format"])
        if actual_format != TEACHER_RECORDS_FORMAT:
            raise RuntimeError(
                f"Unsupported teacher-record format {actual_format!r}; "
                f"expected {TEACHER_RECORDS_FORMAT!r}"
            )

        self.teacher_actions = _as_cpu_float_tensor(
            self.teacher_records["teacher_actions_lerobot"],
            "teacher_actions_lerobot",
        )
        if self.teacher_actions.ndim != 3:
            raise RuntimeError(
                "teacher_actions_lerobot must have shape [N, H, D], got "
                f"{tuple(self.teacher_actions.shape)}"
            )

        self.teacher_horizon = int(self.teacher_records["teacher_horizon"])
        self.supervised_steps = self.teacher_horizon
        self.teacher_action_dim = int(self.teacher_records["action_dim"])
        saved_processor_horizon = int(
            self.teacher_records["processor_action_horizon"]
        )
        saved_action_keys = [str(key) for key in self.teacher_records["action_keys"]]

        expected_shape = (
            len(self.teacher_records["sample_ids"]),
            self.teacher_horizon,
            self.teacher_action_dim,
        )
        if tuple(self.teacher_actions.shape) != expected_shape:
            raise RuntimeError(
                "Teacher tensor shape does not match teacher metadata: "
                f"tensor={tuple(self.teacher_actions.shape)}, metadata={expected_shape}"
            )
        if self.teacher_horizon <= 0:
            raise ValueError(f"teacher_horizon must be positive, got {self.teacher_horizon}")
        if self.teacher_horizon > self.processor_action_horizon:
            raise RuntimeError(
                f"Teacher horizon {self.teacher_horizon} exceeds configured processor "
                f"action horizon {self.processor_action_horizon}"
            )
        if saved_processor_horizon != self.processor_action_horizon:
            raise RuntimeError(
                "Processor action horizon mismatch: teacher file records "
                f"{saved_processor_horizon}, current modality config requests "
                f"{self.processor_action_horizon}"
            )
        if saved_action_keys != self.action_keys:
            raise RuntimeError(
                "Action-key order mismatch. Teacher columns were saved as "
                f"{saved_action_keys}, but current modality config uses {self.action_keys}. "
                "Changing this order would silently corrupt action supervision."
            )

        n = self.teacher_actions.shape[0]
        self.teacher_sample_ids = [
            str(value) for value in self.teacher_records["sample_ids"]
        ]
        self.teacher_sample_indices = torch.as_tensor(
            self.teacher_records["sample_indices"], dtype=torch.long
        ).reshape(-1)
        self.teacher_episode_indices = torch.as_tensor(
            self.teacher_records["episode_indices"], dtype=torch.long
        ).reshape(-1)
        self.teacher_local_steps = torch.as_tensor(
            self.teacher_records["local_steps"], dtype=torch.long
        ).reshape(-1)

        for name, values in [
            ("sample_ids", self.teacher_sample_ids),
            ("sample_indices", self.teacher_sample_indices),
            ("episode_indices", self.teacher_episode_indices),
            ("local_steps", self.teacher_local_steps),
        ]:
            if len(values) != n:
                raise RuntimeError(
                    f"Teacher field {name} has length {len(values)}, expected {n}"
                )
        if len(set(self.teacher_sample_ids)) != n:
            raise RuntimeError("Duplicate sample_id detected in teacher records")

    def _validate_index(self) -> None:
        required = [
            "format",
            "distillation_index",
            "record_id",
            "sample_id",
            "sample_index",
            "episode_index",
            "local_step",
            "source",
            "teacher_row",
        ]
        record_ids: set[str] = set()
        teacher_rows_seen: list[int] = []

        for expected_index, row in enumerate(self.index_rows):
            _require_fields(row, required, f"distillation index row {expected_index}")
            if str(row["format"]) != DISTILLATION_INDEX_FORMAT:
                raise RuntimeError(
                    f"Unsupported index format at row {expected_index}: {row['format']!r}"
                )
            if int(row["distillation_index"]) != expected_index:
                raise RuntimeError(
                    f"distillation_index is not contiguous at row {expected_index}: "
                    f"stored={row['distillation_index']}"
                )

            record_id = str(row["record_id"])
            if record_id in record_ids:
                raise RuntimeError(f"Duplicate record_id in mixed index: {record_id}")
            record_ids.add(record_id)

            source = str(row["source"])
            if source not in _SUPPORTED_SOURCES:
                raise RuntimeError(
                    f"Unsupported source {source!r} at mixed-index row {expected_index}"
                )

            episode_id = int(row["episode_index"])
            if episode_id not in self._episode_position_by_id:
                raise RuntimeError(
                    f"Index row {expected_index} references unknown episode {episode_id}"
                )
            episode_position = self._episode_position_by_id[episode_id]
            episode_length = self.episode_loader.get_episode_length(episode_position)
            local_step = int(row["local_step"])
            if local_step < 0 or local_step >= episode_length:
                raise RuntimeError(
                    f"Invalid local_step for row {expected_index}: episode={episode_id}, "
                    f"step={local_step}, episode_length={episode_length}"
                )

            if not self.allow_padding:
                min_delta = min(
                    min(config.delta_indices) for config in self.modality_configs.values()
                )
                max_delta = max(
                    max(config.delta_indices) for config in self.modality_configs.values()
                )
                if local_step + min_delta < 0 or local_step + max_delta >= episode_length:
                    raise RuntimeError(
                        "A distillation window crosses an episode boundary while "
                        "allow_padding=False. For the current horizon-8 FLARE manifest "
                        "with the GR00T horizon-16 processor, enable data.allow_padding. "
                        f"Offending row={expected_index}, episode={episode_id}, "
                        f"local_step={local_step}, episode_length={episode_length}, "
                        f"delta_range=[{min_delta}, {max_delta}]."
                    )

            if source == "gt":
                if row["teacher_row"] is not None:
                    raise RuntimeError(
                        f"GT row {expected_index} must have teacher_row=null"
                    )
                continue

            teacher_row = int(row["teacher_row"])
            if teacher_row < 0 or teacher_row >= len(self.teacher_actions):
                raise RuntimeError(
                    f"Teacher row pointer out of range at index {expected_index}: "
                    f"teacher_row={teacher_row}"
                )
            teacher_rows_seen.append(teacher_row)

            expected_values = {
                "sample_id": self.teacher_sample_ids[teacher_row],
                "sample_index": int(self.teacher_sample_indices[teacher_row]),
                "episode_index": int(self.teacher_episode_indices[teacher_row]),
                "local_step": int(self.teacher_local_steps[teacher_row]),
            }
            for field, expected_value in expected_values.items():
                actual_value = row[field]
                if field == "sample_id":
                    actual_value = str(actual_value)
                else:
                    actual_value = int(actual_value)
                if actual_value != expected_value:
                    raise RuntimeError(
                        f"Teacher mapping mismatch at mixed-index row {expected_index}: "
                        f"field={field}, index={actual_value!r}, "
                        f"teacher_file={expected_value!r}"
                    )

        expected_teacher_rows = list(range(len(self.teacher_actions)))
        if sorted(teacher_rows_seen) != expected_teacher_rows:
            raise RuntimeError(
                "Teacher rows in distillation_index.jsonl must reference every teacher "
                "record exactly once. "
                f"seen={sorted(teacher_rows_seen)}, expected={expected_teacher_rows}"
            )

    def _build_shards(self) -> None:
        record_indices = np.arange(self.num_records, dtype=np.int64)
        # The serialized index stores all GT rows first and appends teacher rows.
        # Deterministically shuffle before slicing so smaller shards do not become
        # source-pure and bias an epoch toward only GT or only teacher targets.
        rng = np.random.default_rng(self.seed)
        rng.shuffle(record_indices)
        self.shards = [
            record_indices[start : start + self.shard_size]
            for start in range(0, self.num_records, self.shard_size)
        ]
        self.shard_lengths = np.asarray(
            [len(shard) for shard in self.shards], dtype=np.int64
        )
        if len(self.shards) == 0 or bool((self.shard_lengths <= 0).any()):
            raise RuntimeError("FLARE sharding produced an empty shard")

    def __len__(self) -> int:
        """Return the number of shards, as required by ``ShardedDataset``."""

        return len(self.shards)

    def get_shard_length(self, idx: int) -> int:
        return int(self.shard_lengths[idx])

    def get_index_record(self, record_index: int) -> dict[str, Any]:
        """Return a copy of one mixed-index row for verification/debugging."""

        return dict(self.index_rows[record_index])

    def _episode_position(self, episode_id: int) -> int:
        try:
            return self._episode_position_by_id[int(episode_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown episode_index: {episode_id}") from exc

    def _extract_raw_vla_step(
        self,
        episode_data: pd.DataFrame,
        record: dict[str, Any],
    ) -> VLAStepData:
        return extract_step_data(
            episode_data=episode_data,
            step_index=int(record["local_step"]),
            modality_configs=self.modality_configs,
            embodiment_tag=self.embodiment_tag,
            allow_padding=self.allow_padding,
        )

    def _replace_teacher_prefix(
        self,
        vla_step_data: VLAStepData,
        teacher_row: int,
    ) -> VLAStepData:
        teacher = self.teacher_actions[teacher_row].numpy()
        new_actions: dict[str, np.ndarray] = {}
        column_offset = 0

        for key in self.action_keys:
            if key not in vla_step_data.actions:
                raise KeyError(
                    f"Action key {key!r} is missing from extracted VLAStepData"
                )
            official_action = np.asarray(
                vla_step_data.actions[key], dtype=np.float32
            )
            if official_action.ndim != 2:
                raise RuntimeError(
                    f"Action {key!r} must have shape [H, D], got "
                    f"{official_action.shape}"
                )
            if official_action.shape[0] < self.teacher_horizon:
                raise RuntimeError(
                    f"Official action horizon for {key!r} is "
                    f"{official_action.shape[0]}, shorter than teacher horizon "
                    f"{self.teacher_horizon}"
                )

            key_dim = int(official_action.shape[1])
            next_offset = column_offset + key_dim
            if next_offset > self.teacher_action_dim:
                raise RuntimeError(
                    "Action-key dimensions exceed saved teacher action_dim while "
                    f"splitting key {key!r}"
                )

            replaced_action = official_action.copy()
            replaced_action[: self.teacher_horizon] = teacher[
                :, column_offset:next_offset
            ]
            new_actions[key] = replaced_action
            column_offset = next_offset

        if column_offset != self.teacher_action_dim:
            raise RuntimeError(
                "Sum of extracted action-key dimensions does not match teacher "
                f"action_dim: extracted={column_offset}, "
                f"teacher={self.teacher_action_dim}"
            )

        metadata = dict(vla_step_data.metadata)
        metadata.update(
            {
                "flare_source": "teacher",
                "flare_teacher_row": int(teacher_row),
                "flare_teacher_horizon": int(self.teacher_horizon),
            }
        )
        return replace(vla_step_data, actions=new_actions, metadata=metadata)

    def get_raw_record(
        self,
        record_index: int,
        *,
        apply_teacher: bool = True,
    ) -> VLAStepData:
        """Load one raw VLAStepData record for verification.

        This method is intentionally separate from ``get_shard`` so the
        verification script can compare paired GT/teacher observations before
        random image/state augmentation in the processor.
        """

        record = self.index_rows[record_index]
        episode_position = self._episode_position(int(record["episode_index"]))
        episode_data = self.episode_loader[episode_position]
        vla_step_data = self._extract_raw_vla_step(episode_data, record)
        if apply_teacher and str(record["source"]) == "teacher":
            vla_step_data = self._replace_teacher_prefix(
                vla_step_data, int(record["teacher_row"])
            )
        return vla_step_data

    def get_processed_record(self, record_index: int) -> dict[str, Any]:
        """Load and process one indexed record using the configured processor."""

        record = self.index_rows[record_index]
        episode_position = self._episode_position(int(record["episode_index"]))
        episode_data = self.episode_loader[episode_position]
        return self._process_record(episode_data, record)

    def _process_record(
        self,
        episode_data: pd.DataFrame,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("Processor must be set before loading FLARE shards")

        vla_step_data = self._extract_raw_vla_step(episode_data, record)
        source = str(record["source"])
        if source == "teacher":
            vla_step_data = self._replace_teacher_prefix(
                vla_step_data, int(record["teacher_row"])
            )

        messages = [
            {
                "type": MessageType.EPISODE_STEP.value,
                "content": vla_step_data,
            }
        ]
        processed = self.processor(messages)

        if source == "teacher":
            if "action" not in processed or "action_mask" not in processed:
                raise KeyError(
                    "GR00T processor output must contain action and action_mask "
                    "for teacher training records"
                )
            action = processed["action"]
            action_mask = processed["action_mask"]
            if not torch.is_tensor(action) or not torch.is_tensor(action_mask):
                raise TypeError("Processed action and action_mask must be torch tensors")
            if action.ndim != 2 or action_mask.shape != action.shape:
                raise RuntimeError(
                    "Processed action/action_mask must have equal [H, D] shapes, got "
                    f"action={tuple(action.shape)}, mask={tuple(action_mask.shape)}"
                )
            if action.shape[0] < self.teacher_horizon:
                raise RuntimeError(
                    f"Processed action horizon {action.shape[0]} is shorter than "
                    f"teacher horizon {self.teacher_horizon}"
                )

            # The flow-matching model encodes the full action trajectory before
            # applying the loss mask.  Therefore keeping official GT values in
            # the masked tail would leak future expert targets into a teacher
            # sample.  Zero both the numerical tail and its supervision mask.
            action = action.clone()
            action[self.teacher_horizon :] = 0
            processed["action"] = action

            action_mask = action_mask.clone()
            action_mask[self.teacher_horizon :] = 0
            processed["action_mask"] = action_mask

        return processed

    def get_shard(self, idx: int) -> list[dict[str, Any]]:
        """Load and process all indexed records in one shard.

        Records are grouped by episode so each episode/video is decoded once per
        shard.  ``ShardedMixtureDataset`` shuffles the returned datapoints before
        yielding them, so grouping here does not reduce training randomization.
        """

        if idx < 0 or idx >= len(self.shards):
            raise IndexError(f"Shard index {idx} out of bounds")

        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record_index in self.shards[idx]:
            record = self.index_rows[int(record_index)]
            grouped[int(record["episode_index"])].append(record)

        datapoints: list[dict[str, Any]] = []
        for episode_id, records in grouped.items():
            episode_position = self._episode_position(episode_id)
            episode_data = self.episode_loader[episode_position]
            for record in records:
                datapoints.append(self._process_record(episode_data, record))
        return datapoints

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Delegate normalization statistics to the original LeRobot dataset."""

        return self.episode_loader.get_dataset_statistics()

    def get_initial_actions(self):
        """Delegate optional policy initial actions to the original dataset."""

        return self.episode_loader.get_initial_actions()
