# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Optional

import yaml

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType

from .data.data_config import DataConfig, SingleDatasetConfig
from .model import create_model_union_type
from .model.gr00t_n1d7 import Gr00tN1d7Config
from .training.training_config import TrainingConfig


ModelUnionType = create_model_union_type()


@dataclass
class Config:
    """Complete configuration."""

    load_config_path: Optional[str] = None
    model: ModelUnionType = field(default_factory=lambda: Gr00tN1d7Config())
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self, f)

    def load(self, path: Path):
        """Load configuration from YAML file."""
        data = yaml.load(path.read_text(), Loader=yaml.Loader)
        if isinstance(data, dict):  # for training
            self.load_dict(data)
        elif isinstance(data, self.__class__):
            self = data
        else:
            raise ValueError(f"Invalid config file: {path}")
        # config = cls(**config) # if yaml.dump(self.__dict__, ...) is used
        return self

    def load_dict(self, data: dict):
        if "model" in data:
            self.model = self.model.__class__(**data["model"])
        if "data" in data:
            self.data = DataConfig(**data["data"])
            # Ensure nested datasets are converted to dataclass instances
            converted: List[SingleDatasetConfig] = []
            for ds in self.data.datasets:
                if isinstance(ds, dict):
                    converted.append(SingleDatasetConfig(**ds))
                else:
                    converted.append(ds)
            self.data.datasets = converted
        if "training" in data:
            self.training = TrainingConfig(**data["training"])
        return self

    @classmethod
    def from_pretrained(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        data = yaml.load(path.read_text(), Loader=yaml.Loader)
        return data

    def get_deepspeed_config(self) -> dict:
        """Generate DeepSpeed configuration."""
        stage = self.training.deepspeed_stage

        gr00t_dir = Path(__file__).parent.parent
        if stage == 2:
            config = json.load(open(gr00t_dir / "configs/deepspeed/zero2_config.json"))
        elif stage == 3:
            config = json.load(open(gr00t_dir / "configs/deepspeed/zero3_config.json"))
        else:
            raise ValueError(f"Invalid DeepSpeed stage: {stage}")

        return config

    def validate(self):
        """Validate configuration."""
        # Check dataset path(s)
        embodiment_tags = set()
        for d_cfg in self.data.datasets:
            # (Disable general missing-data checks because some workflows support
            # cached remote data sources. FLARE sidecar files are local-only and
            # are validated explicitly below.)
            if d_cfg.dataset_type not in {"physical_embodiment", "flare_distillation"}:
                raise ValueError(f"Unsupported dataset_type: {d_cfg.dataset_type!r}")
            if not d_cfg.embodiment_tag:
                raise ValueError(
                    f"Embodiment tag is empty for dataset paths {d_cfg.dataset_paths}"
                )
            if d_cfg.dataset_type == "flare_distillation":
                if len(d_cfg.dataset_paths) != 1:
                    raise ValueError(
                        "flare_distillation requires exactly one original LeRobot "
                        f"dataset path, got {d_cfg.dataset_paths}"
                    )
                if not d_cfg.distillation_index_path:
                    raise ValueError(
                        "flare_distillation requires distillation_index_path"
                    )
                if not d_cfg.teacher_records_path:
                    raise ValueError(
                        "flare_distillation requires teacher_records_path"
                    )
                for sidecar in [
                    d_cfg.distillation_index_path,
                    d_cfg.teacher_records_path,
                ]:
                    if not Path(sidecar).is_file():
                        raise FileNotFoundError(
                            f"FLARE distillation sidecar does not exist: {sidecar}"
                        )
            embodiment_tags.add(d_cfg.embodiment_tag)

        stripped_modality_configs = {}
        for embodiment_tag in embodiment_tags:
            modality_cfg = self.data.modality_configs.get(embodiment_tag)
            if modality_cfg is None:
                raise ValueError(
                    f"No modality config registered for embodiment tag '{embodiment_tag}'. "
                    f"Available tags: {sorted(self.data.modality_configs.keys())}. "
                    f"Provide --modality-config-path to register a custom modality config, "
                    f"or use one of the pre-registered tags."
                )
            stripped_modality_configs[embodiment_tag] = modality_cfg
        self.data.modality_configs = stripped_modality_configs

        # ensure mix ratios are valid
        total_ratio = sum(d.mix_ratio for d in self.data.datasets)
        if total_ratio <= 0:
            raise ValueError("Sum of mix_ratio must be greater than zero")

        # Fill in default values for action configs
        for embodiment_tag in self.data.modality_configs:
            # Fill in default values for action representation, type and format
            if self.data.modality_configs[embodiment_tag]["action"].action_configs is None:
                self.data.modality_configs[embodiment_tag]["action"].action_configs = [
                    ActionConfig(
                        rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                    )
                ] * len(self.data.modality_configs[embodiment_tag]["action"].modality_keys)

        # Validate precision settings
        if self.training.fp16 and self.training.bf16:
            raise ValueError("Cannot use both fp16 and bf16")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
