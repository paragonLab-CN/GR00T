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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.data.dataset.flare_sidecars import resolve_flare_distillation_sidecars
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    flare_sidecars = resolve_flare_distillation_sidecars(
        distillation_dataset_dir=ft_config.distillation_dataset_dir,
        distillation_index_path=ft_config.distillation_index_path,
        teacher_records_path=ft_config.teacher_records_path,
    )
    flare_requested = flare_sidecars is not None
    require_mixed_data = os.environ.get("FLARE_REQUIRE_MIXED_DATA", "0") == "1"
    if require_mixed_data and not ft_config.keep_full_gt_dataset:
        raise RuntimeError(
            "FLARE_REQUIRE_MIXED_DATA=1 but --keep_full_gt_dataset was not received. "
            "Refusing to silently launch teacher-only training."
        )

    if flare_sidecars is not None:
        print("========== FLARE distillation input ==========")
        print("distillation index :", flare_sidecars.index_path)
        print("teacher records    :", flare_sidecars.teacher_records_path)
        print("mixed records      :", flare_sidecars.total_records)
        print("source counts      :", flare_sidecars.source_counts)
        if flare_sidecars.summary_path is not None:
            print("validated summary  :", flare_sidecars.summary_path)

    if ft_config.keep_full_gt_dataset:
        if flare_sidecars is None:
            raise ValueError(
                "--keep_full_gt_dataset requires a FLARE distillation directory or "
                "explicit sidecar file pair"
            )
        teacher_mix_ratio = float(ft_config.teacher_mix_ratio)
        if not 0.0 < teacher_mix_ratio < 1.0:
            raise ValueError(
                f"teacher_mix_ratio must lie in (0, 1), got {teacher_mix_ratio}"
            )
        if int(flare_sidecars.source_counts.get("gt", 0)) != 0:
            raise ValueError(
                "Keeping the full physical GT dataset requires a teacher-only FLARE "
                "index. Rebuild the sidecars with --index-mode teacher_only."
            )
        if int(flare_sidecars.source_counts.get("teacher", 0)) <= 0:
            raise ValueError("Teacher-only FLARE index contains no teacher records")

        dataset_configs = [
            {
                "dataset_paths": [ft_config.dataset_path],
                "mix_ratio": 1.0 - teacher_mix_ratio,
                "embodiment_tag": embodiment_tag,
                "dataset_type": "physical_embodiment",
            },
            {
                "dataset_paths": [ft_config.dataset_path],
                "mix_ratio": teacher_mix_ratio,
                "embodiment_tag": embodiment_tag,
                "dataset_type": "flare_distillation",
                "distillation_index_path": str(flare_sidecars.index_path),
                "teacher_records_path": str(flare_sidecars.teacher_records_path),
            },
        ]
        print("========== Full-GT + FLARE mixture ==========")
        print("GT sampling fraction      :", 1.0 - teacher_mix_ratio)
        print("teacher sampling fraction :", teacher_mix_ratio)
    else:
        dataset_config = {
            "dataset_paths": [ft_config.dataset_path],
            "mix_ratio": 1.0,
            "embodiment_tag": embodiment_tag,
            "dataset_type": (
                "flare_distillation" if flare_requested else "physical_embodiment"
            ),
        }
        if flare_sidecars is not None:
            dataset_config.update(
                {
                    "distillation_index_path": str(flare_sidecars.index_path),
                    "teacher_records_path": str(flare_sidecars.teacher_records_path),
                }
            )
        dataset_configs = [dataset_config]

    if require_mixed_data:
        expected_teacher_ratio = float(
            os.environ.get("FLARE_EXPECTED_TEACHER_MIX_RATIO", ft_config.teacher_mix_ratio)
        )
        if len(dataset_configs) != 2:
            raise RuntimeError(
                f"Expected exactly two mixed data sources, got {len(dataset_configs)}: {dataset_configs}"
            )
        actual = {item["dataset_type"]: float(item["mix_ratio"]) for item in dataset_configs}
        expected = {
            "physical_embodiment": 1.0 - expected_teacher_ratio,
            "flare_distillation": expected_teacher_ratio,
        }
        if set(actual) != set(expected) or any(
            abs(actual[key] - expected[key]) > 1e-9 for key in expected
        ):
            raise RuntimeError(
                f"Resolved data mixture does not match expectation. actual={actual}, expected={expected}"
            )
        print("FLARE mixed-data contract verified:", actual)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": dataset_configs,
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = os.environ.get("GR00T_BACKBONE_MODEL_NAME", "nvidia/Cosmos-Reason2-2B")
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.allow_padding = ft_config.allow_padding

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    if ft_config.validate_config_only:
        resolved = [
            {
                "dataset_type": item.dataset_type,
                "mix_ratio": float(item.mix_ratio),
                "dataset_paths": list(item.dataset_paths),
                "distillation_index_path": item.distillation_index_path,
                "teacher_records_path": item.teacher_records_path,
            }
            for item in config.data.datasets
        ]
        print("========== Resolved training datasets ==========")
        print(json.dumps(resolved, indent=2))
        print("Configuration validation completed; training was not started.")
        raise SystemExit(0)

    run(config)
