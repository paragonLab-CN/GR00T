# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU batch-metadata and checkpoint-configuration integration regressions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@pytest.fixture
def collator(monkeypatch):
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing

    processor = MagicMock()
    processor.return_value = {
        "input_ids": torch.tensor([[11, 12], [21, 22]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    build_processor = MagicMock(return_value=processor)
    monkeypatch.setattr(processing, "build_processor", build_processor)
    result = processing.Gr00tN1d7DataCollator(
        model_name="fake-vlm", transformers_loading_kwargs={"local_files_only": True}
    )
    build_processor.assert_called_once_with("fake-vlm", {"local_files_only": True})
    assert processor.tokenizer.padding_side == "left"
    return result, processor


def _sample(index):
    return {
        "action": torch.full((4, 3), float(index)),
        "action_mask": np.ones((4, 3), dtype=np.float32),
        "state": np.full((1, 3), float(index), dtype=np.float32),
        "embodiment_id": index,
        "vlm_content": {"text": f"sample {index}", "images": [f"image-{index}"]},
    }


@pytest.mark.parametrize("teacher_first", [False, True])
@pytest.mark.parametrize("scalar_tensors", [False, True])
def test_mixed_physical_gt_and_group_teacher_preserve_batch_axis(
    collator, teacher_first, scalar_tensors
):
    data_collator, processor = collator
    gt = _sample(0)
    teacher = _sample(1)
    teacher.update(
        action_loss_weight=torch.tensor(0.75) if scalar_tensors else 0.75,
        action_loss_source=torch.tensor(1) if scalar_tensors else 1,
    )
    features = [teacher, gt] if teacher_first else [gt, teacher]

    inputs = data_collator(features)["inputs"]

    assert inputs["action"].shape == (2, 4, 3)
    assert inputs["action_mask"].shape == (2, 4, 3)
    assert inputs["state"].shape == (2, 1, 3)
    assert inputs["embodiment_id"].shape == (2,)
    assert inputs["action_loss_weight"].shape == (2,)
    assert inputs["action_loss_source"].shape == (2,)
    assert inputs["action_loss_weight"].dtype == torch.float32
    assert inputs["action_loss_source"].dtype == torch.long
    order = [1, 0] if teacher_first else [0, 1]
    torch.testing.assert_close(inputs["embodiment_id"], torch.tensor(order))
    torch.testing.assert_close(inputs["action_loss_source"], torch.tensor(order))
    expected_weights = torch.tensor([0.75, 1.0] if teacher_first else [1.0, 0.75])
    torch.testing.assert_close(inputs["action_loss_weight"], expected_weights)
    processor.assert_called_once_with(
        text=[feature["vlm_content"]["text"] for feature in features],
        images=[feature["vlm_content"]["images"][0] for feature in features],
        return_tensors="pt",
        padding=True,
    )
    assert "action_loss_weight" not in gt
    assert "action_loss_source" not in gt


def test_legacy_batches_do_not_gain_group_loss_metadata(collator):
    data_collator, _ = collator
    inputs = data_collator([_sample(0), _sample(1)])["inputs"]

    assert "action_loss_weight" not in inputs
    assert "action_loss_source" not in inputs
    assert inputs["action"].shape[0] == 2
    assert inputs["input_ids"].shape[0] == 2


def test_explicit_group_mode_adds_metadata_even_for_all_physical_gt(collator):
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing

    data_collator = processing.Gr00tN1d7DataCollator("fake-vlm", group_teacher_mode=True)
    features = [_sample(0), _sample(1)]

    inputs = data_collator(features)["inputs"]

    torch.testing.assert_close(inputs["action_loss_weight"], torch.ones(2))
    torch.testing.assert_close(inputs["action_loss_source"], torch.zeros(2, dtype=torch.long))
    assert all("action_loss_source" not in feature for feature in features)


@pytest.mark.parametrize("has_group_sidecars", [False, True])
def test_pipeline_sets_group_mode_from_dataset_not_current_batch(collator, has_group_sidecars):
    from gr00t.model.gr00t_n1d7 import setup

    data_collator, _ = collator
    data_collator.group_teacher_mode = not has_group_sidecars
    pipeline = object.__new__(setup.Gr00tN1d7Pipeline)
    pipeline.processor = SimpleNamespace(collator=data_collator)
    pipeline.train_dataset = SimpleNamespace(
        datasets=[
            SimpleNamespace(),
            SimpleNamespace(group_teacher={} if has_group_sidecars else None),
        ]
    )

    selected_collator = pipeline._create_collator()
    inputs = selected_collator([_sample(0), _sample(1)])["inputs"]

    assert selected_collator is data_collator
    assert selected_collator.group_teacher_mode is has_group_sidecars
    assert ("action_loss_source" in inputs) is has_group_sidecars
    assert ("action_loss_weight" in inputs) is has_group_sidecars


@pytest.mark.parametrize("group_mode", [False, True])
def test_pipeline_all_gt_and_teacher_ranks_enter_matching_collectives(
    collator, monkeypatch, group_mode
):
    from gr00t.model.gr00t_n1d7 import setup
    from gr00t.utils import action_loss

    data_collator, _ = collator
    pipeline = object.__new__(setup.Gr00tN1d7Pipeline)
    pipeline.processor = SimpleNamespace(collator=data_collator)
    pipeline.train_dataset = SimpleNamespace(
        datasets=[SimpleNamespace(group_teacher={} if group_mode else None)]
    )
    data_collator = pipeline._create_collator()
    teacher_features = [_sample(2), _sample(3)]
    if group_mode:
        for feature in teacher_features:
            feature["action_loss_source"] = torch.tensor(1)
            feature["action_loss_weight"] = torch.tensor(1.0)
    local_masses = []

    def all_reduce(tensor, op):
        assert op == action_loss.dist.ReduceOp.SUM
        local_masses.append(tensor.clone())
        tensor.copy_(torch.tensor([2.0, 2.0]))

    monkeypatch.setattr(action_loss.dist, "is_available", lambda: True)
    monkeypatch.setattr(action_loss.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(action_loss.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(action_loss.dist, "all_reduce", all_reduce)
    rank_losses = []
    rank_collective_counts = []
    for features in [[_sample(0), _sample(1)], teacher_features]:
        inputs = data_collator(features)["inputs"]
        before = len(local_masses)
        rank_losses.append(
            action_loss.reduce_action_loss(
                inputs["action"].square() * inputs["action_mask"],
                inputs["action_mask"],
                sample_weights=inputs.get("action_loss_weight"),
                sample_sources=inputs.get("action_loss_source"),
            )
        )
        rank_collective_counts.append(len(local_masses) - before)

    assert rank_collective_counts == ([1, 1] if group_mode else [0, 0])
    expected = torch.tensor(0.7 * 0.5 + 0.3 * 6.5 if group_mode else 3.5)
    torch.testing.assert_close(torch.stack(rank_losses).mean(), expected)
    if group_mode:
        torch.testing.assert_close(
            torch.stack(local_masses), torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        )


@pytest.mark.parametrize(
    "field,value,default", [("action_loss_source", 1, 0), ("action_loss_weight", 0.5, 1.0)]
)
def test_metadata_defaults_are_filled_for_every_missing_sample(collator, field, value, default):
    data_collator, _ = collator
    teacher = _sample(1)
    teacher[field] = torch.tensor(value)

    inputs = data_collator([_sample(0), teacher])["inputs"]

    torch.testing.assert_close(inputs[field], torch.tensor([default, value]))
    other = "action_loss_weight" if field == "action_loss_source" else "action_loss_source"
    assert other not in inputs


@pytest.mark.parametrize("group_mode", [False, True])
def test_action_head_forward_consumes_group_metadata(group_mode):
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from transformers.feature_extraction_utils import BatchFeature

    config = Gr00tN1d7Config(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=3,
        max_action_dim=3,
        action_horizon=4,
        max_num_embodiments=2,
        max_seq_len=16,
        use_alternate_vl_dit=False,
        use_vlln=False,
        state_dropout_prob=0.0,
        group_teacher_loss_weight=0.73,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": True,
        },
    )
    head = Gr00tN1d7ActionHead(config)
    mask = torch.ones(2, 4, 3)
    mask[1, 1:] = 0
    action_input = BatchFeature(
        data={
            "state": torch.randn(2, 1, 3),
            "action": torch.randn(2, 4, 3),
            "action_mask": mask,
            "embodiment_id": torch.zeros(2, dtype=torch.long),
        }
    )
    if group_mode:
        action_input["action_loss_source"] = torch.tensor([0, 1])
        action_input["action_loss_weight"] = torch.ones(2)
    backbone_output = BatchFeature(
        data={
            "backbone_features": torch.randn(2, 3, 32),
            "backbone_attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
    )

    output = head(backbone_output, action_input)

    if group_mode:
        sample_means = output["action_loss"].flatten(1).sum(1) / mask.flatten(1).sum(1)
        expected = 0.27 * sample_means[0] + 0.73 * sample_means[1]
    else:
        expected = output["action_loss"].sum() / (mask.sum() + 1e-6)
    torch.testing.assert_close(output["loss"], expected)


def test_cli_exposes_teacher_coefficient_separately_from_sampling_ratio():
    from gr00t.configs.finetune_config import FinetuneConfig
    import tyro

    config = tyro.cli(
        FinetuneConfig,
        args=[
            "--base-model-path",
            "fake-checkpoint",
            "--dataset-path",
            "fake-dataset",
            "--embodiment-tag",
            "libero_sim",
            "--group-teacher-loss-weight",
            "0.73",
            "--teacher-mix-ratio",
            "0.2",
        ],
    )

    assert config.group_teacher_loss_weight == 0.73
    assert config.teacher_mix_ratio == 0.2


@pytest.mark.parametrize("teacher_weight", [-0.01, 1.01, float("nan"), float("inf")])
def test_training_config_rejects_invalid_teacher_coefficient(teacher_weight):
    from gr00t.configs.base_config import get_default_config

    config = get_default_config().load_dict(
        {
            "data": {
                "datasets": [{"dataset_paths": ["fake-dataset"], "embodiment_tag": "libero_sim"}]
            }
        }
    )
    config.model.group_teacher_loss_weight = teacher_weight

    with pytest.raises(ValueError, match="group_teacher_loss_weight must be finite"):
        config.validate()


def test_checkpoint_config_accepts_explicit_teacher_coefficient_override(tmp_path):
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    checkpoint_config = Gr00tN1d7Config(group_teacher_loss_weight=0.1)
    checkpoint_config.save_pretrained(tmp_path)
    loaded = Gr00tN1d7Config.from_pretrained(
        tmp_path, group_teacher_loss_weight=0.73, local_files_only=True
    )

    assert loaded.group_teacher_loss_weight == 0.73
    assert loaded.to_filtered_dict()["group_teacher_loss_weight"] == 0.73


def test_training_checkpoint_loader_forwards_teacher_coefficient(monkeypatch, tmp_path):
    from gr00t.model.gr00t_n1d7 import setup

    requested_weight = 0.73
    pipeline = object.__new__(setup.Gr00tN1d7Pipeline)
    pipeline.config = SimpleNamespace(
        model=SimpleNamespace(
            tune_llm=False,
            tune_visual=False,
            tune_projector=True,
            tune_diffusion_model=True,
            tune_vlln=True,
            state_dropout_prob=0.0,
            backbone_trainable_params_fp32=True,
            load_bf16=False,
            group_teacher_loss_weight=requested_weight,
        ),
        training=SimpleNamespace(
            start_from_checkpoint="fake-checkpoint", skip_weight_loading=False
        ),
    )
    pipeline.transformers_loading_kwargs = {"trust_remote_code": False, "local_files_only": True}
    pipeline.save_cfg_dir = tmp_path

    def from_pretrained(checkpoint, **kwargs):
        assert checkpoint == "fake-checkpoint"
        assert kwargs["group_teacher_loss_weight"] == requested_weight
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(
            group_teacher_loss_weight=kwargs["group_teacher_loss_weight"]
        )
        model.action_head = SimpleNamespace(config=model.config)
        return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}

    monkeypatch.setattr(setup.AutoModel, "from_pretrained", from_pretrained)
    monkeypatch.setattr(setup, "get_rank", lambda: 1)

    model = pipeline._create_model()

    assert model.config.group_teacher_loss_weight == requested_weight
    assert model.action_head.config.group_teacher_loss_weight == requested_weight
