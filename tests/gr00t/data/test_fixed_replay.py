from pathlib import Path

from gr00t.configs.base_config import get_default_config
from gr00t.configs.data.data_config import SingleDatasetConfig
from gr00t.data.dataset.sharded_mixture_dataset import FIXED_BATCH_SOURCE_KEY
from gr00t.experiment.trainer import FixedSourceCountCollator
import pytest


def _fixed_config(tmp_path: Path, gt_count: int = 54, teacher_count: int = 10):
    index_path = tmp_path / "distillation_index.jsonl"
    teacher_path = tmp_path / "teacher_records.pt"
    index_path.write_text("{}\n", encoding="utf-8")
    teacher_path.write_bytes(b"teacher")

    config = get_default_config()
    config.training.global_batch_size = 64
    config.training.num_gpus = 1
    config.data.fixed_gt_samples_per_batch = gt_count
    config.data.fixed_teacher_samples_per_batch = teacher_count
    config.data.datasets = [
        SingleDatasetConfig(
            dataset_paths=["/fake/libero"],
            embodiment_tag="libero_sim",
            dataset_type="physical_embodiment",
            mix_ratio=0.85,
        ),
        SingleDatasetConfig(
            dataset_paths=["/fake/libero"],
            embodiment_tag="libero_sim",
            dataset_type="flare_distillation",
            mix_ratio=0.15,
            distillation_index_path=str(index_path),
            teacher_records_path=str(teacher_path),
        ),
    ]
    return config


def test_fixed_replay_config_accepts_54_10(tmp_path):
    _fixed_config(tmp_path).validate()


def test_fixed_replay_config_accepts_44_20(tmp_path):
    _fixed_config(tmp_path, gt_count=44, teacher_count=20).validate()


def test_fixed_replay_config_accepts_34_30(tmp_path):
    _fixed_config(tmp_path, gt_count=34, teacher_count=30).validate()


def test_fixed_replay_config_rejects_wrong_total(tmp_path):
    config = _fixed_config(tmp_path)
    config.data.fixed_gt_samples_per_batch = 53
    with pytest.raises(ValueError, match="sum to the per-device batch size"):
        config.validate()


def test_fixed_replay_config_rejects_missing_source(tmp_path):
    config = _fixed_config(tmp_path)
    config.data.datasets = config.data.datasets[:1]
    with pytest.raises(ValueError, match="exactly one physical_embodiment"):
        config.validate()


def test_fixed_replay_config_requires_both_counts(tmp_path):
    config = _fixed_config(tmp_path)
    config.data.fixed_teacher_samples_per_batch = None
    with pytest.raises(ValueError, match="must be provided together"):
        config.validate()


@pytest.mark.parametrize(
    ("gt_count", "teacher_count"),
    [(54, 10), (44, 20), (34, 30)],
)
def test_fixed_source_collator_strips_internal_marker(gt_count, teacher_count):
    received = []

    def collator(features):
        received.extend(features)
        return {"payload": [feature["payload"] for feature in features]}

    wrapper = FixedSourceCountCollator(
        collator,
        {"gt": gt_count, "teacher": teacher_count},
    )
    features = [
        {FIXED_BATCH_SOURCE_KEY: "gt", "payload": index} for index in range(gt_count)
    ] + [
        {FIXED_BATCH_SOURCE_KEY: "teacher", "payload": index}
        for index in range(teacher_count)
    ]
    result = wrapper(features)

    assert len(result["payload"]) == 64
    assert all(FIXED_BATCH_SOURCE_KEY not in feature for feature in received)


def test_fixed_source_collator_fails_closed_on_mismatch():
    wrapper = FixedSourceCountCollator(lambda features: features, {"gt": 54, "teacher": 10})
    features = [
        {FIXED_BATCH_SOURCE_KEY: "gt", "payload": index} for index in range(55)
    ] + [
        {FIXED_BATCH_SOURCE_KEY: "teacher", "payload": index} for index in range(9)
    ]

    with pytest.raises(RuntimeError, match="composition mismatch"):
        wrapper(features)

