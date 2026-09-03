#!/usr/bin/env python3
"""Verify the FLARE-VLA indexed distillation dataset through GR00T's pipeline.

The script verifies three layers:

1. Index/teacher sidecars: exact counts, formats, row mappings, shapes, finiteness.
2. Raw paired records: GT and teacher rows load the same image/state/language;
   the teacher row replaces exactly the first H raw LeRobot action steps and
   preserves the official GR00T tail.
3. Processed records: the official GR00T processor returns model-ready action
   tensors, with 16-step GT supervision and 8-step teacher supervision.

Use ``--max-pairs 0`` to run the expensive raw/processed checks for all selected
teacher pairs.  A small positive value is useful as a first smoke check.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor


EXPECTED_INDEX_FORMAT = "flare_distillation_index_v1"
EXPECTED_TEACHER_FORMAT = "flare_distillation_teacher_records_v1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_gr00t_root() -> Path:
    return repo_root() / "patches" / "isaac_gr00t" / "current_active"


def default_distillation_dir() -> Path:
    return (
        default_gr00t_root()
        / "outputs"
        / "flare_distillation_500_131"
    )


def parse_args() -> argparse.Namespace:
    defaults = default_distillation_dir()
    parser = argparse.ArgumentParser(
        description="Verify the indexed 500-GT + 131-teacher FLARE dataset."
    )
    parser.add_argument(
        "--gr00t-root",
        type=Path,
        default=default_gr00t_root(),
        help="Isaac-GR00T repository root containing the gr00t Python package.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Original LIBERO LeRobot dataset root used by the mixed index.",
    )
    parser.add_argument(
        "--base-model-path",
        type=Path,
        required=True,
        help="GR00T checkpoint directory containing processor_config.json.",
    )
    parser.add_argument(
        "--distillation-index",
        type=Path,
        default=defaults / "distillation_index.jsonl",
    )
    parser.add_argument(
        "--teacher-records",
        type=Path,
        default=defaults / "teacher_records_enriched.pt",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="libero_sim",
        help="Registered GR00T embodiment tag.",
    )
    parser.add_argument(
        "--modality-config-path",
        type=Path,
        default=None,
        help="Optional Python file registering a custom modality config.",
    )
    parser.add_argument(
        "--video-backend",
        default="torchcodec",
        help="Video backend used by LeRobotEpisodeLoader.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=631,
        help="FLARE shard size used only for the factory integration check.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=5,
        help="Number of GT/teacher pairs to decode; 0 means all 131 pairs.",
    )
    parser.add_argument("--expected-total", type=int, default=631)
    parser.add_argument("--expected-gt", type=int, default=500)
    parser.add_argument("--expected-teacher", type=int, default=131)
    parser.add_argument("--expected-teacher-horizon", type=int, default=8)
    parser.add_argument("--expected-processor-horizon", type=int, default=16)
    parser.add_argument("--expected-model-horizon", type=int, default=40)
    parser.add_argument("--expected-action-dim", type=int, default=7)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prevent Transformers from downloading files during verification.",
    )
    parser.add_argument(
        "--allow-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Clamp requested horizon-16 indices at episode boundaries. Keep this "
            "enabled for the current horizon-8 FLARE pilot."
        ),
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def add_gr00t_to_python_path(gr00t_root: Path) -> None:
    root = str(gr00t_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_modality_config(path: Path | None) -> None:
    if path is None:
        return
    path = require_file(path, "modality config")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    importlib.import_module(path.stem)
    print("loaded modality config   :", path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
                    f"Expected JSON object at {path}:{line_no}, "
                    f"got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def load_torch_dict(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict in {path}, got {type(value).__name__}")
    return value


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def assert_tensor_finite(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{label} contains NaN or Inf")


def verify_sidecars(
    rows: list[dict[str, Any]],
    teacher: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    assert_equal(len(rows), args.expected_total, "mixed record count")
    assert_equal(str(teacher.get("format")), EXPECTED_TEACHER_FORMAT, "teacher format")

    gt_rows = [row for row in rows if row.get("source") == "gt"]
    teacher_rows = [row for row in rows if row.get("source") == "teacher"]
    assert_equal(len(gt_rows), args.expected_gt, "GT record count")
    assert_equal(len(teacher_rows), args.expected_teacher, "teacher record count")

    for expected_index, row in enumerate(rows):
        assert_equal(row.get("format"), EXPECTED_INDEX_FORMAT, f"index format row {expected_index}")
        assert_equal(
            int(row.get("distillation_index", -1)),
            expected_index,
            f"distillation_index row {expected_index}",
        )

    actions = torch.as_tensor(
        teacher["teacher_actions_lerobot"], dtype=torch.float32
    )
    expected_shape = (
        args.expected_teacher,
        args.expected_teacher_horizon,
        args.expected_action_dim,
    )
    assert_equal(tuple(actions.shape), expected_shape, "teacher action shape")
    assert_tensor_finite(actions, "teacher_actions_lerobot")
    assert_equal(
        int(teacher["teacher_horizon"]),
        args.expected_teacher_horizon,
        "teacher horizon",
    )
    assert_equal(
        int(teacher["processor_action_horizon"]),
        args.expected_processor_horizon,
        "processor action horizon",
    )
    assert_equal(
        int(teacher["model_action_horizon"]),
        args.expected_model_horizon,
        "model action horizon",
    )
    assert_equal(int(teacher["action_dim"]), args.expected_action_dim, "action dim")

    teacher_pointers = sorted(int(row["teacher_row"]) for row in teacher_rows)
    assert_equal(
        teacher_pointers,
        list(range(args.expected_teacher)),
        "teacher-row coverage",
    )

    sample_ids = [str(value) for value in teacher["sample_ids"]]
    sample_indices = torch.as_tensor(teacher["sample_indices"], dtype=torch.long)
    episode_indices = torch.as_tensor(teacher["episode_indices"], dtype=torch.long)
    local_steps = torch.as_tensor(teacher["local_steps"], dtype=torch.long)
    for row in teacher_rows:
        pointer = int(row["teacher_row"])
        assert_equal(str(row["sample_id"]), sample_ids[pointer], "teacher sample_id mapping")
        assert_equal(
            int(row["sample_index"]),
            int(sample_indices[pointer]),
            "teacher sample_index mapping",
        )
        assert_equal(
            int(row["episode_index"]),
            int(episode_indices[pointer]),
            "teacher episode mapping",
        )
        assert_equal(
            int(row["local_step"]),
            int(local_steps[pointer]),
            "teacher local-step mapping",
        )


def build_config_and_processor(args: argparse.Namespace):
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag

    # Import for model/processor registration side effects, matching the official
    # experiment setup.
    import gr00t.model.gr00t_n1d7.setup  # noqa: F401

    embodiment = EmbodimentTag.resolve(args.embodiment_tag)
    embodiment_value = embodiment.value
    dataset_config = {
        "dataset_paths": [str(args.dataset_path)],
        "mix_ratio": 1.0,
        "embodiment_tag": embodiment_value,
        "dataset_type": "flare_distillation",
        "distillation_index_path": str(args.distillation_index),
        "teacher_records_path": str(args.teacher_records),
    }
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [dataset_config],
                "video_backend": args.video_backend,
                "shard_size": args.shard_size,
                "num_shards_per_epoch": 1,
                "episode_sampling_rate": 1.0,
                "allow_padding": args.allow_padding,
                "override_pretraining_statistics": True,
            }
        }
    )

    # Match launch_finetune.py. Disable stochastic augmentation because this is
    # a deterministic paired-record verification, not training.
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.state_dropout_prob = 0.0
    config.model.random_rotation_angle = None
    config.model.color_jitter_params = None
    config.model.extra_augmentation_config = None
    config.training.start_from_checkpoint = str(args.base_model_path)
    config.training.transformers_local_files_only = args.local_files_only
    config.validate()

    loading_kwargs = {
        "trust_remote_code": config.training.transformers_trust_remote_code,
        "local_files_only": config.training.transformers_local_files_only,
    }
    if config.model.model_revision is not None:
        loading_kwargs["revision"] = config.model.model_revision
    if config.training.transformers_cache_dir is not None:
        loading_kwargs["cache_dir"] = config.training.transformers_cache_dir
    if config.training.transformers_access_token is not None:
        loading_kwargs["token"] = config.training.transformers_access_token

    processor = AutoProcessor.from_pretrained(
        str(args.base_model_path),
        modality_configs=config.data.modality_configs,
        use_percentiles=config.model.use_percentiles,
        image_crop_size=config.model.image_crop_size,
        image_target_size=config.model.image_target_size,
        random_rotation_angle=config.model.random_rotation_angle,
        color_jitter_params=config.model.color_jitter_params,
        model_name=config.model.model_name,
        model_type=config.model.backbone_model_type,
        formalize_language=config.model.formalize_language,
        apply_sincos_state_encoding=config.model.apply_sincos_state_encoding,
        max_action_horizon=config.model.action_horizon,
        use_albumentations=config.model.use_albumentations_transforms,
        extra_augmentation_config=config.model.extra_augmentation_config,
        shortest_image_edge=config.model.shortest_image_edge,
        crop_fraction=config.model.crop_fraction,
        transformers_loading_kwargs=loading_kwargs,
        use_alternate_vl_dit=config.model.use_alternate_vl_dit,
        use_relative_action=config.model.use_relative_action,
        exclude_state=config.model.exclude_state,
        state_dropout_prob=config.model.state_dropout_prob,
        use_mean_std=config.model.use_mean_std,
        **loading_kwargs,
    )
    processor.eval()
    return config, processor, embodiment


def values_equal(left: Any, right: Any, *, atol: float = 0.0) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        try:
            left_tensor = torch.as_tensor(left)
            right_tensor = torch.as_tensor(right)
        except (TypeError, ValueError):
            return False
        if left_tensor.shape != right_tensor.shape:
            return False
        if left_tensor.dtype.is_floating_point or right_tensor.dtype.is_floating_point:
            return bool(torch.allclose(left_tensor, right_tensor, rtol=0.0, atol=atol))
        return bool(torch.equal(left_tensor, right_tensor))

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            left_array = np.asarray(left)
            right_array = np.asarray(right)
        except (TypeError, ValueError):
            return False
        if left_array.shape != right_array.shape:
            return False
        if np.issubdtype(left_array.dtype, np.floating) or np.issubdtype(
            right_array.dtype, np.floating
        ):
            return bool(np.allclose(left_array, right_array, rtol=0.0, atol=atol))
        return bool(np.array_equal(left_array, right_array))

    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(values_equal(left[key], right[key], atol=atol) for key in left)

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(values_equal(a, b, atol=atol) for a, b in zip(left, right))

    if isinstance(left, Image.Image) or isinstance(right, Image.Image):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except (TypeError, ValueError):
            return False

    # Other array-like image containers.
    if hasattr(left, "__array__") or hasattr(right, "__array__"):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except (TypeError, ValueError):
            pass

    return left == right


def require_values_equal(left: Any, right: Any, label: str, *, atol: float = 0.0) -> None:
    if not values_equal(left, right, atol=atol):
        raise AssertionError(f"Paired values differ: {label}")


def concatenate_actions(step, action_keys: list[str]) -> np.ndarray:
    pieces = []
    for key in action_keys:
        if key not in step.actions:
            raise KeyError(f"Missing raw action key {key!r}")
        value = np.asarray(step.actions[key], dtype=np.float32)
        if value.ndim != 2:
            raise AssertionError(f"Raw action {key!r} has invalid shape {value.shape}")
        pieces.append(value)
    return np.concatenate(pieces, axis=-1)


def verify_raw_pair(dataset, gt_index: int, teacher_index: int) -> None:
    teacher_record = dataset.get_index_record(teacher_index)
    teacher_row = int(teacher_record["teacher_row"])

    gt_step = dataset.get_raw_record(gt_index, apply_teacher=True)
    teacher_official = dataset.get_raw_record(teacher_index, apply_teacher=False)
    teacher_replaced = dataset.get_raw_record(teacher_index, apply_teacher=True)

    require_values_equal(gt_step.text, teacher_official.text, "raw text GT/teacher")
    require_values_equal(gt_step.text, teacher_replaced.text, "raw text after replacement")
    require_values_equal(gt_step.states, teacher_official.states, "raw states GT/teacher")
    require_values_equal(gt_step.states, teacher_replaced.states, "raw states after replacement")
    require_values_equal(gt_step.images, teacher_official.images, "raw images GT/teacher")
    require_values_equal(gt_step.images, teacher_replaced.images, "raw images after replacement")
    require_values_equal(gt_step.masks, teacher_official.masks, "raw masks GT/teacher")
    require_values_equal(gt_step.masks, teacher_replaced.masks, "raw masks after replacement")
    require_values_equal(gt_step.actions, teacher_official.actions, "official raw GT actions")

    official_actions = concatenate_actions(gt_step, dataset.action_keys)
    replaced_actions = concatenate_actions(teacher_replaced, dataset.action_keys)
    saved_teacher = dataset.teacher_actions[teacher_row].numpy()
    horizon = dataset.teacher_horizon

    if replaced_actions.shape[1] != dataset.teacher_action_dim:
        raise AssertionError(
            "Concatenated raw action dim mismatch: "
            f"actual={replaced_actions.shape[1]}, expected={dataset.teacher_action_dim}"
        )
    if not np.allclose(
        replaced_actions[:horizon], saved_teacher, rtol=0.0, atol=1e-7
    ):
        max_error = float(np.max(np.abs(replaced_actions[:horizon] - saved_teacher)))
        raise AssertionError(
            f"Teacher prefix does not match saved raw teacher action; max error={max_error}"
        )
    if not np.allclose(
        replaced_actions[horizon:], official_actions[horizon:], rtol=0.0, atol=0.0
    ):
        raise AssertionError("Teacher replacement changed the official action tail")


def verify_processed_pair(
    dataset,
    gt_index: int,
    teacher_index: int,
    args: argparse.Namespace,
) -> tuple[int, int]:
    gt = dataset.get_processed_record(gt_index)
    teacher = dataset.get_processed_record(teacher_index)

    required_keys = {"state", "action", "action_mask", "embodiment_id", "vlm_content"}
    missing_gt = required_keys - set(gt)
    missing_teacher = required_keys - set(teacher)
    if missing_gt or missing_teacher:
        raise AssertionError(
            f"Missing processed keys: GT={sorted(missing_gt)}, teacher={sorted(missing_teacher)}"
        )

    for label, processed in [("GT", gt), ("teacher", teacher)]:
        action = processed["action"]
        action_mask = processed["action_mask"]
        if not torch.is_tensor(action) or not torch.is_tensor(action_mask):
            raise TypeError(f"{label} action/action_mask are not tensors")
        assert_equal(tuple(action.shape), tuple(action_mask.shape), f"{label} action-mask shape")
        assert_equal(
            int(action.shape[0]),
            args.expected_model_horizon,
            f"{label} model action horizon",
        )
        assert_tensor_finite(action, f"{label} processed action")
        assert_tensor_finite(action_mask, f"{label} processed action mask")

    require_values_equal(gt["state"], teacher["state"], "processed state", atol=1e-7)
    require_values_equal(gt["embodiment_id"], teacher["embodiment_id"], "embodiment id")
    require_values_equal(gt["vlm_content"], teacher["vlm_content"], "processed VLM content")

    gt_mask_sum = int(torch.round(gt["action_mask"].sum()).item())
    teacher_mask_sum = int(torch.round(teacher["action_mask"].sum()).item())
    expected_gt_mask_sum = args.expected_processor_horizon * args.expected_action_dim
    expected_teacher_mask_sum = args.expected_teacher_horizon * args.expected_action_dim
    assert_equal(gt_mask_sum, expected_gt_mask_sum, "GT active action-mask elements")
    assert_equal(
        teacher_mask_sum,
        expected_teacher_mask_sum,
        "teacher active action-mask elements",
    )

    if bool((teacher["action_mask"][args.expected_teacher_horizon :] != 0).any()):
        raise AssertionError("Teacher action_mask contains non-zero values after horizon 8")
    if bool((teacher["action"][args.expected_teacher_horizon :] != 0).any()):
        raise AssertionError(
            "Teacher processed action contains non-zero values after horizon 8; "
            "this could leak the official GT tail into the diffusion transformer"
        )
    return gt_mask_sum, teacher_mask_sum


def select_teacher_pairs(dataset, max_pairs: int) -> list[tuple[int, int]]:
    gt_by_sample: dict[str, int] = {}
    teacher_pairs: list[tuple[int, int]] = []
    teacher_indices: list[int] = []

    for index, row in enumerate(dataset.index_rows):
        sample_id = str(row["sample_id"])
        if row["source"] == "gt":
            gt_by_sample[sample_id] = index
        elif row["source"] == "teacher":
            teacher_indices.append(index)

    for teacher_index in teacher_indices:
        sample_id = str(dataset.index_rows[teacher_index]["sample_id"])
        if sample_id not in gt_by_sample:
            raise AssertionError(f"Teacher sample {sample_id} has no paired GT record")
        teacher_pairs.append((gt_by_sample[sample_id], teacher_index))

    if max_pairs < 0:
        raise ValueError(f"max_pairs must be >= 0, got {max_pairs}")
    if max_pairs == 0 or max_pairs >= len(teacher_pairs):
        return teacher_pairs

    # Evenly cover the teacher list instead of checking only its first rows.
    selected = np.linspace(0, len(teacher_pairs) - 1, num=max_pairs, dtype=int)
    return [teacher_pairs[int(index)] for index in selected]


def main() -> None:
    args = parse_args()
    args.gr00t_root = require_dir(args.gr00t_root, "GR00T root")
    args.dataset_path = require_dir(args.dataset_path, "LeRobot dataset")
    args.base_model_path = require_dir(args.base_model_path, "base model checkpoint")
    args.distillation_index = require_file(args.distillation_index, "distillation index")
    args.teacher_records = require_file(args.teacher_records, "teacher records")

    add_gr00t_to_python_path(args.gr00t_root)
    load_modality_config(args.modality_config_path)

    rows = load_jsonl(args.distillation_index)
    teacher = load_torch_dict(args.teacher_records)
    verify_sidecars(rows, teacher, args)

    config, processor, _ = build_config_and_processor(args)

    from gr00t.data.dataset.factory import DatasetFactory
    from gr00t.data.dataset.flare_distillation_dataset import FlareDistillationDataset

    mixture, eval_dataset = DatasetFactory(config).build(processor=processor)
    if eval_dataset is not None:
        raise AssertionError("FLARE training factory unexpectedly returned an eval dataset")
    if len(mixture.datasets) != 1:
        raise AssertionError(f"Expected one underlying dataset, got {len(mixture.datasets)}")
    dataset = mixture.datasets[0]
    if not isinstance(dataset, FlareDistillationDataset):
        raise AssertionError(
            f"DatasetFactory returned {type(dataset).__name__}, expected FlareDistillationDataset"
        )

    assert_equal(dataset.num_records, args.expected_total, "factory dataset record count")
    assert_equal(dataset.source_counts.get("gt", 0), args.expected_gt, "factory GT count")
    assert_equal(
        dataset.source_counts.get("teacher", 0),
        args.expected_teacher,
        "factory teacher count",
    )
    assert_equal(dataset.teacher_horizon, args.expected_teacher_horizon, "factory teacher horizon")
    assert_equal(
        dataset.processor_action_horizon,
        args.expected_processor_horizon,
        "factory processor horizon",
    )

    pairs = select_teacher_pairs(dataset, args.max_pairs)
    if not pairs:
        raise AssertionError("No GT/teacher pairs selected for verification")

    observed_gt_mask_sums: set[int] = set()
    observed_teacher_mask_sums: set[int] = set()
    print("========== FLARE dataset verification ==========")
    print("mixed records            :", dataset.num_records)
    print("source counts            :", dataset.source_counts)
    print("teacher tensor shape     :", tuple(dataset.teacher_actions.shape))
    print("dataset shards           :", len(dataset))
    print("pairs decoded            :", len(pairs))
    print("allow padding            :", args.allow_padding)

    for pair_number, (gt_index, teacher_index) in enumerate(pairs, start=1):
        verify_raw_pair(dataset, gt_index, teacher_index)
        gt_mask_sum, teacher_mask_sum = verify_processed_pair(
            dataset, gt_index, teacher_index, args
        )
        observed_gt_mask_sums.add(gt_mask_sum)
        observed_teacher_mask_sums.add(teacher_mask_sum)
        teacher_record = dataset.get_index_record(teacher_index)
        print(
            f"pair {pair_number:03d}/{len(pairs):03d} PASS: "
            f"sample_id={teacher_record['sample_id']} "
            f"teacher_row={teacher_record['teacher_row']} "
            f"episode={teacher_record['episode_index']} "
            f"step={teacher_record['local_step']}"
        )

    expected_gt_mask = args.expected_processor_horizon * args.expected_action_dim
    expected_teacher_mask = args.expected_teacher_horizon * args.expected_action_dim
    assert_equal(observed_gt_mask_sums, {expected_gt_mask}, "observed GT mask sums")
    assert_equal(
        observed_teacher_mask_sums,
        {expected_teacher_mask},
        "observed teacher mask sums",
    )

    teacher_fraction = args.expected_teacher / args.expected_total
    if not math.isclose(teacher_fraction, 131 / 631, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Unexpected teacher fraction for the current protocol")

    print("processed action horizon :", args.expected_model_horizon)
    print("GT active mask elements  :", expected_gt_mask)
    print("teacher active elements  :", expected_teacher_mask)
    print("teacher fraction         :", f"{teacher_fraction:.6f}")
    print("status                   : PASS")


if __name__ == "__main__":
    main()
