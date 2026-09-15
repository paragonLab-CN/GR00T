"""Strict resolved-config validation for controlled FLARE experiments."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path

from omegaconf import OmegaConf
import yaml


def _collect_differences(reference, candidate, path=""):
    if isinstance(reference, dict) and isinstance(candidate, dict):
        differences = []
        for key in sorted(set(reference) | set(candidate)):
            child_path = f"{path}.{key}" if path else key
            if key not in reference:
                differences.append(f"{child_path}: unexpected {candidate[key]!r}")
            elif key not in candidate:
                differences.append(f"{child_path}: missing (expected {reference[key]!r})")
            else:
                differences.extend(
                    _collect_differences(reference[key], candidate[key], child_path)
                )
        return differences
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            return [f"{path}: list length {len(candidate)} != {len(reference)}"]
        differences = []
        for index, (ref_item, candidate_item) in enumerate(zip(reference, candidate)):
            differences.extend(
                _collect_differences(ref_item, candidate_item, f"{path}[{index}]")
            )
        return differences
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        if math.isclose(float(reference), float(candidate), rel_tol=0.0, abs_tol=1e-12):
            return []
    elif reference == candidate:
        return []
    return [f"{path}: {candidate!r} != {reference!r}"]


def validate_reference_config(config, flare_sidecars) -> None:
    """Fail unless only the explicitly allowed experiment fields differ.

    Enabled by ``FLARE_REFERENCE_CONFIG_PATH`` so normal finetuning is unchanged.
    """
    reference_path_value = os.environ.get("FLARE_REFERENCE_CONFIG_PATH")
    if not reference_path_value:
        return

    reference_path = Path(reference_path_value).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference resolved config does not exist: {reference_path}")
    with reference_path.open("r", encoding="utf-8") as handle:
        reference = yaml.safe_load(handle)
    if not isinstance(reference, dict):
        raise TypeError(f"Reference config must be a YAML mapping: {reference_path}")

    candidate = OmegaConf.to_container(
        OmegaConf.create(config.__dict__), resolve=True, enum_to_str=True
    )
    candidate["max_steps"] = config.training.max_steps
    candidate["save_steps"] = config.training.save_steps

    expected_color = json.loads(os.environ["FLARE_EXPECTED_COLOR_JITTER_PARAMS"])
    expected_extra = json.loads(os.environ["FLARE_EXPECTED_EXTRA_AUGMENTATION_CONFIG"])
    if candidate["model"]["color_jitter_params"] != expected_color:
        raise RuntimeError(
            "Resolved ColorJitter does not match the experiment contract: "
            f"actual={candidate['model']['color_jitter_params']}, expected={expected_color}"
        )
    if candidate["model"]["extra_augmentation_config"] != expected_extra:
        raise RuntimeError(
            "Resolved blur config does not match the experiment contract: "
            f"actual={candidate['model']['extra_augmentation_config']}, expected={expected_extra}"
        )

    reference_by_type = {
        item["dataset_type"]: item for item in reference["data"]["datasets"]
    }
    candidate_by_type = {
        item["dataset_type"]: item for item in candidate["data"]["datasets"]
    }
    expected_types = {"physical_embodiment", "flare_distillation"}
    if set(reference_by_type) != expected_types or set(candidate_by_type) != expected_types:
        raise RuntimeError(
            "Reference comparison requires exactly physical_embodiment and "
            "flare_distillation datasets"
        )
    if flare_sidecars is None or flare_sidecars.summary_path is None:
        raise RuntimeError("Sampler compensation validation requires a validated FLARE summary")
    with flare_sidecars.summary_path.open("r", encoding="utf-8") as handle:
        sidecar_summary = json.load(handle)
    teacher_horizon = int(sidecar_summary["gr00t_interface"]["teacher_horizon"])
    gt_horizon = len(config.data.modality_configs["libero_sim"]["action"].delta_indices)
    if teacher_horizon != 8 or gt_horizon != 16:
        raise RuntimeError(
            "Controlled FLARE experiments require teacher horizon 8 and GT horizon 16; "
            f"got teacher={teacher_horizon}, GT={gt_horizon}"
        )

    gt_ratio = float(candidate_by_type["physical_embodiment"]["mix_ratio"])
    teacher_ratio = float(candidate_by_type["flare_distillation"]["mix_ratio"])
    reference_teacher = float(reference_by_type["flare_distillation"]["mix_ratio"])
    reference_total = sum(float(item["mix_ratio"]) for item in reference_by_type.values())
    reference_teacher /= reference_total

    fixed_gt = config.data.fixed_gt_samples_per_batch
    fixed_teacher = config.data.fixed_teacher_samples_per_batch
    if fixed_gt is not None and fixed_teacher is not None:
        fixed_total = fixed_gt + fixed_teacher
        effective_teacher = fixed_teacher / fixed_total
        nearest_integer_tolerance = 0.5 / fixed_total
        if not math.isclose(
            effective_teacher,
            reference_teacher,
            rel_tol=0.0,
            abs_tol=nearest_integer_tolerance,
        ):
            raise RuntimeError(
                "Fixed teacher count is not the nearest-integer implementation "
                f"of the v1 ratio: effective_teacher={effective_teacher:.16f}, "
                f"expected={reference_teacher:.16f}"
            )
        print(
            "Fixed replay composition verified: "
            f"GT/teacher={fixed_gt}/{fixed_teacher}, "
            f"effective GT/teacher={1.0 - effective_teacher:.3%}/{effective_teacher:.3%}, "
            f"horizons={gt_horizon}/{teacher_horizon}"
        )
    else:
        gt_weight = gt_ratio / gt_horizon
        teacher_weight = teacher_ratio / teacher_horizon
        effective_teacher = teacher_weight / (gt_weight + teacher_weight)
        if not math.isclose(effective_teacher, reference_teacher, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                "Supervised-step-corrected mixture differs from the v1 reference: "
                f"effective_teacher={effective_teacher:.16f}, expected={reference_teacher:.16f}"
            )
        print(
            "Sampler compatibility compensation verified: "
            f"raw GT/teacher={gt_ratio:.16f}/{teacher_ratio:.16f}, "
            f"horizons={gt_horizon}/{teacher_horizon}, "
            f"effective GT/teacher={1.0 - effective_teacher:.2%}/{effective_teacher:.2%}"
        )

    normalized = deepcopy(candidate)
    normalized["data"].pop("fixed_gt_samples_per_batch", None)
    normalized["data"].pop("fixed_teacher_samples_per_batch", None)
    normalized["training"]["output_dir"] = reference["training"]["output_dir"]
    normalized["training"]["experiment_name"] = reference["training"]["experiment_name"]
    expected_save_only = os.environ.get("FLARE_EXPECTED_SAVE_ONLY_MODEL")
    if expected_save_only is not None:
        if expected_save_only not in {"0", "1"}:
            raise ValueError("FLARE_EXPECTED_SAVE_ONLY_MODEL must be 0 or 1")
        expected_value = expected_save_only == "1"
        if candidate["training"]["save_only_model"] != expected_value:
            raise RuntimeError(
                "Resolved save_only_model does not match the persistence contract: "
                f"actual={candidate['training']['save_only_model']}, "
                f"expected={expected_value}"
            )
    normalized["training"]["save_only_model"] = reference["training"].get(
        "save_only_model", False
    )
    normalized["model"]["color_jitter_params"] = reference["model"]["color_jitter_params"]
    normalized["model"]["extra_augmentation_config"] = reference["model"].get(
        "extra_augmentation_config"
    )
    for item in normalized["data"]["datasets"]:
        item["mix_ratio"] = reference_by_type[item["dataset_type"]]["mix_ratio"]

    differences = _collect_differences(reference, normalized)
    if differences:
        preview = "\n".join(f"  - {item}" for item in differences[:30])
        raise RuntimeError(
            "Resolved config differs from the v1 reference outside the allowlist:\n"
            f"{preview}"
        )
    print(
        "Resolved-config reference comparison passed; only experiment identity, "
        "planned image augmentation, sampler compensation, and checkpoint "
        "persistence may differ."
    )
