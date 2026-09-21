#!/usr/bin/env python3
"""Audit FLARE-VLA teachers for Filtered Direct Augmentation.

LIBERO-PLUS variant: fork of the shared v3 audit (see
scripts/libero_90/04_scoring/06_audit_teacher_quality.py) with additional
filters: per-component (xyz/rot) action-distance, gripper inversion,
relative-jerk exemption, and late-window MSE instead of the single terminal step.

This script implements the FLARE-Filtered Direct Augmentation policy:

1. Do not use per-sample confidence weights as the main training signal.
2. Do not hard-filter by relative MSE margin.
3. Do not hard-filter by candidate diversity.
4. Remove only teachers with unstable score agreement or obvious quality outliers:
   per-component action-distance outlier, gripper inversion, relative-jerk
   outlier, late-window MSE outlier, or oracle-gap outlier.
5. Save the selected teacher actions as direct extra training samples in
   LeRobot convention.

Outputs:
  * teacher_quality_audit.csv / .parquet      per-sample metrics and flags
  * teacher_quality_task_summary.csv          per-task aggregation
  * teacher_quality_summary.json              global thresholds and counts
  * teacher_quality_selected.jsonl            selected direct-augmentation rows
  * teacher_quality_filtered_records.pt       selected teacher tensors
  * teacher_quality_arrays.npz                compact arrays for plotting

The ``confidence_weight`` key is kept as a backward-compatible alias and is set
to 1.0 for selected teachers and 0.0 for rejected teachers. It is not a learned
or quality-dependent weight in this direct-augmentation policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


FORMAT_SUMMARY = "flare_teacher_quality_summary_v3_direct_augmentation"
FORMAT_RECORDS = "flare_teacher_quality_filtered_records_v3_direct_augmentation"
POLICY_NAME = "flare_filtered_direct_augmentation"
SCORING_RUN_FORMAT = "flare_lewm_scoring_run_summary_v1"
VALIDATION_SUMMARY_FORMAT = "flare_validation_summary_v1"
VALIDATION_RECORDS_FORMAT = "flare_validation_teacher_records_v1"
MANIFEST_FORMAT = "libero_plus_sparse_manifest_v1"
EXPECTED_MANIFEST_SAMPLES = 487_880
EXPECTED_SUITE_COUNTS = {
    "libero_spatial": 109_403,
    "libero_object": 135_456,
    "libero_goal": 98_964,
    "libero_10": 144_057,
}
SUITES = set(EXPECTED_SUITE_COUNTS)

SCRIPT_PATH = Path(__file__).resolve()
FLARE_ROOT = SCRIPT_PATH.parents[2]
ISAAC_ROOT = FLARE_ROOT.parent
DEFAULT_MANIFEST = (
    FLARE_ROOT / "outputs/libero_plus/manifests/libero_plus_sparse_manifest.jsonl"
)
DEFAULT_SCORING_DIR = FLARE_ROOT / "outputs/libero_plus/teacher_scoring"
DEFAULT_OUTPUT_DIR = FLARE_ROOT / "outputs/libero_plus/teacher_scoring_v2"
DEFAULT_CANDIDATE_DIR = ISAAC_ROOT / "outputs/libero_plus/sparse_candidates/candidates"


# -----------------------------------------------------------------------------
# Generic I/O helpers
# -----------------------------------------------------------------------------


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_jsonl_and_count(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                count += 1
    return digest.hexdigest(), count


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    keys = [
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "median",
        "p70",
        "p75",
        "p80",
        "p90",
        "p95",
        "p99",
        "max",
    ]
    if arr.size == 0:
        return {key: math.nan for key in keys}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p70": float(np.quantile(arr, 0.70)),
        "p75": float(np.quantile(arr, 0.75)),
        "p80": float(np.quantile(arr, 0.80)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def finite_quantile(values: np.ndarray | pd.Series, q: float, name: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise RuntimeError(f"Cannot compute {name} quantile; all values are non-finite")
    return float(np.quantile(arr, q))


def write_parquet_best_effort(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    try:
        df.to_parquet(path, index=False, compression="snappy")
        return {"written": True, "path": str(path), "error": None}
    except Exception as exc:
        return {"written": False, "path": str(path), "error": repr(exc)}


def validate_scoring_run_header(
    manifest_path: Path,
    scoring_run_path: Path,
    expected_samples: int,
) -> tuple[str, int, dict[str, Any]]:
    manifest_sha256, manifest_samples = sha256_jsonl_and_count(manifest_path)
    scoring_run = load_json(scoring_run_path)

    if scoring_run.get("format") != SCORING_RUN_FORMAT:
        raise RuntimeError(f"Unexpected scoring run format: {scoring_run.get('format')!r}")
    if scoring_run.get("status") != "PASS":
        raise RuntimeError("Scoring run is not marked PASS")
    if scoring_run.get("manifest_format") != MANIFEST_FORMAT:
        raise RuntimeError("Scoring run does not use the full LIBERO-Plus manifest")
    if Path(str(scoring_run.get("manifest", ""))).resolve() != manifest_path:
        raise RuntimeError("Scoring run points to a different manifest")
    if scoring_run.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("Scoring run manifest SHA-256 mismatch")
    if int(scoring_run.get("manifest_samples", -1)) != manifest_samples:
        raise RuntimeError("Scoring run manifest sample count mismatch")
    if scoring_run.get("selection_is_full_manifest") is not True:
        raise RuntimeError("Scoring run is a partial MAX_SAMPLES run")
    if int(scoring_run.get("selected", -1)) != manifest_samples:
        raise RuntimeError("Scoring run did not select every manifest sample")
    if int(scoring_run.get("scored_parts", -1)) != manifest_samples:
        raise RuntimeError("Scoring run does not contain every score part")
    if int(scoring_run.get("failed", -1)) != 0:
        raise RuntimeError("Scoring run contains failed samples")
    if expected_samples > 0 and manifest_samples != expected_samples:
        raise RuntimeError(
            f"Manifest contains {manifest_samples} samples, expected {expected_samples}"
        )
    return manifest_sha256, manifest_samples, scoring_run


def validate_scoring_contract(
    manifest_path: Path,
    scoring_run_path: Path,
    scores_path: Path,
    teacher_records_path: Path,
    stepwise_path: Path,
    validation_summary_path: Path,
    scores: pd.DataFrame,
    records: dict[str, Any],
    stepwise: Any,
    validation_summary: dict[str, Any],
    manifest_sha256: str,
    manifest_samples: int,
    scoring_run: dict[str, Any],
) -> dict[str, Any]:
    if validation_summary.get("format") != VALIDATION_SUMMARY_FORMAT:
        raise RuntimeError("Unexpected validation summary format")
    if validation_summary.get("status") != "PASS":
        raise RuntimeError("Validation summary is not marked PASS")
    if validation_summary.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("Validation summary manifest SHA-256 mismatch")
    if int(validation_summary.get("requested_samples", -1)) != manifest_samples:
        raise RuntimeError("Validation summary requested-sample count mismatch")
    if int(validation_summary.get("scored_samples", -1)) != manifest_samples:
        raise RuntimeError("Validation summary scored-sample count mismatch")
    if validation_summary.get("missing_samples"):
        raise RuntimeError("Validation summary contains missing samples")

    checkpoint_sha256 = str(scoring_run.get("checkpoint_sha256", ""))
    gt_h5_sha256 = str(scoring_run.get("gt_h5_sha256", ""))
    for key, expected in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("gt_h5_sha256", gt_h5_sha256),
    ):
        if len(expected) != 64:
            raise RuntimeError(f"Scoring run has invalid {key}")
        if validation_summary.get(key) != expected:
            raise RuntimeError(f"Validation summary {key} mismatch")
        if records.get(key) != expected:
            raise RuntimeError(f"Teacher records {key} mismatch")

    if records.get("format") != VALIDATION_RECORDS_FORMAT:
        raise RuntimeError(f"Unexpected validation teacher format: {records.get('format')!r}")
    if records.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("Teacher records manifest SHA-256 mismatch")

    required_columns = {
        "sample_id",
        "sample_index",
        "suite",
        "gt_latent_key",
        "task_index",
        "task_text",
        "episode_index",
        "local_step",
    }
    missing_columns = required_columns - set(scores.columns)
    if missing_columns:
        raise RuntimeError(f"Scores CSV is missing columns: {sorted(missing_columns)}")

    n = len(scores)
    if n != manifest_samples:
        raise RuntimeError(f"Scores CSV has {n} rows, expected {manifest_samples}")
    sample_ids = scores["sample_id"].astype(str).tolist()
    if len(set(sample_ids)) != n:
        raise RuntimeError("Scores CSV contains duplicate sample_id values")
    sample_indices = scores["sample_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_indices, np.arange(n, dtype=np.int64)):
        raise RuntimeError("Scores CSV sample_index is not contiguous manifest order")

    suites = scores["suite"].astype(str)
    suite_counts = Counter(suites)
    if manifest_samples == EXPECTED_MANIFEST_SAMPLES:
        if dict(suite_counts) != EXPECTED_SUITE_COUNTS:
            raise RuntimeError(
                f"Scores CSV suite counts are invalid: {dict(suite_counts)}"
            )
    elif set(suite_counts) != SUITES:
        raise RuntimeError(f"Scores CSV suite coverage is invalid: {dict(suite_counts)}")
    if not np.array_equal(
        suites.to_numpy(),
        scores["gt_latent_key"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Scores CSV suite/gt_latent_key mismatch")
    task_count = int(scores["task_index"].nunique())
    episode_count = int(scores["episode_index"].nunique())
    if task_count != int(validation_summary.get("tasks", -1)):
        raise RuntimeError("Scores CSV task count does not match validation summary")
    if episode_count != int(validation_summary.get("episodes", -1)):
        raise RuntimeError("Scores CSV episode count does not match validation summary")
    if (
        manifest_samples == EXPECTED_MANIFEST_SAMPLES
        and set(scores["task_index"].astype(int)) != set(range(40))
    ):
        raise RuntimeError("Scores CSV does not cover canonical task indices 0..39")

    record_ids = [str(value) for value in records.get("sample_ids", [])]
    if record_ids != sample_ids:
        raise RuntimeError("Teacher records sample order does not match scores CSV")
    record_indices = torch.as_tensor(records.get("sample_indices")).numpy()
    if not np.array_equal(record_indices, sample_indices):
        raise RuntimeError("Teacher records sample indices do not match scores CSV")
    record_suites = [str(value) for value in records.get("suites", [])]
    if record_suites != suites.tolist():
        raise RuntimeError("Teacher records suite order does not match scores CSV")
    record_tasks = torch.as_tensor(records.get("task_indices")).numpy()
    if not np.array_equal(
        record_tasks,
        scores["task_index"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Teacher records task indices do not match scores CSV")

    candidate_k = int(validation_summary.get("candidate_k", -1))
    horizon = int(validation_summary.get("future_horizon", -1))
    if (candidate_k, horizon) != (4, 8):
        raise RuntimeError(f"Expected K=4,H=8, got K={candidate_k},H={horizon}")
    tensor_shapes = {
        "teacher_actions_cosine_lerobot": (n, horizon, 7),
        "teacher_actions_mse_lerobot": (n, horizon, 7),
        "gt_actions_lerobot": (n, horizon, 7),
        "cosine_scores": (n, candidate_k),
        "mse_scores": (n, candidate_k),
    }
    for key, expected_shape in tensor_shapes.items():
        value = torch.as_tensor(records.get(key))
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"Teacher records {key} shape={tuple(value.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Teacher records {key} contains NaN/Inf")

    if list(stepwise["sample_ids"].astype(str)) != sample_ids:
        raise RuntimeError("Stepwise metrics sample order does not match scores CSV")
    stepwise_shapes = {
        "cosine_per_step": (n, candidate_k, horizon),
        "mse_per_step": (n, candidate_k, horizon),
        "oracle_cosine_per_step": (n, horizon),
        "oracle_mse_per_step": (n, horizon),
        "teacher_cosine_per_step": (n, horizon),
        "teacher_mse_per_step": (n, horizon),
        "teacher_by_cosine": (n,),
        "teacher_by_mse": (n,),
    }
    for key, expected_shape in stepwise_shapes.items():
        value = stepwise[key]
        if value.shape != expected_shape:
            raise RuntimeError(
                f"Stepwise {key} shape={value.shape}, expected {expected_shape}"
            )
        if not np.isfinite(value).all():
            raise RuntimeError(f"Stepwise {key} contains NaN/Inf")

    input_paths = {
        "manifest": manifest_path,
        "scoring_run_summary": scoring_run_path,
        "scores_csv": scores_path,
        "teacher_records": teacher_records_path,
        "stepwise_npz": stepwise_path,
        "validation_summary": validation_summary_path,
    }
    return {
        "manifest_sha256": manifest_sha256,
        "manifest_samples": manifest_samples,
        "checkpoint_sha256": checkpoint_sha256,
        "gt_h5_sha256": gt_h5_sha256,
        "suite_counts": dict(sorted(suite_counts.items())),
        "input_sha256": {
            key: sha256_file(value) for key, value in input_paths.items()
        },
    }


# -----------------------------------------------------------------------------
# Action metrics
# -----------------------------------------------------------------------------


def pairwise_metrics(actions: np.ndarray) -> dict[str, float]:
    """Pairwise candidate diversity for one sample.

    actions: [K,H,7] in LeRobot convention. Diversity is diagnostic only under
    FLARE-Filtered Direct Augmentation, not a hard filter.
    """
    k = actions.shape[0]
    if k < 2:
        return {}
    all_l2, xyz_l2, rot_l2, grip_abs = [], [], [], []
    for i in range(k):
        for j in range(i + 1, k):
            diff = actions[i] - actions[j]
            all_l2.append(float(np.linalg.norm(diff, axis=-1).mean()))
            xyz_l2.append(float(np.linalg.norm(diff[:, 0:3], axis=-1).mean()))
            rot_l2.append(float(np.linalg.norm(diff[:, 3:6], axis=-1).mean()))
            grip_abs.append(float(np.abs(diff[:, 6]).mean()))
    arr_all = np.asarray(all_l2)
    return {
        "candidate_pairwise_action_l2_mean": float(arr_all.mean()),
        "candidate_pairwise_action_l2_min": float(arr_all.min()),
        "candidate_pairwise_action_l2_max": float(arr_all.max()),
        "candidate_pairwise_xyz_l2_mean": float(np.mean(xyz_l2)),
        "candidate_pairwise_rot_l2_mean": float(np.mean(rot_l2)),
        "candidate_pairwise_gripper_abs_mean": float(np.mean(grip_abs)),
        "candidate_action_std_mean": float(actions.std(axis=0).mean()),
        "candidate_xyz_std_mean": float(actions[:, :, 0:3].std(axis=0).mean()),
        "candidate_rot_std_mean": float(actions[:, :, 3:6].std(axis=0).mean()),
        "candidate_gripper_std_mean": float(actions[:, :, 6].std(axis=0).mean()),
    }


def action_distance_metrics(teacher: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    diff = teacher - gt
    teacher_gripper_open = teacher[:, 6] >= 0.5
    gt_gripper_open = gt[:, 6] >= 0.5
    return {
        "teacher_gt_action_mse": float(np.mean(diff**2)),
        "teacher_gt_action_l1": float(np.mean(np.abs(diff))),
        "teacher_gt_xyz_mse": float(np.mean(diff[:, 0:3] ** 2)),
        "teacher_gt_rot_mse": float(np.mean(diff[:, 3:6] ** 2)),
        "teacher_gt_gripper_mse": float(np.mean(diff[:, 6] ** 2)),
        "teacher_gt_gripper_l1": float(np.mean(np.abs(diff[:, 6]))),
        "teacher_gt_gripper_match_rate": float(np.mean(teacher_gripper_open == gt_gripper_open)),
        "teacher_gt_gripper_mismatch_steps": int(np.sum(teacher_gripper_open != gt_gripper_open)),
    }


def smoothness_metrics(action: np.ndarray, prefix: str) -> dict[str, float]:
    """Velocity/acceleration/jerk norms for an [H,7] action chunk."""
    out: dict[str, float] = {}

    def add_stats(name: str, values: np.ndarray) -> None:
        if values.size == 0:
            out[f"{prefix}_{name}_mean"] = math.nan
            out[f"{prefix}_{name}_max"] = math.nan
        else:
            out[f"{prefix}_{name}_mean"] = float(values.mean())
            out[f"{prefix}_{name}_max"] = float(values.max())

    vel = np.diff(action, n=1, axis=0)
    acc = np.diff(action, n=2, axis=0)
    jerk = np.diff(action, n=3, axis=0)

    for name, arr in [("velocity", vel), ("acceleration", acc), ("jerk", jerk)]:
        add_stats(f"{name}_l2", np.linalg.norm(arr, axis=-1))
        add_stats(f"{name}_xyz_l2", np.linalg.norm(arr[:, 0:3], axis=-1) if arr.size else arr)
        add_stats(f"{name}_rot_l2", np.linalg.norm(arr[:, 3:6], axis=-1) if arr.size else arr)
        add_stats(f"{name}_gripper_abs", np.abs(arr[:, 6]) if arr.size else arr)
    return out


def load_candidate_actions(
    candidate_dir: Path | None,
    sample_ids: list[str],
    candidate_k: int,
    horizon: int,
) -> tuple[np.ndarray | None, dict[str, str]]:
    """Load [N,K,H,7] candidate actions if a candidate directory is supplied."""
    if candidate_dir is None:
        return None, {}
    if not candidate_dir.is_dir():
        raise FileNotFoundError(candidate_dir)
    actions = []
    paths: dict[str, str] = {}
    for sid in sample_ids:
        path = candidate_dir / f"{sid}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = load_torch(path)
        arr = torch.as_tensor(payload["candidate_actions_concat"]).float().numpy()
        if arr.shape != (1, candidate_k, horizon, 7):
            raise RuntimeError(
                f"{path} candidate_actions_concat shape={arr.shape}, "
                f"expected {(1, candidate_k, horizon, 7)}"
            )
        actions.append(arr[0])
        paths[sid] = str(path)
    return np.stack(actions).astype(np.float32), paths


# -----------------------------------------------------------------------------
# Selection policy
# -----------------------------------------------------------------------------


def build_rejection_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["flag_finite_core"]):
        reasons.append("non_finite_core_metric")
    if not bool(row["flag_finite_actions"]):
        reasons.append("non_finite_action")
    if not bool(row["flag_teacher_agreement"]):
        reasons.append("cosine_mse_teacher_disagree")
    if not bool(row["flag_xyz_distance_ok"]):
        reasons.append("xyz_distance_outlier")
    if not bool(row["flag_rot_distance_ok"]):
        reasons.append("rot_distance_outlier")
    if not bool(row["flag_gripper_ok"]):
        reasons.append("gripper_mismatch_outlier")
    if not bool(row["flag_jerk_ok"]):
        reasons.append("teacher_jerk_outlier")
    if not bool(row["flag_late_mse_ok"]):
        reasons.append("late_mse_outlier")
    if not bool(row["flag_oracle_gap_ok"]):
        reasons.append("large_oracle_gap")
    return ";".join(reasons) if reasons else "selected"


def compute_quality_level(row: pd.Series) -> str:
    if bool(row["use_for_direct_augmentation"]):
        if bool(row["flag_high_quality"]):
            return "high"
        return "medium"
    if bool(row["flag_finite_core"]) and bool(row["flag_finite_actions"]) and bool(row["flag_teacher_agreement"]):
        return "low"
    return "reject"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--scoring-run-summary",
        type=Path,
        default=DEFAULT_SCORING_DIR / "scoring_run_summary.json",
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=DEFAULT_SCORING_DIR / "validation_scores.csv",
    )
    parser.add_argument(
        "--teacher-records",
        type=Path,
        default=DEFAULT_SCORING_DIR / "validation_teacher_records.pt",
    )
    parser.add_argument(
        "--stepwise-npz",
        type=Path,
        default=DEFAULT_SCORING_DIR / "validation_stepwise_metrics.npz",
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=DEFAULT_SCORING_DIR / "validation_summary.json",
    )
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=EXPECTED_MANIFEST_SAMPLES,
    )
    parser.add_argument("--primary-metric", choices=["mse", "cosine"], default="mse")
    parser.add_argument(
        "--agreement-policy",
        choices=["require", "warn", "ignore"],
        default="require",
        help="require rejects cosine/MSE teacher disagreements; warn/ignore keep them if all other filters pass.",
    )
    parser.add_argument("--action-distance-quantile", type=float, default=0.90)
    parser.add_argument("--jerk-quantile", type=float, default=0.90)
    parser.add_argument("--terminal-mse-quantile", type=float, default=0.90)
    parser.add_argument("--oracle-gap-quantile", type=float, default=0.90)
    parser.add_argument("--gripper-min-match-rate", type=float, default=0.3,
        help="Reject teachers whose gripper-vs-GT match rate is below this fraction (binary channel; default rejects severely inverted grippers).")
    parser.add_argument("--high-quality-quantile", type=float, default=0.80)
    parser.add_argument("--high-relative-margin", type=float, default=0.02)
    parser.add_argument("--high-relative-improvement", type=float, default=0.05)
    parser.add_argument("--no-candidate-actions", action="store_true", help="Do not load candidate .pt files; diversity diagnostics will be unavailable.")
    args = parser.parse_args()

    input_names = (
        "manifest",
        "scoring_run_summary",
        "scores_csv",
        "teacher_records",
        "stepwise_npz",
        "validation_summary",
    )
    for name in input_names:
        path = getattr(args, name).expanduser().resolve()
        setattr(args, name, path)
        if not path.is_file():
            raise FileNotFoundError(path)
    args.candidate_dir = args.candidate_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.expected_samples < 1:
        raise ValueError("--expected-samples must be >= 1")
    if not args.no_candidate_actions and not args.candidate_dir.is_dir():
        raise FileNotFoundError(args.candidate_dir)
    for name in [
        "action_distance_quantile",
        "jerk_quantile",
        "terminal_mse_quantile",
        "oracle_gap_quantile",
        "high_quality_quantile",
    ]:
        value = float(getattr(args, name))
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in (0,1)")
    if not (0.0 <= args.gripper_min_match_rate <= 1.0):
        raise ValueError("--gripper-min-match-rate must lie in [0,1]")

    manifest_sha256, manifest_samples, scoring_run = validate_scoring_run_header(
        manifest_path=args.manifest,
        scoring_run_path=args.scoring_run_summary,
        expected_samples=args.expected_samples,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores_csv)
    records = load_torch(args.teacher_records)
    stepwise = np.load(args.stepwise_npz, allow_pickle=False)
    summary = load_json(args.validation_summary)

    provenance = validate_scoring_contract(
        manifest_path=args.manifest,
        scoring_run_path=args.scoring_run_summary,
        scores_path=args.scores_csv,
        teacher_records_path=args.teacher_records,
        stepwise_path=args.stepwise_npz,
        validation_summary_path=args.validation_summary,
        scores=scores,
        records=records,
        stepwise=stepwise,
        validation_summary=summary,
        manifest_sha256=manifest_sha256,
        manifest_samples=manifest_samples,
        scoring_run=scoring_run,
    )

    sample_ids = scores["sample_id"].astype(str).tolist()
    n = len(scores)
    candidate_k = int(summary["candidate_k"])
    horizon = int(summary["future_horizon"])

    teacher_by_cos = records["teacher_by_cosine"].numpy().astype(np.int64)
    teacher_by_mse = records["teacher_by_mse"].numpy().astype(np.int64)
    teacher_idx = teacher_by_mse if args.primary_metric == "mse" else teacher_by_cos

    teacher_actions_cos = records["teacher_actions_cosine_lerobot"].numpy().astype(np.float32)
    teacher_actions_mse = records["teacher_actions_mse_lerobot"].numpy().astype(np.float32)
    teacher_actions = teacher_actions_mse if args.primary_metric == "mse" else teacher_actions_cos
    gt_actions = records["gt_actions_lerobot"].numpy().astype(np.float32)

    cosine_scores = records["cosine_scores"].numpy().astype(np.float32)
    mse_scores = records["mse_scores"].numpy().astype(np.float32)
    best_mse = mse_scores.min(axis=1)
    second_mse = np.partition(mse_scores, 1, axis=1)[:, 1]
    worst_mse = mse_scores.max(axis=1)
    mean_mse = mse_scores.mean(axis=1)
    best_cos = cosine_scores.max(axis=1)
    second_cos = -np.partition(-cosine_scores, 1, axis=1)[:, 1]
    worst_cos = cosine_scores.min(axis=1)
    mean_cos = cosine_scores.mean(axis=1)

    eps = 1e-12
    audit = scores.copy()
    audit["selection_policy"] = POLICY_NAME
    audit["primary_metric"] = args.primary_metric
    audit["agreement_policy"] = args.agreement_policy
    audit["teacher_idx_primary"] = teacher_idx
    audit["relative_mse_margin"] = (second_mse - best_mse) / (second_mse + eps)
    audit["relative_mse_improvement_vs_mean"] = (mean_mse - best_mse) / (mean_mse + eps)
    audit["relative_mse_improvement_vs_worst"] = (worst_mse - best_mse) / (worst_mse + eps)
    audit["relative_cosine_margin"] = (best_cos - second_cos) / (np.maximum(1.0 - second_cos, eps))
    audit["relative_cosine_improvement_vs_mean"] = (best_cos - mean_cos) / (np.maximum(1.0 - mean_cos, eps))
    audit["oracle_gap_mse"] = best_mse - scores["oracle_mse_score"].to_numpy()
    audit["oracle_gap_cosine"] = scores["oracle_cosine_score"].to_numpy() - best_cos
    audit["oracle_gap_ratio_mse"] = audit["oracle_gap_mse"] / (best_mse + eps)
    audit["oracle_gap_ratio_cosine"] = audit["oracle_gap_cosine"] / (np.maximum(1.0 - best_cos, eps))

    # Stepwise metrics for the chosen teacher.
    mse_per_step = stepwise["mse_per_step"].astype(np.float32)
    cosine_per_step = stepwise["cosine_per_step"].astype(np.float32)
    row_index = np.arange(n)
    teacher_mse_step = mse_per_step[row_index, teacher_idx]
    teacher_cos_step = cosine_per_step[row_index, teacher_idx]
    audit["early_mse"] = teacher_mse_step[:, 0:2].mean(axis=1)
    audit["mid_mse"] = teacher_mse_step[:, 2:5].mean(axis=1)
    audit["late_mse"] = teacher_mse_step[:, 5:8].mean(axis=1)
    audit["terminal_mse"] = teacher_mse_step[:, 7]
    audit["early_cosine"] = teacher_cos_step[:, 0:2].mean(axis=1)
    audit["mid_cosine"] = teacher_cos_step[:, 2:5].mean(axis=1)
    audit["late_cosine"] = teacher_cos_step[:, 5:8].mean(axis=1)
    audit["terminal_cosine"] = teacher_cos_step[:, 7]
    audit["mse_growth_terminal_over_early"] = audit["terminal_mse"] / (audit["early_mse"] + eps)

    # Teacher-vs-GT action distance and smoothness.
    action_rows = []
    for i in range(n):
        row = {}
        row.update(action_distance_metrics(teacher_actions[i], gt_actions[i]))
        row.update(smoothness_metrics(teacher_actions[i], "teacher"))
        row.update(smoothness_metrics(gt_actions[i], "gt"))
        action_rows.append(row)
    action_df = pd.DataFrame(action_rows)
    audit = pd.concat([audit, action_df], axis=1)
    audit["teacher_gt_velocity_ratio"] = audit["teacher_velocity_l2_mean"] / (audit["gt_velocity_l2_mean"] + eps)
    audit["teacher_gt_acceleration_ratio"] = audit["teacher_acceleration_l2_mean"] / (audit["gt_acceleration_l2_mean"] + eps)
    audit["teacher_gt_jerk_ratio"] = audit["teacher_jerk_l2_mean"] / (audit["gt_jerk_l2_mean"] + eps)

    candidate_actions = None
    candidate_paths = {}
    if not args.no_candidate_actions:
        candidate_actions, candidate_paths = load_candidate_actions(args.candidate_dir, sample_ids, candidate_k, horizon)
        diversity_rows = [pairwise_metrics(candidate_actions[i]) for i in range(n)]
        diversity_df = pd.DataFrame(diversity_rows)
        audit = pd.concat([audit, diversity_df], axis=1)
        audit["candidate_path"] = [candidate_paths[sid] for sid in sample_ids]
    else:
        audit["candidate_path"] = ""

    thresholds = {
        "selection_policy": POLICY_NAME,
        "direct_augmentation_no_quality_weight": True,
        "primary_metric": args.primary_metric,
        "agreement_policy": args.agreement_policy,
        "action_distance_quantile": float(args.action_distance_quantile),
        "teacher_gt_action_mse_max": finite_quantile(audit["teacher_gt_action_mse"], args.action_distance_quantile, "teacher_gt_action_mse"),
        "teacher_gt_xyz_mse_max": finite_quantile(audit["teacher_gt_xyz_mse"], args.action_distance_quantile, "teacher_gt_xyz_mse"),
        "teacher_gt_rot_mse_max": finite_quantile(audit["teacher_gt_rot_mse"], args.action_distance_quantile, "teacher_gt_rot_mse"),
        "gripper_min_match_rate": float(args.gripper_min_match_rate),
        "jerk_quantile": float(args.jerk_quantile),
        "teacher_jerk_l2_mean_max": finite_quantile(audit["teacher_jerk_l2_mean"], args.jerk_quantile, "teacher_jerk_l2_mean"),
        "teacher_jerk_xyz_l2_mean_max": finite_quantile(audit["teacher_jerk_xyz_l2_mean"], args.jerk_quantile, "teacher_jerk_xyz_l2_mean"),
        "teacher_jerk_rot_l2_mean_max": finite_quantile(audit["teacher_jerk_rot_l2_mean"], args.jerk_quantile, "teacher_jerk_rot_l2_mean"),
        "terminal_mse_quantile": float(args.terminal_mse_quantile),
        "late_mse_max": finite_quantile(audit["late_mse"], args.terminal_mse_quantile, "late_mse"),
        "terminal_mse_max": finite_quantile(audit["terminal_mse"], args.terminal_mse_quantile, "terminal_mse"),
        "oracle_gap_quantile": float(args.oracle_gap_quantile),
        "oracle_gap_mse_max": finite_quantile(audit["oracle_gap_mse"], args.oracle_gap_quantile, "oracle_gap_mse"),
        "high_quality_quantile": float(args.high_quality_quantile),
        "high_teacher_gt_action_mse_max": finite_quantile(audit["teacher_gt_action_mse"], args.high_quality_quantile, "high teacher_gt_action_mse"),
        "high_teacher_jerk_l2_mean_max": finite_quantile(audit["teacher_jerk_l2_mean"], args.high_quality_quantile, "high teacher_jerk_l2_mean"),
        "high_terminal_mse_max": finite_quantile(audit["terminal_mse"], args.high_quality_quantile, "high terminal_mse"),
        "high_oracle_gap_mse_max": finite_quantile(audit["oracle_gap_mse"], args.high_quality_quantile, "high oracle_gap_mse"),
        "high_relative_margin": float(args.high_relative_margin),
        "high_relative_improvement": float(args.high_relative_improvement),
        "margin_hard_filter_enabled": False,
        "candidate_diversity_hard_filter_enabled": False,
    }

    finite_cols = [
        "best_mse_score",
        "best_cosine_score",
        "relative_mse_margin",
        "relative_mse_improvement_vs_mean",
        "teacher_gt_action_mse",
        "teacher_gt_xyz_mse",
        "teacher_gt_rot_mse",
        "teacher_jerk_l2_mean",
        "teacher_jerk_xyz_l2_mean",
        "teacher_jerk_rot_l2_mean",
        "late_mse",
        "terminal_mse",
        "oracle_gap_mse",
    ]
    audit["flag_finite_core"] = np.isfinite(audit[finite_cols].to_numpy()).all(axis=1)
    audit["flag_finite_actions"] = np.isfinite(teacher_actions.reshape(n, -1)).all(axis=1)
    audit["flag_teacher_agreement"] = audit["teacher_agreement"].astype(bool)
    if args.agreement_policy == "require":
        audit["flag_agreement_ok"] = audit["flag_teacher_agreement"]
    else:
        audit["flag_agreement_ok"] = True
    # A2: decompose the combined action-distance outlier into per-component xyz /
    # rotation quantiles (the gripper channel is handled by flag_gripper_ok below).
    audit["flag_xyz_distance_ok"] = audit["teacher_gt_xyz_mse"] <= thresholds["teacher_gt_xyz_mse_max"]
    audit["flag_rot_distance_ok"] = audit["teacher_gt_rot_mse"] <= thresholds["teacher_gt_rot_mse_max"]
    # B3: relative-jerk exemption — reject only when the teacher is jerkier than GT
    # AND lands in the absolute top decile on either the xyz or rotation jerk.
    audit["flag_jerk_xyz_ok"] = audit["teacher_jerk_xyz_l2_mean"] <= thresholds["teacher_jerk_xyz_l2_mean_max"]
    audit["flag_jerk_rot_ok"] = audit["teacher_jerk_rot_l2_mean"] <= thresholds["teacher_jerk_rot_l2_mean_max"]
    audit["flag_jerk_ok"] = (
        (audit["teacher_gt_jerk_ratio"] <= 1.0)
        | (audit["flag_jerk_xyz_ok"] & audit["flag_jerk_rot_ok"])
    )
    # A1: reject teachers with a severely inverted gripper channel.
    audit["flag_gripper_ok"] = audit["teacher_gt_gripper_match_rate"] >= thresholds["gripper_min_match_rate"]
    # B4: filter on the late-window MSE (mean of the last 3 steps) instead of the
    # noisy single terminal step.
    audit["flag_late_mse_ok"] = audit["late_mse"] <= thresholds["late_mse_max"]
    audit["flag_oracle_gap_ok"] = audit["oracle_gap_mse"] <= thresholds["oracle_gap_mse_max"]

    selected_flags = [
        "flag_finite_core",
        "flag_finite_actions",
        "flag_agreement_ok",
        "flag_xyz_distance_ok",
        "flag_rot_distance_ok",
        "flag_gripper_ok",
        "flag_jerk_ok",
        "flag_late_mse_ok",
        "flag_oracle_gap_ok",
    ]
    audit["use_for_direct_augmentation"] = audit[selected_flags].all(axis=1)
    # Backward-compatible alias used by the existing distillation-index builder.
    audit["use_for_distillation"] = audit["use_for_direct_augmentation"]

    audit["flag_high_quality"] = (
        audit["use_for_direct_augmentation"]
        & (audit["relative_mse_margin"] >= thresholds["high_relative_margin"])
        & (audit["relative_mse_improvement_vs_mean"] >= thresholds["high_relative_improvement"])
        & (audit["teacher_gt_action_mse"] <= thresholds["high_teacher_gt_action_mse_max"])
        & (audit["teacher_jerk_l2_mean"] <= thresholds["high_teacher_jerk_l2_mean_max"])
        & (audit["terminal_mse"] <= thresholds["high_terminal_mse_max"])
        & (audit["oracle_gap_mse"] <= thresholds["high_oracle_gap_mse_max"])
        & (audit["teacher_gt_gripper_mismatch_steps"] == 0)
    )
    audit["teacher_quality_level"] = audit.apply(compute_quality_level, axis=1)
    # Compatibility fields: selected teachers are direct extra samples, not
    # confidence-weighted samples. Keep the old field with 1/0 values so the
    # existing index builder can read the file without semantic ambiguity.
    audit["direct_augmentation_weight"] = audit["use_for_direct_augmentation"].astype(float)
    audit["confidence_weight"] = audit["direct_augmentation_weight"]
    audit["confidence_level"] = audit["teacher_quality_level"]
    audit["rejection_reasons"] = audit.apply(build_rejection_reasons, axis=1)

    # Keep suite identity in every aggregate; task_index is globally unique in
    # the current manifest, but this prevents future suite-local IDs colliding.
    task_summary = audit.groupby(
        ["suite", "task_index", "task_text"], as_index=False
    ).agg(
        samples=("sample_id", "count"),
        selected=("use_for_direct_augmentation", "sum"),
        direct_aug_rate=("use_for_direct_augmentation", "mean"),
        high_quality=("teacher_quality_level", lambda s: int((s == "high").sum())),
        medium_quality=("teacher_quality_level", lambda s: int((s == "medium").sum())),
        low_quality=("teacher_quality_level", lambda s: int((s == "low").sum())),
        reject_quality=("teacher_quality_level", lambda s: int((s == "reject").sum())),
        mean_relative_mse_margin=("relative_mse_margin", "mean"),
        mean_relative_mse_improvement=("relative_mse_improvement_vs_mean", "mean"),
        mean_best_mse=("best_mse_score", "mean"),
        mean_oracle_mse=("oracle_mse_score", "mean"),
        mean_oracle_gap_mse=("oracle_gap_mse", "mean"),
        mean_teacher_gt_action_mse=("teacher_gt_action_mse", "mean"),
        mean_teacher_gt_xyz_mse=("teacher_gt_xyz_mse", "mean"),
        mean_teacher_gt_rot_mse=("teacher_gt_rot_mse", "mean"),
        mean_gripper_match_rate=("teacher_gt_gripper_match_rate", "mean"),
        mean_teacher_jerk=("teacher_jerk_l2_mean", "mean"),
        mean_jerk_ratio=("teacher_gt_jerk_ratio", "mean"),
        mean_late_mse=("late_mse", "mean"),
        mean_terminal_mse=("terminal_mse", "mean"),
    )

    audit_csv = output_dir / "teacher_quality_audit.csv"
    audit_parquet = output_dir / "teacher_quality_audit.parquet"
    task_csv = output_dir / "teacher_quality_task_summary.csv"
    task_parquet = output_dir / "teacher_quality_task_summary.parquet"
    suite_csv = output_dir / "teacher_quality_suite_summary.csv"
    suite_parquet = output_dir / "teacher_quality_suite_summary.parquet"
    selected_jsonl = output_dir / "teacher_quality_selected.jsonl"
    arrays_npz = output_dir / "teacher_quality_arrays.npz"
    filtered_pt = output_dir / "teacher_quality_filtered_records.pt"
    direct_records_pt = output_dir / "teacher_quality_direct_records.pt"
    summary_json = output_dir / "teacher_quality_summary.json"

    atomic_json(
        summary_json,
        {
            "format": FORMAT_SUMMARY,
            "status": "BUILDING",
            "manifest_sha256": provenance["manifest_sha256"],
        },
    )
    audit.to_csv(audit_csv, index=False)
    parquet_status = write_parquet_best_effort(audit, audit_parquet)
    task_summary.to_csv(task_csv, index=False)
    task_parquet_status = write_parquet_best_effort(task_summary, task_parquet)
    suite_summary = audit.groupby("suite", as_index=False).agg(
        samples=("sample_id", "count"),
        tasks=("task_index", "nunique"),
        episodes=("episode_index", "nunique"),
        selected=("use_for_direct_augmentation", "sum"),
        direct_aug_rate=("use_for_direct_augmentation", "mean"),
        mean_best_mse=("best_mse_score", "mean"),
        mean_oracle_mse=("oracle_mse_score", "mean"),
        mean_teacher_gt_action_mse=("teacher_gt_action_mse", "mean"),
    )
    suite_summary.to_csv(suite_csv, index=False)
    suite_parquet_status = write_parquet_best_effort(suite_summary, suite_parquet)

    selected = audit[audit["use_for_direct_augmentation"]].copy()
    with selected_jsonl.open("w", encoding="utf-8") as handle:
        for _, row in selected.iterrows():
            payload = {
                "sample_id": row["sample_id"],
                "sample_index": int(row["sample_index"]),
                "suite": str(row["suite"]),
                "gt_latent_key": str(row["gt_latent_key"]),
                "episode_index": int(row["episode_index"]),
                "local_step": int(row["local_step"]),
                "task_index": int(row["task_index"]),
                "teacher_idx_primary": int(row["teacher_idx_primary"]),
                "teacher_quality_level": row["teacher_quality_level"],
                "confidence_level": row["confidence_level"],
                "direct_augmentation_weight": 1.0,
                "confidence_weight": 1.0,
                "relative_mse_margin": float(row["relative_mse_margin"]),
                "relative_mse_improvement_vs_mean": float(row["relative_mse_improvement_vs_mean"]),
                "teacher_gt_action_mse": float(row["teacher_gt_action_mse"]),
                "teacher_gt_xyz_mse": float(row["teacher_gt_xyz_mse"]),
                "teacher_gt_rot_mse": float(row["teacher_gt_rot_mse"]),
                "teacher_gt_gripper_match_rate": float(row["teacher_gt_gripper_match_rate"]),
                "teacher_gt_gripper_mismatch_steps": int(row["teacher_gt_gripper_mismatch_steps"]),
                "teacher_jerk_l2_mean": float(row["teacher_jerk_l2_mean"]),
                "teacher_gt_jerk_ratio": float(row["teacher_gt_jerk_ratio"]),
                "late_mse": float(row["late_mse"]),
                "terminal_mse": float(row["terminal_mse"]),
                "oracle_gap_mse": float(row["oracle_gap_mse"]),
                "selection_policy": POLICY_NAME,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    selected_positions = selected.index.to_numpy(dtype=np.int64)
    selected_sample_indices = selected["sample_index"].to_numpy(dtype=np.int64)
    np.savez_compressed(
        arrays_npz,
        sample_ids=audit["sample_id"].to_numpy(dtype="U32"),
        use_for_direct_augmentation=audit["use_for_direct_augmentation"].to_numpy(dtype=np.bool_),
        use_for_distillation=audit["use_for_distillation"].to_numpy(dtype=np.bool_),
        direct_augmentation_weight=audit["direct_augmentation_weight"].to_numpy(dtype=np.float32),
        teacher_quality_level=audit["teacher_quality_level"].to_numpy(dtype="U16"),
        teacher_idx_primary=audit["teacher_idx_primary"].to_numpy(dtype=np.int64),
        relative_mse_margin=audit["relative_mse_margin"].to_numpy(dtype=np.float32),
        relative_mse_improvement_vs_mean=audit["relative_mse_improvement_vs_mean"].to_numpy(dtype=np.float32),
        teacher_gt_action_mse=audit["teacher_gt_action_mse"].to_numpy(dtype=np.float32),
        teacher_gt_xyz_mse=audit["teacher_gt_xyz_mse"].to_numpy(dtype=np.float32),
        teacher_gt_rot_mse=audit["teacher_gt_rot_mse"].to_numpy(dtype=np.float32),
        teacher_gt_gripper_match_rate=audit["teacher_gt_gripper_match_rate"].to_numpy(dtype=np.float32),
        teacher_jerk_l2_mean=audit["teacher_jerk_l2_mean"].to_numpy(dtype=np.float32),
        teacher_gt_jerk_ratio=audit["teacher_gt_jerk_ratio"].to_numpy(dtype=np.float32),
        late_mse=audit["late_mse"].to_numpy(dtype=np.float32),
        terminal_mse=audit["terminal_mse"].to_numpy(dtype=np.float32),
        oracle_gap_mse=audit["oracle_gap_mse"].to_numpy(dtype=np.float32),
        selected_positions=selected_positions,
        selected_sample_indices=selected_sample_indices,
    )

    # Save teacher actions for the selected subset in LeRobot convention.
    selected_weights = np.ones(len(selected), dtype=np.float32)
    filtered = {
        "format": FORMAT_RECORDS,
        "selection_policy": POLICY_NAME,
        "direct_augmentation_no_quality_weight": True,
        "primary_metric": args.primary_metric,
        "thresholds": thresholds,
        "manifest_sha256": provenance["manifest_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "gt_h5_sha256": provenance["gt_h5_sha256"],
        "sample_ids": selected["sample_id"].tolist(),
        "sample_indices": torch.from_numpy(selected_sample_indices),
        "suites": selected["suite"].astype(str).tolist(),
        "gt_latent_keys": selected["gt_latent_key"].astype(str).tolist(),
        "task_indices": torch.from_numpy(
            selected["task_index"].to_numpy(dtype=np.int64)
        ),
        "teacher_idx_primary": torch.from_numpy(selected["teacher_idx_primary"].to_numpy(dtype=np.int64)),
        "teacher_quality_level": selected["teacher_quality_level"].tolist(),
        "confidence_level": selected["confidence_level"].tolist(),
        "direct_augmentation_weight": torch.from_numpy(selected_weights),
        # Compatibility with 08_build_distillation_dataset.py.
        "confidence_weight": torch.from_numpy(selected_weights),
        "teacher_actions_lerobot": torch.from_numpy(teacher_actions[selected_positions].astype(np.float32)),
        "gt_actions_lerobot": torch.from_numpy(gt_actions[selected_positions].astype(np.float32)),
        "relative_mse_margin": torch.from_numpy(selected["relative_mse_margin"].to_numpy(dtype=np.float32)),
        "relative_mse_improvement_vs_mean": torch.from_numpy(selected["relative_mse_improvement_vs_mean"].to_numpy(dtype=np.float32)),
        "teacher_gt_action_mse": torch.from_numpy(selected["teacher_gt_action_mse"].to_numpy(dtype=np.float32)),
        "teacher_gt_xyz_mse": torch.from_numpy(selected["teacher_gt_xyz_mse"].to_numpy(dtype=np.float32)),
        "teacher_gt_rot_mse": torch.from_numpy(selected["teacher_gt_rot_mse"].to_numpy(dtype=np.float32)),
        "teacher_gt_gripper_match_rate": torch.from_numpy(selected["teacher_gt_gripper_match_rate"].to_numpy(dtype=np.float32)),
        "teacher_jerk_l2_mean": torch.from_numpy(selected["teacher_jerk_l2_mean"].to_numpy(dtype=np.float32)),
        "teacher_gt_jerk_ratio": torch.from_numpy(selected["teacher_gt_jerk_ratio"].to_numpy(dtype=np.float32)),
        "late_mse": torch.from_numpy(selected["late_mse"].to_numpy(dtype=np.float32)),
        "terminal_mse": torch.from_numpy(selected["terminal_mse"].to_numpy(dtype=np.float32)),
        "oracle_gap_mse": torch.from_numpy(selected["oracle_gap_mse"].to_numpy(dtype=np.float32)),
    }
    atomic_torch(filtered_pt, filtered)
    # Same content with a name that makes the direct-augmentation semantics clear.
    atomic_torch(direct_records_pt, filtered)
    audit_output_sha256 = {
        "selected_jsonl": sha256_file(selected_jsonl),
        "filtered_records": sha256_file(filtered_pt),
        "direct_records": sha256_file(direct_records_pt),
    }

    reason_counts: dict[str, int] = {}
    for reasons_text in audit["rejection_reasons"]:
        for reason in str(reasons_text).split(";"):
            if not reason:
                continue
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    quality_counts = audit["teacher_quality_level"].value_counts().to_dict()
    selected_rate = float(audit["use_for_direct_augmentation"].mean())
    global_summary = {
        "format": FORMAT_SUMMARY,
        "status": "PASS",
        "inputs": {
            "manifest": str(args.manifest),
            "scoring_run_summary": str(args.scoring_run_summary),
            "scores_csv": str(args.scores_csv),
            "teacher_records": str(args.teacher_records),
            "stepwise_npz": str(args.stepwise_npz),
            "validation_summary": str(args.validation_summary),
            "candidate_dir": None if args.no_candidate_actions else str(args.candidate_dir),
            "sha256": provenance["input_sha256"],
        },
        "provenance": {
            "manifest_format": MANIFEST_FORMAT,
            "manifest_sha256": provenance["manifest_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "gt_h5_sha256": provenance["gt_h5_sha256"],
        },
        "selection_policy": POLICY_NAME,
        "policy_explanation": (
            "LIBERO-PLUS Filtered Direct Augmentation: remove unstable teacher "
            "agreement, top-quantile per-component xyz/rotation action-distance, "
            "gripper majority-inversion, relative-jerk, late-window MSE, and "
            "oracle-gap outliers. Relative margin and candidate diversity are "
            "diagnostics, not hard filters. Selected teachers are appended as "
            "equal-weight extra training samples."
        ),
        "primary_metric": args.primary_metric,
        "agreement_policy": args.agreement_policy,
        "samples": int(n),
        "tasks": int(audit["task_index"].nunique()),
        "episodes": int(audit["episode_index"].nunique()),
        "suite_counts": provenance["suite_counts"],
        "selected_for_direct_augmentation": {
            "count": int(audit["use_for_direct_augmentation"].sum()),
            "rate": selected_rate,
        },
        # Backward-compatible summary field.
        "selected_for_distillation": {
            "count": int(audit["use_for_distillation"].sum()),
            "rate": selected_rate,
        },
        "teacher_quality_counts": {str(k): int(v) for k, v in quality_counts.items()},
        "rejection_reason_counts": {str(k): int(v) for k, v in sorted(reason_counts.items())},
        "thresholds": thresholds,
        "metric_distributions": {
            "relative_mse_margin": numeric_summary(audit["relative_mse_margin"].to_numpy()),
            "relative_mse_improvement_vs_mean": numeric_summary(audit["relative_mse_improvement_vs_mean"].to_numpy()),
            "oracle_gap_mse": numeric_summary(audit["oracle_gap_mse"].to_numpy()),
            "oracle_gap_ratio_mse": numeric_summary(audit["oracle_gap_ratio_mse"].to_numpy()),
            "teacher_gt_action_mse": numeric_summary(audit["teacher_gt_action_mse"].to_numpy()),
            "teacher_gt_xyz_mse": numeric_summary(audit["teacher_gt_xyz_mse"].to_numpy()),
            "teacher_gt_rot_mse": numeric_summary(audit["teacher_gt_rot_mse"].to_numpy()),
            "teacher_gt_gripper_match_rate": numeric_summary(audit["teacher_gt_gripper_match_rate"].to_numpy()),
            "teacher_gt_gripper_mismatch_steps": numeric_summary(audit["teacher_gt_gripper_mismatch_steps"].to_numpy()),
            "teacher_velocity_l2_mean": numeric_summary(audit["teacher_velocity_l2_mean"].to_numpy()),
            "teacher_acceleration_l2_mean": numeric_summary(audit["teacher_acceleration_l2_mean"].to_numpy()),
            "teacher_jerk_l2_mean": numeric_summary(audit["teacher_jerk_l2_mean"].to_numpy()),
            "teacher_jerk_xyz_l2_mean": numeric_summary(audit["teacher_jerk_xyz_l2_mean"].to_numpy()),
            "teacher_jerk_rot_l2_mean": numeric_summary(audit["teacher_jerk_rot_l2_mean"].to_numpy()),
            "teacher_gt_jerk_ratio": numeric_summary(audit["teacher_gt_jerk_ratio"].to_numpy()),
            "late_mse": numeric_summary(audit["late_mse"].to_numpy()),
            "terminal_mse": numeric_summary(audit["terminal_mse"].to_numpy()),
        },
        "candidate_diversity_distributions": {},
        "outputs": {
            "audit_csv": str(audit_csv),
            "audit_parquet": parquet_status,
            "task_summary_csv": str(task_csv),
            "task_summary_parquet": task_parquet_status,
            "suite_summary_csv": str(suite_csv),
            "suite_summary_parquet": suite_parquet_status,
            "selected_jsonl": str(selected_jsonl),
            "arrays_npz": str(arrays_npz),
            "filtered_records_pt": str(filtered_pt),
            "direct_records_pt": str(direct_records_pt),
            "sha256": audit_output_sha256,
        },
    }
    for col in [
        "candidate_pairwise_action_l2_mean",
        "candidate_pairwise_action_l2_min",
        "candidate_pairwise_xyz_l2_mean",
        "candidate_pairwise_rot_l2_mean",
        "candidate_pairwise_gripper_abs_mean",
    ]:
        if col in audit.columns:
            global_summary["candidate_diversity_distributions"][col] = numeric_summary(audit[col].to_numpy())

    # Helpful ranked lists for quick inspection.
    global_summary["top_problem_samples"] = {
        "largest_teacher_gt_action_mse": audit.sort_values("teacher_gt_action_mse", ascending=False)[["sample_id", "task_index", "teacher_gt_action_mse", "relative_mse_margin", "terminal_mse", "rejection_reasons"]].head(10).to_dict(orient="records"),
        "largest_teacher_jerk": audit.sort_values("teacher_jerk_l2_mean", ascending=False)[["sample_id", "task_index", "teacher_jerk_l2_mean", "teacher_gt_action_mse", "terminal_mse", "rejection_reasons"]].head(10).to_dict(orient="records"),
        "largest_terminal_mse": audit.sort_values("terminal_mse", ascending=False)[["sample_id", "task_index", "terminal_mse", "best_mse_score", "oracle_mse_score", "rejection_reasons"]].head(10).to_dict(orient="records"),
        "largest_oracle_gap_mse": audit.sort_values("oracle_gap_mse", ascending=False)[["sample_id", "task_index", "oracle_gap_mse", "best_mse_score", "oracle_mse_score", "rejection_reasons"]].head(10).to_dict(orient="records"),
        "largest_teacher_gt_rot_mse": audit.sort_values("teacher_gt_rot_mse", ascending=False)[["sample_id", "task_index", "teacher_gt_rot_mse", "teacher_gt_xyz_mse", "teacher_gt_gripper_match_rate", "rejection_reasons"]].head(10).to_dict(orient="records"),
        "lowest_teacher_gt_gripper_match_rate": audit.sort_values("teacher_gt_gripper_match_rate", ascending=True)[["sample_id", "task_index", "teacher_gt_gripper_match_rate", "teacher_gt_gripper_mismatch_steps", "rejection_reasons"]].head(10).to_dict(orient="records"),
        "largest_late_mse": audit.sort_values("late_mse", ascending=False)[["sample_id", "task_index", "late_mse", "terminal_mse", "best_mse_score", "rejection_reasons"]].head(10).to_dict(orient="records"),
        "lowest_relative_margin": audit.sort_values("relative_mse_margin", ascending=True)[["sample_id", "task_index", "relative_mse_margin", "teacher_agreement", "teacher_quality_level", "rejection_reasons"]].head(10).to_dict(orient="records"),
    }

    atomic_json(summary_json, global_summary)

    print("========== Teacher quality audit ==========")
    print("samples                         :", n)
    print("selection policy                :", POLICY_NAME)
    print("primary metric                  :", args.primary_metric)
    print("agreement policy                :", args.agreement_policy)
    print("selected for direct augmentation:", int(audit["use_for_direct_augmentation"].sum()), f"({selected_rate:.3f})")
    print("teacher quality counts          :", quality_counts)
    print("rejection reason counts         :", reason_counts)
    print("audit csv                       :", audit_csv)
    print("summary                         :", summary_json)
    print("filtered teacher records        :", filtered_pt)
    print("direct teacher records          :", direct_records_pt)


if __name__ == "__main__":
    main()
