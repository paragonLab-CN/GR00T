#!/usr/bin/env python3
"""Audit FLARE-VLA teachers for Filtered Direct Augmentation.

This script implements the FLARE-Filtered Direct Augmentation policy:

1. Do not use per-sample confidence weights as the main training signal.
2. Do not hard-filter by relative MSE margin.
3. Do not hard-filter by candidate diversity.
4. Remove only teachers with unstable score agreement or obvious quality outliers:
   action-distance outlier, jerk outlier, terminal-MSE outlier, or oracle-gap
   outlier.
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
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


FORMAT_SUMMARY = "flare_teacher_quality_summary_v3_direct_augmentation"
FORMAT_RECORDS = "flare_teacher_quality_filtered_records_v3_direct_augmentation"
POLICY_NAME = "flare_filtered_direct_augmentation"


# -----------------------------------------------------------------------------
# Generic I/O helpers
# -----------------------------------------------------------------------------


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    if not bool(row["flag_action_distance_ok"]):
        reasons.append("action_distance_outlier")
    if not bool(row["flag_jerk_ok"]):
        reasons.append("teacher_jerk_outlier")
    if not bool(row["flag_terminal_mse_ok"]):
        reasons.append("terminal_mse_outlier")
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
    parser.add_argument("--scores-csv", type=Path, default=Path("/root/autodl-tmp/le-wm/outputs/flare_validation_500/validation_scores.csv"))
    parser.add_argument("--teacher-records", type=Path, default=Path("/root/autodl-tmp/le-wm/outputs/flare_validation_500/validation_teacher_records.pt"))
    parser.add_argument("--stepwise-npz", type=Path, default=Path("/root/autodl-tmp/le-wm/outputs/flare_validation_500/validation_stepwise_metrics.npz"))
    parser.add_argument("--validation-summary", type=Path, default=Path("/root/autodl-tmp/le-wm/outputs/flare_validation_500/validation_summary.json"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T/outputs/flare_validation_500/candidates"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/le-wm/outputs/flare_validation_500/teacher_quality_audit_direct"))
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
    parser.add_argument("--high-quality-quantile", type=float, default=0.70)
    parser.add_argument("--high-relative-margin", type=float, default=0.05)
    parser.add_argument("--high-relative-improvement", type=float, default=0.10)
    parser.add_argument("--no-candidate-actions", action="store_true", help="Do not load candidate .pt files; diversity diagnostics will be unavailable.")
    args = parser.parse_args()

    for path in [args.scores_csv, args.teacher_records, args.stepwise_npz, args.validation_summary]:
        if not path.exists():
            raise FileNotFoundError(path)
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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores_csv)
    records = load_torch(args.teacher_records)
    stepwise = np.load(args.stepwise_npz, allow_pickle=False)
    summary = json.loads(args.validation_summary.read_text(encoding="utf-8"))

    sample_ids = scores["sample_id"].astype(str).tolist()
    n = len(scores)
    candidate_k = int(summary["candidate_k"])
    horizon = int(summary["future_horizon"])
    if n == 0:
        raise RuntimeError("scores CSV is empty")

    rec_ids = [str(x) for x in records["sample_ids"]]
    if rec_ids != sample_ids:
        raise RuntimeError("sample_ids in validation_teacher_records.pt do not match validation_scores.csv order")
    if list(stepwise["sample_ids"].astype(str)) != sample_ids:
        raise RuntimeError("sample_ids in validation_stepwise_metrics.npz do not match validation_scores.csv order")

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
        "jerk_quantile": float(args.jerk_quantile),
        "teacher_jerk_l2_mean_max": finite_quantile(audit["teacher_jerk_l2_mean"], args.jerk_quantile, "teacher_jerk_l2_mean"),
        "terminal_mse_quantile": float(args.terminal_mse_quantile),
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
        "teacher_jerk_l2_mean",
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
    audit["flag_action_distance_ok"] = audit["teacher_gt_action_mse"] <= thresholds["teacher_gt_action_mse_max"]
    audit["flag_jerk_ok"] = audit["teacher_jerk_l2_mean"] <= thresholds["teacher_jerk_l2_mean_max"]
    audit["flag_terminal_mse_ok"] = audit["terminal_mse"] <= thresholds["terminal_mse_max"]
    audit["flag_oracle_gap_ok"] = audit["oracle_gap_mse"] <= thresholds["oracle_gap_mse_max"]

    selected_flags = [
        "flag_finite_core",
        "flag_finite_actions",
        "flag_agreement_ok",
        "flag_action_distance_ok",
        "flag_jerk_ok",
        "flag_terminal_mse_ok",
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
    )
    audit["teacher_quality_level"] = audit.apply(compute_quality_level, axis=1)
    # Compatibility fields: selected teachers are direct extra samples, not
    # confidence-weighted samples. Keep the old field with 1/0 values so the
    # existing index builder can read the file without semantic ambiguity.
    audit["direct_augmentation_weight"] = audit["use_for_direct_augmentation"].astype(float)
    audit["confidence_weight"] = audit["direct_augmentation_weight"]
    audit["confidence_level"] = audit["teacher_quality_level"]
    audit["rejection_reasons"] = audit.apply(build_rejection_reasons, axis=1)

    # Per-task summary.
    task_summary = audit.groupby(["task_index", "task_text"], as_index=False).agg(
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
        mean_teacher_jerk=("teacher_jerk_l2_mean", "mean"),
        mean_terminal_mse=("terminal_mse", "mean"),
    )

    audit_csv = output_dir / "teacher_quality_audit.csv"
    audit_parquet = output_dir / "teacher_quality_audit.parquet"
    task_csv = output_dir / "teacher_quality_task_summary.csv"
    task_parquet = output_dir / "teacher_quality_task_summary.parquet"
    selected_jsonl = output_dir / "teacher_quality_selected.jsonl"
    arrays_npz = output_dir / "teacher_quality_arrays.npz"
    filtered_pt = output_dir / "teacher_quality_filtered_records.pt"
    direct_records_pt = output_dir / "teacher_quality_direct_records.pt"
    summary_json = output_dir / "teacher_quality_summary.json"

    audit.to_csv(audit_csv, index=False)
    parquet_status = write_parquet_best_effort(audit, audit_parquet)
    task_summary.to_csv(task_csv, index=False)
    task_parquet_status = write_parquet_best_effort(task_summary, task_parquet)

    selected = audit[audit["use_for_direct_augmentation"]].copy()
    with selected_jsonl.open("w", encoding="utf-8") as handle:
        for _, row in selected.iterrows():
            payload = {
                "sample_id": row["sample_id"],
                "sample_index": int(row["sample_index"]),
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
                "teacher_jerk_l2_mean": float(row["teacher_jerk_l2_mean"]),
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
        teacher_jerk_l2_mean=audit["teacher_jerk_l2_mean"].to_numpy(dtype=np.float32),
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
        "sample_ids": selected["sample_id"].tolist(),
        "sample_indices": torch.from_numpy(selected_sample_indices),
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
        "teacher_jerk_l2_mean": torch.from_numpy(selected["teacher_jerk_l2_mean"].to_numpy(dtype=np.float32)),
        "terminal_mse": torch.from_numpy(selected["terminal_mse"].to_numpy(dtype=np.float32)),
        "oracle_gap_mse": torch.from_numpy(selected["oracle_gap_mse"].to_numpy(dtype=np.float32)),
    }
    atomic_torch(filtered_pt, filtered)
    # Same content with a name that makes the direct-augmentation semantics clear.
    atomic_torch(direct_records_pt, filtered)

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
            "scores_csv": str(args.scores_csv),
            "teacher_records": str(args.teacher_records),
            "stepwise_npz": str(args.stepwise_npz),
            "validation_summary": str(args.validation_summary),
            "candidate_dir": None if args.no_candidate_actions else str(args.candidate_dir),
        },
        "selection_policy": POLICY_NAME,
        "policy_explanation": (
            "Filtered Direct Augmentation: remove only unstable teacher agreement "
            "or top-quantile action/jerk/terminal/oracle-gap outliers. "
            "Relative margin and candidate diversity are diagnostics, not hard filters. "
            "Selected teachers are appended as equal-weight extra training samples."
        ),
        "primary_metric": args.primary_metric,
        "agreement_policy": args.agreement_policy,
        "samples": int(n),
        "tasks": int(audit["task_index"].nunique()),
        "episodes": int(audit["episode_index"].nunique()),
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
            "teacher_velocity_l2_mean": numeric_summary(audit["teacher_velocity_l2_mean"].to_numpy()),
            "teacher_acceleration_l2_mean": numeric_summary(audit["teacher_acceleration_l2_mean"].to_numpy()),
            "teacher_jerk_l2_mean": numeric_summary(audit["teacher_jerk_l2_mean"].to_numpy()),
            "terminal_mse": numeric_summary(audit["terminal_mse"].to_numpy()),
        },
        "candidate_diversity_distributions": {},
        "outputs": {
            "audit_csv": str(audit_csv),
            "audit_parquet": parquet_status,
            "task_summary_csv": str(task_csv),
            "task_summary_parquet": task_parquet_status,
            "selected_jsonl": str(selected_jsonl),
            "arrays_npz": str(arrays_npz),
            "filtered_records_pt": str(filtered_pt),
            "direct_records_pt": str(direct_records_pt),
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
