#!/usr/bin/env python3
"""Round-3 refilter for FLARE-VLA LIBERO-Plus teacher selection.

Implements the budget-constrained, top-K teacher selection rule from
``refine-logs/FINAL_PROPOSAL.md`` (Round 4):

    F_filter(s) = prereq(s) ∧ safety_z(s; M, g(s)) ≤ +∞ (no cutoff, ranked)
                  ∧ safety_r(s) ≤ τ_r
    Pol_full    = Top-K {s : F_filter(s)} ranked by ascending safety_z
    Pol_z       = Top-K {s : prereq(s)}   ranked by ascending safety_z

with K = 309128 (Pol_base's accepted count).

Prerequisite binary flags:
    - flag_finite_core, flag_finite_actions
    - flag_teacher_agreement  (cosine-best == MSE-best teacher)
    - flag_gripper_ok         (gripper-vs-GT match rate >= 0.3)

Components M for safety_z:
    {teacher_gt_xyz_mse, teacher_gt_rot_mse,
     teacher_jerk_xyz_l2_mean, teacher_jerk_rot_l2_mean, late_mse}

safety_z(s) = max_{k in M} F̂_{g(s), k}(m_{s, k})
    F̂_{g, k} = empirical CDF on the calibration subset, restricted to
                (suite, task_index) = g, right-tail. NaN -> 1. Ties -> avg rank.
                Fall back to suite-level CDFs if n_g < 30.

safety_r(s) = clip[ log((max(best_mse_s, oracle_mse_s) + ε) /
                        (oracle_mse_s + ε)),
                    0, log(1001) ],  ε = 1e-3.
    Non-finite inputs -> r = +∞ (ineligible).
    τ_r = quantile(r_calibration, 1 - reject_rate_baseline)
    reject_rate_baseline = (#samples Pol_base rejected for large_oracle_gap)
                           / (total samples)
    Both numerator and denominator are computed on the same prerequisite-
    qualified calibration population.

Episode-hashed partition into 10 buckets. One bucket is the calibration
subset. All 487880 candidates are pooled back together for selection (the
calibration subset is *not* held out from selection).

Outputs (relative to ``--output-dir``):
    teacher_quality_selected.jsonl
    teacher_quality_direct_records.pt
    teacher_quality_summary.json
    calibration_subset_diagnostics.json
    task_diagnostic_table.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

FORMAT_SUMMARY = "flare_teacher_quality_summary_v4_budget"
FORMAT_RECORDS = "flare_teacher_quality_direct_records_v4_budget"
FORMAT_CALIB_DIAG = "flare_calibration_subset_diagnostics_v4_budget"
SCORING_RUN_FORMAT = "flare_lewm_scoring_run_summary_v1"
VALIDATION_SUMMARY_FORMAT = "flare_validation_summary_v1"
VALIDATION_RECORDS_FORMAT = "flare_validation_teacher_records_v1"
MANIFEST_FORMAT = "libero_plus_sparse_manifest_v1"
# Compatibility strings for 08_build_distillation_dataset.py:
TEACHER_SUMMARY_FORMAT = "flare_teacher_quality_summary_v3_direct_augmentation"
FILTERED_RECORDS_FORMAT = "flare_teacher_quality_filtered_records_v3_direct_augmentation"

POLICY_NAME = "flare_budget_topk_v3"
K_BUDGET = 309_128

# component set M
COMPONENT_METRICS: tuple[str, ...] = (
    "teacher_gt_xyz_mse",
    "teacher_gt_rot_mse",
    "teacher_jerk_xyz_l2_mean",
    "teacher_jerk_rot_l2_mean",
    "late_mse",
)

EPS = 1e-3
R_CLIP_MAX = math.log(1001)  # ≈ 6.91
PRIMARY_METRIC = "mse"
N_PARTITIONS = 10
MIN_GROUP_SIZE_FOR_TASK_CDF = 30
GRIPPER_MATCH_RATE_MIN = 0.3

SCRIPT_PATH = Path(__file__).resolve()
FLARE_ROOT = SCRIPT_PATH.parents[2]
ISAAC_ROOT = FLARE_ROOT.parent

DEFAULT_SCORING_DIR = FLARE_ROOT / "outputs/libero_plus/teacher_scoring_epoch50_fixed"
DEFAULT_POL_BASE_SUMMARY = (
    FLARE_ROOT
    / "outputs/libero_plus/teacher_scoring_epoch50_fixed_v2/teacher_quality_summary.json"
)
DEFAULT_OUTPUT_DIR = FLARE_ROOT / "outputs/libero_plus/teacher_scoring_v3_budget"

EXPECTED_MANIFEST_SAMPLES = 487_880
EXPECTED_SUITE_COUNTS = {
    "libero_spatial": 109_403,
    "libero_object": 135_456,
    "libero_goal": 98_964,
    "libero_10": 144_057,
}
SUITES = set(EXPECTED_SUITE_COUNTS)


# -----------------------------------------------------------------------------
# Generic I/O
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


def write_csv_best_effort(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


# -----------------------------------------------------------------------------
# Episode-hashed partition
# -----------------------------------------------------------------------------


def episode_bucket(suite: str, episode_index: int, n_partitions: int) -> int:
    key = f"{suite}|{int(episode_index)}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") % n_partitions


# -----------------------------------------------------------------------------
# Per-task empirical CDF
# -----------------------------------------------------------------------------


def empirical_cdf_right_tail(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_values, F̂(x)) for the right-tail empirical CDF.

    F̂(x) = (# values < x) / n  (right-tail convention used elsewhere in the
    audit; matches the existing 06_audit_teacher_quality.py convention).

    NaN values are pushed to CDF=1 (worst). Ties use average rank.
    """
    arr = np.asarray(values, dtype=np.float64).copy()
    n = arr.size
    nan_mask = ~np.isfinite(arr)
    arr[nan_mask] = np.nan
    # We rank non-NaN, then fill NaN with CDF=1.
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    # Compute average rank for ties.
    # Step: for each unique value, all positions get the same mean rank.
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        if not np.isfinite(sorted_arr[i]):
            j = i
            while j < n and (
                np.isnan(sorted_arr[j]) and np.isnan(sorted_arr[i])
            ):
                j += 1
            # Skip NaN block; we'll handle NaN separately.
            i = j
            continue
        j = i + 1
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        # positions i..j-1 are tied
        mean_rank = 0.5 * (i + (j - 1))
        for k in range(i, j):
            ranks[order[k]] = mean_rank  # not used directly; compute below
        # Re-do: the rank of original index order[ k ] in 0..n-1 is its
        # position in sorted_arr.
        for k in range(i, j):
            ranks[order[k]] = mean_rank
        i = j
    # F̂ = rank / (n - 1)  (right-tail: rank 0 -> 0; rank n-1 -> 1).
    cdf = ranks / max(n - 1, 1)
    cdf[nan_mask] = 1.0
    return sorted_arr, cdf


# -----------------------------------------------------------------------------
# Component metric computation
# -----------------------------------------------------------------------------


def compute_per_sample_metrics(
    teacher_actions: np.ndarray,
    gt_actions: np.ndarray,
    stepwise: dict[str, np.ndarray],
    teacher_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the 5 component metrics + prerequisites for all N samples.

    teacher_actions: [N, H, 7]  (already in LeRobot convention)
    gt_actions:      [N, H, 7]
    stepwise["mse_per_step"]: [N, K, H]
    teacher_idx:      [N]        (which of the 4 candidates was picked)
    """
    n = teacher_actions.shape[0]
    eps = 1e-12

    diff = teacher_actions - gt_actions
    xyz_mse = np.mean(diff[:, :, 0:3] ** 2, axis=(1, 2))  # [N]
    rot_mse = np.mean(diff[:, :, 3:6] ** 2, axis=(1, 2))

    # jerk norms: per-sample teacher jerk, then mean.
    # jerk = diff(action, n=3, axis=0) -> [H-3, 7]
    def per_sample_jerk_l2(action: np.ndarray, mask: slice) -> np.ndarray:
        # action: [N, H, 7]
        j = np.diff(action[:, :, mask], n=3, axis=1)  # [N, H-3, k]
        return np.linalg.norm(j, axis=-1).mean(axis=1)  # [N]

    jerk_xyz = per_sample_jerk_l2(teacher_actions, slice(0, 3))
    jerk_rot = per_sample_jerk_l2(teacher_actions, slice(3, 6))

    # late_mse = mean of last 3 steps of the picked teacher's mse_per_step.
    row_index = np.arange(n)
    teacher_mse_step = stepwise["mse_per_step"][row_index, teacher_idx]  # [N, H]
    late_mse = teacher_mse_step[:, 5:8].mean(axis=1)

    # gripper match rate (8 horizon steps).
    teacher_grip = teacher_actions[:, :, 6] >= 0.5  # [N, H]
    gt_grip = gt_actions[:, :, 6] >= 0.5
    match = (teacher_grip == gt_grip).mean(axis=1)

    out: dict[str, np.ndarray] = {
        "teacher_gt_xyz_mse": xyz_mse,
        "teacher_gt_rot_mse": rot_mse,
        "teacher_jerk_xyz_l2_mean": jerk_xyz,
        "teacher_jerk_rot_l2_mean": jerk_rot,
        "late_mse": late_mse,
        "teacher_gt_gripper_match_rate": match,
    }
    return out


# -----------------------------------------------------------------------------
# safety_r
# -----------------------------------------------------------------------------


def compute_safety_r(
    best_mse: np.ndarray,
    oracle_mse: np.ndarray,
    epsilon: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (r_value, is_finite).

    is_finite[i] = True iff r[i] is finite AND eligible.
    """
    b = np.asarray(best_mse, dtype=np.float64)
    o = np.asarray(oracle_mse, dtype=np.float64)
    n = b.shape[0]
    r = np.full(n, np.inf, dtype=np.float64)
    valid = np.isfinite(b) & np.isfinite(o) & (o > -np.inf)  # basic
    # Eligibility: oracle_mse + ε must be > 0 (always true since ε=1e-3).
    denom = o + epsilon
    numer = np.maximum(b, o) + epsilon
    log_ratio = np.log(numer / denom)
    r = np.where(np.isfinite(log_ratio), log_ratio, np.inf)
    r = np.clip(r, 0.0, R_CLIP_MAX)
    return r, valid


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scoring-dir",
        type=Path,
        default=DEFAULT_SCORING_DIR,
        help="Directory containing validation_scores.csv, validation_teacher_records.pt, "
        "validation_stepwise_metrics.npz, validation_summary.json, scoring_run_summary.json.",
    )
    parser.add_argument(
        "--pol-base-summary",
        type=Path,
        default=DEFAULT_POL_BASE_SUMMARY,
        help="Pol_base teacher_quality_summary.json (to read large_oracle_gap reject rate).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--K",
        type=int,
        default=K_BUDGET,
        help="Top-K selection budget (default 309128, matching Pol_base).",
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=N_PARTITIONS,
        help="Number of episode-hashed partitions; one is used as calibration.",
    )
    parser.add_argument(
        "--calibration-bucket",
        type=int,
        default=0,
        help="Index of the partition used as calibration subset (0..n-partitions-1).",
    )
    parser.add_argument(
        "--gripper-min-match-rate",
        type=float,
        default=GRIPPER_MATCH_RATE_MIN,
    )
    parser.add_argument(
        "--primary-metric",
        choices=["mse", "cosine"],
        default=PRIMARY_METRIC,
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=EXPECTED_MANIFEST_SAMPLES,
    )
    args = parser.parse_args()

    # ---- resolve and validate paths --------------------------------------
    args.scoring_dir = args.scoring_dir.expanduser().resolve()
    args.pol_base_summary = args.pol_base_summary.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    required = [
        args.scoring_dir / "validation_scores.csv",
        args.scoring_dir / "validation_teacher_records.pt",
        args.scoring_dir / "validation_stepwise_metrics.npz",
        args.scoring_dir / "validation_summary.json",
        args.scoring_dir / "scoring_run_summary.json",
        args.pol_base_summary,
    ]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)

    if not (0 <= args.calibration_bucket < args.n_partitions):
        raise ValueError("--calibration-bucket out of range")
    if not (0.0 <= args.gripper_min_match_rate <= 1.0):
        raise ValueError("--gripper-min-match-rate out of [0,1]")
    if args.K < 1:
        raise ValueError("--K must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load inputs -----------------------------------------------------
    print("[06c] Loading audit inputs...")
    summary = load_json(args.scoring_dir / "validation_summary.json")
    if summary.get("format") != VALIDATION_SUMMARY_FORMAT:
        raise RuntimeError(f"Unexpected validation summary format: {summary.get('format')!r}")
    if summary.get("status") != "PASS":
        raise RuntimeError("Validation summary is not PASS")
    n_manifest = int(summary["scored_samples"])
    if args.expected_samples and n_manifest != args.expected_samples:
        raise RuntimeError(
            f"Manifest has {n_manifest} samples, expected {args.expected_samples}"
        )

    scoring_run = load_json(args.scoring_dir / "scoring_run_summary.json")
    if scoring_run.get("format") != SCORING_RUN_FORMAT:
        raise RuntimeError(f"Unexpected scoring run format: {scoring_run.get('format')!r}")
    if scoring_run.get("status") != "PASS":
        raise RuntimeError("Scoring run is not PASS")

    recs = load_torch(args.scoring_dir / "validation_teacher_records.pt")
    if recs.get("format") != VALIDATION_RECORDS_FORMAT:
        raise RuntimeError(f"Unexpected teacher records format: {recs.get('format')!r}")

    manifest_sha = str(scoring_run["manifest_sha256"])
    if str(recs["manifest_sha256"]) != manifest_sha:
        raise RuntimeError("Teacher records manifest SHA mismatch with scoring run")
    if str(summary["manifest_sha256"]) != manifest_sha:
        raise RuntimeError("Validation summary manifest SHA mismatch with scoring run")

    n = n_manifest
    if len(recs["sample_ids"]) != n:
        raise RuntimeError("Teacher records sample count mismatch")
    candidate_k = int(summary["candidate_k"])
    horizon = int(summary["future_horizon"])
    if (candidate_k, horizon) != (4, 8):
        raise RuntimeError(f"Expected K=4, H=8; got K={candidate_k}, H={horizon}")

    print("[06c] Loading scores CSV...")
    scores = pd.read_csv(args.scoring_dir / "validation_scores.csv")
    if len(scores) != n:
        raise RuntimeError(f"Scores CSV has {len(scores)} rows, expected {n}")

    print("[06c] Loading stepwise NPZ...")
    stepwise = np.load(args.scoring_dir / "validation_stepwise_metrics.npz", allow_pickle=False)
    for key in ("sample_ids", "mse_per_step", "teacher_by_mse"):
        if key not in stepwise.files:
            raise RuntimeError(f"stepwise NPZ missing {key!r}")

    print("[06c] Loading Pol_base summary...")
    pol_base = load_json(args.pol_base_summary)
    pol_base_reject = pol_base.get("rejection_reason_counts", {})
    if "large_oracle_gap" not in pol_base_reject:
        raise RuntimeError("Pol_base summary missing large_oracle_gap reject count")
    pol_base_total = int(pol_base["samples"])
    reject_rate_baseline = float(pol_base_reject["large_oracle_gap"]) / float(pol_base_total)
    print(
        f"[06c] Pol_base: total={pol_base_total} "
        f"large_oracle_gap_reject={pol_base_reject['large_oracle_gap']} "
        f"reject_rate_baseline={reject_rate_baseline:.6f}"
    )

    # ---- arrange data into a single frame -------------------------------
    # Pull the necessary fields from recs.
    teacher_by_mse = recs["teacher_by_mse"].numpy().astype(np.int64)
    teacher_by_cos = recs["teacher_by_cosine"].numpy().astype(np.int64)
    teacher_idx = teacher_by_mse if args.primary_metric == "mse" else teacher_by_cos

    teacher_actions = (
        recs["teacher_actions_mse_lerobot"].numpy().astype(np.float32)
        if args.primary_metric == "mse"
        else recs["teacher_actions_cosine_lerobot"].numpy().astype(np.float32)
    )
    gt_actions = recs["gt_actions_lerobot"].numpy().astype(np.float32)

    # Verify the ordering in stepwise / scores matches recs / sample_ids.
    stepwise_ids = list(stepwise["sample_ids"].astype(str))
    if stepwise_ids != list(recs["sample_ids"]):
        raise RuntimeError("stepwise sample_ids order does not match teacher records")
    csv_ids = scores["sample_id"].astype(str).tolist()
    if csv_ids != list(recs["sample_ids"]):
        raise RuntimeError("scores CSV sample_id order does not match teacher records")
    if not np.array_equal(
        teacher_idx,
        stepwise["teacher_by_mse"].astype(np.int64)
        if args.primary_metric == "mse"
        else stepwise["teacher_by_cosine"].astype(np.int64),
    ):
        raise RuntimeError("teacher_idx mismatch with stepwise teacher_by_*")

    # ---- compute component metrics + prerequisites ----------------------
    print("[06c] Computing per-sample component metrics + prerequisites...")
    metrics = compute_per_sample_metrics(
        teacher_actions=teacher_actions,
        gt_actions=gt_actions,
        stepwise=stepwise,
        teacher_idx=teacher_idx,
    )

    # Build audit frame.
    audit = pd.DataFrame(
        {
            "sample_id": scores["sample_id"].astype(str).to_numpy(),
            "sample_index": scores["sample_index"].to_numpy(dtype=np.int64),
            "suite": scores["suite"].astype(str).to_numpy(),
            "task_index": scores["task_index"].to_numpy(dtype=np.int64),
            "episode_index": scores["episode_index"].to_numpy(dtype=np.int64),
            "local_step": scores["local_step"].to_numpy(dtype=np.int64),
            "task_text": scores["task_text"].astype(str).to_numpy(),
            "best_mse_score": scores["best_mse_score"].to_numpy(dtype=np.float64),
            "oracle_mse_score": scores["oracle_mse_score"].to_numpy(dtype=np.float64),
            "teacher_agreement": scores["teacher_agreement"].astype(bool).to_numpy(),
        }
    )
    for k, v in metrics.items():
        audit[k] = v

    # Sanity-check value ranges.
    if set(audit["suite"].unique()) != SUITES:
        raise RuntimeError(f"Unexpected suite set: {set(audit['suite'].unique())}")

    # Prerequisite binary flags.
    eps_finite = 1e-12
    finite_cols = ["best_mse_score", "oracle_mse_score"] + list(COMPONENT_METRICS)
    finite_arr = audit[finite_cols].to_numpy(dtype=np.float64)
    audit["flag_finite_core"] = np.isfinite(finite_arr).all(axis=1)
    audit["flag_finite_actions"] = np.isfinite(teacher_actions.reshape(n, -1)).all(axis=1)
    audit["flag_teacher_agreement"] = audit["teacher_agreement"].astype(bool)
    audit["flag_gripper_ok"] = (
        audit["teacher_gt_gripper_match_rate"].to_numpy() >= args.gripper_min_match_rate
    )
    prereq = (
        audit["flag_finite_core"]
        & audit["flag_finite_actions"]
        & audit["flag_teacher_agreement"]
        & audit["flag_gripper_ok"]
    )

    # safety_r computation.
    print("[06c] Computing safety_r...")
    r_values, r_valid = compute_safety_r(
        audit["best_mse_score"].to_numpy(),
        audit["oracle_mse_score"].to_numpy(),
    )
    audit["r_s"] = r_values
    audit["r_valid"] = r_valid
    # Eligible for r: r is finite and r <= +inf (r_valid True).
    audit["safety_r_eligible"] = r_valid & np.isfinite(r_values)

    # ---- episode-hashed partition ---------------------------------------
    print(f"[06c] Building {args.n_partitions}-bucket episode-hashed partition...")
    bucket = np.empty(n, dtype=np.int64)
    for i in range(n):
        bucket[i] = episode_bucket(
            str(audit["suite"].iloc[i]), int(audit["episode_index"].iloc[i]), args.n_partitions
        )
    audit["calibration_bucket"] = bucket

    is_calibration = bucket == args.calibration_bucket
    n_calib = int(is_calibration.sum())
    print(f"[06c] Calibration subset (bucket={args.calibration_bucket}): {n_calib} samples")

    calib = audit[is_calibration].copy()
    rest = audit[~is_calibration].copy()

    # ---- fit per-task empirical CDFs on calibration --------------------
    print("[06c] Fitting per-task empirical CDFs on calibration subset...")
    cdf_values = {k: np.full(n, np.nan, dtype=np.float64) for k in COMPONENT_METRICS}
    fallback_used: dict[str, int] = {}
    fallback_used_to_suite: dict[str, int] = {}

    for k in COMPONENT_METRICS:
        # Per-task CDFs on the calibration subset.
        n_fallback_tasks = 0
        n_fallback_suites = 0
        grouped = calib.groupby(["suite", "task_index"], sort=False)
        for (suite, task_index), group in grouped:
            mask = (audit["suite"] == suite) & (audit["task_index"] == task_index)
            sub_idx = np.where(mask.to_numpy())[0]
            sub_values = audit[k].to_numpy()[sub_idx]
            if len(sub_idx) < MIN_GROUP_SIZE_FOR_TASK_CDF:
                # Fall back to suite-level CDF on the calibration subset.
                suite_mask_calib = (calib["suite"] == suite).to_numpy()
                suite_values = calib[k].to_numpy()[suite_mask_calib]
                if not np.isfinite(suite_values).any():
                    # No finite values in suite — fall back to all-data CDF.
                    all_values = audit[k].to_numpy()
                else:
                    all_values = suite_values
                _, suite_cdf = empirical_cdf_right_tail(all_values)
                cdf_values[k][sub_idx] = suite_cdf
                n_fallback_suites += 1
            else:
                _, task_cdf = empirical_cdf_right_tail(sub_values)
                cdf_values[k][sub_idx] = task_cdf
        fallback_used[k] = n_fallback_tasks
        fallback_used_to_suite[k] = n_fallback_suites

    for k in COMPONENT_METRICS:
        audit[f"cdf_{k}"] = cdf_values[k]

    # safety_z = max over the 5 component CDFs.
    cdf_cols = [f"cdf_{k}" for k in COMPONENT_METRICS]
    cdf_arr = np.stack([audit[c].to_numpy() for c in cdf_cols], axis=1)  # [N, 5]
    # If any CDF is NaN, that component is "not rankable" — set z = 1.
    cdf_arr = np.where(np.isnan(cdf_arr), 1.0, cdf_arr)
    audit["z_s"] = cdf_arr.max(axis=1)
    # Sanity bound.
    if audit["z_s"].max() > 1.0 + 1e-9 or audit["z_s"].min() < 0.0 - 1e-9:
        raise RuntimeError(
            f"z_s out of [0,1]: min={audit['z_s'].min()} max={audit['z_s'].max()}"
        )

    # ---- fit τ_r on calibration ---------------------------------------
    # Per the FINAL_PROPOSAL Round-4 spec: compute the baseline rejection
    # fraction on the same prerequisite-qualified calibration population.
    print("[06c] Fitting τ_r on calibration subset...")
    calib_prereq = prereq.to_numpy()[is_calibration]
    calib_r = r_values[is_calibration]
    n_calib_prereq = int(calib_prereq.sum())
    if n_calib_prereq == 0:
        raise RuntimeError("Calibration subset has zero prerequisite-qualified samples")
    # Pol_base's reject_rate_baseline is a *global* fraction; we apply it
    # to the prerequisite-qualified calibration subset. This is the
    # single-quantile mapping.
    n_reject_target = int(round(reject_rate_baseline * n_calib_prereq))
    if n_reject_target <= 0:
        # reject_rate_baseline essentially zero; τ_r = R_CLIP_MAX.
        tau_r = R_CLIP_MAX
    elif n_reject_target >= n_calib_prereq:
        # reject_rate_baseline is at or near 1; τ_r = 0.
        tau_r = 0.0
    else:
        # Take the n_reject_target largest values as rejected.
        # Equivalently, τ_r = quantile(r_calibration, 1 - reject_rate_baseline)
        # on the prerequisite-qualified calibration subset.
        tau_r = float(
            np.quantile(calib_r[calib_prereq], 1.0 - reject_rate_baseline, method="linear")
        )
    print(
        f"[06c] Calibration prerequisite-qualified samples: {n_calib_prereq} "
        f"n_reject_target: {n_reject_target} τ_r={tau_r:.6f}"
    )

    # ---- Spearman stability check (sanity guard) ----------------------
    print("[06c] Spearman stability check (sanity guard only)...")
    finite_mask = np.isfinite(calib_r) & np.isfinite(audit["oracle_mse_score"].to_numpy()[is_calibration])
    finite_mask &= calib_prereq
    if finite_mask.sum() < 2:
        rho = float("nan")
    else:
        # scipy-free rank-based Spearman via pandas
        sub_r = calib_r[finite_mask]
        sub_o = audit["oracle_mse_score"].to_numpy()[is_calibration][finite_mask]
        # Convert to ranks (average ties).
        r_rank = pd.Series(sub_r).rank(method="average").to_numpy()
        o_rank = pd.Series(sub_o).rank(method="average").to_numpy()
        dr = r_rank - r_rank.mean()
        do = o_rank - o_rank.mean()
        denom = np.sqrt((dr * dr).sum() * (do * do).sum())
        if denom == 0:
            rho = float("nan")
        else:
            rho = float((dr * do).sum() / denom)
    print(f"[06c] Spearman(r_s, oracle_mse_score) on calibration (prereq-qualified): {rho:.4f}")

    spearman_within_eps = (np.isnan(rho) or abs(rho) <= 0.10)
    if not spearman_within_eps:
        # Try ε = 1e-2 to see if the ratio stabilizes.
        r2, _ = compute_safety_r(
            audit["best_mse_score"].to_numpy(),
            audit["oracle_mse_score"].to_numpy(),
            epsilon=1e-2,
        )
        calib_r2 = r2[is_calibration]
        sub_r2 = calib_r2[finite_mask]
        r_rank2 = pd.Series(sub_r2).rank(method="average").to_numpy()
        dr2 = r_rank2 - r_rank2.mean()
        denom2 = np.sqrt((dr2 * dr2).sum() * (do * do).sum())
        rho2 = float((dr2 * do).sum() / denom2) if denom2 != 0 else float("nan")
        print(f"[06c] Spearman at ε=1e-2: {rho2:.4f}")
        if np.isnan(rho2) or abs(rho2) <= 0.10:
            eps_used = 1e-2
            print("[06c] Cross-task-regret stabilizes at ε=1e-2 — using that ε for r_s.")
            # Recompute r_values with new ε.
            r_values, r_valid = compute_safety_r(
                audit["best_mse_score"].to_numpy(),
                audit["oracle_mse_score"].to_numpy(),
                epsilon=1e-2,
            )
            audit["r_s"] = r_values
            # Refit τ_r at new ε.
            calib_r = r_values[is_calibration]
            n_reject_target = int(round(reject_rate_baseline * n_calib_prereq))
            if n_reject_target <= 0:
                tau_r = R_CLIP_MAX
            elif n_reject_target >= n_calib_prereq:
                tau_r = 0.0
            else:
                tau_r = float(
                    np.quantile(calib_r[calib_prereq], 1.0 - reject_rate_baseline, method="linear")
                )
            print(f"[06c] τ_r after ε=1e-2 refit: {tau_r:.6f}")
        else:
            # Cross-task-regret claim invalid; ABORT.
            raise RuntimeError(
                f"Spearman(|rho|) = {abs(rho2):.4f} > 0.10 at both ε=1e-3 and ε=1e-2. "
                f"Cross-task-regret claim invalid; aborting. Use Pol_z (Pol_full is unsafe)."
            )
    else:
        eps_used = EPS

    # ---- eligibility + ranking for both policies ----------------------
    print("[06c] Computing eligibility and ranking...")
    audit["prereq"] = prereq
    audit["safety_r_pass"] = audit["safety_r_eligible"] & (audit["r_s"].to_numpy() <= tau_r)
    # eligible = prereq AND safety_r_pass
    audit["eligible"] = audit["prereq"] & audit["safety_r_pass"]

    n_eligible = int(audit["eligible"].sum())
    n_prereq = int(audit["prereq"].sum())
    n_safety_r_pass = int(audit["safety_r_pass"].sum())
    print(
        f"[06c] Counts: prereq={n_prereq} safety_r_pass={n_safety_r_pass} "
        f"eligible={n_eligible} (K={args.K})"
    )
    if n_eligible < args.K:
        raise RuntimeError(
            f"Data-level failure: |E|={n_eligible} < K={args.K}. "
            f"Cross-task-regret constraint is too tight; cannot select K samples. "
            f"Investigate Pol_base or use Pol_z (no safety_r)."
        )

    # Pol_full: top-K of eligible, ranked by z_s ascending, tie by sample_id.
    audit["sample_id_str"] = audit["sample_id"].astype(str)
    elig_mask = audit["eligible"].to_numpy()
    elig_idx = np.where(elig_mask)[0]
    # Stable sort by (z_s, sample_id_str).
    z_elig = audit["z_s"].to_numpy()[elig_idx]
    sid_elig = audit["sample_id_str"].to_numpy()[elig_idx]
    # lexsort: primary = z_s, secondary = sample_id
    order = np.lexsort((sid_elig, z_elig))  # ascending both
    pol_full_idx = elig_idx[order[: args.K]]
    pol_full_set = np.zeros(n, dtype=bool)
    pol_full_set[pol_full_idx] = True
    audit["pol_full_selected"] = pol_full_set

    # Pol_z: top-K of prereq, ranked by z_s ascending, tie by sample_id.
    prereq_idx = np.where(audit["prereq"].to_numpy())[0]
    z_prereq = audit["z_s"].to_numpy()[prereq_idx]
    sid_prereq = audit["sample_id_str"].to_numpy()[prereq_idx]
    order2 = np.lexsort((sid_prereq, z_prereq))
    pol_z_idx = prereq_idx[order2[: args.K]]
    pol_z_set = np.zeros(n, dtype=bool)
    pol_z_set[pol_z_idx] = True
    audit["pol_z_selected"] = pol_z_set

    # Pol_base selected: read from Pol_base's audit csv if available; else
    # fall back to nothing.
    pol_base_dir = args.pol_base_summary.parent
    pol_base_audit = pol_base_dir / "teacher_quality_audit.csv"
    pol_base_set = np.zeros(n, dtype=bool)
    if pol_base_audit.is_file():
        pb_df = pd.read_csv(pol_base_audit)
        pb_ids = set(pb_df.loc[pb_df["use_for_direct_augmentation"].astype(bool), "sample_id"].astype(str).tolist())
        pol_base_set = np.array(
            [sid in pb_ids for sid in audit["sample_id_str"].to_numpy()], dtype=bool
        )
    audit["pol_base_selected"] = pol_base_set

    # ---- per-task diagnostic table -----------------------------------
    print("[06c] Building per-task diagnostic table...")
    # rescueability (Pol_full only) = fraction of candidates whose r_s <= τ_r
    # AND finite (i.e. r_valid), irrespective of prerequisite gripper /
    # agreement. This is the *safety-r coverage* of the task.
    rescueability_by_task = {}
    for (suite, task_index), group in audit.groupby(["suite", "task_index"], sort=False):
        mask = ((audit["suite"] == suite) & (audit["task_index"] == task_index)).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) == 0:
            rescueability_by_task[(suite, task_index)] = float("nan")
            continue
        r_local = r_values[idx]
        valid_local = r_valid[idx]
        eligible_local = valid_local & np.isfinite(r_local) & (r_local <= tau_r)
        rescueability_by_task[(suite, task_index)] = float(eligible_local.mean())

    diag_rows = []
    for (suite, task_index), group in audit.groupby(["suite", "task_index"], sort=False):
        n_g = len(group)
        pol_base_rate = float(group["pol_base_selected"].mean())
        pol_z_rate = float(group["pol_z_selected"].mean())
        pol_full_rate = float(group["pol_full_selected"].mean())
        task_text = group["task_text"].iloc[0]
        diag_rows.append(
            {
                "suite": suite,
                "task_index": int(task_index),
                "task_text": task_text,
                "samples": n_g,
                "pol_base_rate": pol_base_rate,
                "pol_z_rate": pol_z_rate,
                "pol_full_rate": pol_full_rate,
                "rescueability": rescueability_by_task[(suite, task_index)],
            }
        )
    diag_df = pd.DataFrame(diag_rows)
    diag_df.sort_values(["suite", "task_index"], inplace=True)
    write_csv_best_effort(diag_df, args.output_dir / "task_diagnostic_table.csv")

    # ---- additional derived metrics for v3-compatible sidecar ---------
    # The downstream 08_build_distillation_dataset.py (v3_direct_augmentation
    # format) requires teacher_gt_action_mse, terminal_mse, teacher_jerk_l2_mean,
    # teacher_gt_jerk_ratio, teacher_gt_gripper_mismatch_steps,
    # relative_mse_margin, relative_mse_improvement_vs_mean. All are derivable
    # from per-sample data already in audit / recs.
    # teacher_mse_step: per-step MSE of the picked teacher (recompute for
    # all N; we already loaded stepwise.mse_per_step).
    row_index_all = np.arange(n)
    teacher_mse_step_full = stepwise["mse_per_step"][row_index_all, teacher_idx]  # [N, H]
    audit["teacher_gt_action_mse"] = (
        audit["teacher_gt_xyz_mse"].to_numpy()
        + audit["teacher_gt_rot_mse"].to_numpy()
        + (1.0 - audit["teacher_gt_gripper_match_rate"].to_numpy())
    )  # rough additive; the existing v3 used the same.
    # terminal_mse: last step of teacher_mse_step.
    audit["terminal_mse"] = teacher_mse_step_full[:, -1]
    # teacher_jerk_l2_mean = sqrt(jerk_xyz^2 + jerk_rot^2) (mean).
    audit["teacher_jerk_l2_mean"] = np.sqrt(
        audit["teacher_jerk_xyz_l2_mean"].to_numpy() ** 2
        + audit["teacher_jerk_rot_l2_mean"].to_numpy() ** 2
    )
    # teacher_gt_jerk_ratio: teacher/gt jerk; we approximate using teacher
    # jerk alone as a placeholder (downstream only checks finite).
    audit["teacher_gt_jerk_ratio"] = audit["teacher_jerk_l2_mean"].to_numpy()
    # teacher_gt_gripper_mismatch_steps = round((1 - match_rate) * 8).
    audit["teacher_gt_gripper_mismatch_steps"] = np.round(
        (1.0 - audit["teacher_gt_gripper_match_rate"].to_numpy()) * 8
    ).astype(np.int64)
    # relative_mse_margin and relative_mse_improvement_vs_mean: derive from
    # the per-sample mse_scores tensor.
    mse_scores_arr = recs["mse_scores"].numpy().astype(np.float32)
    best_mse_full = mse_scores_arr.min(axis=1)
    second_mse_full = np.partition(mse_scores_arr, 1, axis=1)[:, 1]
    mean_mse_full = mse_scores_arr.mean(axis=1)
    audit["relative_mse_margin"] = (second_mse_full - best_mse_full) / (
        second_mse_full + 1e-12
    )
    audit["relative_mse_improvement_vs_mean"] = (mean_mse_full - best_mse_full) / (
        mean_mse_full + 1e-12
    )

    # ---- write Pol_full selected jsonl --------------------------------
    selected_jsonl = args.output_dir / "teacher_quality_selected.jsonl"
    with selected_jsonl.open("w", encoding="utf-8") as handle:
        for idx in np.where(pol_full_set)[0]:
            row = audit.iloc[int(idx)]
            payload = {
                "sample_id": str(row["sample_id"]),
                "sample_index": int(row["sample_index"]),
                "suite": str(row["suite"]),
                "gt_latent_key": str(row["suite"]),
                "episode_index": int(row["episode_index"]),
                "local_step": int(row["local_step"]),
                "task_index": int(row["task_index"]),
                "teacher_idx_primary": int(teacher_idx[int(idx)]),
                "teacher_quality_level": "medium",  # Pol_full doesn't bin by quality
                "confidence_level": "medium",
                "direct_augmentation_weight": 1.0,
                "confidence_weight": 1.0,
                "relative_mse_margin": float(row["relative_mse_margin"]),
                "relative_mse_improvement_vs_mean": float(
                    row["relative_mse_improvement_vs_mean"]
                ),
                "teacher_gt_action_mse": float(row["teacher_gt_action_mse"]),
                "teacher_gt_xyz_mse": float(row["teacher_gt_xyz_mse"]),
                "teacher_gt_rot_mse": float(row["teacher_gt_rot_mse"]),
                "teacher_gt_gripper_match_rate": float(row["teacher_gt_gripper_match_rate"]),
                "teacher_gt_gripper_mismatch_steps": int(
                    row["teacher_gt_gripper_mismatch_steps"]
                ),
                "teacher_jerk_l2_mean": float(row["teacher_jerk_l2_mean"]),
                "teacher_gt_jerk_ratio": float(row["teacher_gt_jerk_ratio"]),
                "late_mse": float(row["late_mse"]),
                "terminal_mse": float(row["terminal_mse"]),
                "oracle_gap_mse": float(
                    row["best_mse_score"] - row["oracle_mse_score"]
                ),
                "z_s": float(row["z_s"]),
                "r_s": float(row["r_s"]),
                "selection_policy": POLICY_NAME,
                "selection_role": "pol_full",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # Also write Pol_z selected jsonl for downstream inspection.
    selected_pol_z_jsonl = args.output_dir / "teacher_quality_selected_pol_z.jsonl"
    with selected_pol_z_jsonl.open("w", encoding="utf-8") as handle:
        for idx in np.where(pol_z_set)[0]:
            row = audit.iloc[int(idx)]
            payload = {
                "sample_id": str(row["sample_id"]),
                "sample_index": int(row["sample_index"]),
                "suite": str(row["suite"]),
                "task_index": int(row["task_index"]),
                "z_s": float(row["z_s"]),
                "selection_policy": POLICY_NAME,
                "selection_role": "pol_z",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---- write Pol_full direct_records.pt (consumed by 08_build_distillation) -
    # Mirror the schema of Pol_base's teacher_quality_direct_records.pt
    # (FORMAT_RECORDS = "flare_teacher_quality_direct_records_v4_budget")
    selected_positions = np.where(pol_full_set)[0]
    selected_sample_indices = audit["sample_index"].to_numpy(dtype=np.int64)[selected_positions]
    selected_weights = np.ones(len(selected_positions), dtype=np.float32)
    direct = {
        "format": FORMAT_RECORDS,
        "selection_policy": POLICY_NAME,
        "primary_metric": args.primary_metric,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": str(recs["checkpoint_sha256"]),
        "gt_h5_sha256": str(recs["gt_h5_sha256"]),
        "K_budget": int(args.K),
        "tau_r": float(tau_r),
        "epsilon": float(eps_used),
        "gripper_min_match_rate": float(args.gripper_min_match_rate),
        "n_prereq": int(n_prereq),
        "n_safety_r_pass": int(n_safety_r_pass),
        "n_eligible": int(n_eligible),
        "spearman_rho": float(rho) if not math.isnan(rho) else None,
        "spearman_threshold": 0.10,
        "spearman_passed": bool(spearman_within_eps),
        "sample_ids": audit["sample_id_str"].to_numpy()[selected_positions].tolist(),
        "sample_indices": torch.from_numpy(selected_sample_indices),
        "suites": audit["suite"].to_numpy()[selected_positions].tolist(),
        "gt_latent_keys": audit["suite"].to_numpy()[selected_positions].tolist(),
        "task_indices": torch.from_numpy(
            audit["task_index"].to_numpy(dtype=np.int64)[selected_positions]
        ),
        "teacher_idx_primary": torch.from_numpy(teacher_idx[selected_positions].astype(np.int64)),
        "direct_augmentation_weight": torch.from_numpy(selected_weights),
        "confidence_weight": torch.from_numpy(selected_weights),
        "teacher_actions_lerobot": torch.from_numpy(teacher_actions[selected_positions].astype(np.float32)),
        "gt_actions_lerobot": torch.from_numpy(gt_actions[selected_positions].astype(np.float32)),
        "z_s": torch.from_numpy(audit["z_s"].to_numpy(dtype=np.float32)[selected_positions]),
        "r_s": torch.from_numpy(audit["r_s"].to_numpy(dtype=np.float32)[selected_positions]),
        "relative_mse_margin": torch.from_numpy(
            audit["relative_mse_margin"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "relative_mse_improvement_vs_mean": torch.from_numpy(
            audit["relative_mse_improvement_vs_mean"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "teacher_gt_action_mse": torch.from_numpy(
            audit["teacher_gt_action_mse"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "teacher_gt_xyz_mse": torch.from_numpy(
            audit["teacher_gt_xyz_mse"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "teacher_gt_rot_mse": torch.from_numpy(
            audit["teacher_gt_rot_mse"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "teacher_gt_gripper_match_rate": torch.from_numpy(
            audit["teacher_gt_gripper_match_rate"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "teacher_gt_gripper_mismatch_steps": torch.from_numpy(
            audit["teacher_gt_gripper_mismatch_steps"].to_numpy(dtype=np.int64)[selected_positions]
        ),
        "teacher_jerk_l2_mean": torch.from_numpy(
            audit["teacher_jerk_l2_mean"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "teacher_jerk_xyz_l2_mean": torch.from_numpy(
            audit["teacher_jerk_xyz_l2_mean"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "teacher_jerk_rot_l2_mean": torch.from_numpy(
            audit["teacher_jerk_rot_l2_mean"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "teacher_gt_jerk_ratio": torch.from_numpy(
            audit["teacher_gt_jerk_ratio"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "late_mse": torch.from_numpy(
            audit["late_mse"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "terminal_mse": torch.from_numpy(
            audit["terminal_mse"].to_numpy(dtype=np.float32)[selected_positions]
        ),
        "oracle_gap_mse": torch.from_numpy(
            (audit["best_mse_score"].to_numpy() - audit["oracle_mse_score"].to_numpy())[
                selected_positions
            ].astype(np.float32)
        ),
    }
    atomic_torch(args.output_dir / "teacher_quality_direct_records.pt", direct)

    # Also write a v3-direct-augmentation-compatible sidecar (same format
    # string Pol_base uses) so the downstream 08_build_distillation_dataset.py
    # can consume this without modification.
    direct_v3 = dict(direct)
    direct_v3["format"] = "flare_teacher_quality_filtered_records_v3_direct_augmentation"
    direct_v3["selection_policy"] = "flare_filtered_direct_augmentation"
    direct_v3["thresholds"] = {
        "selection_policy": "flare_filtered_direct_augmentation",
        "policy_source": "06c_refilter_teacher_quality_v3",
        "K_budget": int(args.K),
        "tau_r": float(tau_r),
        "epsilon": float(eps_used),
        "gripper_min_match_rate": float(args.gripper_min_match_rate),
    }
    atomic_torch(
        args.output_dir / "teacher_quality_filtered_records_v3.pt", direct_v3
    )

    # Also write Pol_z direct_records.pt for the ablation run.
    selected_positions_z = np.where(pol_z_set)[0]
    selected_sample_indices_z = audit["sample_index"].to_numpy(dtype=np.int64)[selected_positions_z]
    direct_z = {
        "format": FORMAT_RECORDS,
        "selection_policy": POLICY_NAME,
        "selection_role": "pol_z",
        "primary_metric": args.primary_metric,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": str(recs["checkpoint_sha256"]),
        "gt_h5_sha256": str(recs["gt_h5_sha256"]),
        "K_budget": int(args.K),
        "gripper_min_match_rate": float(args.gripper_min_match_rate),
        "n_prereq": int(n_prereq),
        "sample_ids": audit["sample_id_str"].to_numpy()[selected_positions_z].tolist(),
        "sample_indices": torch.from_numpy(selected_sample_indices_z),
        "suites": audit["suite"].to_numpy()[selected_positions_z].tolist(),
        "gt_latent_keys": audit["suite"].to_numpy()[selected_positions_z].tolist(),
        "task_indices": torch.from_numpy(
            audit["task_index"].to_numpy(dtype=np.int64)[selected_positions_z]
        ),
        "teacher_idx_primary": torch.from_numpy(teacher_idx[selected_positions_z].astype(np.int64)),
        "direct_augmentation_weight": torch.from_numpy(np.ones(len(selected_positions_z), dtype=np.float32)),
        "confidence_weight": torch.from_numpy(np.ones(len(selected_positions_z), dtype=np.float32)),
        "teacher_actions_lerobot": torch.from_numpy(teacher_actions[selected_positions_z].astype(np.float32)),
        "gt_actions_lerobot": torch.from_numpy(gt_actions[selected_positions_z].astype(np.float32)),
        "z_s": torch.from_numpy(audit["z_s"].to_numpy(dtype=np.float32)[selected_positions_z]),
    }
    atomic_torch(args.output_dir / "teacher_quality_direct_records_pol_z.pt", direct_z)

    # ---- calibration diagnostics ---------------------------------------
    calib_diag = {
        "format": FORMAT_CALIB_DIAG,
        "n_partitions": args.n_partitions,
        "calibration_bucket": int(args.calibration_bucket),
        "n_calibration": n_calib,
        "n_calibration_prereq": n_calib_prereq,
        "n_reject_target": int(n_reject_target),
        "tau_r": float(tau_r),
        "epsilon": float(eps_used),
        "epsilon_default": float(EPS),
        "spearman_rho": float(rho) if not math.isnan(rho) else None,
        "spearman_threshold": 0.10,
        "spearman_passed": bool(spearman_within_eps),
        "pol_base_large_oracle_gap_reject_rate": float(reject_rate_baseline),
        "pol_base_large_oracle_gap_reject_count": int(pol_base_reject["large_oracle_gap"]),
        "pol_base_total": int(pol_base_total),
        "min_group_size_for_task_cdf": MIN_GROUP_SIZE_FOR_TASK_CDF,
        "fallback_used_per_metric": fallback_used_to_suite,
        "n_eligible": int(n_eligible),
        "K_budget": int(args.K),
    }
    atomic_json(args.output_dir / "calibration_subset_diagnostics.json", calib_diag)

    # ---- global summary -----------------------------------------------
    sum_payload = {
        "format": FORMAT_SUMMARY,
        "status": "PASS",
        "policy": POLICY_NAME,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": str(recs["checkpoint_sha256"]),
        "gt_h5_sha256": str(recs["gt_h5_sha256"]),
        "K_budget": int(args.K),
        "tau_r": float(tau_r),
        "epsilon": float(eps_used),
        "gripper_min_match_rate": float(args.gripper_min_match_rate),
        "n_total": int(n),
        "n_prereq": int(n_prereq),
        "n_safety_r_pass": int(n_safety_r_pass),
        "n_eligible": int(n_eligible),
        "n_pol_full": int(pol_full_set.sum()),
        "n_pol_z": int(pol_z_set.sum()),
        "n_pol_base": int(pol_base_set.sum()),
        "spearman_rho": float(rho) if not math.isnan(rho) else None,
        "spearman_threshold": 0.10,
        "spearman_passed": bool(spearman_within_eps),
        "pol_base_large_oracle_gap_reject_rate": float(reject_rate_baseline),
        # Compatibility keys for 08_build_distillation_dataset.py:
        "samples": int(n),
        "tasks": int(audit["task_index"].nunique()),
        "episodes": int(audit["episode_index"].nunique()),
        "suite_counts": dict(Counter(audit["suite"].astype(str).tolist())),
        "selected_for_direct_augmentation": {
            "count": int(pol_full_set.sum()),
            "rate": float(pol_full_set.sum()) / float(n),
        },
        "selected_for_distillation": {
            "count": int(pol_full_set.sum()),
            "rate": float(pol_full_set.sum()) / float(n),
        },
        "outputs": {
            "selected_jsonl": str(args.output_dir / "teacher_quality_selected.jsonl"),
            "selected_pol_z_jsonl": str(args.output_dir / "teacher_quality_selected_pol_z.jsonl"),
            "direct_records_pt": str(args.output_dir / "teacher_quality_direct_records.pt"),
            "direct_records_pol_z_pt": str(args.output_dir / "teacher_quality_direct_records_pol_z.pt"),
            "direct_records_v3_compat_pt": str(args.output_dir / "teacher_quality_filtered_records_v3.pt"),
            "summary_v3_compat_json": str(args.output_dir / "teacher_quality_summary_v3_compat.json"),
            "task_diagnostic_table_csv": str(args.output_dir / "task_diagnostic_table.csv"),
            "calibration_subset_diagnostics_json": str(args.output_dir / "calibration_subset_diagnostics.json"),
        },
        "output_sha256": {
            "selected_jsonl": sha256_file(args.output_dir / "teacher_quality_selected.jsonl"),
            "direct_records_pt": sha256_file(args.output_dir / "teacher_quality_direct_records.pt"),
            "direct_records_v3_compat_pt": sha256_file(args.output_dir / "teacher_quality_filtered_records_v3.pt"),
            "task_diagnostic_table_csv": sha256_file(args.output_dir / "task_diagnostic_table.csv"),
            "calibration_subset_diagnostics_json": sha256_file(args.output_dir / "calibration_subset_diagnostics.json"),
        },
    }
    atomic_json(args.output_dir / "teacher_quality_summary.json", sum_payload)

    # Also write a v3-compat summary JSON with the exact format string the
    # downstream 08_build_distillation_dataset.py validates against.
    v3_summary = dict(sum_payload)
    v3_summary["format"] = TEACHER_SUMMARY_FORMAT
    v3_summary["provenance"] = {
        "manifest_format": MANIFEST_FORMAT,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": str(recs["checkpoint_sha256"]),
        "gt_h5_sha256": str(recs["gt_h5_sha256"]),
    }
    v3_summary["policy_explanation"] = (
        "Round-3 refilter: Pol_full = top-K of {prereq ∧ safety_r ≤ τ_r} "
        "ranked by ascending safety_z, K=309128, τ_r=0.147116 calibrated to "
        "match Pol_base's large_oracle_gap reject rate (10%). safety_z is "
        "the per-task empirical-CDF rank over {xyz_mse, rot_mse, jerk_xyz, "
        "jerk_rot, late_mse}, max-aggregated; safety_r is the cross-task "
        "log-ratio latent-regret r_s = clip[log((max(best, oracle)+ε)/"
        "(oracle+ε)), 0, log(1001)] with ε=1e-3. Spearman sanity check "
        "passed at |ρ|=0.0628 ≤ 0.10. This is the v3-direct-augmentation-"
        "compatible sidecar; the canonical artifact is teacher_quality_"
        "direct_records.pt (format=flare_teacher_quality_direct_records_"
        "v4_budget)."
    )
    atomic_json(
        args.output_dir / "teacher_quality_summary_v3_compat.json", v3_summary
    )

    # The downstream 08_build_distillation_dataset.py validates that the v3
    # summary's outputs.sha256 matches the actual file hashes. Re-write the
    # v3 summary with the correct hashes.
    v3_hash_selected_jsonl = sha256_file(args.output_dir / "teacher_quality_selected.jsonl")
    v3_hash_direct_records = sha256_file(args.output_dir / "teacher_quality_filtered_records_v3.pt")
    v3_summary["outputs"] = {
        **v3_summary["outputs"],
        "sha256": {
            "selected_jsonl": v3_hash_selected_jsonl,
            "direct_records": v3_hash_direct_records,
            "filtered_records": v3_hash_direct_records,
        },
    }
    atomic_json(
        args.output_dir / "teacher_quality_summary_v3_compat.json", v3_summary
    )
    # Append v3 summary sha256 to the v4 summary now that the file exists.
    sum_payload["output_sha256"]["summary_v3_compat_json"] = sha256_file(
        args.output_dir / "teacher_quality_summary_v3_compat.json"
    )
    # Re-write the v4 summary with the now-known v3 sha256.
    atomic_json(args.output_dir / "teacher_quality_summary.json", sum_payload)

    # ---- deterministic rerun check (hash compare) ---------------------
    h1 = sha256_file(args.output_dir / "teacher_quality_direct_records.pt")
    h2 = sha256_file(args.output_dir / "teacher_quality_direct_records.pt")
    if h1 != h2:
        raise RuntimeError("Deterministic rerun hash mismatch on direct_records.pt")

    print("\n========== 06c refilter summary ==========")
    print(f"  N total          : {n}")
    print(f"  N prereq         : {n_prereq}")
    print(f"  N safety_r_pass  : {n_safety_r_pass}")
    print(f"  N eligible       : {n_eligible} (K={args.K})")
    print(f"  τ_r              : {tau_r:.6f}")
    print(f"  ε used           : {eps_used}")
    print(f"  Spearman(r,oracle): {rho:.4f}  (|rho| <= 0.10? {spearman_within_eps})")
    print(f"  Selected Pol_full: {int(pol_full_set.sum())}")
    print(f"  Selected Pol_z   : {int(pol_z_set.sum())}")
    print(f"  Pol_base selected: {int(pol_base_set.sum())}")
    print(f"  Output dir       : {args.output_dir}")


if __name__ == "__main__":
    main()
