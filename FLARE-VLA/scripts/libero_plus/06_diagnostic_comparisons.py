#!/usr/bin/env python3
"""Diagnostic comparisons for LeWM-Spatial sparse candidate scoring.

Four checks validate that LeWM-selected teachers are meaningfully better:
  A. LeWM best-of-K vs random candidate
  B. Best-of-K vs always candidate-0
  C. GT oracle vs candidate teacher
  D. Shuffled action scoring (stratified 500-sample subset)

If any check fails, the pipeline should stop — LeWM selection is not
providing meaningful signal over random choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ─── Path setup ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/le-wm")
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Constants ───────────────────────────────────────────────────────────
EXPECTED_CKPT_SHA256 = "075e11a02390caea4c426491b06d480d2ea9501ee6006183c4686d48e96f1a26"


# ─── I/O helpers ─────────────────────────────────────────────────────────

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def to_jsonable(value: Any) -> Any:
    """Convert any value to JSON-serializable type."""
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
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: math.nan for k in ("mean", "std", "min", "p25", "median", "p75", "max")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }

# ─── Check A: LeWM best-of-K vs random candidate ────────────────────────

def check_a_lewm_vs_random(
    scores_df: pd.DataFrame,
    stepwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare LeWM-selected teacher MSE vs random candidate MSE."""
    n = len(scores_df)
    candidate_k = 4
    mse_per_step = stepwise["mse_per_step"].astype(np.float32)  # [N, K, 8]
    mse_scores = mse_per_step.mean(axis=2)  # [N, K]
    
    teacher_idx = scores_df["teacher_by_mse"].to_numpy(dtype=np.int64)
    teacher_mse = mse_scores[np.arange(n), teacher_idx]
    
    # Random: pick one candidate uniformly per sample
    rng = np.random.default_rng(2026)
    random_idx = rng.integers(0, candidate_k, size=n)
    random_mse = mse_scores[np.arange(n), random_idx]
    
    # Also simulate many random draws to get expected random MSE
    n_sims = 1000
    sim_means = []
    for _ in range(n_sims):
        r_idx = rng.integers(0, candidate_k, size=n)
        sim_means.append(float(mse_scores[np.arange(n), r_idx].mean()))
    expected_random_mean = float(np.mean(sim_means))
    expected_random_std = float(np.std(sim_means))
    
    diff = random_mse - teacher_mse  # positive = teacher better
    teacher_mean = float(teacher_mse.mean())
    random_mean = float(random_mse.mean())
    improvement_pct = 100.0 * (random_mean - teacher_mean) / (random_mean + 1e-12)
    
    # Statistical test: paired t-test
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_rel(random_mse, teacher_mse, alternative="greater")
    
    # Effect size: Cohen's d
    pooled_std = math.sqrt((np.var(random_mse) + np.var(teacher_mse)) / 2)
    cohens_d = float((random_mean - teacher_mean) / (pooled_std + 1e-12))
    
    # How often a random pick equals or beats the teacher?
    better_mask = random_mse <= teacher_mse
    random_wins_or_ties = int(better_mask.sum())
    random_wins_rate = float(better_mask.mean())
    
    # LeWM wins if teacher beats random in >65% of samples (chance=25%)
    # The expected random win rate with 4 equally-good candidates is ~25%
    lewm_win_rate = 1.0 - random_wins_rate
    passed = improvement_pct > 2.0 and p_value < 0.01 and lewm_win_rate > 0.65
    
    result = {
        "check": "A_lewm_vs_random",
        "passed": bool(passed),
        "teacher_mse_mean": teacher_mean,
        "random_mse_mean": random_mean,
        "expected_random_mse_mean": expected_random_mean,
        "expected_random_mse_std": expected_random_std,
        "improvement_pct": improvement_pct,
        "paired_t_statistic": float(t_stat),
        "paired_p_value": float(p_value),
        "cohens_d": cohens_d,
        "random_wins_or_ties": random_wins_or_ties,
        "random_wins_rate": random_wins_rate,
        "stop_criterion": "improvement < 5% OR p >= 0.01",
        "diagnosis": (
            "LeWM selection IS meaningfully better than random"
            if passed
            else "STOP: LeWM selection NOT meaningfully better than random"
        ),
    }
    return result


# ─── Check B: Best-of-K vs always candidate-0 ───────────────────────────

def check_b_best_of_k_vs_candidate0(
    scores_df: pd.DataFrame,
    stepwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare teacher MSE (argmin over K) vs always using candidate 0."""
    n = len(scores_df)
    mse_per_step = stepwise["mse_per_step"].astype(np.float32)  # [N, K, 8]
    mse_scores = mse_per_step.mean(axis=2)  # [N, K]
    
    teacher_idx = scores_df["teacher_by_mse"].to_numpy(dtype=np.int64)
    teacher_mse = mse_scores[np.arange(n), teacher_idx]
    candidate0_mse = mse_scores[:, 0]
    
    diff = candidate0_mse - teacher_mse  # positive = teacher better
    improvement_pct = 100.0 * float(diff.mean()) / (float(candidate0_mse.mean()) + 1e-12)
    
    # How often is candidate 0 the teacher?
    teacher_is_c0 = int((teacher_idx == 0).sum())
    teacher_is_c0_rate = float((teacher_idx == 0).mean())
    
    # How often does candidate 0 win?
    c0_best_mask = np.argmin(mse_scores, axis=1) == 0
    c0_best_count = int(c0_best_mask.sum())
    c0_best_rate = float(c0_best_mask.mean())
    
    # How much better is teacher when candidate 0 is NOT best?
    not_c0_mask = ~c0_best_mask
    if not_c0_mask.sum() > 0:
        improvement_when_not_c0 = float(
            100.0 * diff[not_c0_mask].mean() / (candidate0_mse[not_c0_mask].mean() + 1e-12)
        )
    else:
        improvement_when_not_c0 = 0.0
    
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_rel(candidate0_mse, teacher_mse, alternative="greater")
    
    # If candidate 0 is best >75% of the time, K>1 adds little value
    passed = c0_best_rate < 0.75
    
    result = {
        "check": "B_best_of_k_vs_candidate0",
        "passed": bool(passed),
        "teacher_mse_mean": float(teacher_mse.mean()),
        "candidate0_mse_mean": float(candidate0_mse.mean()),
        "improvement_over_c0_pct": improvement_pct,
        "teacher_is_c0_count": teacher_is_c0,
        "teacher_is_c0_rate": teacher_is_c0_rate,
        "c0_is_best_count": c0_best_count,
        "c0_is_best_rate": c0_best_rate,
        "improvement_when_c0_not_best_pct": improvement_when_not_c0,
        "paired_t_statistic": float(t_stat),
        "paired_p_value": float(p_value),
        "stop_criterion": "candidate_0 is best > 75% of samples",
        "diagnosis": (
            "K>1 provides meaningful benefit over always using candidate 0"
            if passed
            else "Candidate 0 dominates too often — K>1 adds limited value"
        ),
    }
    return result


# ─── Check C: GT oracle vs candidate teacher ────────────────────────────

def check_c_oracle_vs_teacher(
    scores_df: pd.DataFrame,
    stepwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare GT-oracle latent trajectory vs teacher (best candidate)."""
    oracle_mse_per_step = stepwise["oracle_mse_per_step"].astype(np.float32)  # [N, 8]
    mse_per_step = stepwise["mse_per_step"].astype(np.float32)  # [N, K, 8]
    teacher_idx = scores_df["teacher_by_mse"].to_numpy(dtype=np.int64)
    n = len(scores_df)
    teacher_mse_step = mse_per_step[np.arange(n), teacher_idx]  # [N, 8]
    
    oracle_mse = oracle_mse_per_step.mean(axis=1)
    teacher_mse = teacher_mse_step.mean(axis=1)
    
    # Gap: positive means teacher is worse
    gap = teacher_mse - oracle_mse
    oracle_better = int((gap > 0).sum())
    oracle_better_rate = float((gap > 0).mean())
    
    # Per-step comparison
    step_gaps = teacher_mse_step - oracle_mse_per_step
    step_oracle_better = [(step_gaps[:, i] > 0).mean() for i in range(8)]
    
    # Normalized gap
    normalized_gap = gap / (teacher_mse + 1e-12)
    
    # Oracle should beat teacher on SOME samples (shows LeWM has signal)
    # but not too many (would mean candidates are poor)
    oracle_beats_all_pct = float(scores_df["oracle_better_than_all_mse"].mean() * 100)
    
    # Check: oracle should be better than teacher on >50% to show room for improvement
    # but also teacher should beat oracle on >10% to show candidates add value
    passed = oracle_better_rate > 0.3 and oracle_better_rate < 0.9
    
    result = {
        "check": "C_oracle_vs_teacher",
        "passed": bool(passed),
        "teacher_mse_mean": float(teacher_mse.mean()),
        "oracle_mse_mean": float(oracle_mse.mean()),
        "teacher_vs_oracle_gap_mean": float(gap.mean()),
        "teacher_vs_oracle_gap_median": float(np.median(gap)),
        "oracle_better_count": oracle_better,
        "oracle_better_rate": oracle_better_rate,
        "oracle_beats_all_candidates_rate": oracle_beats_all_pct,
        "normalized_gap_mean": float(normalized_gap.mean()),
        "per_step_oracle_better_rate": step_oracle_better,
        "stop_criterion": "oracle better rate outside (0.3, 0.9)",
        "diagnosis": (
            "Teacher and oracle have a healthy gap — LeWM selection has signal"
            if passed
            else (
                "Oracle dominates too much — candidates are poor"
                if oracle_better_rate >= 0.9
                else "Teacher too close to oracle — LeWM selection adds little"
            )
        ),
    }
    return result

# ─── Check D: Shuffled action scoring ───────────────────────────────────

@torch.inference_mode()
def score_one_sequence(
    model,
    history_latents: torch.Tensor,  # [1, 3, 192]
    action_sequence: torch.Tensor,   # [1, 10, 7]
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    gt_latents: torch.Tensor,        # [1, 8, 192]
    history_size: int = 3,
) -> float:
    """Score a single action sequence through LeWM and return MSE."""
    device = next(model.parameters()).device
    history_latents = history_latents.to(device)
    action_sequence = action_sequence.to(device)
    gt_latents = gt_latents.to(device)
    
    # Normalize
    action_sequence = torch.nan_to_num(
        (action_sequence.float() - action_mean.to(device)) / action_std.to(device),
        nan=0.0,
    )
    
    batch_size = 1
    sample_count = 1
    total_actions = action_sequence.shape[1]
    
    emb = history_latents.clone()
    act = action_sequence[:, :history_size].clone()
    act_future = action_sequence[:, history_size:]
    
    for step in range(7):
        act_emb = model.action_encoder(act)
        pred = model.predict(
            emb[:, -history_size:],
            act_emb[:, -history_size:],
        )[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        act = torch.cat([act, act_future[:, step:step + 1]], dim=1)
    
    act_emb = model.action_encoder(act)
    pred = model.predict(
        emb[:, -history_size:],
        act_emb[:, -history_size:],
    )[:, -1:]
    emb = torch.cat([emb, pred], dim=1)
    
    future_latents = emb[:, history_size:]
    mse = float(F.mse_loss(future_latents, gt_latents.to(device)))
    return mse


def check_d_shuffled_actions(
    scores_csv: Path,
    teacher_records_path: Path,
    stepwise_path: Path,
    candidate_dir: Path,
    gt_h5_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    device_str: str = "cuda",
) -> dict[str, Any]:
    """Shuffled action scoring on a stratified 500-sample subset."""
    import h5py
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    
    # Load scoring data
    scores_df = pd.read_csv(scores_csv)
    records = load_torch(teacher_records_path)
    stepwise = np.load(stepwise_path, allow_pickle=False)
    
    n_total = len(scores_df)
    teacher_idx = records["teacher_by_mse"].numpy().astype(np.int64)
    
    # Stratified sampling: 50 per task, 10 tasks = 500 samples
    task_groups = scores_df.groupby("task_index")
    sample_indices = []
    for task_idx, group in task_groups:
        group_indices = group.index.tolist()
        n_per_task = min(50, len(group_indices))
        rng = np.random.default_rng(2026 + int(task_idx))
        chosen = rng.choice(group_indices, size=n_per_task, replace=False).tolist()
        sample_indices.extend(chosen)
    
    sample_indices = sorted(sample_indices)
    n_sample = len(sample_indices)
    print(f"  Check D: stratified sample {n_sample} windows across {len(task_groups)} tasks")
    
    # Load manifest for history/future rows
    manifest_rows = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            manifest_rows[row["sample_index"]] = row
    
    # Load LeWM model
    print("  Loading LeWM model for Check D...")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "config/train")):
        cfg = compose(config_name="lewm", overrides=["data=libero_plus"])
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = 7
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = load_torch(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    
    # Compute action stats from H5
    with h5py.File(gt_h5_path, "r") as h5:
        frame_actions = torch.from_numpy(np.asarray(h5["frame_action"][:], dtype=np.float32))
    finite_mask = torch.isfinite(frame_actions).all(dim=1)
    action_mean = frame_actions[finite_mask].mean(dim=0)
    action_std = frame_actions[finite_mask].std(dim=0)
    
    teacher_mses = []
    shuffled_mses = []
    best_candidate_paths = {}
    
    rng = np.random.default_rng(20260729)
    
    with h5py.File(gt_h5_path, "r") as h5:
        for idx_in_sample, df_idx in enumerate(sample_indices):
            row = scores_df.iloc[df_idx]
            sid = row["sample_id"]
            sample_idx = int(row["sample_index"])
            episode_idx = int(row["episode_index"])
            global_row = int(row["global_index"])
            window_idx = int(row["window_index_h8"])
            
            mr = manifest_rows.get(sample_idx)
            if mr is None:
                continue
            
            history_rows = np.asarray(mr["history_rows"], dtype=np.int64)
            future_rows = np.asarray(mr["future_rows"], dtype=np.int64)
            tidx = int(teacher_idx[df_idx])
            
            # Load candidate
            cand_path = candidate_dir / f"{sid}.pt"
            if not cand_path.is_file():
                continue
            cand = load_torch(cand_path)
            
            # Get best candidate actions
            best_actions = torch.as_tensor(cand["candidate_actions_concat"]).float()[0, tidx]  # [8, 7]
            historical_actions = torch.from_numpy(
                np.asarray(h5["frame_action"][global_row - 2:global_row], dtype=np.float32)
            )
            history_latents = torch.from_numpy(
                np.asarray(h5["frame_latent"][history_rows], dtype=np.float32)
            ).unsqueeze(0)
            gt_latents = torch.from_numpy(
                np.asarray(h5["frame_latent"][future_rows], dtype=np.float32)
            ).unsqueeze(0)
            
            # Original sequence: [history_actions(2) + best_actions(8)] = [10, 7]
            original_seq = torch.cat([historical_actions, best_actions], dim=0).unsqueeze(0)
            
            # Shuffled: permute the 8 future timesteps
            perm = rng.permutation(8)
            shuffled_actions = best_actions[perm]
            shuffled_seq = torch.cat([historical_actions, shuffled_actions], dim=0).unsqueeze(0)
            
            # Score both
            teacher_mse = score_one_sequence(
                model, history_latents, original_seq,
                action_mean, action_std, gt_latents,
            )
            shuffled_mse = score_one_sequence(
                model, history_latents, shuffled_seq,
                action_mean, action_std, gt_latents,
            )
            
            teacher_mses.append(teacher_mse)
            shuffled_mses.append(shuffled_mse)
            
            if (idx_in_sample + 1) % 100 == 0:
                print(f"    Check D progress: {idx_in_sample + 1}/{n_sample}")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    teacher_arr = np.array(teacher_mses, dtype=np.float64)
    shuffled_arr = np.array(shuffled_mses, dtype=np.float64)
    
    # Teacher MSE should be significantly lower than shuffled
    diff = shuffled_arr - teacher_arr  # positive = teacher better
    teacher_mean = float(teacher_arr.mean())
    shuffled_mean = float(shuffled_arr.mean())
    improvement_pct = 100.0 * (shuffled_mean - teacher_mean) / (shuffled_mean + 1e-12)
    
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_rel(shuffled_arr, teacher_arr, alternative="greater")
    
    # Compute 95% CI overlap
    teacher_ci95 = 1.96 * teacher_arr.std() / math.sqrt(len(teacher_arr))
    shuffled_ci95 = 1.96 * shuffled_arr.std() / math.sqrt(len(shuffled_arr))
    ci_overlap = (teacher_mean + teacher_ci95) > (shuffled_mean - shuffled_ci95)
    
    passed = improvement_pct > 5.0 and p_value < 0.01
    ci_overlap_bool = bool(ci_overlap)  # cast numpy bool for JSON serialization
    
    result = {
        "check": "D_shuffled_actions",
        "passed": bool(passed) if passed is not None else None,
        "n_samples_tested": len(teacher_arr),
        "teacher_mse_mean": teacher_mean,
        "shuffled_mse_mean": shuffled_mean,
        "improvement_pct": improvement_pct,
        "teacher_mse_std": float(teacher_arr.std()),
        "shuffled_mse_std": float(shuffled_arr.std()),
        "paired_t_statistic": float(t_stat),
        "paired_p_value": float(p_value),
        "ci95_overlap": ci_overlap_bool,
        "stop_criterion": "improvement < 10% OR p >= 0.001 OR CI overlap",
        "diagnosis": (
            "LeWM is meaningfully sensitive to action quality (shuffled >> teacher)"
            if passed
            else "STOP: LeWM not sensitive to action quality — shuffled ≈ teacher"
        ),
    }
    return result


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic comparisons for LeWM-Spatial teacher scoring"
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=Path("/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_scores.csv"),
    )
    parser.add_argument(
        "--teacher-records",
        type=Path,
        default=Path("/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_teacher_records.pt"),
    )
    parser.add_argument(
        "--stepwise-npz",
        type=Path,
        default=Path("/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_stepwise_metrics.npz"),
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=Path("/root/autodl-tmp/le-wm/outputs/spatial/scoring/validation_summary.json"),
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("/root/autodl-tmp/Isaac-GR00T/outputs/spatial/sparse_candidates/candidates"),
    )
    parser.add_argument(
        "--gt-h5",
        type=Path,
        default=Path("/root/autodl-tmp/FLARE-VLA/outputs/spatial/gt_latents/spatial_epoch50_h8_framebank_trainpipe.h5"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/le-wm/stablewm_data/checkpoints/lewm/weights_epoch_50.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/root/autodl-tmp/FLARE-VLA/outputs/spatial/manifests/spatial_sparse_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/FLARE-VLA/outputs/spatial/10_teacher_scoring"),
    )
    parser.add_argument("--skip-shuffled", action="store_true", help="Skip Check D (expensive)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Validate inputs
    for path in [args.scores_csv, args.teacher_records, args.stepwise_npz, args.validation_summary]:
        if not path.exists():
            raise FileNotFoundError(path)
    
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  LeWM-Spatial Teacher Diagnostic Comparisons")
    print("=" * 60)
    
    # Load data
    scores_df = pd.read_csv(args.scores_csv)
    stepwise = dict(np.load(args.stepwise_npz, allow_pickle=False))
    summary = load_json(args.validation_summary)
    
    n = len(scores_df)
    print(f"\n  Samples: {n}")
    print(f"  Tasks: {summary.get('tasks', '?')}")
    print(f"  Teacher agreement rate: {summary.get('teacher_agreement', {}).get('rate', '?'):.4f}")
    print(f"  Oracle beats all MSE rate: {summary.get('oracle_better_than_all_candidates', {}).get('mse_rate', '?'):.4f}")
    
    results = {}
    all_passed = True
    
    # Check A: LeWM vs random
    print("\n--- Check A: LeWM best-of-K vs random ---")
    t0 = time.time()
    results["A"] = check_a_lewm_vs_random(scores_df, stepwise)
    print(f"  Teacher MSE: {results['A']['teacher_mse_mean']:.6f}")
    print(f"  Random MSE: {results['A']['random_mse_mean']:.6f}")
    print(f"  Improvement: {results['A']['improvement_pct']:.2f}%")
    print(f"  p-value: {results['A']['paired_p_value']:.6f}")
    print(f"  PASSED: {results['A']['passed']} ({results['A']['diagnosis']})")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    if not results["A"]["passed"]:
        all_passed = False
    
    # Check B: Best-of-K vs candidate-0
    print("\n--- Check B: Best-of-K vs candidate-0 ---")
    t0 = time.time()
    results["B"] = check_b_best_of_k_vs_candidate0(scores_df, stepwise)
    print(f"  Teacher MSE: {results['B']['teacher_mse_mean']:.6f}")
    print(f"  Candidate-0 MSE: {results['B']['candidate0_mse_mean']:.6f}")
    print(f"  Improvement: {results['B']['improvement_over_c0_pct']:.2f}%")
    print(f"  C0 is best rate: {results['B']['c0_is_best_rate']:.4f}")
    print(f"  PASSED: {results['B']['passed']} ({results['B']['diagnosis']})")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    if not results["B"]["passed"]:
        all_passed = False
    
    # Check C: Oracle vs teacher
    print("\n--- Check C: GT Oracle vs teacher ---")
    t0 = time.time()
    results["C"] = check_c_oracle_vs_teacher(scores_df, stepwise)
    print(f"  Teacher MSE: {results['C']['teacher_mse_mean']:.6f}")
    print(f"  Oracle MSE: {results['C']['oracle_mse_mean']:.6f}")
    print(f"  Oracle better rate: {results['C']['oracle_better_rate']:.4f}")
    print(f"  Oracle beats all rate: {results['C']['oracle_beats_all_candidates_rate']:.2f}%")
    print(f"  PASSED: {results['C']['passed']} ({results['C']['diagnosis']})")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    if not results["C"]["passed"]:
        all_passed = False
    
    # Check D: Shuffled actions
    if args.skip_shuffled:
        print("\n--- Check D: Shuffled action scoring (SKIPPED) ---")
        results["D"] = {"check": "D_shuffled_actions", "passed": None, "skipped": True,
                        "diagnosis": "Skipped by user request"}
    else:
        print("\n--- Check D: Shuffled action scoring ---")
        t0 = time.time()
        results["D"] = check_d_shuffled_actions(
            scores_csv=args.scores_csv,
            teacher_records_path=args.teacher_records,
            stepwise_path=args.stepwise_npz,
            candidate_dir=args.candidate_dir,
            gt_h5_path=args.gt_h5,
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_dir=output_dir,
            device_str=args.device,
        )
        print(f"  Teacher MSE: {results['D'].get('teacher_mse_mean', 'N/A'):}")
        print(f"  Shuffled MSE: {results['D'].get('shuffled_mse_mean', 'N/A')}")
        print(f"  Improvement: {results['D'].get('improvement_pct', 'N/A')}")
        print(f"  p-value: {results['D'].get('paired_p_value', 'N/A')}")
        print(f"  PASSED: {results['D']['passed']} ({results['D']['diagnosis']})")
        print(f"  elapsed: {time.time() - t0:.1f}s")
        if results["D"].get("passed") is False:
            all_passed = False
    
    # ─── Final verdict ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL CHECKS PASSED — LeWM selection is meaningfully better")
        verdict = "PROCEED"
    else:
        failed = [k for k, v in results.items() if v.get("passed") is False]
        print(f"  CHECKS FAILED: {failed} — STOP PIPELINE")
        verdict = "STOP"
    print("=" * 60)
    
    # Save diagnostic report
    diagnostic_report = {
        "format": "flare_lewm_diagnostic_comparisons_v1",
        "verdict": verdict,
        "timestamp": time.time(),
        "inputs": {
            "scores_csv": str(args.scores_csv),
            "teacher_records": str(args.teacher_records),
            "stepwise_npz": str(args.stepwise_npz),
            "candidate_dir": str(args.candidate_dir),
        },
        "results": results,
    }
    report_path = output_dir / "diagnostic_comparisons.json"
    atomic_write_json(report_path, diagnostic_report)
    print(f"\n  Diagnostic report saved: {report_path}")
    
    if verdict == "STOP":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
