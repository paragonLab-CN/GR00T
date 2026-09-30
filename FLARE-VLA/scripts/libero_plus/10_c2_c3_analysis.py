#!/usr/bin/env python3
"""C2/C3 compute-only analysis for the FLARE-VLA teacher-selection anatomy.

C2: correlation + PCA factor structure over the per-trajectory audit parquet.
C3: join per-task selection features with per-task downstream success of the
    gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20 model; per-block
    attribution.

Data (per user correction): teacher_scoring_v2 is the teacher set distilled
into the gt_replay_44_20 model (LeWM epoch-9 scorer, suite-aligned, N=487,880,
selected 65.2%). Downstream success = paired fixed-noise eval seed 2026,
checkpoint-10000.

No sklearn: PCA via numpy SVD on the standardized feature matrix.
"""
import json
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

ROOT = "/root/autodl-tmp/Isaac-GR00T"
AUDIT = f"{ROOT}/FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/teacher_quality_audit.parquet"
TASK_SUM = f"{ROOT}/FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/teacher_quality_task_summary.parquet"
EVAL = (f"{ROOT}/outputs/eval_libero_plus/"
        "gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20_checkpoint_10000_fixed_noise_seed2026/"
        "summary_all.json")
OUT_JSON = f"{ROOT}/FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/c2_c3_analysis.json"

# ---------------------------------------------------------------- feature blocks
BLOCKS = {
    "oracle_gap": ["oracle_mse_score", "oracle_cosine_score", "oracle_gap_mse",
                   "oracle_gap_cosine", "early_mse", "mid_mse", "late_mse", "terminal_mse"],
    "candidate_diversity": ["candidate_mse_std", "candidate_cosine_std", "candidate_mse_mean",
                            "candidate_cosine_mean", "mse_margin", "cosine_margin",
                            "relative_mse_margin", "candidate_0_vs_1_max_abs_diff"],
    "teacher_gt_divergence": ["teacher_gt_action_mse", "teacher_gt_xyz_mse",
                              "teacher_gt_rot_mse", "teacher_gt_gripper_mse",
                              "teacher_gt_gripper_match_rate"],
    "dynamics": ["teacher_velocity_l2_mean", "teacher_acceleration_l2_mean",
                 "teacher_jerk_l2_mean", "teacher_gt_velocity_ratio",
                 "teacher_gt_acceleration_ratio", "teacher_gt_jerk_ratio"],
}
ALL_FEATS = sorted({f for fs in BLOCKS.values() for f in fs})

STOP = set(("the a an on in of to and it there please now first next located adjacent within "
            "system organization proper our we for so host show cooking recipe instructions says "
            "suggest grab grasp pick place put move set position would you mind could id really "
            "appreciate ensure").split())


def sig(s):
    s = re.sub(r"[^a-z ]", " ", s.lower())
    s = re.sub(r"\s+(tb|table)\s*\d+\s*$", "", s)
    s = re.sub(r"\s+view.*$", "", s)
    s = re.sub(r"\s+initstate.*$", "", s)
    return set(w for w in s.split() if w not in STOP)


def load_eval_per_task():
    d = json.load(open(EVAL))
    rows = []
    for t in d["tasks"]:
        rows.append({"suite": t["suite"], "language": t["language"],
                     "category": t["category"], "difficulty": t["difficulty"],
                     "eval_success": float(t["success_rate"])})
    return pd.DataFrame(rows)


def join_eval_to_train(eval_df, train_ts):
    """Per-suite content-word containment join. Returns eval_df + task_index + conf."""
    train_sig = [(r["suite"], r["task_index"], r["task_text"], sig(r["task_text"]))
                 for _, r in train_ts.iterrows()]
    maps = []
    for _, e in eval_df.iterrows():
        es, esuite = sig(e["language"]), e["suite"]
        best, bi, btxt, bmargin = -1.0, None, None, 0.0
        scored = []
        for (su, i, t, tsg) in train_sig:
            if su != esuite or not tsg:
                continue
            cont = len(tsg & es) / max(1, len(tsg))
            fz = SequenceMatcher(None, " ".join(sorted(es)), " ".join(sorted(tsg))).ratio()
            score = 0.6 * cont + 0.4 * fz
            scored.append((score, i, t))
        scored.sort(reverse=True)
        top = scored[0]
        margin = scored[0][0] - (scored[1][0] if len(scored) > 1 else 0.0)
        maps.append({"task_index": top[1], "join_conf": round(top[0], 3),
                     "join_margin": round(margin, 3)})
    m = pd.DataFrame(maps)
    return pd.concat([eval_df.reset_index(drop=True), m], axis=1)


def main():
    out = {"data": {"audit": AUDIT.split(ROOT + "/")[-1],
                    "task_summary": TASK_SUM.split(ROOT + "/")[-1],
                    "eval": EVAL.split(ROOT + "/")[-1],
                    "scorer_checkpoint": "LeWM weights_epoch_9.pt (from scoring_run_summary.json)",
                    "note": "teacher_scoring_v2 = the teacher set distilled into gt_replay_44_20"}}

    # ------------------------------------------------------------ C2: factor structure
    print("loading audit columns for C2 ...")
    df = pd.read_parquet(AUDIT, columns=ALL_FEATS + ["task_index"])
    n_total = len(df)
    X = df[ALL_FEATS].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(X.to_numpy()).all(axis=1)
    Xc = X[mask]
    out["c2"] = {"n_total": int(n_total), "n_finite": int(len(Xc)),
                 "n_dropped_nonfinite": int(n_total - len(Xc)), "features": ALL_FEATS}

    # Spearman correlation matrix (rank-based, robust to heavy tails)
    print("spearman correlation ...")
    from scipy.stats import spearmanr
    R, _ = spearmanr(Xc.to_numpy())
    R = np.atleast_2d(R)
    corr_df = pd.DataFrame(R, index=ALL_FEATS, columns=ALL_FEATS)

    # key cross-block correlations
    def c(a, b):
        return float(corr_df.loc[a, b])
    out["c2"]["key_correlations"] = {
        "candidate_mse_std_vs_teacher_gt_action_mse": c("candidate_mse_std", "teacher_gt_action_mse"),
        "candidate_mse_std_vs_oracle_gap_mse": c("candidate_mse_std", "oracle_gap_mse"),
        "candidate_mse_std_vs_oracle_mse": c("candidate_mse_std", "oracle_mse_score"),
        "teacher_gt_xyz_mse_vs_oracle_gap_mse": c("teacher_gt_xyz_mse", "oracle_gap_mse"),
        "teacher_jerk_ratio_vs_oracle_gap_mse": c("teacher_gt_jerk_ratio", "oracle_gap_mse"),
        "best_proxy_oracle_mse_vs_candidate_mse_std": c("oracle_mse_score", "candidate_mse_std"),
    }

    # PCA via SVD on standardized data
    print("PCA ...")
    Xs = (Xc - Xc.mean()) / Xc.std(ddof=0)
    Xs = Xs.fillna(0.0)
    # subsample for SVD speed if huge (deterministic stride)
    if len(Xs) > 200000:
        Xs = Xs.iloc[::len(Xs) // 200000]
    U, S, Vt = np.linalg.svd(Xs.to_numpy(), full_matrices=False)
    var = (S ** 2) / (len(Xs) - 1)
    explained = var / var.sum()
    loadings = pd.DataFrame(Vt.T, index=ALL_FEATS,
                            columns=[f"PC{i+1}" for i in range(len(ALL_FEATS))])
    out["c2"]["pca"] = {
        "n_used": int(len(Xs)),
        "explained_variance_ratio_top8": [round(float(v), 4) for v in explained[:8]],
        "cumvar_top8": [round(float(v), 4) for v in np.cumsum(explained)[:8]],
    }
    # top loadings per of first 4 PCs
    tops = {}
    for pc in ["PC1", "PC2", "PC3", "PC4"]:
        ser = loadings[pc].abs().sort_values(ascending=False)
        tops[pc] = [(f, round(float(loadings.loc[f, pc]), 3)) for f in ser.index[:5]]
    out["c2"]["pca"]["top_loadings"] = tops

    # diversity ⊥ oracle-gap verdict
    cc = out["c2"]["key_correlations"]["candidate_mse_std_vs_oracle_gap_mse"]
    out["c2"]["diversity_independent_of_oracle_gap"] = bool(abs(cc) < 0.3)
    out["c2"]["diversity_oracle_gap_corr"] = round(cc, 4)

    corr_df.to_csv(f"{ROOT}/FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/c2_feature_spearman.csv")
    loadings.to_csv(f"{ROOT}/FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/c2_pca_loadings.csv")

    # ------------------------------------------------------------ C3: feature -> downstream
    print("C3 join + attribution ...")
    train_ts = pd.read_parquet(TASK_SUM)
    eval_df = load_eval_per_task()
    joined = join_eval_to_train(eval_df, train_ts)
    n_conf = int((joined["join_conf"] >= 0.6).sum())
    out["c3"] = {"eval_tasks": int(len(eval_df)),
                 "joined_confident_ge0.6": n_conf,
                 "join_note": "per-suite content-word containment + fuzzy; conf>=0.6 kept",
                 "downstream_model": "gt_replay_44_20 checkpoint-10000 fixed-noise seed2026"}

    # aggregate eval success per train task_index (a train task may map to several eval tasks)
    eval_per_train = (joined[joined["join_conf"] >= 0.6]
                      .groupby("task_index")["eval_success"].mean().rename("eval_success"))
    tt = train_ts.set_index("task_index").join(eval_per_train, how="inner")
    out["c3"]["train_tasks_with_downstream"] = int(len(tt))

    # per-task selection features to correlate with downstream success
    feat_cols = ["direct_aug_rate", "mean_oracle_gap_mse", "mean_best_mse", "mean_oracle_mse",
                 "mean_teacher_gt_action_mse", "mean_teacher_gt_xyz_mse", "mean_teacher_gt_rot_mse",
                 "mean_teacher_jerk", "mean_jerk_ratio", "mean_late_mse", "mean_terminal_mse",
                 "mean_gripper_match_rate", "mean_relative_mse_margin", "mean_relative_mse_improvement"]
    feat_cols = [c for c in feat_cols if c in tt.columns]
    from scipy.stats import spearmanr as sr
    per_feat = {}
    for c in feat_cols:
        sub = tt[[c, "eval_success"]].dropna()
        if len(sub) >= 5 and sub[c].std() > 0:
            rho, p = sr(sub[c], sub["eval_success"])
            per_feat[c] = {"spearman_rho": round(float(rho), 3), "p": round(float(p), 4),
                           "n": int(len(sub))}
    out["c3"]["per_feature_vs_downstream"] = per_feat

    # per-block composite (mean of z-scored member features present in task summary)
    block_map = {
        "oracle_gap": ["mean_oracle_gap_mse", "mean_oracle_mse", "mean_late_mse", "mean_terminal_mse"],
        "teacher_gt_divergence": ["mean_teacher_gt_action_mse", "mean_teacher_gt_xyz_mse", "mean_teacher_gt_rot_mse"],
        "dynamics": ["mean_teacher_jerk", "mean_jerk_ratio"],
        "candidate_diversity": ["mean_relative_mse_margin", "mean_relative_mse_improvement"],
        "selection_rate": ["direct_aug_rate"],
    }
    per_block = {}
    for blk, cols in block_map.items():
        cols = [c for c in cols if c in tt.columns]
        if not cols:
            continue
        Z = tt[cols].apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
        comp = Z.mean(axis=1)
        sub = pd.DataFrame({"comp": comp, "y": tt["eval_success"]}).dropna()
        rho, p = sr(sub["comp"], sub["y"])
        per_block[blk] = {"spearman_rho": round(float(rho), 3), "p": round(float(p), 4),
                          "n": int(len(sub)), "members": cols}
    out["c3"]["per_block_vs_downstream"] = per_block

    joined.to_csv(f"{ROOT}/FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/c3_eval_to_train_join.csv", index=False)
    tt.to_csv(f"{ROOT}/FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/c3_train_task_features_with_downstream.csv")

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", OUT_JSON)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
