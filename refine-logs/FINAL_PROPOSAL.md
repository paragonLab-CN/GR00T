# Research Proposal: Budget-Constrained Task-Relative Teacher Filtering for FLARE-VLA on LIBERO-Plus

**Status**: Round-4 READY (GPT-6-Astra review, weighted overall 9.01/10).
**Date**: 2026-10-12.
**Pipeline**: `research-refine-pipeline` Phase 1 (method) + Phase 3 (experiment plan) below.

## Problem Anchor
- **Bottom-line problem**: The current FLARE-VLA teacher filter (`06_audit_teacher_quality.py`, policy `flare_filtered_direct_augmentation`) removes outliers with **fixed global quantiles** on per-component action distance, late-window MSE, oracle gap, jerk, and gripper mismatch. The downstream GR00T-N1.7 LIBERO-Plus distillation run (`08_train_gr00t_flare_44_20.sh`) therefore accepts ~63% of candidates (309128 / 487880 in `teacher_scoring_epoch50_fixed_v2`) as equal-weight action targets. Inspection of `teacher_quality_task_summary.csv` shows this global policy is **highly non-uniform across the 40 LIBERO-Plus tasks**: `direct_aug_rate` ranges from 0.045 (task 7, "open the middle drawer of the cabinet") to 0.776 (task 6, "pick up the black bowl from table center"). Tasks where the global rate collapses below 0.30 visibly suffer in `08c_train_gr00t_flare_44_20.sh`, so success-rate improvements on those tasks are the bottleneck.
- **Must-solve bottleneck**: Replace the global fixed-quantile outlier policy with a **suite/task-conditional, latent-score-aware teacher filter** that (a) raises acceptance where confident teacher trajectories exist under current evidence, (b) lowers acceptance where the candidate set is unreliable, and (c) keeps the mechanism small enough to defend in a paper while **raising LIBERO-Plus overall success rate** — with the previously failing `direct_aug_rate < 0.20` tasks as the dominant diagnostic target.
- **Non-goals**: We are NOT changing LeWM training, the candidate generator, the GR00T backbone, the policy gradient, or the teacher horizon. We are NOT introducing new inference-time computation at deployment. We are NOT adding a learned reward model or RL fine-tuning. The paper story remains a **distillation-data curation paper**, not a new VLA architecture.
- **Constraints**: Single host with NVIDIA RTX-class GPU; offline teacher scoring already produced once for ~488k samples at a fixed epoch-50 LeWM checkpoint (`weights_epoch_50.pt`, sha256 `381bc4725e71a5728d172d84e6c706cffa7dc0f01220af1ac0dcfd440dbc5825`). GT latent H5 (`gt_latents/libero_plus_epoch50_h8_framebank_trainpipe.h5`, sha256 `2d74cc68...`) is fixed. Manifest (`libero_plus_sparse_manifest.jsonl`, sha256 `ac3d919a...`) is fixed. Refilter cost must be < 1 GPU-hour on the existing CSV/PT/NPZ audit inputs.
- **Success condition**: On the same backbone and same training budget, the *refiltered* distillation set must yield:
  1. A **measurable overall LIBERO-Plus success-rate improvement** against the current `08c_train_gr00t_flare_44_20.sh` baseline (sign-test on 40 tasks, paired t-test on suite means).
  2. A **reduction of the worst-task gap**: the lowest-`direct_aug_rate` tasks must either gain >5 percentage points success, or, if not, show a documented diagnostic showing why no filter could rescue them.
  3. A **paper-claim-ready comparison** that the new policy is both statistically and qualitatively stronger than the global policy on the same teacher set, scoring budget, and training budget.

## Method Thesis
- **One-sentence thesis**: replace the global-quantile outlier filter with a **fixed-budget top-K** teacher selection rule that operates on two eligibility predicates — per-task component-badness (`safety_z`) and one shared cross-task latent-regret constraint (`safety_r`) — calibrated independently on a small calibration subset, then evaluated end-to-end on LIBERO-Plus under paired training seeds.
- **Why this is the smallest adequate intervention**: two eligibility predicates, one independent scalar cutoff for `safety_r`, deterministic top-K selection; no new training, no new model, no new forward pass.
- **Why this route is timely**: the dominant current debate in VLA is data curation at scale. Our contribution is a *demonstrated principle* with end-to-end LIBERO-Plus evaluation.

## Contribution Focus
- **Dominant contribution**: a **fixed-budget, paired-seed, end-to-end LIBERO-Plus evaluation** of a teacher-filter that combines per-task empirical-CDF rank normalization with a cross-task log-ratio latent-regret gate.
- **Optional supporting contribution**: a per-(suite, task) diagnostic table with rescueability (candidates passing `safety_r` alone) as a continuous moderator.

## Proposed Method

### Complexity Budget
- **Frozen**: LeWM (epoch-50, fixed SHA), the four audit artifacts, GR00T-N1.7 LIBERO-Plus student, manifest, GT-latent H5, candidate generator.
- **New trainable components**: **none**.
- **Tempting additions intentionally not used**: no learned reward model / VLM judge; no RL on filtered teachers; no per-task learnable threshold; no new forward pass.

### System Overview
```
audit inputs (CSV/PT/NPZ; 487880 windows)
   │
   ▼
episode-hashed partition into 10 buckets; pool ALL back into the candidate set
after the calibration step is done
   │
   ▼
on the calibration subset (1-of-10 buckets; ~47k windows):
   ├─ fit empirical CDFs per (suite, task_index) for M = {xyz, rot, jerk_xyz, jerk_rot, late_mse}
   ├─ fit τ_r by matching Pol_base's marginal `large_oracle_gap` reject rate
   ├─ verify |ρ_Spearman(r_s, oracle_mse_s)| ≤ 0.10 (predeclared)
   └─ on failure: ABORT cross-task-regret claim; fallback to Pol_z
   │
   ▼
on all 487880 candidates:
   for s:
     prereq = flag_finite_core & flag_finite_actions
              & flag_teacher_agreement & flag_gripper_ok
     safety_z(s) = max_{k in M} F̂_{g(s),k}(m_{s,k})
     safety_r(s) = clip[ log((max(best_mse_s, oracle_mse_s)+ε)
                              / (oracle_mse_s+ε)), 0, log(1001) ]
     eligible = prereq & safety_r(s) ≤ τ_r
   │
   ▼
Pol_full = top-K=309128 of eligible by ascending safety_z, ties by sample_id ASC
   ▼
Pol_z    = top-K=309128 of {prereq} by ascending safety_z, ties by sample_id ASC
   │
   ▼
write teacher_quality_direct_records.pt — same downstream scripts unchanged
   │
   ▼
08_build_distillation_dataset.py — UNMODIFIED
   │
   ▼
08c_train_gr00t_flare_44_20.sh × {0,1,2} for Pol_base and Pol_full; × {1} for Pol_z
   │
   ▼
hierarchical bootstrap + 40-task sign test + non-inferiority on easy decile
```

### Core Mechanism (Final, Exact)

#### Prerequisite Flags (Binary)
- `flag_finite_core`, `flag_finite_actions` — finite-metric prerequisite.
- `flag_teacher_agreement` — cosine-best teacher == MSE-best teacher.
- `flag_gripper_ok` — gripper-vs-GT match rate ≥ 0.3.

#### `safety_z` (Per-Task Component-Badness)
- `M = {teacher_gt_xyz_mse, teacher_gt_rot_mse, teacher_jerk_xyz_l2_mean, teacher_jerk_rot_l2_mean, late_mse}`.
- `z_s = max_{k in M} F̂_{g(s),k}(m_{s,k})` where `F̂_{g,k}` is the empirical CDF on the calibration subset, restricted to `(suite, task_index) = g`, right-tail.
- NaN → CDF = 1. Ties → average rank. Suite-level fallback if `n_g < 30`.

#### `safety_r` (Cross-Task Log-Ratio Latent-Regret)
\[
r_s \;=\; \operatorname{clip}\!\left[\log\!\left(\frac{\max(\text{best\_mse}_s, \text{oracle\_mse}_s)+\epsilon}{\text{oracle\_mse}_s+\epsilon}\right),\; 0,\; \log(1001)\right],\quad\epsilon=10^{-3}.
\]
- Eligible iff `safety_r(s) ≤ τ_r`. Non-finite → ineligible.
- `τ_r = quantile(r_calibration, 1 − reject_rate_baseline)` where `reject_rate_baseline = (#samples Pol_base rejected for `large_oracle_gap`) / (total samples)`. Both numerator and denominator on the same prerequisite-qualified calibration population.

#### Selection
**Pol_full**:
1. Apply prerequisite flags ∧ `safety_r(s) ≤ τ_r`; let E = eligible.
2. If `|E| < K=309128`: ABORT with documented data-level failure.
3. Top-K of E by ascending `z_s`; ties by `sample_id ASC`.

**Pol_z**:
\[
\text{Pol}_z = \text{TopK}_{z_s}\{s : \text{prerequisite}(s)\},\quad K=309128.
\]
Same K, same tie-breaking; no `safety_r`.

### Failure Modes and Diagnostics
- **`n_g < 30`**: suite-level empirical-CDF fallback; documented.
- **`|E| < K`**: ABORT; report. (Data-level failure, not method-level.)
- **`|ρ_Spearman(r_s, oracle_mse_s)| > 0.10`** at both `ε = 10^-3` and `ε = 10^-2`: ABORT cross-task-regret claim; default to Pol_z; report.

### Optional Supporting Component
- Per-(suite, task) diagnostic: `samples`, `Pol_base_rate`, `Pol_z_rate`, `Pol_full_rate`, `rescueability` (candidates passing `safety_r` and finite, irrespective of prerequisite gripper/agreement), `Pol_base_success`, `Pol_full_success`, `Δ`. Reported as paper figure + moderator regression of per-task Δ on `rescueability`.

### Modern Primitive Usage
**None.** LeWM is already trained. Frontier primitives here would obscure the contribution.

### Integration into Base Generator / Downstream Pipeline
- **Changes**: only the audit step (`06c_refilter_teacher_quality_v3.py`). Downstream `08_build_distillation_dataset.py` and `08c_train_gr00t_flare_44_20.sh` are **byte-identical**.

### Training Plan
- 6 paired seed runs (Pol_base × Pol_full × {0,1,2}) + 1 single-seed Pol_z ablation.
- ≈ 7 × 20k steps ≈ 100-120 GPU-hours on a 4090/A100.
- Refilter cost: <0.5 GPU-hour, mostly CPU-side pandas.

### Claim-Driven Validation
- **C1 (Main)**: Pol_full > Pol_base on LIBERO-Plus overall success rate, same accepted-sample budget.
  - Primary inferential unit: 40 per-task Δ, sign test.
  - Secondary: per-suite paired t-test; hierarchical bootstrap 95% CI.
  - Predeclared non-inferiority: easy-decile (`Pol_base.direct_aug_rate > 0.70`) lower 95% one-sided bound for mean Δ > −1.5 pp.
- **C2 (Supporting)**: Pol_z ablation isolates `safety_r`.
- **Predeclared moderator**: per-task Δ regressed on `rescueability`; report coefficients and 95% CI.

### Compute & Timeline
- **GPU-hours**: <0.5 refilter + 7 × 20k fine-tunes ≈ 100-120 GPU-hours on a 4090/A100.
- **Timeline**: ≈ 10 working days, single GPU.

### Novelty and Elegance Argument
- **Closest work**:
  1. Existing `flare_filtered_direct_augmentation` (Pol_base) — exact delta is per-task empirical-CDF + cross-task log-ratio latent regret + fixed-budget top-K + paired training seeds.
  2. Classical per-stratum stratified outlier filtering (textbook).
  3. FLARE-VLA / FLARE original teacher selection (same scoring primitive, different normalization).
- **Claim phrasing**: *under a fixed-data budget, task-relative calibration removes task-dependent metric scale, while a shared latent-regret constraint prevents unconditional per-task quota filling.*

## Implementation File Plan
- New file: `FLARE-VLA/scripts/libero_plus/06c_refilter_teacher_quality_v3.py`.
- Inputs: `validation_scores.csv`, `validation_teacher_records.pt`, `validation_stepwise_metrics.npz`, `validation_summary.json`, Pol_base `teacher_quality_summary.json`.
- Outputs:
  - `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/teacher_quality_selected.jsonl`
  - `.../teacher_quality_direct_records.pt`
  - `.../teacher_quality_summary.json` (thresholds, edge policy, calibration diagnostics, Spearman check result)
  - `.../calibration_subset_diagnostics.json`
  - `.../task_diagnostic_table.csv` (rescueability column)
- Downstream: `08_build_distillation_dataset.py` and `08c_train_gr00t_flare_44_20.sh` are **byte-identical**.
