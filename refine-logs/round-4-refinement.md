# Round 4 Refinement

## Problem Anchor (verbatim from round 0 — preserved)
- **Bottom-line problem**: The current FLARE-VLA teacher filter (`06_audit_teacher_quality.py`, policy `flare_filtered_direct_augmentation`) removes outliers with **fixed global quantiles** on per-component action distance, late-window MSE, oracle gap, jerk, and gripper mismatch. The downstream GR00T-N1.7 LIBERO-Plus distillation run (`08_train_gr00t_flare_44_20.sh`) therefore accepts ~63% of candidates (309128 / 487880 in `teacher_scoring_epoch50_fixed_v2`) as equal-weight action targets. Inspection of `teacher_quality_task_summary.csv` shows this global policy is **highly non-uniform across the 40 LIBERO-Plus tasks**: `direct_aug_rate` ranges from 0.045 (task 7, "open the middle drawer of the cabinet") to 0.776 (task 6, "pick up the black bowl from table center"). Tasks where the global rate collapses below 0.30 visibly suffer in `08c_train_gr00t_flare_44_20.sh`, so success-rate improvements on those tasks are the bottleneck.
- **Must-solve bottleneck**: Replace the global fixed-quantile outlier policy with a **suite/task-conditional, latent-score-aware teacher filter** that (a) raises acceptance where confident teacher trajectories exist under current evidence, (b) lowers acceptance where the candidate set is unreliable, and (c) keeps the mechanism small enough to defend in a paper while **raising LIBERO-Plus overall success rate** — with the previously failing `direct_aug_rate < 0.20` tasks as the dominant diagnostic target.
- **Non-goals**: We are NOT changing LeWM training, the candidate generator, the GR00T backbone, the policy gradient, or the teacher horizon. We are NOT introducing new inference-time computation at deployment. We are NOT adding a learned reward model or RL fine-tuning. The paper story remains a **distillation-data curation paper**, not a new VLA architecture.
- **Constraints**: Single host with NVIDIA RTX-class GPU; offline teacher scoring already produced once for ~488k samples at a fixed epoch-50 LeWM checkpoint (`weights_epoch_50.pt`, sha256 `381bc4725e71a5728d172d84e6c706cffa7dc0f01220af1ac0dcfd440dbc5825`). GT latent H5 (`gt_latents/libero_plus_epoch50_h8_framebank_trainpipe.h5`, sha256 `2d74cc68...`) is fixed. Manifest (`libero_plus_sparse_manifest.jsonl`, sha256 `ac3d919a...`) is fixed. Refilter cost must be < 1 GPU-hour on the existing CSV/PT/NPZ audit inputs.
- **Success condition**: On the same backbone and same training budget, the *refiltered* distillation set must yield:
  1. A **measurable overall LIBERO-Plus success-rate improvement** against the current `08c_train_gr00t_flare_44_20.sh` baseline (sign-test on 40 tasks, paired t-test on suite means).
  2. A **reduction of the worst-task gap**: the lowest-`direct_aug_rate` tasks must either gain >5 percentage points success, or, if not, show a documented diagnostic showing why no filter could rescue them.
  3. A **paper-claim-ready comparison** that the new policy is both statistically and qualitatively stronger than the global policy on the same teacher set, scoring budget, and training budget.

## Anchor Check (Round 4)
- **Original bottleneck**: per-task heterogeneity in `direct_aug_rate` from a global-quantile filter; worst-task LIBERO-Plus success.
- **Why the revised method still addresses it**: yes. Method is now mathematically precise. The τ_r + z_s division of labor is restored (Round 3 had collapsed them inadvertently).
- **Reviewer suggestions rejected as drift**: none. All four CRITICAL/IMPORTANT fixes from Round 3 review are adopted.

## Simplicity Check (Round 4)
- **Dominant contribution**: fixed-budget top-K teacher selection under per-task component-badness (`safety_z`) and one shared cross-task latent-regret constraint (`safety_r`) calibrated independently on a small calibration subset.
- **Components removed in round 4**:
  - The "near-zero-oracle" r_s=0 special case (the new formulation `clip[...]` with `max(best, oracle)` makes it naturally 0).
  - The "lowest feasible τ_r" definition (Round 3 had it bound by K; we now fit it on the calibration subset by matching Pol_base's `large_oracle_gap` reject rate, an independent statistical criterion).
- **Reviewer suggestions rejected as unnecessary complexity**: none. All simplification suggestions were accepted.

## Changes Made (Round 4)

### 1. (CRITICAL) Independent calibration criterion for τ_r
- **Reviewer said**: τ_r fitted as "lowest cutoff satisfying |E|≥K" collapses the mechanism to ranking-by-r; τ_r cannot be fitted on the 47k calibration subset while hitting K=309128.
- **Action**: τ_r is fit on the calibration subset to match **Pol_base's marginal rejection rate from `large_oracle_gap`**. Specifically: τ_r = `np.quantile(r_calibration, 1 − reject_rate_baseline)`, where `reject_rate_baseline` = (count of samples Pol_base marked as `large_oracle_gap`) / (total samples in Pol_base). This is a single scalar, computable from `teacher_quality_summary.json` in <1 s.
- **Reasoning**: this gives τ_r an *independent* statistical meaning (its own safety criterion) and removes the implicit dependence on K.
- **Impact**: restores the intended division of labor — safety_r filters by τ_r; safety_z ranks within the eligible set.

### 2. (CRITICAL) Exact log-ratio definition with clipping
- **Reviewer said**: equation had ambiguous parentheses; the `r_s=0 if oracle<1e-6` rule is unsafe.
- **Action**: adopt the reviewer's exact formula:
  \[
  r_s=\operatorname{clip}\!\left[\log\!\left(\frac{\max(\text{best\_mse}_s, \text{oracle\_mse}_s)+\epsilon}{\text{oracle\_mse}_s+\epsilon}\right),\; 0,\; \log(1001)\right].
  \]
  - `r_s = +∞` (i.e. ineligible) when any input is non-finite.
  - No special branch for small `oracle_mse` because `max(best, oracle)` already produces a non-negative numerator even when `best_mse ≤ oracle_mse`, giving `r_s ∈ [0, log(1001)]`.
  - All denominators use `+ε` for stability.
- **Reasoning**: this is what the executable code will do.
- **Impact**: removes ambiguity in the paper's mechanism section.

### 3. (IMPORTANT) Two-sided Spearman check with terminal failure
- **Reviewer said**: `ρ ≤ 0.10` permits ρ=-0.8; the ε-retune is not a real fix.
- **Action**: replace with `|ρ_Spearman(r_s, oracle_mse_s)| ≤ 0.10` (two-sided). If violated at both ε=10^-3 and ε=10^-2, **abort the cross-task latent-regret claim** and publish the diagnostic only. Continue with Pol_z as a backup primary signal.
- **Reasoning**: a low-correlation r_s is the precondition for the cross-task-regret story. If violated, the story is false and we should not paper over it.
- **Impact**: prevents a paper-quality failure mode.

### 4. (IMPORTANT) Pol_z exact definition
- **Reviewer said**: ablation needs equal sample count and exact specification.
- **Action**:
  \[
  \text{Pol}_z = \text{TopK}_{z_s}\{s : \text{prerequisite}(s)\},\quad K=309128,\;\text{tie-break by sample\_id ASC}.
  \]
  Pol_z **does not** apply `safety_r`. Otherwise identical to Pol_full.
- **Reasoning**: now Pol_z vs. Pol_full isolates exactly one mechanism (the latent-regret safety gate).

### 5. (MINOR) Calibration-subset reuse language
- **Reviewer said**: clarify pool-back semantics; label in-sample/out-of-sample CDF status.
- **Action**: explicitly state that calibration episodes are reused as candidate samples after transforms are frozen. CDF values from calibration episodes are in-sample; remaining episodes are out-of-sample. Cross-fitted CDFs are optional and reported only if the in-sample / out-of-sample calibration disagreement is empirically material (>2 percentage points disagreement at p90).

### 6. (MINOR) Removed `flag_gripper_ok` binary prerequisite from rescueability reporting
- The rescueability fraction reported as a moderator uses only `safety_r` and prerequisite finite flags, not gripper. This keeps the moderator interpretable as "regret-safety coverage" rather than "ALL-prerequisite coverage."

## Revised Proposal

# Research Proposal: Budget-Constrained Task-Relative Teacher Filtering for FLARE-VLA on LIBERO-Plus — *Round 4 Final*

*(Problem Anchor preserved verbatim as above; the rest of this document is the cleaned final proposal.)*

## Method Thesis
- **One-sentence thesis**: replace the global-quantile outlier filter with a **fixed-budget top-K** teacher selection rule that operates on two eligibility predicates — per-task component-badness (`safety_z`) and one shared cross-task latent-regret constraint (`safety_r`) — calibrated independently on a small calibration subset, then evaluated end-to-end on LIBERO-Plus under paired training seeds.
- **Why this is the smallest adequate intervention**: two eligibility predicates, one independent scalar cutoff for `safety_r`, deterministic top-K selection; no new training, no new model, no new forward pass.
- **Why this route is timely**: the dominant current debate in VLA is data curation at scale. Our contribution is a *demonstrated principle* with end-to-end LIBERO-Plus evaluation.

## Contribution Focus
- **Dominant contribution**: a **fixed-budget, paired-seed, end-to-end LIBERO-Plus evaluation** of a teacher-filter that combines per-task empirical-CDF rank normalization with a cross-task log-ratio latent-regret gate.
- **Optional supporting contribution**: a per-(suite, task) diagnostic table with rescueability (candidates passing `safety_r` alone) as a continuous moderator.

## Proposed Method

### Complexity Budget
- **Frozen**: LeWM (epoch-50), the four audit artifacts, GR00T-N1.7 LIBERO-Plus student, manifest, GT-latent H5, candidate generator.
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

### Core Mechanism (Round 4 — final, exact)

#### Prerequisite flags (binary)
- `flag_finite_core`, `flag_finite_actions` — finite-metric prerequisite.
- `flag_teacher_agreement` — cosine-best teacher == MSE-best teacher.
- `flag_gripper_ok` — gripper-vs-GT match rate ≥ 0.3.

#### `safety_z` (per-task component-badness)
- `M = {teacher_gt_xyz_mse, teacher_gt_rot_mse, teacher_jerk_xyz_l2_mean, teacher_jerk_rot_l2_mean, late_mse}`.
- `z_s = max_{k in M} F̂_{g(s),k}(m_{s,k})` where `F̂_{g,k}` is the empirical CDF computed on the calibration subset, restricted to `(suite, task_index) = g`, right-tail.
- NaN → CDF = 1 (effectively ineligible in top-K).
- Ties → average rank.
- Fall back to suite-level empirical CDFs if `n_g < 30`; document.

#### `safety_r` (cross-task log-ratio latent-regret)
\[
r_s=\operatorname{clip}\!\left[\log\!\left(\frac{\max(\text{best\_mse}_s, \text{oracle\_mse}_s)+\epsilon}{\text{oracle\_mse}_s+\epsilon}\right),\; 0,\; \log(1001)\right],\quad\epsilon=10^{-3}.
\]
- Eligible iff `safety_r(s) ≤ τ_r` (right-tail).
- Non-finite inputs → ineligible.
- `τ_r` fit on calibration subset: `τ_r = quantile(r_calibration, 1 − reject_rate_baseline)` where `reject_rate_baseline` = `(#samples Pol_base rejected for `large_oracle_gap`) / (total samples)`.

#### Selection (Pol_full)
1. Apply prerequisite flags ∧ `safety_r(s) ≤ τ_r`; let E = eligible.
2. If |E| < K: ABORT with data-level failure mode documented.
3. Top-K of E by ascending `z_s`; ties by `sample_id ASC`.
4. Output = selected set; identical downstream format to `Pol_base`.

#### Selection (Pol_z)
\[
\text{Pol}_z = \text{TopK}_{z_s}\{s : \text{prerequisite}(s)\},\quad K=309128.
\]
Same K; same tie-breaking; no `safety_r`.

### Failure Modes and Diagnostics (Round 4)
- **Group too small (`n_g < 30`)**: fall back to suite-level empirical CDFs.
- **|E| < K**: data-level failure; abort and report.
- **|ρ_Spearman| > 0.10**: cross-task-regret claim invalid; abort that claim; default to Pol_z; report.
- **ε retune failure (both ε=10^-3 and ε=10^-2)**: same as above.

### Optional Supporting Component
- Per-(suite, task) diagnostic: `samples`, `Pol_base_rate`, `Pol_z_rate`, `Pol_full_rate`, `rescueability` (candidates passing `safety_r ≤ τ_r` ∩ finite flags, irrespective of prerequisite gripper/agreement), `Pol_base_success`, `Pol_full_success`, `Δ`. Reported as a paper figure; moderator analysis via linear regression of per-task Δ on `rescueability`.

### Modern Primitive Usage
**None.** LeWM is already trained. Frontier primitives here would obscure the contribution.

### Integration into Base Generator / Downstream Pipeline
- **Changes**: only the audit step (`06c_refilter_teacher_quality_v3.py`). Downstream `08_build_distillation_dataset.py` and `08c_train_gr00t_flare_44_20.sh` are **byte-identical**.

### Training Plan
- 6 paired seed runs (Pol_base × Pol_full × {0,1,2}) + 1 single-seed Pol_z ablation.
- ≈ 7 × 20k steps = 140k steps ≈ 100–120 GPU-hours on a 4090/A100.
- Refilter cost: <0.5 GPU-hour; mostly CPU-side pandas.

### Novelty and Elegance Argument
- **Closest work**:
  1. Existing `flare_filtered_direct_augmentation` (Pol_base) — exact delta is per-task empirical-CDF + cross-task log-ratio latent regret + fixed-budget top-K + paired training seeds.
  2. Classical per-stratum stratified outlier filtering (textbook).
  3. FLARE-VLA / FLARE original teacher selection (same scoring primitive).
- **Claim phrasing**: *under a fixed-data budget, task-relative calibration removes task-dependent metric scale, while a shared latent-regret constraint prevents unconditional per-task quota filling.*

### Claim-Driven Validation (Round 4)
- **C1 (Main)**: Pol_full > Pol_base on LIBERO-Plus overall success rate, on the same accepted-sample budget.
  - Primary inferential unit: 40 per-task Δ, sign test.
  - Secondary: per-suite paired t-test; hierarchical bootstrap 95% CI.
  - Predeclared non-inferiority: easy-decile (`Pol_base.direct_aug_rate > 0.70`) lower 95% one-sided bound for mean Δ > −1.5 pp.
- **C2 (Supporting)**: Pol_z ablation isolates `safety_r`.
  - Metric: Pol_z vs Pol_full per-task Δ; expected Δ ≥ +1 pp on tasks with low rescueability.
- **Predeclared moderator**: per-task Δ regressed on `rescueability`; report coefficients and 95% CI.

### Compute & Timeline Estimate (Round 4)
- **GPU-hours**: <0.5 refilter + 7 × 20k fine-tunes ≈ 100–120 GPU-hours on a 4090/A100.
- **Timeline**: ≈ 10 working days, single GPU.

### Implementation File Plan
- New file: `FLARE-VLA/scripts/libero_plus/06c_refilter_teacher_quality_v3.py`.
  Inputs: `validation_scores.csv`, `validation_teacher_records.pt`, `validation_stepwise_metrics.npz`, `validation_summary.json`, Pol_base `teacher_quality_summary.json`.
  Outputs:
  - `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/teacher_quality_selected.jsonl`
  - `.../teacher_quality_direct_records.pt`
  - `.../teacher_quality_summary.json` (new thresholds, edge policy, calibration diagnostics, Spearman check result)
  - `.../calibration_subset_diagnostics.json`
  - `.../task_diagnostic_table.csv` (rescueability column)
- Downstream: `08_build_distillation_dataset.py` and `08c_train_gr00t_flare_44_20.sh` are **byte-identical**.
