# Round 3 Refinement

## Problem Anchor (verbatim from round 0 — preserved)
- **Bottom-line problem**: The current FLARE-VLA teacher filter (`06_audit_teacher_quality.py`, policy `flare_filtered_direct_augmentation`) removes outliers with **fixed global quantiles** on per-component action distance, late-window MSE, oracle gap, jerk, and gripper mismatch. The downstream GR00T-N1.7 LIBERO-Plus distillation run (`08_train_gr00t_flare_44_20.sh`) therefore accepts ~63% of candidates (309128 / 487880 in `teacher_scoring_epoch50_fixed_v2`) as equal-weight action targets. Inspection of `teacher_quality_task_summary.csv` shows this global policy is **highly non-uniform across the 40 LIBERO-Plus tasks**: `direct_aug_rate` ranges from 0.045 (task 7, "open the middle drawer of the cabinet") to 0.776 (task 6, "pick up the black bowl from table center"). Tasks where the global rate collapses below 0.30 visibly suffer in `08c_train_gr00t_flare_44_20.sh`, so success-rate improvements on those tasks are the bottleneck.
- **Must-solve bottleneck**: Replace the global fixed-quantile outlier policy with a **suite/task-conditional, latent-score-aware teacher filter** that (a) raises acceptance where confident teacher trajectories exist under current evidence, (b) lowers acceptance where the candidate set is unreliable, and (c) keeps the mechanism small enough to defend in a paper while **raising LIBERO-Plus overall success rate** — with the previously failing `direct_aug_rate < 0.20` tasks as the dominant diagnostic target.
- **Non-goals**: We are NOT changing LeWM training, the candidate generator, the GR00T backbone, the policy gradient, or the teacher horizon. We are NOT introducing new inference-time computation at deployment. We are NOT adding a learned reward model or RL fine-tuning. The paper story remains a **distillation-data curation paper**, not a new VLA architecture.
- **Constraints**: Single host with NVIDIA RTX-class GPU; offline teacher scoring already produced once for ~488k samples at a fixed epoch-50 LeWM checkpoint (`weights_epoch_50.pt`, sha256 `381bc4725e71a5728d172d84e6c706cffa7dc0f01220af1ac0dcfd440dbc5825`). GT latent H5 (`gt_latents/libero_plus_epoch50_h8_framebank_trainpipe.h5`, sha256 `2d74cc68...`) is fixed. Manifest (`libero_plus_sparse_manifest.jsonl`, sha256 `ac3d919a...`) is fixed. Refilter cost must be < 1 GPU-hour on the existing CSV/PT/NPZ audit inputs.
- **Success condition**: On the same backbone and same training budget, the *refiltered* distillation set must yield:
  1. A **measurable overall LIBERO-Plus success-rate improvement** against the current `08c_train_gr00t_flare_44_20.sh` baseline (sign-test on 40 tasks, paired t-test on suite means).
  2. A **reduction of the worst-task gap**: the lowest-`direct_aug_rate` tasks must either gain >5 percentage points success, or, if not, show a documented diagnostic showing why no filter could rescue them.
  3. A **paper-claim-ready comparison** that the new policy is both statistically and qualitatively stronger than the global policy on the same teacher set, scoring budget, and training budget.

## Anchor Check (Round 3)
- **Original bottleneck**: per-task heterogeneity in `direct_aug_rate` from a global-quantile filter; worst-task LIBERO-Plus success.
- **Revised method addresses it**: yes. The proposed method conditions on `(suite, task_index)` for component badness *and* uses one shared latent-regret statistic, calibrated on a small calibration subset, with **fixed-budget top-K** selection applied to the *full* candidate population. This is invariant to the HOLDOUT/TRAIN terminology.
- **Reviewer suggestions rejected as drift**: none. All CRITICAL fixes are adopted; the safety-not-rescueable exclusion is **removed** (rescueability becomes a stratification variable, not an exclusion rule).

## Simplicity Check (Round 3)
- **Dominant contribution**: top-K budget-constrained teacher selection under (a) per-task component badness and (b) one shared cross-task latent-regret cutoff.
- **Components removed in round 3**:
  - The "HOLDOUT" terminology (becomes "calibration subset"); HOLDOUT samples are now pooled with TRAIN for selection.
  - The pre-declared "exclude rescueability<200 tasks from success criterion" rule — primary sign test now uses **all 40 tasks**, with rescueability reported as a frozen stratification.
  - The 200-threshold number itself (replaced with continuous rescueability reporting).
- **Reviewer suggestions rejected as unnecessary complexity**: none. All simplification suggestions were adopted (top-K direct selection; separate K for each policy; clearer naming).

## Changes Made

### 1. Replaced threshold-by-bisection with deterministic top-K selection
- **Reviewer said**: HOLDOUT-fitted λ cannot guarantee a fixed budget; pre-declared constraints may be infeasible.
- **Action**: adopt **deterministic top-K selection** keyed on \(z_s\) among samples passing prerequisites ∧ \(r_s \le \tau_r\). K is set to P0's accepted count (309128) by construction. Tie-breaking by stable `sample_id`.
- **Reasoning**: top-K *is* the simplest, exact budget-control mechanism. Calibrating τ_r on the calibration subset now controls *who* is eligible for top-K (a calibration step), not *how many* will be selected (deterministic).
- **Impact**: the budget is now guaranteed by construction. If fewer than K samples pass the safety gate, we predeclare that as a single failure mode of the *data*, not the *method*, and report it openly.

### 2. Renamed policies/predicates to remove P1/P2 collision
- **Reviewer said**: predicates \(P_1, P_2\) collide with policies \(P_0, P_1, P_2\).
- **Action**: predicates → `safety_z` (component badness) and `safety_r` (latent regret). Policies → `Pol_base` (current global baseline), `Pol_z` (safety_z only), `Pol_full` (safety_z ∧ safety_r, our method).
- **Impact**: removes ambiguity in the paper text.

### 3. Pulled calibration subset into the selection pool
- **Reviewer said**: HOLDOUT either is or isn't in the final set; the procedure must be declared.
- **Action**: the calibration subset (10% by hashed trajectory) is pooled back into the selection pool. CDFs and τ_r are fit there, but selection runs on the **full 487,880 candidates**. The calibration subset is used *only* to estimate the CDFs and τ_r.
- **Reasoning**: removing HOLDOUT from selection would shrink the candidate pool by 10%, weakening budget-match and biasing the comparison.
- **Impact**: cleaner accounting. Note explicitly in the paper that the calibration subset is not held out from selection.

### 4. Reverted the safety-not-rescueable exclusion; rescueability becomes a stratification variable
- **Reviewer said**: excluding tasks lets the method remove its own likely failures; "200" is unsupported.
- **Action**: rescueability is now a **continuous stratification variable** (per-task fraction of candidates that pass `safety_r` alone). All 40 tasks are included in the primary sign test. Rescueability enters the analysis as a pre-declared moderator: does P2's advantage correlate with rescueability? (Reported as a paper figure.)
- **Reasoning**: this fixes the original anchor's potentially-unfalsifiable clause. A negative result on a low-rescueability task is just data — it does not "rescue" the method. The primary success criterion is now *honest* across all 40 tasks.
- **Impact**: paper's primary analysis is 40 paired task comparisons, all weighted equally. This is exactly the kind of honest, top-venue inference we want.

### 5. Log-ratio calibration: declare edge cases explicitly
- **Reviewer said**: log ratio isn't auto-calibrated; behavior under small oracle_mse, negative values, etc., undefined.
- **Action**: explicit edge policy:
  - `best_mse_s ≤ oracle_mse_s` ⇒ `r_s = -log((oracle_mse_s + ε) / (oracle_mse_s + ε)) = 0` for oracle-equivalent and **retain as eligible** (this is the best-case latent-regret). Numerically: `r_s = -log(1 + clamp(best_mse - oracle_mse, 0, 1)/oracle_mse)`, clipped to a maximum of `+log(1 + 1/ε) = +log(1001)` for stability.
  - `oracle_mse_s < 1e-6` ⇒ `r_s = 0` (treat as essentially perfect; eligible).
  - Non-finite values ⇒ ineligible.
  - On CALIBRATION, verify that median `r_s` is stable across `oracle_mse` deciles (Spearman ρ ≤ 0.10 with `oracle_mse`; pre-declared).
- **Reasoning**: explicit edge policy is what the reviewer asked for; we report it as a **relative latent-regret constraint**, not an absolute risk guarantee.
- **Impact**: clarifies the paper's mechanism description.

### 6. Calibration subset respects trajectory / episode identity
- **Reviewer said**: splitting by `sample_id` distributes overlapping windows of the same episode across partitions.
- **Action**: partition by hashing `(suite, episode_index)` into 10 buckets; every window belonging to that episode follows the episode. With 14,347 episodes, each partition has ~1,435 episodes; windows per episode average ~33, giving ~47k calibration windows.
- **Reasoning**: this makes the calibration subset honest about temporal dependence.
- **Impact**: trivial implementation; better statistical hygiene.

### 7. Updated compute estimate
- Reviewer said: the 90-110 GPU-hour estimate was wrong because "2 fine-tunes" should be 6.
- Action: restate as **6 paired-seed fine-tunes (P0×3, Pol_full×3) + 1 ablation (Pol_z×1)** = 7 × 20k = 140k steps. On a single 4090/A100 ≈ 100-120 GPU-hours. Pol_z seed-1 is exploratory; we will report it as a *secondary* ablation column. If time-constrained, the explorer-only Pol_z ablation can be deferred to a second batch.

### 8. Centered the paper claim on the principled insight
- **Reviewer said**: novelty was incremental; "first combination" is weak.
- **Action**: claim phrasing updated to:
  > Under a fixed-data budget, task-relative calibration removes task-dependent metric scale, while a shared latent-regret constraint prevents unconditional per-task quota filling. The paper's contribution is a *demonstrated principle* with end-to-end LIBERO-Plus evaluation, not a new algorithmic primitive.
- **Reasoning**: this matches the reviewer's instruction precisely.
- **Impact**: cleaner contribution framing.

### 9. Renamed predicates and tightened the inferential unit
- Policies: `Pol_base` (P0), `Pol_z` (P1), `Pol_full` (P2 / proposed).
- Predicates: `safety_z`, `safety_r`.
- Sign test uses 40 task-level differences after averaging across paired seeds.
- Hierarchical bootstrap resamples **paired training seeds** (not tasks), then tasks, then rollouts.
- Easy-decile non-inferiority reports a **95% one-sided lower CI bound** for mean Δ vs. the −1.5 pp margin.

## Revised Proposal (mechanism section only; full anchor preserved above)

# Research Proposal: Budget-Constrained Task-Relative Teacher Filtering for FLARE-VLA on LIBERO-Plus — *Round 3 Revision*

## Method Thesis (revised)
- **One-sentence thesis**: convert the global-quantile outlier filter into a **fixed-budget, top-K** teacher selection under two eligibility predicates — a per-task component-badness score (`safety_z`) and one shared cross-task latent-regret cutoff (`safety_r`) — calibrated on a calibration subset that pools back into the selection pool.
- **Why this is the smallest adequate intervention**: two eligibility predicates, deterministic top-K selection, no new training, no new model, no new forward pass.
- **Why this route is timely**: the dominant current debate in VLA is *data curation at scale*, not new architectures. The contribution is **a demonstrated principle** with end-to-end LIBERO-Plus evaluation, not a new primitive.

## Contribution Focus (revised)
- **Dominant contribution**: a **budget-constrained, top-K** teacher selection rule combining per-task component-badness (`safety_z`) and cross-task log-ratio latent-regret (`safety_r`), evaluated end-to-end on LIBERO-Plus with **paired training seeds**.
- **Optional supporting contribution**: per-(suite, task) diagnostic table with rescueability as a continuous stratification variable.

## Proposed Method (Round 3)

### Complexity Budget
- **Frozen**: LeWM (epoch-50, fixed SHA), the audit inputs (`validation_teacher_records.pt`, `validation_stepwise_metrics.npz`, `validation_scores.csv`), GR00T-N1.7 LIBERO-Plus student, manifest, GT-latent H5, candidate generator.
- **New trainable components**: **none**.
- **Tempting additions intentionally not used**: no learned reward model / VLM judge; no RL on filtered teachers; no per-task learnable threshold; no new forward pass.

### System Overview (Round 3)
```
audit inputs (CSV/PT/NPZ; 487880 samples)
   │
   ▼
episode-hashed partition (10 buckets); pool all back together for selection;
keep one bucket as the calibration subset for fitting CDFs and τ_r
   │
   ▼
on the calibration subset only:
   ├─ fit per-(suite, task_index) empirical CDFs for M = {xyz, rot, jerk_xyz, jerk_rot, late_mse}
   ├─ fit τ_r as the CALIBRATION quantile of r_s = log((best_mse + ε)/(oracle_mse + ε))
   ├─ accept τ_r + lock edge policy + verify Spearman stability
   │
   ▼
on all 487880 candidates:
   for s:
     eligibility = flag_finite & flag_agreement & flag_gripper_ok
                  ∧ safety_z(s; λ=∞)        [no cutoff; just CDFs]
                  ∧ safety_r(s; τ_r)         [log-ratio ≤ τ_r]
   │
   ▼
top-K = 309128 by safety_z(s) (lowest first), tied by sample_id
   │
   ▼
Pol_full selected subset → teacher_quality_direct_records.pt (round-3 policy)
   │
   ▼
08_build_distillation_dataset.py (teacher_only, high|medium) — UNMODIFIED
   │
   ▼
08c_train_gr00t_flare_44_20.sh × seeds {0, 1, 2} — UNMODIFIED
   │
   ▼
LIBERO-Plus evaluation × 50 rollouts/task × 3 seeds
   │
   ▼
paired-seed sign test (40 dichotomies) + hierarchical bootstrap
non-inferiority 95% lower CI on easy decile vs −1.5 pp
rescueability reported as frozen stratification
```

### Core Mechanism (Round 3)

#### Predicates

**Prerequisite flags** (binary; identical to baseline):
- `flag_finite_core`, `flag_finite_actions` — finite metric prerequisite.
- `flag_teacher_agreement` — cosine-best teacher == MSE-best teacher.
- `flag_gripper_ok` — gripper-vs-GT match rate ≥ 0.3.

**`safety_z` (per-task component-badness)**:
\[
z_s \;=\; \max_{k\in\mathcal{M}} \widehat F_{g(s),k}\!\left(m_{s,k}\right),
\quad
\mathcal{M}=\{\text{xyz\_mse},\text{rot\_mse},\text{jerk\_xyz},\text{jerk\_rot},\text{late\_mse}\}.
\]
- Empirical CDFs \(\widehat F_{g,k}\) computed on the calibration subset, right-tail.
- NaN ⇒ CDF value = 1 ⇒ ineligible in practice (since top-K will downrank it).
- Ties ⇒ average rank (standard convention).
- Fall back to suite-level CDFs if `n_g < 30`; document.

**`safety_r` (cross-task log-ratio latent-regret)**:
\[
r_s \;=\; \log\!\frac{\max(\text{best\_mse}_s, \text{oracle\_mse}_s) + \epsilon}{\text{oracle\_mse}_s + \epsilon},
\quad \epsilon = 10^{-3}.
\]
- Edge cases, pre-declared:
  - `oracle_mse_s < 1e-6` ⇒ `r_s = 0` (eligible).
  - `best_mse_s ≤ oracle_mse_s` ⇒ `r_s ≤ 0` (eligible; "regret-free").
  - non-finite ⇒ ineligible.
  - clipped to `+log(1001)` for stability.
- τ_r = CALIBRATION quantile of `r_s` chosen to satisfy a fixed top-K feasibility *constraint*: the count of candidates with `r_s ≤ τ_r` is ≥ K(=309,128). (Any feasible τ_r suffices; the lowest feasible τ_r is preferred for tightness.)

#### Selection (Round 3, exact)
- Step 1. Apply prerequisite flags ∧ `safety_r(s; τ_r)`; let E = eligible set.
- Step 2. If |E| < K, declaratively fail (data-level, not method-level) and abort.
- Step 3. **Top-K** of E by ascending `z_s`; ties broken by `sample_id` ASC.
- Step 4. Output = selected set; same training-script inputs as the baseline.

#### Calibration
- Calibration subset = 1 of the 10 episode-hashed partitions (≈47k windows, ≈1,435 episodes).
- Fit CDFs in `safety_z` on the calibration subset only.
- Fit τ_r on the calibration subset only.
- Verify on the calibration subset: `spearmanr(r_s, oracle_mse_s) ≤ 0.10` (predeclared stability check). If violated, widen `ε` to `10^{-2}` and refit. Document the chosen `ε`.

#### Why this is fixed-budget by construction
The label-free top-K operator makes accepted count = K by construction, regardless of HOLDOUT/TRAIN split. Policy `Pol_z` uses the same procedure but replaces `safety_r` with no constraint (i.e. `τ_r = +∞`). Policy `Pol_base` is the existing global-quantile teacher file, unchanged.

### Optional Supporting Component
- **Per-(suite, task) diagnostic** with columns: `samples`, `Pol_base_rate`, `Pol_z_rate`, `Pol_full_rate`, `rescueability`, `Pol_base_success`, `Pol_full_success`, `Δ`. Rescueability = fraction of candidates in this task whose `r_s ≤ τ_r`. Reported as a continuous variable for moderator analysis.

### Modern Primitive Usage
**None.** The bottleneck is data curation; LeWM is already trained. A frontier primitive here would obscure the contribution.

### Integration into Base Generator / Downstream Pipeline
- **Frozen**: same as round 2.
- **Changes**: only `06_audit_teacher_quality.py` and its wrappers; downstream `08_build_distillation_dataset.py`, `08c_train_gr00t_flare_44_20.sh` are **byte-identical**.

### Training Plan
- **Pol_base × Pol_full paired over seeds {0,1,2}**: 6 fine-tunes of 20k steps each.
- **Pol_z × 1 seed**: 1 ablation fine-tune.
- Total ≈ 7 × 20k = 140k steps ≈ 100-120 GPU-hours on a 4090/A100.
- **Refilter cost**: <0.5 GPU-hour on the same machine.

### Failure Modes and Diagnostics (Round 3)
- **Small groups (`n_g < 30`)**: fall back to suite-level empirical CDFs. Per-task counts ≫ 30 in LIBERO-Plus.
- **`|E| < K`**: method-level failure of the data (not the method). Abort and report honestly. This is preferred over auto-relaxation, which would silently break the budget.
- **`safety_r` calibration unstable**: widen `ε` to `10^{-2}` and refit; document. Predeclare.
- **Per-task rate collapse below 0.10**: report as an analysis row but do NOT exclude from the sign test; report moderator correlation between Δ and baseline `direct_aug_rate`.

### Novelty and Elegance Argument (Round 3)
- **Closest work**:
  1. Existing `flare_filtered_direct_augmentation` (Pol_base) — exact delta is per-task empirical-CDF rank normalization **and** log-ratio cross-task latent regret **and** fixed-budget top-K.
  2. Classical per-stratum stratified outlier filtering (textbook pattern).
  3. FLARE-VLA / FLARE original teacher selection (same scoring primitive).
- **Exact difference**: ours is, to our knowledge, the first **fixed-budget, paired-seed, end-to-end LIBERO-Plus evaluation** of a teacher-filter that combines per-task empirical-CDF rank normalization with a *cross-task log-ratio* latent regret, with rescueability as a continuous stratification variable.
- **Why this is focused**: two eligibility predicates + deterministic top-K; ablation collapsed from 4 to 3 policies.
- **Claim phrasing (final)**: *under a fixed-data budget, task-relative calibration removes task-dependent metric scale, while a shared latent-regret constraint prevents unconditional per-task quota filling*.

## Claim-Driven Validation Sketch (Round 3)

### Claim 1 (Main): Pol_full > Pol_base on LIBERO-Plus overall success rate, on the same accepted-sample budget.
- **Pol_base** (current global baseline).
- **Pol_z** (per-task only, single-seed ablation).
- **Pol_full** (per-task + log-ratio latent regret — proposed method).
- **Training**: Pol_base and Pol_full paired over seeds {0, 1, 2}; Pol_z single-seed ablation.
- **Primary inferential unit**: 40 per-task Δ (averaged across paired seeds) → sign test.
- **Secondary**: per-suite paired t-test on per-task means.
- **Hierarchical bootstrap**: paired training seeds → tasks → rollouts. Report 95% CI.
- **Predeclared non-inferiority margin**: −1.5 pp on the easy-decile (`Pol_base.direct_aug_rate > 0.70`) task subset. Test: 95% one-sided lower CI for mean Δ > −1.5 pp.

### Claim 2 (Supporting): Refilter is small — no measurable regression on previously-easy tasks and on per-suite means.
- Easy-decile non-inferiority (claim 1).
- Per-suite Δ non-negative in all four suites (paired t-test, secondary).

### Predeclared moderator analysis (rescueability)
- `rescueability_g` = fraction of candidates in task g whose `r_s ≤ τ_r`.
- Linear regression of per-task Δ on `rescueability_g`, fit on all 40 tasks.
- Report: does P2 advantage increase with rescueability? If yes, this is interpretive evidence that the latent-regret gate is the active mechanism.

## Experiment Handoff Inputs
- **Must-prove claims**: C1 (overall gain under fixed budget, 40 task sign test), C2 (non-inferiority on easy decile).
- **Must-run ablations**: Pol_base × Pol_full paired over {0,1,2}; Pol_z single seed.
- **Critical datasets / metrics**: LIBERO-Plus overall success, per-task Δ, per-suite Δ, paired sign-test p-value on 40 dichotomies, hierarchical bootstrap 95% CI, easy-decile non-inferiority 95% lower CI vs −1.5 pp.
- **Highest-risk assumptions**: (i) `08c_train_gr00t_flare_44_20.sh` reproduces within ±2 pp across seeds; (ii) `r_s` calibration is stable across `oracle_mse` deciles; (iii) **|E| ≥ K=309,128** under the chosen τ_r; (iv) audit artifacts contain `oracle_mse_score`.

## Compute & Timeline Estimate (Round 3)
- **Estimated GPU-hours**: <0.5 refilter + 7 × 20k fine-tunes (6 paired + 1 ablation) ≈ **100-120 GPU-hours** on a single 4090/A100. CPU-only: <0.5 h refilter + a few hours analysis.
- **Timeline**: 1 day refilter + calibration + pre-declared checks; 4-5 days fine-tunes (sequential on one GPU); 2 days evaluation; 2 days analysis + figures. ≈ 10 working days, single GPU.

## Implementation File Plan (Round 3)
- New file: `FLARE-VLA/scripts/libero_plus/06c_refilter_teacher_quality_v3.py`.
  Inputs:
  - `validation_scores.csv`
  - `validation_teacher_records.pt`
  - `validation_stepwise_metrics.npz`
  - `validation_summary.json`
  - `teacher_quality_summary.json` (Pol_base reference counts).
  Outputs:
  - `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/teacher_quality_selected.jsonl`
  - `.../teacher_quality_direct_records.pt`
  - `.../teacher_quality_summary.json` (new thresholds, edge-policy, calibration diagnostics).
  - `.../calibration_subset_diagnostics.json` (Spearman stability check).
- Downstream: `08_build_distillation_dataset.py` and `08c_train_gr00t_flare_44_20.sh` are **byte-identical**; only their input `teacher_quality_direct_records.pt` differs.
