# Round 2 Refinement

## Problem Anchor (verbatim from round 0 — preserved)
- **Bottom-line problem**: The current FLARE-VLA teacher filter (`06_audit_teacher_quality.py`, policy `flare_filtered_direct_augmentation`) removes outliers with **fixed global quantiles** on per-component action distance, late-window MSE, oracle gap, jerk, and gripper mismatch. The downstream GR00T-N1.7 LIBERO-Plus distillation run (`08_train_gr00t_flare_44_20.sh`) therefore accepts ~63% of candidates (309128 / 487880 in `teacher_scoring_epoch50_fixed_v2`) as equal-weight action targets. Inspection of `teacher_quality_task_summary.csv` shows this global policy is **highly non-uniform across the 40 LIBERO-Plus tasks**: `direct_aug_rate` ranges from 0.045 (task 7, "open the middle drawer of the cabinet") to 0.776 (task 6, "pick up the black bowl from table center"). Tasks where the global rate collapses below 0.30 visibly suffer in `08c_train_gr00t_flare_44_20.sh`, so success-rate improvements on those tasks are the bottleneck.
- **Must-solve bottleneck**: Replace the global fixed-quantile outlier policy with a **suite/task-conditional, latent-score-aware teacher filter** that (a) raises acceptance where confident teacher trajectories exist under current evidence, (b) lowers acceptance where the candidate set is unreliable, and (c) keeps the mechanism small enough to defend in a paper while **raising LIBERO-Plus overall success rate** — with the previously failing `direct_aug_rate < 0.20` tasks as the dominant diagnostic target.
- **Non-goals**: We are NOT changing LeWM training, the candidate generator, the GR00T backbone, the policy gradient, or the teacher horizon. We are NOT introducing new inference-time computation at deployment. We are NOT adding a learned reward model or RL fine-tuning. The paper story remains a **distillation-data curation paper**, not a new VLA architecture.
- **Constraints**: Single host with NVIDIA RTX-class GPU; offline teacher scoring already produced once for ~488k samples at a fixed epoch-50 LeWM checkpoint (`weights_epoch_50.pt`, sha256 `381bc4725e71a5728d172d84e6c706cffa7dc0f01220af1ac0dcfd440dbc5825`). GT latent H5 (`gt_latents/libero_plus_epoch50_h8_framebank_trainpipe.h5`, sha256 `2d74cc68...`) is fixed. Manifest (`libero_plus_sparse_manifest.jsonl`, sha256 `ac3d919a...`) is fixed. Refilter cost must be < 1 GPU-hour on the existing CSV/PT/NPZ audit inputs.
- **Success condition**: On the same backbone and same training budget, the *refiltered* distillation set must yield:
  1. A **measurable overall LIBERO-Plus success-rate improvement** against the current `08c_train_gr00t_flare_44_20.sh` baseline (sign-test on 40 tasks, paired t-test on suite means).
  2. A **reduction of the worst-task gap**: the lowest-`direct_aug_rate` tasks must either gain >5 percentage points success, or, if not, show a documented diagnostic showing why no filter could rescue them.
  3. A **paper-claim-ready comparison** that the new policy is both statistically and qualitatively stronger than the global policy on the same teacher set, scoring budget, and training budget.

## Anchor Check
- **Original bottleneck**: per-task heterogeneity in `direct_aug_rate` from a global-quantile filter; worst-task LIBERO-Plus success.
- **Why the revised method still addresses it**: still a single-script refilter on the same audit artifacts; now (i) normalizes per-task nuisance-scale variance via **task-relative component badness**, and (ii) preserves cross-task trust calibration via a **single shared latent-regret constraint**.
- **Reviewer suggestions rejected as drift**: none. All three CRITICAL fixes were adopted; all IMPORTANT fixes adopted.

## Simplicity Check
- **Dominant contribution after revision**: a budget-controlled teacher filter that **combines a per-task component-badness score with one shared cross-task latent-regret gate**, evaluated against the unchanged baseline under fixed selection count.
- **Components removed**: the `best_group_relative` clause (1.5×IQR and rank-OR); oracle-gap from Layer-1 quantiles; `B-nosuite-nooracle` ablation column.
- **Reviewer suggestions rejected as unnecessary complexity**: NONE — every simplification suggestion was accepted.
- **Why the remaining mechanism is still the smallest adequate route**: two predicates, both on already-computed metrics; no learned components; no new forward passes; evaluation budget matched to baseline by construction.

## Changes Made

### 1. New policy framing: budget-constrained conditional calibration under a global latent-risk constraint
- **Reviewer said**: novelty framing is too weak ("first combination of two heuristics").
- **Action**: reframe as a single principle — *task conditioning removes nuisance-scale variance, a shared latent-regret constraint preserves cross-task trust calibration* — and make **fixed-budget selection** part of the method, not an analysis trick.
- **Reasoning**: fixed-budget comparison isolates *curation quality* from *dataset volume*; the reviewer is right that 309k vs 200k accepted samples cannot speak about either of the two competing interpretations.
- **Impact on core method**: λ (the task-relative badness cutoff) is now chosen **on a held-out 10% audit split** by bisection so total accepted = P0's 309128 (within ±1%). The latent-regret cutoff τ_r is chosen on the same held-out split to satisfy P0's overall reject rate of known-bad cases.

### 2. Replaced per-task oracle-gate quantile with a global latent-regret statistic
- **Reviewer said**: per-task oracle quantile cannot identify unreliable tasks; duplicates baseline behavior.
- **Action**: define a single **cross-task excess latent error**
  \[
  r_i = \log \frac{\text{best\_mse}_i + \epsilon}{\text{oracle\_mse}_i + \epsilon},\quad \epsilon = 10^{-3},
  \]
  threshold globally with one cutoff τ_r (calibrated to match P0's reject rate of `large_oracle_gap`).
- **Reasoning**: r_i is a *latent-regret* signal — how much worse the picked teacher is than the oracle action chunk evaluated under the same latent world model. This is genuinely new: the existing audit reports `oracle_gap_mse` as a per-sample *additive* gap; we convert it to a *ratio* in log space, which is approximately scale-free across tasks.
- **Impact on core method**: L1 (per-group component badness) and L2 (global r-regret) are now distinct (not duplicative). Reviewer CRITICAL #1 fixed.

### 3. Collapsed three gates into one task-relative component badness score
- **Reviewer said**: three gates are arbitrary + unstable; minimum+IQR shrinks with group size; rank-OR shrinks dataset.
- **Action**: define
  \[
  z_i = \max_{k \in \mathcal{M}} \widehat F_{g(i), k}\!\left(m_{i,k}\right)
  \]
  where \(\mathcal{M} = \{\text{teacher\_gt\_xyz\_mse}, \text{teacher\_gt\_rot\_mse}, \text{teacher\_jerk\_xyz\_l2\_mean}, \text{teacher\_jerk\_rot\_l2\_mean}, \text{late\_mse}\}\); \(\widehat F_{g,k}\) is the empirical CDF of metric *k* restricted to group *g* (computed on the training audit split only). Use **right-tail** direction (higher metric ⇒ worse) for all five. Accept iff \(z_i \le \lambda\). NaN assigned CDF = 1 (worst, eliminated). Ties broken by mean of tied ranks (standard empirical-CDF convention).
- **Reasoning**: per-task empirical-CDF normalization is exactly the textbook per-stratum rank-normalization the reviewer asked for; the max collapses five metrics into one interpretable scalar; the cutoff λ is what now matches the budget.
- **Impact on core method**: reviewer's CRITICAL #3 fixed; `flag_finite_*` and `flag_teacher_agreement` and `flag_gripper_ok` (binary channels) become *prerequisite* flags outside the badness score (pass them all ⇒ evaluation uses empirical CDF; fail any ⇒ reject directly). This preserves the no-inverted-gripper safety net.

### 4. Fixed-budget selection via held-out calibration split
- **Reviewer said**: anchor vs. validation policy splits, target subset inconsistencies, compute/ablation inconsistent.
- **Action**: split audit samples into TRAIN (90%) / HOLDOUT (10%) *deterministically by sample_id*. All threshold fitting (λ, τ_r) happens on HOLDOUT; final policy is then evaluated on TRAIN. This is one held-out split per metric, not used in the existing 06 script — fits in <5 minutes of CPU work.
- **Reasoning**: removing selection-by-evaluation leakage is mandatory for a paper.
- **Impact**: removes the un-falsifiable "no filter could rescue" clause. We instead define a **rescueability statistic**: for each task with `direct_aug_rate < 0.20`, the count of candidates that would pass the *global latent-regret gate alone* (excluding task-relative badness). If that count is small (<200 samples/task), we report the task as **safety-not-rescueable** and treat as negative evidence — but it does not count toward the success criterion; the success criterion only counts tasks where rescueability ≥ 200.

### 5. Three policies, not four, with paired training seeds
- **Reviewer said**: B-nosuite-nooracle mislabeled.
- **Action**: rename to **(P0)** Current global baseline; **(P1)** Task-conditional component badness, no latent-regret gate; **(P2)** Task-conditional component badness + global latent-regret gate (proposed method). Train P0 and P2 with **paired training seeds** {0, 1, 2} (3 seeds × 20k steps each = ~48 GPU-hours total on a 4090). P1 is an exploration-only single-seed run that supports the ablation table.
- **Reasoning**: matched seeds remove training variance from the comparison.
- **Impact**: compute roughly doubles from round-1 plan to ~48 GPU-hours. This is still tractable on a single host and is the minimum to defend a top-venue claim.

### 6. Hierarchical bootstrap statistical plan
- **Reviewer said**: paired t-test on suite means has no power; non-inferiority margin must be pre-declared.
- **Action**: hierarchical bootstrap over (training seed ∈ {0,1,2}) × (task ∈ 40) × (eval rollout ∈ 50 trials). Report:
  - overall LIBERO-Plus success-rate **95% CI** (mean ± 1.96·SE_bootstrap);
  - per-task Δ **sign-test p-value** (40 dichotomies) as a primary summary;
  - per-suite paired t-test on per-task means as a secondary summary;
  - predeclared non-inferiority margin of **−1.5 pp** on the easy-decile task subset (defined once before training); one-sided 95% lower bound of mean Δ on the easy decile must exceed −1.5 pp.
- **Reasoning**: this is the smallest statistical plan that supports a top-venue causal claim with three seeds.
- **Impact**: longer analysis script (~200 lines), but no extra GPU cost.

### 7. Removed acceptance-floor safeguard; replaced with rescueability statistic
- **Reviewer said**: `q ≤ 0.95` clamp is mathematically backward.
- **Action**: delete clamp entirely. For tasks with rescueability ≥ 200 (defined in #4 above), the success criterion applies. For tasks with rescueability < 200, we report a "safety-not-rescueable" diagnostic and exclude from the success criterion count.
- **Reasoning**: low rescueability is a legitimate negative finding; pretending to "rescue" them would be paper fraud.
- **Impact**: failure-mode list shrinks to two entries (small groups; tight latent-regret cutoff), both with concrete mitigations.

### 8. Filtering unit and implementation specifics
- **Reviewer said (MINOR)**: specify filtering unit, NaN, ties, etc.
- **Action**: filtering **unit = per sample (per window)**, identical to current 06 script. NaN metrics → empirical CDF = 1 (rejected). Ties → average rank (standard convention). Gripper-match-rate kept as **binary prerequisite flag** (gripper is a discrete channel, not amenable to empirical-CDF). Teacher-agreement kept as **binary prerequisite** (unstable scoring → reject). Sample-level finiteness flag is unchanged. All thresholds fit on HOLDOUT only, then locked.

## Revised Proposal

# Research Proposal: Budget-Constrained Task-Relative Teacher Filtering for FLARE-VLA on LIBERO-Plus

## Problem Anchor
*(verbatim as in round 0, no edits)*

## Technical Gap
1. **Where the current pipeline breaks**: the existing `flare_filtered_direct_augmentation` policy uses a *single global threshold* per metric (e.g. `teacher_gt_xyz_mse` ≤ p90 over the 487k-sample population). Tasks where GT trajectories are inherently more variable (drawer/door, articulated objects) end up under-served. The downstream student cannot recover.
2. **Why naive fixes fail**: lowering global quantiles uniformly imports noise; per-task oracle-quantile thresholds cannot identify unreliable tasks; per-task rank normalization *alone* is silent about cross-task trust.
3. **Smallest adequate intervention**: a two-predicate filter — (P1) task-relative component badness removes nuisance-scale variance; (P2) a *cross-task* latent-regret constraint preserves trust calibration. **Fixed-budget**: total accepted ≈ current baseline's 309,128 ± 1%.
4. **Why this is *not* contribution sprawl**: both predicates reuse already-computed metrics. Zero new model components, zero new forward passes.
5. **Why no frontier primitive is required**: pure data-curation. LeWM is already trained; VLM judges / reward models / RL fine-tuning would obscure, not sharpen.
6. **Core technical claim**: a budget-constrained filter combining per-task component badness with a global latent-regret gate yields higher LIBERO-Plus success rate than the global baseline, particularly on under-served tasks, **at the same training budget and same accepted-sample budget as the baseline**.

## Method Thesis
- **One-sentence thesis**: convert the global-quantile outlier filter into a fixed-budget combination of a per-task component-badness score and one shared cross-task latent-regret gate, calibrated on a held-out audit split, and ship the resulting distillation set as the new FLARE-VLA training data.
- **Why this is the smallest adequate intervention**: two predicates, no new training, no new model, no new forward pass.
- **Why this route is timely**: the dominant current debate in VLA is *data curation at scale*, not new architectures. This is a *budget-controlled conditional calibration under a global latent-risk constraint*.

## Contribution Focus
- **Dominant contribution**: a **budget-constrained task-relative + cross-task latent-regret** teacher filter for FLARE-VLA, evaluated end-to-end on LIBERO-Plus.
- **Optional supporting contribution**: a per-(suite, task) diagnostic table linking rescueability, selection rate, and student gain — analysis artifact, not a method.

## Proposed Method

### Complexity Budget
- **Frozen**: LeWM (epoch-50, fixed SHA), the audit inputs (`validation_teacher_records.pt`, `validation_stepwise_metrics.npz`, `validation_scores.csv`), GR00T-N1.7 LIBERO-Plus student, manifest, GT-latent H5, candidate generator.
- **New trainable components**: **none**.
- **Tempting additions intentionally not used**: no learned reward model / VLM judge; no RL on filtered teachers; no per-task learnable threshold; no new forward pass.

### System Overview
```
audit inputs (CSV/PT/NPZ; 487880 samples)
   │
   ▼
deterministic split: TRAIN (90%) / HOLDOUT (10%)
   │
   ▼
on HOLDOUT only:
   ├─ fit λ  (task-relative badness cutoff → bisection to P0's 309128 ± 1%)
   └─ fit τ_r (latent-regret cutoff → match P0's overall reject rate)
   │
   ▼
apply to TRAIN:
   for s in TRAIN:
     prerequisite ok iff flag_finite_core & flag_finite_actions
                       & flag_teacher_agreement & flag_gripper_ok
     g ← (suite, task_index) of s
     z_s = max_{k ∈ M} F̂_{g,k}(m_{s,k})     with M = {xyz_mse, rot_mse, jerk_xyz, jerk_rot, late_mse}
     r_s = log((best_mse_s + ε) / (oracle_mse_s + ε))
     accept s iff  prerequisite ok
                ∧ z_s ≤ λ
                ∧ r_s ≤ τ_r
   │
   ▼
selected subset → teacher_quality_direct_records.pt (new policy)
   │
   ▼
08_build_distillation_dataset.py (teacher_only, high|medium)
   │
   ▼
08c_train_gr00t_flare_44_20.sh × seeds {0, 1, 2}   (identical training recipe)
   │
   ▼
LIBERO-Plus evaluation × 50 rollouts/task
   │
   ▼
hierarchical bootstrap + sign test + non-inferiority on easy decile
```

### Core Mechanism
- **Inputs**: per-sample metrics in `validation_teacher_records.pt` + `validation_stepwise_metrics.npz`. Group key = `(suite, task_index)`.
- **Filtering unit**: per-sample (per validation window). Binary flags (gripper match, teacher agreement, finiteness) are prerequisite.
- **Prerequisite flags** (same as current, retained for safety):
  - `flag_finite_core`, `flag_finite_actions` — finite-action prerequisite.
  - `flag_teacher_agreement` — reject if cosine-best ≠ MSE-best teacher.
  - `flag_gripper_ok` — reject if `teacher_gt_gripper_match_rate < 0.3`.
- **Predicate P1 (per-task component badness)**:
  \[
  z_s = \max_{k\in\mathcal{M}} \widehat F_{g(s),k}\!\left(m_{s,k}\right),\quad
  \mathcal{M}=\{\text{xyz\_mse}, \text{rot\_mse}, \text{jerk\_xyz}, \text{jerk\_rot}, \text{late\_mse}\}.
  \]
  All metrics right-tail. NaN → CDF = 1. Ties → average rank. Empirical CDFs fit on HOLDOUT only.
- **Predicate P2 (cross-task latent regret)**:
  \[
  r_s = \log \frac{\text{best\_mse}_s + \epsilon}{\text{oracle\_mse}_s + \epsilon}, \quad \epsilon = 10^{-3}.
  \]
  Single global cutoff τ_r; right-tail. Captures *relative* regret to the latent world model.
- **Accept**: prerequisite ∧ \(z_s \le \lambda\) ∧ \(r_s \le \tau_r\).
- **Budget**: λ chosen by bisection on HOLDOUT to match P0's accepted count within ±1%. τ_r chosen on HOLDOUT to match P0's reject rate from `large_oracle_gap`. Both locked before TRAIN evaluation.

### Optional Supporting Component
- **Diagnostic table** (analysis, not a model): per-(suite, task) rows reporting `samples`, `baseline_selection_rate`, `new_selection_rate`, `baseline_success`, `new_success`, `Δ`, `rescueability`. Doubles as a paper figure and as a tool for future authors. No extra compute.

### Modern Primitive Usage
- **None**. The bottleneck is data curation; the latent scorer is already trained.

### Integration into Base Generator / Downstream Pipeline
- **Frozen**: student, scorer, manifest, GT-H5, candidate generator.
- **Changes**: only the audit step (`06_audit_teacher_quality.py`) and its wrapper commands (`06b_*`). Downstream `08_build_distillation_dataset.py` and `08c_train_gr00t_flare_44_20.sh` are **byte-identical**, only their input file changes.

### Training Plan
- **No new training** beyond three seeds of `08c_train_gr00t_flare_44_20.sh`. Refilter: <1 GPU-hour (mostly CPU-side `pandas.groupby`).

### Failure Modes and Diagnostics
- **Small groups (`n_g < 30`)**: fall back to suite-level empirical CDFs. (Empirical-CDF mode is robust; bin count `≤ 30` is theoretical floor — well below LIBERO-Plus per-task counts of 4k–14k.)
- **Tight latent-regret cutoff**: clamp τ_r at the HOLDOUT-fitted value's 99th percentile to bound run-away shrinkage. Document in JSON summary.
- **Rescueability diagnostic**: for any task with rescueability < 200, the task is reported as **safety-not-rescueable** and excluded from the success-criterion count.

### Novelty and Elegance Argument
- **Closest work**:
  1. Existing `flare_filtered_direct_augmentation` (paper's exact baseline) — exact delta is per-task rank normalization replacing global quantiles **and** a log-ratio latent-regret replacing additive oracle-gap.
  2. Classical per-stratum stratified outlier filtering (textbook).
  3. FLARE (cheang et al., 2023) original teacher selection (same scoring primitive, different normalization).
- **Exact difference**: ours is, to our knowledge, the first *budget-controlled* per-task rank normalization with a *cross-task log-ratio latent-regret gate*, evaluated end-to-end on LIBERO-Plus.
- **Why this is focused**: two predicates on already-computed metrics; the ablation column dropped from four policies to three.

## Claim-Driven Validation Sketch

### Claim 1 (Main): Budget-constrained per-task component badness + cross-task log-ratio latent-regret gate raises LIBERO-Plus success rate over the global-quantile baseline, on the same accepted-sample budget.
- **P0** — current global baseline (existing `teacher_quality_direct_records.pt`).
- **P1** — per-task component badness without latent-regret gate (ablation column).
- **P2** — per-task component badness + global log-ratio latent-regret gate (the proposed method).
- **Training**: P0 and P2 with paired seeds {0,1,2}; P1 single seed (ablation evidence only).
- **Metric**: LIBERO-Plus overall success rate (40 tasks × 50 trials × 3 seeds). Predeclared non-inferiority margin −1.5 pp on easy decile. Per-task Δ sign-test (40 dichotomies) primary, paired t-test on suite means secondary.
- **Success evidence (predeclared)**: overall Δ ≥ +3 pp with 95% bootstrap CI excluding 0; per-task sign-test p < 0.05 favoring P2 ≥ 25 of 40 tasks; easy-decile non-inferiority lower bound > −1.5 pp.

### Claim 2 (Supporting): Refilter is small — no measurable degradation on previously-easy tasks and on per-suite means.
- **Metric**: per-task Δ on easy decile (`baseline_direct_aug_rate > 0.70`); per-suite Δ.
- **Evidence**: easy-decile non-inferiority (claim 1); per-suite paired t-test non-negative in all four suites.

## Experiment Handoff Inputs
- **Must-prove claims**: C1 (overall gain under fixed budget), C2 (no regression on easy decile).
- **Must-run ablations**: P0, P2 paired over seeds {0,1,2}. P1 single-seed exploration. Hierarchical bootstrap inference.
- **Critical datasets / metrics**: LIBERO-Plus overall success, per-task Δ, per-suite Δ, hierarchical bootstrap CI.
- **Highest-risk assumptions**: (i) `08c_train_gr00t_flare_44_20.sh` reproduces within ±2 pp across seeds; (ii) the HOLDOUT split is sufficient for fitting two scalar cutoffs; (iii) audit artifacts already contain `oracle_mse_score`.

## Compute & Timeline Estimate
- Estimated GPU-hours: <1 refilter + 6 × 20k-step fine-tunes (P0+P2 × 3 seeds) + ~16 hours ablation/support ≈ 90–110 GPU-hours on a single 4090/A100. CPU-only: <0.5 h for refilter + a few hours for analysis.
- Timeline: 1 day refilter; 3-4 days fine-tunes (sequential on one GPU); 1 day eval; 2 days analysis + figures.

## Implementation Note (file-level)
- One new Python file: `FLARE-VLA/scripts/libero_plus/06b_refilter_teacher_quality_v2.py`.
- It reads `validation_scores.csv`, `validation_teacher_records.pt`, `validation_stepwise_metrics.npz`, `validation_summary.json` from the existing `teacher_scoring_epoch50_fixed/` directory, performs the deterministic split, fits λ and τ_r on HOLDOUT, applies to TRAIN, and writes
  `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/task_summary.csv|json|selected.jsonl|direct_records.pt`.
- Downstream `08_build_distillation_dataset.py` and `08c_train_gr00t_flare_44_20.sh` are **unmodified**.
