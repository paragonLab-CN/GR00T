# Research Proposal: Suite-Conditional, Latent-Score-Aware Teacher Filtering for FLARE-VLA on LIBERO-Plus

## Problem Anchor
- **Bottom-line problem**: The current FLARE-VLA teacher filter (`06_audit_teacher_quality.py`, policy `flare_filtered_direct_augmentation`) removes outliers with **fixed global quantiles** on per-component action distance, late-window MSE, oracle gap, jerk, and gripper mismatch. The downstream GR00T-N1.7 LIBERO-Plus distillation run (`08_train_gr00t_flare_44_20.sh`) therefore accepts ~63% of candidates (309128 / 487880 in `teacher_scoring_epoch50_fixed_v2`) as equal-weight action targets. Inspection of `teacher_quality_task_summary.csv` shows this global policy is **highly non-uniform across the 40 LIBERO-Plus tasks**: `direct_aug_rate` ranges from 0.045 (task 7, "open the middle drawer of the cabinet") to 0.776 (task 6, "pick up the black bowl from table center"). Tasks where the global rate collapses below 0.30 visibly suffer in `08c_train_gr00t_flare_44_20.sh`, so success-rate improvements on those tasks are the bottleneck.
- **Must-solve bottleneck**: Replace the global fixed-quantile outlier policy with a **suite/task-conditional, latent-score-aware teacher filter** that (a) raises acceptance where confident teacher trajectories exist under current evidence, (b) lowers acceptance where the candidate set is unreliable, and (c) keeps the mechanism small enough to defend in a paper while **raising LIBERO-Plus overall success rate** — with the previously failing `direct_aug_rate < 0.20` tasks as the dominant diagnostic target.
- **Non-goals**: We are NOT changing LeWM training, the candidate generator, the GR00T backbone, the policy gradient, or the teacher horizon. We are NOT introducing new inference-time computation at deployment. We are NOT adding a learned reward model or RL fine-tuning. The paper story remains a **distillation-data curation paper**, not a new VLA architecture.
- **Constraints**: Single host with NVIDIA RTX-class GPU; offline teacher scoring already produced once for ~488k samples at a fixed epoch-50 LeWM checkpoint (`weights_epoch_50.pt`, sha256 `381bc4725e71a5728d172d84e6c706cffa7dc0f01220af1ac0dcfd440dbc5825`). GT latent H5 (`gt_latents/libero_plus_epoch50_h8_framebank_trainpipe.h5`, sha256 `2d74cc68...`) is fixed. Manifest (`libero_plus_sparse_manifest.jsonl`, sha256 `ac3d919a...`) is fixed. Refilter cost must be < 1 GPU-hour on the existing CSV/PT/NPZ audit inputs.
- **Success condition**: On the same backbone and same training budget, the *refiltered* distillation set must yield:
  1. A **measurable overall LIBERO-Plus success-rate improvement** against the current `08c_train_gr00t_flare_44_20.sh` baseline (sign-test on 40 tasks, paired t-test on suite means).
  2. A **reduction of the worst-task gap**: the lowest-`direct_aug_rate` tasks must either gain >5 percentage points success, or, if not, show a documented diagnostic showing why no filter could rescue them.
  3. A **paper-claim-ready comparison** that the new policy is both statistically and qualitatively stronger than the global policy on the same teacher set, scoring budget, and training budget.

## Technical Gap
1. **Where the current pipeline breaks**: The existing `flare_filtered_direct_augmentation` policy uses a single global threshold per metric (e.g. `teacher_gt_xyz_mse` ≤ p90 over the *entire* 487k-sample population). Tasks where GT trajectories are inherently more variable (e.g. drawer/door tasks, articulated-object tasks) end up under-served: their *truthful* teacher trajectories are rejected because they look like outliers relative to easier tasks (pick-up-and-place). This creates per-task selection rate heterogeneity that the downstream student simply cannot recover from.
2. **Why naive fixes fail**:
   - Simply lowering all quantiles (e.g. p90 → p95) imports noisy teachers uniformly — would help fragile tasks but poison robust tasks.
   - Reweighting teachers by confidence has been explicitly rejected in the current policy (see docstring of `06_audit_teacher_quality.py`).
   - Per-task quantile normalization *alone* is a known fix but is purely classical; it gives no oracle-gated signal and is silent about whether the surviving teachers are actually trustworthy.
3. **Smallest adequate intervention**: A two-layer filter:
   - **Layer 1 — suite/task-conditioned normalization**: thresholds are computed within each (suite, task_index) group (or within a robust unit) instead of globally. This is a one-line change to the `finite_quantile` calls in `06_audit_teacher_quality.py`. Already partially explored in `06b_audit_teacher_quality_no_xyz.sh` (off by `--no-xyz-distance-filter`), but the per-group structure is the missing piece.
   - **Layer 2 — oracle-gated acceptance (a true "latent-score-aware" filtering)**: only accept a teacher when its late-window MSE is simultaneously **close to the candidate-set best** AND **the oracle is no better**. Concretely, the per-group threshold becomes `late_mse ≤ group(p, late_mse)` AND `oracle_gap_mse ≤ group(p, oracle_gap_mse)` AND a new **relative-to-best-of-group** check `gap_to_group_best ≤ τ` (a fixed interpretable constant, e.g. `τ = 2 × interquartile range (IQR)`). This is the paper-grade novelty: the filter now uses the **latent world-model scoring gap** explicitly, so it is no longer a pure L2 distillation filter.
4. **Why this is *not* contribution sprawl**: Both layers reuse already-computed metrics in `validation_teacher_records.pt` and `validation_stepwise_metrics.npz`. There are **zero new model components**, **zero new forward passes**, and **zero new trajectories**. The paper claim is "different filter ⇒ different dataset ⇒ different student," which is a *data-centric* ablation story.
5. **Why no frontier primitive is required**: This is a pure data-curation/refiltering contribution. No LLM/VLM/Diffusion/RL primitive fits the bottleneck — the latent scorer (LeWM) is already trained. Adding a learned reward model or VLM-judge would inflate the paper without addressing why the downstream student fails on already-filtered data.
6. **Core technical claim**: A teacher-acceptance policy that combines (i) per-(suite, task) quantile normalization with (ii) oracle-gated accept-reject can be executed in <1 GPU-hour from existing audit outputs, and yields **higher student success on LIBERO-Plus than the current global-quantile baseline** — particularly on under-served tasks — without changing the student, the scorer, or the training budget.
7. **Required evidence**: (a) head-to-head student success on LIBERO-Plus, (b) per-task selection rate change before/after, (c) one ablation isolating the per-group normalization from the oracle-gated check.

## Method Thesis
- **One-sentence thesis**: Replace the global-quantile outlier filter with a (suite, task)-conditional + oracle-gated acceptance policy that operates on the *existing* teacher-scoring artifacts, and ship the resulting distillation set as the new FLARE-VLA training data.
- **Why this is the smallest adequate intervention**: it changes only the selection rule that converts `validation_teacher_records.pt` + `validation_stepwise_metrics.npz` into `teacher_quality_direct_records.pt`. No new training, no new model, no new horizons.
- **Why this route is timely in the foundation-model era**: the dominant current debate in VLA is **data curation at scale**, not new architectures. FLARE-VLA already has the curriculum machinery; the gap is that one research-grade curriculum *policy* fits evidence better than the global-quantile baseline.

## Contribution Focus
- **Dominant contribution**: A **suite/task-conditional + oracle-gated teacher filter** for FLARE-VLA's distillation step, evaluated end-to-end on LIBERO-Plus success rate.
- **Optional supporting contribution**: A diagnostic table reporting per-(suite, task) "selection rate vs. student gain", which doubles as a paper figure and a tool for future authors.
- **Explicit non-contributions**: not changing the student, the scorer, the candidate generator, or adding any new forward pass.

## Proposed Method
### Complexity Budget
- **Frozen / reused backbone**:
  - LeWM scoring artifacts: `validation_teacher_records.pt`, `validation_stepwise_metrics.npz`, `validation_scores.csv`, `validation_summary.json`.
  - GR00T-N1.7 LIBERO-Plus student (the four `libero_{spatial,object,goal,10}/` checkpoints).
  - Manifest: `manifests_epoch50_fixed/libero_plus_sparse_manifest.jsonl`.
  - GT latent H5: `gt_latents/libero_plus_epoch50_h8_framebank_trainpipe.h5`.
- **New trainable components**: **none**.
- **Tempting additions intentionally not used**:
  - No learned reward model / VLM-judge (`06_audit` docstring explicitly rejects per-sample confidence weights).
  - No RL fine-tuning of GR00T on filtered teachers.
  - No per-task learnable acceptance threshold (we use a robust statistic of the group, not a learned parameter, to avoid a meta-learner).
  - No new forward pass over the candidate set.

### System Overview
```
existing audit inputs (CSV/PT/NPZ)
  │
  ▼
Layer 1 — group partition by (suite, task_index)
  │       recompute per-group quantile thresholds for:
  │         teacher_gt_xyz_mse, teacher_gt_rot_mse,
  │         teacher_jerk_xyz_l2_mean, teacher_jerk_rot_l2_mean,
  │         late_mse, oracle_gap_mse
  │
  ▼
Layer 2 — oracle-gated accept rule:
  │       for each sample s in group g:
  │         relative-gap(s) = (best_mse[g] - best_mse[s]) / (best_mse[g] + eps)
  │         oracle-gap(s)    = best_mse[s] - oracle_mse[s]
  │         accept  iff  use_for_direct_augmentation_base
  │                   AND (relative-gap(s) <= τ  OR  rank_in_group(s) ≤ ρ)
  │                   AND oracle_gap_mse(s)  <= group(p, oracle_gap_mse)
  │
  ▼
selected subset  →  teacher_quality_direct_records.pt (new policy)
  │
  ▼
08_build_distillation_dataset.py (teacher_only index mode, quality=high|medium)
  │
  ▼
08c_train_gr00t_flare_44_20.sh   (identical training recipe to baseline)
  │
  ▼
LIBERO-Plus evaluation
```

### Core Mechanism
- **Inputs**:
  - Per-sample metrics already produced by `05_score_libero_plus.py`: `best_mse_score`, `oracle_mse_score`, `teacher_gt_*_mse`, `teacher_jerk_*`, `late_mse`, `terminal_mse`, `oracle_gap_mse`, group keys `suite` and `task_index`.
- **Architecture / policy**:
  - `F_filter(s) = base_quality(s, group_g) ∧ oracle_gate(s, group_g) ∧ best_group_relative(s, group_g)` where the three predicates are defined below.
  - `base_quality`: same flags as the current `use_for_direct_augmentation` (finite, agreement ok, xyz/rot-distance ≤ group p90, gripper ok, jerk ok, late-MSE ≤ group p90), but **all quantile thresholds are computed within `(suite, task_index)` instead of globally**.
  - `oracle_gate(s, g)`: `oracle_gap_mse(s) ≤ Quantile_g(oracle_gap_mse, q_oracle)` (default `q_oracle = 0.90`).
  - `best_group_relative(s, g)`: either `(best_mse_score[s] - best_of_group[g]) ≤ τ_ijk` (interpretable hard constant, default `τ_ijk = 1.5 × IQR_g(best_mse)`), or, equivalently, the rank of `s` within `g` on `best_mse_score` is in the top `ρ` share (default `ρ = 0.5`).
- **Output**: `use_for_direct_augmentation` column rewritten; everything else (teacher_actions_lerobot, confidence_weight aliases, etc.) preserved so the downstream `08_build_distillation_dataset.py` reads the new file with zero code changes.
- **Training signal / loss**: none — pure data filtering.
- **Why this is the main novelty**: The current `06_audit_teacher_quality.py` global-quantile policy explicitly *ignores* per-group structure; this proposal makes grouping a first-class structural choice and adds a *latent-score-aware* accept/reject that uses the **gap** between teacher best-MSE and the GR00T oracle action MSE, which is the relevant quantity for distillation quality.

### Optional Supporting Component
- **Diagnostic table**: a CSV/LaTeX-ready per-(suite, task) table showing (samples, baseline selection rate, new selection rate, baseline student success, new student success, Δ). This is a paper figure, not a model — it costs zero training time and is computable from existing audit outputs.

### Modern Primitive Usage
- **Which LLM / VLM / Diffusion / RL-era primitive is used**: **none of them**. The bottleneck is data curation; the latent scorer is already trained.
- **Exact role in the pipeline**: N/A.
- **Why it is more natural than an old-school alternative**: It would be *less* natural to bolt on a learned reward model or VLM judge here — the latent rollout-MSE gap is the natural quantity the paper's existing pipeline already produces. Adding a frontier primitive here would obscure, not sharpen, the contribution.

### Integration into Base Generator / Downstream Pipeline
- **What is frozen**: same as "Complexity Budget / Frozen".
- **What changes**: only `06_audit_teacher_quality.py` (and its `06b_*` commands). The downstream `08_build_distillation_dataset.py` and `08_train_gr00t_flare_44_20.sh` are **byte-identical**, only their *input* `teacher_quality_direct_records.pt` differs.
- **Inference order**: identical to current.

### Training Plan
- **No new training**: the student is trained exactly as in `08c_train_gr00t_flare_44_20.sh`.
- **Refilter cost**: <1 GPU-hour on a single RTX class GPU; the script is essentially a `pandas.groupby` + threshold computation + boolean column rewrite.

### Failure Modes and Diagnostics
- **Failure mode**: per-group thresholds may be unstable for tasks with very few samples.
  - **Detection**: enforce a minimum `n_per_group ≥ 30` (LIBERO-Plus sample counts are ≫ 1000 per task, so this is a safe bound).
  - **Fallback**: when `n_per_group < 30`, fall back to the *suite-level* threshold for that group.
- **Failure mode**: too-aggressive filtering removes legitimate rare-but-good teachers in rare tasks.
  - **Detection**: monitor `direct_aug_rate` per task; flag any task whose rate drops below 0.10 (current minimum is 0.045).
  - **Fallback**: clamp per-group quantile `q` ≤ 0.95 to preserve a floor of acceptance.
- **Failure mode**: oracle-gated step rejects globally because `oracle_mse` is small and the threshold is too tight.
  - **Detection**: monitor that ≥ 35 of 40 tasks retain at least 40% of baseline acceptance.
  - **Fallback**: introduce a `--oracle-gap-quantile` clamp (default 0.90) and emit the per-task rate to the JSON summary.

### Novelty and Elegance Argument
- **Closest work**:
  1. Current FLARE-VLA `06_audit_teacher_quality.py` itself — exact delta is replacing global quantiles with per-group quantiles and adding an oracle gate.
  2. Classical "stratified sampling / per-stratum outlier filtering" (D'Orazio, 2021 etc.) — the per-group normalization is a textbook pattern; we are deliberately *not* inventing it.
  3. FLARE (cheang2023) original teacher selection — same scoring primitive, different normalization.
- **Exact difference**: Ours is the first (to our knowledge) to combine **per-(suite, task) normalization** with an **oracle-gap gate** in a VLA distillation filter, and to evaluate it on LIBERO-Plus end-to-end.
- **Why this is a focused mechanism-level contribution rather than a module pile-up**: the data flow has two new clauses in one Python function; both clauses use already-computed metrics; no new model, no new forward pass, no new training loop.
- **Why the paper still feels focused**: a single ablation block (`policy comparison`) plus a per-task diagnostic table plus a paper-style comparison figure. Three artifacts, one mechanism.

## Claim-Driven Validation Sketch
### Claim 1 (Main): Per-(suite, task) normalization + oracle gap gate raises LIBERO-Plus success rate over the global-quantile baseline.
- **Minimal experiment**: two student fine-tuning runs identical except for the filter input → `08_train_gr00t_flare_44_20.sh` on the *original* selected set vs. the *refiltered* selected set; same seed; same 20k steps.
- **Baselines / ablations**:
  - **B-baseline**: today's global-quantile policy (already implemented; reuse `teacher_quality_direct_records.pt`).
  - **B-nosuite**: the same filters but with global quantiles (i.e. just adding the oracle-gate clause). This isolates the per-group normalization contribution.
  - **B-nooracle**: per-group quantiles but no oracle gate. Isolates the oracle-gate contribution.
  - **B-nosuite-nooracle** (sanity): identical to baseline — should yield identical selection.
- **Metric**: LIBERO-Plus overall success rate (40 tasks × 50 trials); per-task Δ; per-suite Δ (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`). Statistical test: paired Wilcoxon across 40 tasks.
- **Expected evidence**: overall Δ ≥ +3 percentage points; per-task Δ on the worst-decile (currently `direct_aug_rate < 0.30`) ≥ +5 points in at least 6 of 12 such tasks; per-suite Δ non-negative in all four suites.

### Claim 2 (Supporting): The refilter is *small* — no measurable degradation on previously-easy tasks.
- **Minimal experiment**: same as Claim 1 but inspecting the previously-easy task decile (today's `direct_aug_rate > 0.7`).
- **Baselines / ablations**: same baselines as Claim 1.
- **Metric**: per-task success on the easy decile; expected Δ ≥ −1 percentage point relative to baseline (no regression).
- **Expected evidence**: at most one task in the easy decile regresses by more than 1 point, and the median regression is in the noise band.

## Experiment Handoff Inputs
- **Must-prove claims**: (C1) per-group normalization + oracle gap gate beats global-quantile on overall success and on the worst-decile tasks; (C2) refilter does not regress on the easy tasks.
- **Must-run ablations**: B-baseline (today), B-nosuite (oracle gate only), B-nooracle (per-group only), B-nosuite-nooracle (sanity = baseline). Plus the *best* policy from the ablation, evaluated against B-baseline.
- **Critical datasets / metrics**: LIBERO-Plus overall success rate, per-task success rate, per-suite success rate, paired Wilcoxon p-value.
- **Highest-risk assumptions**: that `08c_train_gr00t_flare_44_20.sh` is *not* already saturated on LIBERO-Plus; that the `08b_train_gr00t_flare_epoch50.sh` baseline numbers from the master log are reproducible to within ±2 percentage points; that the audit artifacts (`teacher_quality_direct_records.pt`) contain all required group keys (`suite`, `task_index`) and per-sample metrics.

## Compute & Timeline Estimate
- **Estimated GPU-hours**: <1 refilter + 2 × 20k step GR00T fine-tuning ≈ 16-20 GPU-hours on a single 4090/A100 + a few CPU hours for analysis.
- **Data / annotation cost**: zero — reuses already-scored teacher tensors.
- **Timeline**: 1 day for refilter + ablation re-runs; 1 day for student fine-tunes (parallelizable on one GPU via background queue if memory allows sequential); 1 day for evaluation + paper-figure generation.
