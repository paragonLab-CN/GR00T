# Paper Plan (v3 — quantitative analysis of the teacher-selection logic; no new baselines)

**Title** (working): *What Does the Teacher-Selection Signal Actually Select? A Quantitative Anatomy of Latent-Scored Trajectory Filtering for VLA Distillation*
**One-sentence contribution**: We dissect a latent-world-model teacher-selection pipeline into a set of interpretable per-trajectory quantities (oracle gap, candidate diversity, dynamics plausibility, teacher–GT divergence), show which of them actually predict downstream generalization (and which are silently filtering the wrong thing), and distill this into a validated "beneficial-diversity vs harmful-ambiguity" score — all without changing the underlying policy or adding new baselines.
**Venue**: ICLR (MAX_PAGES = 9, main body to Conclusion end, excluding references & appendix)
**Type**: empirical / diagnostic (analysis paper; the FLARE pipeline is the *instrument*, not the contribution)
**Date**: 2026-09-25
**Section count**: 7

> ⚠️ **GATE / HONESTY NOTES (read first)**
> 1. `CLAIMS_FROM_RESULTS.md` = `verdict: REVIEW_UNAVAILABLE` → no claim adjudicated. Draft mode → **all claims `[unadjudicated]`**. Re-run `/result-to-claim` (Codex/manual-review backend) before submission.
> 2. **This framing's strength**: it needs NO new models/baselines. The core analysis reuses (a) the 489,632-sample, 138-feature teacher-quality audit (`teacher_scoring_epoch50_v2/teacher_quality_audit.parquet`), (b) the 40-task summary, and (c) per-task downstream success from the fixed-noise evals. The novel work is the **correlation/attribution analysis linking per-trajectory metrics to per-task downstream Δ**.
> 3. **The one hard dependency**: the analysis needs per-task (or per-axis) downstream success for *more than one* teacher-filtering configuration, so that "metric X → downstream Δ" has variance to explain. With only the existing single eval per recipe, some claims are correlational across 40 tasks (acceptable) but the causal "if you filter by metric M you get Δ" needs ≥1 additional filtering-configuration run (cheap: re-filter + retrain, no new baselines).
> 4. No Codex MCP this session → Step-6 cross-review not performed.

## Why this framing (and why it's stronger than the head-sweep)

- **Keeps your existing investment**: the FLARE pipeline, the LeWM scorer, and the 489k-sample audit are the *measurement instrument*. The paper's contribution is understanding, not a new model.
- **The audit is unusually rich**: 138 per-trajectory features already computed (oracle_gap_mse/cosine, candidate_{0..3} pairwise diffs & spread, teacher-vs-GT action/xyz/rot/gripper MSE, teacher-vs-GT velocity/acceleration/jerk ratios, early/mid/late/terminal MSE, quality levels, rejection reasons). Almost nobody characterizes a selection signal at this resolution.
- **Turns a weakness into the contribution**: the fact that FLARE doesn't reliably beat GT-only (and that 61% of windows have GT beating all candidates) becomes the *finding* — the signal selects trajectories that are mostly NOT better than GT, and we can quantify exactly when/why.
- **Directly answers your sub-hypothesis 3** ("quantify beneficial generalization vs harmful ambiguity") as the paper's core, with 1 & 2 reframed as measurable analyses of the *same* signal rather than new-model experiments.

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|---|---|---|---|
| C1. The selection signal is dominated by **agreement/plausibility**, not **improvement-over-GT**: 53% of trajectories pass filtering, yet GT beats all candidates in 61% of windows (oracle_mse 0.0042 vs best-candidate 0.0139) → the filter selects "self-consistent" teachers, not "better-than-demo" teachers | teacher_scoring_epoch50/validation_summary.json + _v2/teacher_quality_summary.json (verified) | `[unadjudicated]` — verified, existing data | §4.1 |
| C2. The 138 per-trajectory features decompose into a few interpretable factors (oracle gap, candidate diversity, dynamics plausibility, teacher–GT divergence); candidate diversity and oracle gap are largely **independent** axes | teacher_quality_audit.parquet (138 cols × 489k) — factor/correlation analysis | `[unadjudicated]` **needs analysis** (compute, no new training) | §4.2 |
| C3. Per-task downstream generalization correlates with only a **subset** of the selection features — candidate diversity & low oracle gap predict gains; high teacher–GT divergence & jerk predict harm — and the *current* filter weights them suboptimally | task_summary.parquet (40 tasks) × per-task eval success | `[unadjudicated]` **needs analysis** (join + correlate; needs per-task success for ≥1 filtering config) | §4.3 |
| C4. The "beneficial-diversity vs harmful-ambiguity" decomposition (§3.3) reveals the filter is **structurally misaligned**: beneficial diversity is rare (~3% at D≥P75) yet the current filter keeps **0** of it while filling 41% of its selections with redundant imitation | computed on 489,632-window epoch-50 audit (this session); four-quadrant proportions verified | `[unadjudicated]` — **measured, existing data**; the downstream *causal* claim (re-filtering by it helps) still **needs 1 re-filtered retrain + eval** | §3.3, §4.4 |
| C5. Teacher quality is ceiling-bounded by the open-loop, same-capability candidate generator (GT beats all K in 61%) — so selection can only reorder, not exceed, this ceiling | teacher_scoring_epoch50 (verified) | `[unadjudicated]` — verified, existing | §5 |

## Structure

### §0 Abstract
- **What we achieve**: a quantitative anatomy of a latent-scored teacher-selection signal — what it measures, what it actually selects, and which components predict downstream generalization.
- **Why it matters / is hard**: teacher/trajectory filtering is a load-bearing but opaque step in self-distillation and data-curation pipelines; selection signals are usually trusted without verifying they select for *improvement* rather than *self-consistency*.
- **How we do it**: instrument the full pipeline to log 138 per-trajectory quantities over ~490k candidate windows, then correlate each with per-task downstream success under a paired fixed-noise eval.
- **Evidence**: C1–C4.
- **Most remarkable result**: the signal selects "self-consistent" trajectories (53% pass) while GT beats all candidates in 61% of windows — i.e. the filter and the goal are misaligned, and a small feature subset explains most downstream variance.
- **Length**: 200–240 words. **Self-contained**: yes.

### §1 Introduction
- **Opening hook**: Data-curation and self-distillation pipelines increasingly rely on a learned scorer to decide which teacher trajectories to keep. We rarely ask: what is that scorer actually selecting for?
- **Gap / challenge**: selection signals are treated as a black box that "picks good trajectories"; it is easy to build a filter that is confident yet selects for self-consistency rather than improvement over demonstrations.
- **One-sentence contribution**: (title block).
- **Approach**: treat the FLARE latent-scored selection pipeline as a measurement instrument; log everything; attribute downstream effect to interpretable features.
- **Key questions**: (Q1) Does the filter select better-than-GT trajectories or merely self-consistent ones? (Q2) Which interpretable quantities carry the signal? (Q3) Can we predict, per trajectory, whether it will help or harm downstream — i.e. separate beneficial diversity from harmful ambiguity?
- **Contributions** (map to matrix):
  1. Evidence that a latent-scored filter selects self-consistency, not improvement-over-GT (C1).
  2. A factor decomposition of the selection signal into interpretable, largely-independent axes (C2).
  3. Attribution of per-task downstream generalization to a small feature subset, and a validated beneficial-vs-harmful score (C3, C4).
- **Results preview**: 53% pass-rate vs 61% GT-better — the misalignment in one pair of numbers.
- **Hero figure**: Fig 1.
- **Length**: 1.5 pages.
- **Key citations**: data-curation / influence / DataComp-style filtering `[VERIFY]`, self-distillation / best-of-n `[VERIFY]`, world-model scoring (LeWM) `[VERIFY]`, VLA (GR00T/OpenVLA) `[VERIFY]`.
- **Front-loading check**: skim reader gets the misalignment finding + the 3 questions before the method.

### §2 Related Work
- **Subtopics** (by family/question):
  1. Data selection / curation & influence functions for large models (what makes a sample "good").
  2. Self-distillation, best-of-n, and reward/scorer-guided data filtering.
  3. World-model / latent-model scoring of trajectories.
  4. Multimodality & ambiguity in imitation (why some "diverse" data is harmful).
- **Positioning**: we *open the box* on a specific, widely-used style of selection signal and validate it against downstream generalization, rather than proposing a new filter and reporting only its end score.
- **Length**: 1 page, 4 synthesis paragraphs.

### §3 Method / Setup
- **§3.1 The instrument**: FLARE pipeline — LeWM latent world model (epoch-50 scorer), K=4 open-loop candidates per window (horizon 8, denoise 4, independent noise), selection = agreement + outlier rejection.
- **§3.2 Instrumentation / logged quantities**: the 138 per-trajectory features, grouped a priori into 4 blocks — (i) **oracle gap** (oracle_mse/cosine, oracle_better flags, early/mid/late/terminal); (ii) **candidate diversity** (candidate_*_std, pairwise diffs, margins); (iii) **dynamics plausibility** (teacher/GT velocity, accel, jerk ratios, terminal error); (iv) **teacher–GT divergence** (action/xyz/rot/gripper MSE).
- **§3.3 The beneficial-vs-harmful score (core, C4)** — revised definition, validated on 489,632 windows.

  Three orthogonal, directly-measurable axes per window (all from the existing audit parquet):
  | Axis | Definition | Field | What it measures |
  |---|---|---|---|
  | **Q — independent quality** | `Q = best_mse_score` (lower = better) | `best_mse_score` | physical validity of the trajectory, **GT-free**: how well LeWM's predicted latent rollout matches the GT latent framebank, regardless of which path is taken |
  | **D — action-space divergence** | `D = teacher_gt_xyz_mse` | `teacher_gt_xyz_mse` | how far the teacher's end-effector path diverges from the demo in **action space** (is it a *different* path?) |
  | **G — gap vs GT** | `G = oracle_gap_mse = oracle_mse − best_mse` | `oracle_gap_mse` | how much better the GT trajectory itself is; `G ≤ 0` = teacher matches/beats GT |
  | (aux) **J — dynamics** | `teacher_gt_jerk_ratio` | jerk ratio | smoothness; an independent harm axis |

  **Why not candidate-latent-spread**: measured `corr(candidate_mse_std, teacher_gt_action_mse) = 0.106` — K candidates fanning out in *latent* space does **not** mean they offer different *action* solutions. "Multipath" is physically about action/EE divergence, so D uses action-space distance.

  **Definitions (per trajectory):**
  - **Beneficial diversity** = `D ≥ d_hi` **AND** `Q ≤ q_hi` **AND** `G ≤ g_lo` **AND** `J ≤ j_hi` — a path that *differs* from the demo, is *physically valid* in its own right, and is *not worse than* GT. This is the only part that can supply generalization signal beyond GT.
  - **Harmful ambiguity, class A** = `D ≥ d_hi` **AND** (`Q ≥ q_lo` **OR** `G > g_hi`) — diverges from GT *and* (low quality / clearly worse than GT); mode-averaging pulls the policy toward worse modes.
  - **Harmful ambiguity, class B** = `J > j_hi` — jittery/dynamically implausible, harmful regardless of Q/D.
  - **Redundant imitation** (neither) = `D < d_lo` **AND** `Q ≤ q_hi` — safe but adds no information beyond GT.
  - Thresholds: `d_hi`=P90, `d_lo`=P50 of D; `q_hi`=P50, `q_lo`=P90 of Q; `g_lo`=0; `j_hi`=P90 of J.

  **Measured four-quadrant proportions (489,632 windows, epoch-50 audit):**
  | D threshold | beneficial diversity |
  |---|---|
  | D ≥ P50 | 7.95% |
  | D ≥ P75 | **3.11%** |
  | D ≥ P90 | 0.98% |
  Relaxing G (D≥P75, high-quality, smooth): G≤0 → 3.11%, G≤P50 → 3.87%, G≤P75 → 6.33%.

  **The structural finding (this is the paper's hook):** the current filter uses `xyz_distance_outlier` (far-from-GT) as a *rejection* reason, so it is **structurally disjoint from beneficial diversity**. Of the 261,380 selected teachers: 41.0% are *redundant imitation* (near-GT + high-quality), 15,574 are harmful class-B (jitter), and **0 are beneficial diversity** — the filter keeps the safe copies and rejects 100% of the genuinely different-but-valid paths. Beneficial diversity is not just rare (~3%); it is *exactly what the current filter throws away*.
- **§3.4 Evaluation protocol**: paired fixed-diffusion-noise eval (112 tasks × 10, seed 2026); per-task and per-category success; the join between per-task selection features and per-task downstream success.
- **Length**: 2 pages.

### §4 Experiments / Main Results
- **§4.1 What the filter selects** (C1): pass-rate vs GT-better-rate; quality-level distribution (high 26.5k / medium 234.8k / low 185.5k / reject 42.7k); rejection-reason breakdown. **Existing data.**
- **§4.2 Factor structure** (C2): correlation/PCA over the 138 features on 489k samples → a few interpretable factors; show diversity ⊥ oracle gap. **Compute-only.**
- **§4.3 Feature → downstream attribution** (C3): per-task correlation between each feature block and per-task success; identify the small subset carrying the signal; show the *current* filter under-weights oracle gap. **Compute-only, needs per-task success join.**
- **§4.4 Beneficial-vs-harmful decomposition** (C4): the §3.3 quadrant proportions and the structural-misalignment table (0% of selected = beneficial; 41% redundant; 15,574 harmful class-B) — **measured, existing data**. The remaining causal step: does re-filtering to keep beneficial diversity (and drop harmful A/B) improve the downstream tradeoff? **That needs 1 re-filtered retrain + paired eval; no new baseline.**
- **§4.5 The ceiling** (C5): oracle-better-61% bounds what selection can achieve. **Existing data.**
- **Data sources**: `teacher_scoring_epoch50{,_v2}/*.parquet/json`, `outputs/eval_libero_plus/*four_model_comparison.json`, per-task eval dumps.
- **Length**: 3 pages.

### §5 Analysis / Discussion
- Why the misalignment happens: selection optimizes agreement+plausibility; improvement-over-GT requires the generator to sometimes beat the demo, which open-loop same-model sampling rarely does (C5).
- Which axes/tasks benefit vs are harmed (Robot Initial States 0.369 vs appearance buckets) and whether the score predicts it.
- Limitations (honest): single benchmark; per-task attribution is correlational across 40 tasks; C4's causal claim needs the re-filtered retrain; score is LeWM-specific; epoch-50 downstream model was interrupted mid-training (unverified benefit).

### §6 Conclusion
- Restate: a latent-scored teacher filter can be confidently misaligned — selecting self-consistency over improvement — and a small interpretable feature subset both explains and fixes the alignment.
- Limitations + future: causal validation via re-filtered retrains, cross-benchmark, closed-loop/planner teachers to lift the ceiling.

**Page budget**: 1.5+1+2+3+1+0.5 = 9.0 = MAX_PAGES ✓ (7 sections)

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Hero (2-panel) | LEFT: the selection pipeline as instrument (K candidates → LeWM score → filter) with the 4 feature blocks annotated. RIGHT: the misalignment in one plot — x = oracle_gap (GT-better → candidate-better), histogram of 489k windows, 61% shaded "GT beats all candidates", overlaid with the 53% that PASS the filter → shows the filter keeps many sub-GT trajectories. Caption: the filter selects self-consistent, not better-than-GT, teachers. | teacher_scoring_epoch50{,_v2} | HIGH |
| Fig 2 | Heatmap / PCA | factor structure of 138 features → ~4 interpretable axes; highlight diversity ⊥ oracle-gap (C2). | audit.parquet | HIGH |
| Fig 3 | Scatter + quadrant | beneficial-vs-harmful score: x=candidate diversity, y=oracle gap; quadrants labeled (beneficial=high-div+low-gap, harmful=high-div+high-gap); point color = measured per-task downstream Δ (C3/C4). | audit.parquet × per-task eval | HIGH |
| Fig 4 | Bar | per-feature-block correlation with downstream success → which blocks carry signal (C3). | task_summary × eval | MEDIUM |
| Table 1 | Filter anatomy | quality-level counts, rejection-reason counts, pass-rate (C1). | teacher_quality_summary.json | HIGH |
| Table 2 | Score validation | correlation/AUC of beneficial-vs-harmful score vs downstream Δ; (if run) re-filtered retrain vs original (C4). | to-compute | HIGH |

**Hero figure detail (Fig 1)**: a skim reader must see, in the right panel alone, that (a) most windows' best candidate is worse than GT and (b) the filter nonetheless keeps over half of them. That single juxtaposition is the paper's hook and motivates all the analysis.

## Citation Plan
- §1: GR00T N1.7 `[VERIFY]`, OpenVLA `[VERIFY]`, LeWM `[VERIFY]`, a data-curation/influence reference `[VERIFY]`, a best-of-n / self-distillation reference `[VERIFY]`.
- §2: DataComp / filtering `[VERIFY]`, influence functions `[VERIFY]`, reward-model data selection `[VERIFY]`, multimodality/mode-averaging in BC `[VERIFY]`, world-model planning `[VERIFY]`.
- §3: LeWM (instrument), GR00T (policy), LIBERO / LIBERO-Plus benchmark `[VERIFY]`.
- **Rule**: never generate BibTeX from memory; verify authors/year/venue. All `[VERIFY]` until checked.

## Reviewer Feedback
- **Not performed** — Codex MCP unavailable this session; no independent cross-model review. Run Step-6 with a working backend before submission.

## Next Steps (evidence to close BEFORE writing — priority order)
- [ ] **Re-adjudicate claims**: re-run `/result-to-claim` with a working Codex/manual-review backend (remove `[unadjudicated]`).
- [ ] **C2 (compute-only)**: correlation/PCA over the 138-feature × 489k audit parquet → factor structure. No training needed.
- [ ] **C3 (compute-only)**: join `teacher_quality_task_summary.parquet` (40 tasks) with per-task downstream success from the evals; compute per-feature-block correlations. Needs per-task success dumps (verify they exist for ≥1 config).
- [ ] **C4 (the one real experiment)**: implement the §3.3 beneficial-vs-harmful score from existing audit fields, re-filter the distillation set, retrain (same recipe, no new baseline), paired eval → validates the score causally.
- [ ] **Resolve the 54:10 discrepancy** (0.8063 vs 0.7812) and add a 2nd diffusion seed, so any downstream number cited has a variance bound.
- [ ] Then: /paper-figure → /paper-write (after claims adjudicated) → /paper-compile.
