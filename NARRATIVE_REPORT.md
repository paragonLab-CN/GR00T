# NARRATIVE REPORT — What Does the Teacher-Selection Signal Actually Select?

**Project**: Isaac-GR00T / FLARE-VLA (LIBERO-Plus future-latent teacher filtering)
**Date**: 2026-09-30
**Target venue**: ICLR (main body ≤ 9 pages, excluding references & appendix)
**Type**: empirical / diagnostic analysis paper
**Working title**: *What Does the Teacher-Selection Signal Actually Select? A Quantitative Anatomy of Latent-Scored Trajectory Filtering for VLA Distillation*

> ⚠️ **Honesty / gate notes (read first)**
> 1. All downstream success rates come from **real LIBERO simulation** (`gr00t/eval/sim/LIBERO/libero_env.py` → `check_success()`), not from the latent proxy. The LeWM latent score is used *only* as the teacher-selection instrument and is labeled as such throughout.
> 2. Claims have been **adjudicated by Codex** (`gpt-5.5`, xhigh; `gpt-6-astra` unavailable → fallback) on 2026-09-30 — see `CLAIMS_FROM_RESULTS.md` (verdict ADJUDICATED) and `findings.md`. Verdicts per claim are in §6. Integrity audit `EXPERIMENT_AUDIT.md`: integrity **pass** / design **WARN** (the "teacher < GT" design flaw is this paper's *finding*, not a defect). **Overall framing must stay narrowed**: the misalignment is *measured and supported*; that it is the *cause* of the downstream failure is *plausible but unproven* until the C4 causal ablation runs.
> 3. Numbers below are copied verbatim from the referenced JSON/parquet artifacts; each carries a `source:` pointer so the `/paper-claim-audit` step can re-verify against raw files.
> 4. **Scorer convention (decided 2026-09-30): all headline numbers are reported on the DEPLOYED filter — LeWM epoch-9 scorer (`teacher_scoring_v2`, N=487,880), which is the teacher set actually distilled into the `gt_replay_44_20` model.** The more-converged epoch-50 scorer (`teacher_scoring_epoch50_v2`, N=489,632) is reported as a **sensitivity analysis** in §5.5. Rationale: the paper analyzes the filter *as deployed*; epoch-50 shows how the diagnosis sharpens as the scorer converges.

---

## 1. One-paragraph narrative

Self-distillation and data-curation pipelines increasingly trust a learned scorer to decide which teacher trajectories to keep — yet rarely verify that the scorer selects for *improvement over demonstrations* rather than mere *self-consistency*. We treat the FLARE-VLA pipeline (a LeWM latent world model scoring K=4 open-loop GR00T candidates per window) as a **measurement instrument**, logging 138 interpretable per-trajectory quantities over 487,880 windows. On the deployed filter, **65.2%** of trajectories pass selection and teacher self-agreement is high (0.94), yet the ground-truth demonstration still beats all four candidates in **44.4%** of windows and the admitted candidates are only marginally sub-GT on average (oracle latent MSE 0.01127 vs 0.01155, ~1.03×) — i.e. the filter is best understood as a **conservative self-consistency selector** that mostly recycles near-GT, redundant copies while adding little demonstrated new signal. Decomposing the signal into approximately-orthogonal axes (independent quality Q, action-space divergence D, gap-vs-GT G, dynamics J) localizes the misalignment precisely: **beneficial diversity** — paths that differ from the demo yet are physically valid and no worse than GT — is rare (5.5% at D≥P75), the filter *does* preserve about half of it at moderate divergence, but it **fails completely at the high-divergence tail** (0 of 9,173 valid paths kept at D≥P90) while over-admitting **35.4%** redundant imitation and 6.8% harmful jitter. Downstream, every FLARE variant trails plain GT-only behavioral cloning (0.7964; the main recipe significantly, −4.2pt). A sensitivity analysis with a more-converged (epoch-50) scorer sharpens the same diagnosis (GT beats all candidates in 61% of windows, a 3.3× MSE gap) — the failure mode is not an artifact of an undertrained scorer.

---

## 2. The instrument (what is being measured)

FLARE-VLA pipeline (LIBERO-Plus variant), all under `FLARE-VLA/`:

1. **LeWM latent world model** trained 50 epochs on LIBERO-90 (`commands/libero_plus/02_train_lewm.sh`). The **epoch-50** checkpoint is the scoring instrument (justified in `EXPERIMENT_AUDIT.md` §appendix: epoch-50 separates GT from candidates far more cleanly than epoch-9; epoch-100 not recommended due to diminishing returns and long-horizon instability).
2. **GT latent framebank**: real demonstration futures encoded to latents, horizon 8 (`03_extract_gt_latent.sh`) — this is *real* data, not model output.
3. **Candidate generation**: GR00T-N1.7-LIBERO (the *same* model that becomes the student) generates **K=4** open-loop candidate action trajectories per window, `denoising_steps=4`, `action_horizon=8`, independent noise (`05_generate_candidates.sh` / `scripts/flare/generate_k_candidates_openloop.py`). No closed-loop environment correction.
4. **Scoring & selection** (`06_score_candidates.sh`, `06b_audit_teacher_quality.sh`): LeWM rolls each candidate's future latent and scores alignment against the GT latent framebank; the best-scoring candidate becomes the "teacher". Selection = teacher-agreement requirement + outlier rejection across several thresholds.
5. **Distillation / training** (`08_train_gr00t_flare.sh` family): selected teachers are mixed with GT demonstrations and the GR00T policy is fine-tuned.

**Key asymmetry to surface in the paper**: the LeWM scorer can only *rank* the K candidates — it cannot improve them. When all K candidates are worse than GT (44% of windows under the deployed epoch-9 scorer; 61% under the converged epoch-50 scorer), the picked teacher is still sub-GT. This is the ceiling that bounds the entire selection approach.

---

## 3. Logged quantities (the 138-feature audit)

Source: `FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_v2/teacher_quality_audit.parquet` (489,632 rows × 138 cols), summarized in `teacher_quality_summary.json`. The features decompose a priori into four interpretable blocks:

| Block | Example fields | What it measures |
|---|---|---|
| **(i) Oracle gap** | `oracle_mse`, `oracle_cosine`, `oracle_better` flags, early/mid/late/terminal MSE | How much better the GT trajectory itself is |
| **(ii) Candidate diversity** | `candidate_*_std`, pairwise diffs, margins | How spread the K candidates are in latent space |
| **(iii) Dynamics plausibility** | teacher/GT velocity / acceleration / jerk ratios, terminal error | Physical smoothness / plausibility |
| **(iv) Teacher–GT divergence** | teacher_gt action / xyz / rot / gripper MSE | How far the teacher path diverges from the demo in action space |

> **Dataset correspondence (important).** Two audit suites exist, both over the same K=4 open-loop candidate pool:
> - **`teacher_scoring_v2` — LeWM epoch-9 scorer, N=487,880, selected 65.2% (317,941). This is the DEPLOYED filter: the teacher set actually distilled into the `gt_replay_44_20` downstream model. All headline numbers (C1–C4) are reported on this suite.**
> - `teacher_scoring_epoch50_v2` — LeWM epoch-50 scorer, N=489,632, selected 53.4%. Reported as a **sensitivity analysis** (§5.5): the more-converged scorer sharpens the same diagnosis.
>
> C2 (factor) and C3 (downstream attribution) run on `teacher_scoring_v2` × the 44_20 checkpoint-10000 fixed-noise eval.

### 3.1 The beneficial-vs-harmful score (core analytic object)

Three orthogonal, directly-measurable axes per window (all from the existing audit parquet), plus one auxiliary:

| Axis | Definition | Field | Interpretation |
|---|---|---|---|
| **Q** — independent quality | `best_mse_score` (lower better) | `best_mse_score` | Physical validity of the candidate, **GT-free**: how well LeWM's predicted latent rollout matches the GT latent framebank regardless of path taken |
| **D** — action-space divergence | `teacher_gt_xyz_mse` | `teacher_gt_xyz_mse` | How far the teacher's end-effector path diverges from the demo (is it a *different* path?) |
| **G** — gap vs GT | `oracle_gap_mse = oracle_mse − best_mse` | `oracle_gap_mse` | How much better GT itself is; `G ≤ 0` ⇒ teacher matches/beats GT |
| **J** — dynamics (aux) | `teacher_gt_jerk_ratio` | jerk ratio | Smoothness; an independent harm axis |

**Design note (why action-space D, not candidate-latent-spread):** measured `corr(candidate_mse_std, teacher_gt_action_mse) = 0.106` — K candidates fanning out in *latent* space does **not** mean they offer different *action* solutions. "Multipath" is physically about EE/action divergence, so D uses action-space distance.

**Per-trajectory definitions** (thresholds: `d_hi`=P90, `d_lo`=P50 of D; `q_hi`=P50, `q_lo`=P90 of Q; `g_lo`=0; `j_hi`=P90 of J):

- **Beneficial diversity** = `D ≥ d_hi` ∧ `Q ≤ q_hi` ∧ `G ≤ g_lo` ∧ `J ≤ j_hi` — differs from demo, physically valid in its own right, and not worse than GT. The only component that can supply generalization signal beyond GT.
- **Harmful ambiguity, class A** = `D ≥ d_hi` ∧ (`Q ≥ q_lo` ∨ `G > g_hi`) — diverges from GT *and* (low quality / clearly worse); mode-averaging pulls the policy toward worse modes.
- **Harmful ambiguity, class B** = `J > j_hi` — jittery / dynamically implausible, harmful regardless of Q/D.
- **Redundant imitation** = `D < d_lo` ∧ `Q ≤ q_hi` — safe but adds no information beyond GT.

**Measured four-quadrant proportions (489,632 windows, epoch-50 audit):**

| D threshold | beneficial diversity |
|---|---|
| D ≥ P50 | 7.95% |
| D ≥ P75 | **3.11%** |
| D ≥ P90 | 0.98% |

Relaxing G (D≥P75, high-quality, smooth): G≤0 → 3.11%, G≤P50 → 3.87%, G≤P75 → 6.33%.

---

## 4. Core findings (claims → evidence)

### C1 — The filter selects self-consistency, not improvement-over-GT  *(deployed filter, epoch-9)*
- **65.2%** of trajectories pass filtering (`use_for_direct_augmentation` rate = 0.6517, count 317,941 / 487,880).
- Yet GT beats **all four** candidates in **44.4%** of windows (both_rate = 0.4439; mse_rate 0.4607, cosine_rate 0.4596) — recomputed from the audit parquet.
- GT latent MSE mean **0.01127** vs best-candidate **0.01155** → under epoch-9 the candidates the filter admits are only **marginally** sub-GT (1.03×), i.e. the filter is largely admitting **redundant near-GT copies**, not better paths.
- Teacher agreement is high (`teacher_agreement` rate = 0.9435) but agreement measures self-consistency, *not* superiority over GT.
- Teacher-quality distribution of selected: high 5,478 / medium 312,463 / low 142,355 / reject 27,584. Rejection-reason counts: `xyz_distance_outlier` 48,788; `rot_distance_outlier` 48,788; `large_oracle_gap` 48,788; `late_mse_outlier` 48,788; `gripper_mismatch_outlier` 9,275; `teacher_jerk_outlier` 31,171; `cosine_mse_teacher_disagree` 27,584.
- **Source**: `teacher_scoring_v2/teacher_quality_audit.parquet` (recomputed), `teacher_scoring_v2/teacher_quality_summary.json`. **Status**: **yes, narrowed** (med-high, Codex 2026-09-30) — the honest framing is "selects self-consistent, near-demonstration teachers with little *new* information beyond GT", **not** "selects sub-GT teachers" (GT beats all candidates in only 44% of windows and the MSE gap is a near-tie 1.03× under epoch-9; the stronger "GT crushes candidates" claim is epoch-50 sensitivity only, §5.5).

### C2 — The 138 features decompose into a few largely-independent interpretable factors
- **Ran 2026-09-30** on `teacher_scoring_v2` (N=487,880, 0 dropped non-finite) via Spearman correlation + SVD-based PCA (no sklearn). Script: `FLARE-VLA/scripts/libero_plus/10_c2_c3_analysis.py`; outputs `c2_feature_spearman.csv`, `c2_pca_loadings.csv`, `c2_c3_analysis.json`.
- **Diversity ⊥ oracle-gap confirmed**: `corr(candidate_mse_std, oracle_gap_mse) = 0.056` (≈ 0, far below the 0.3 independence threshold); `corr(candidate_mse_std, teacher_gt_action_mse) = 0.013`. Latent spread is essentially uncoupled from both the oracle gap and action-space divergence.
- **Compact factor structure**: top-4 principal components explain **70.1%** of variance (PC1 28.7% / PC2 20.4% / PC3 12.1% / PC4 8.9%; top-8 → 89.4%). The PCs separate cleanly onto the a-priori blocks:
  - **PC1 (28.7%)** — oracle/latent-quality axis: loads on candidate_mse_mean, candidate_cosine_mean, late/terminal/mid MSE.
  - **PC2 (20.4%)** — teacher–GT divergence + dynamics: gripper match/mse, action_mse, acceleration, jerk.
  - **PC3 (12.1%)** — candidate diversity / margins: cosine_margin, mse_margin, relative_mse_margin, candidate_*_std.
  - **PC4 (8.9%)** — action-space divergence + oracle gap: teacher_gt_xyz_mse, action_mse, oracle_gap_mse/cosine.
- **Interpretation**: the 138 logged features reduce to ~4 largely-orthogonal, interpretable axes — exactly the Q / D / G / J decomposition in §3.1. This validates using those axes (rather than the raw 138) for the beneficial-vs-harmful score.
- **Status**: **yes** (factor structure + diversity⊥oracle-gap now measured). Replaces prior `partial`.

### C3 — Per-task downstream generalization vs selection features: **no monotonic signal found**
- **Ran 2026-09-30**: joined `teacher_scoring_v2` 40-task summary to per-task downstream success of the `gt_replay_44_20` checkpoint-10000 fixed-noise eval, via per-suite content-word containment + fuzzy matching. 107/112 eval tasks join confidently (conf≥0.6); **35 train tasks** have ≥1 downstream measurement (mean 3.06 eval tasks per train task, range 1–8).
- **Result — a genuine negative**: across the 35 train tasks, **no feature block correlates significantly with downstream success** (Spearman, all p > 0.39):
  | block | ρ | p |
  |---|---|---|
  | candidate_diversity | +0.148 | 0.397 |
  | dynamics | +0.135 | 0.439 |
  | teacher_gt_divergence | −0.125 | 0.476 |
  | selection_rate | +0.080 | 0.648 |
  | **oracle_gap** | **−0.029** | **0.867** |
- Strongest single feature is `mean_teacher_jerk` (ρ=+0.270, p=0.117) — still non-significant.
- **Robustness**: not a join artifact — restricting to exact-match tasks (n=28) *raises* oracle_gap to ρ=+0.247 (p=0.206) but with the **opposite sign** to the "low oracle-gap predicts gains" hypothesis; downstream success has real variance (0.45–1.0). So the absence of signal is not floor/ceiling collapse.
- **Interpretation**: at the 40-task granularity, per-task selection statistics do **not** linearly predict per-task downstream success. The hypothesized "small feature subset carries the downstream signal" is **not supported** at this resolution — the attribution is either non-monotonic, below the noise floor at n=35, or operates at finer (per-window) granularity than task means. This weakens the original C3 and is itself a finding: the per-task selection features are weak proxies for downstream effect.
- **Status**: **no** (hypothesis not supported; measured, not merely unrun). Replaces prior `no (untested)` — now an *evidenced* negative.

### C4 — The beneficial-vs-harmful decomposition reveals structural misalignment (the paper's hook)  *(deployed filter, epoch-9)*
Recomputed on `teacher_scoring_v2` (N=487,880 valid) with the §3.1 axes (Q=`best_mse_score`, D=`teacher_gt_xyz_mse`, G=`oracle_gap_mse`, J=`teacher_gt_jerk_ratio`; thresholds d50/d75/d90, q50, j90 from the epoch-9 distributions):
- **Beneficial diversity is rare**: D≥P50 → 12.28% (59,897); D≥P75 → **5.51%** (26,903); D≥P90 → 1.88% (9,173).
- **The filter discards the most-divergent valid paths entirely**: at the strictest divergence threshold (D≥P90) the filter keeps **0** of the 9,173 beneficial-diverse windows. At D≥P75 it keeps 43.3% (11,651 / 26,903); at D≥P50, 64.5%.
- **Redundant imitation still dominates selections**: of the **317,941 selected teachers**, **35.4% are redundant imitation** (near-GT + high-quality) and **6.8% are harmful class-B** (jitter, n=21,617).
- The filter's `xyz_distance_outlier` (far-from-GT) rejection is the structural cause: it is the reason the most-divergent-but-valid paths (the only ones that could add signal beyond GT) are systematically removed.
- **Status**: **partial** (medium / med-low, Codex 2026-09-30) — descriptive misalignment **measured on the deployed filter**. **Narrowed claim**: the filter preserves some moderate beneficial diversity (43–65% at D≤P75) but **fails at the high-divergence tail** (0 of 9,173 at D≥P90) and **over-admits redundant near-GT copies** (35.4%). It is *not* defensible as "the filter broadly discards beneficial diversity". The *causal* claim (re-filtering improves the downstream tradeoff) still needs **1 re-filtered retrain + paired eval**. Sections §3.3, §4.4.

### C5 — Teacher quality is ceiling-bounded by the open-loop, same-capability generator
- Because candidates are open-loop samples from the *same* GR00T model (denoising 4, horizon 8), and GT beats all K in 44% (epoch-9) / 61% (epoch-50) of windows, selection can only *reorder* candidates, never exceed the GT ceiling.
- **Source**: `teacher_scoring_v2/teacher_quality_audit.parquet` (deployed) + `generate_k_candidates_openloop.py`. **Status**: **yes/partial** (med-high) — "selection can only pick among the generated candidates, never recover quality absent from that pool". Section §5.

---

## 5. Downstream results (real LIBERO simulation)

All from **paired, fixed-diffusion-noise** evals: **112 tasks × 10 episodes = 1120 episodes/model**, `environment_seed=2026`, explicit per-request diffusion-noise seeds (`master_seed + task_index*1e6 + episode_index*1e4 + action_chunk_index`). Paired per-task and per-episode (McNemar) statistics available.

### 5.1 Four-model comparison (checkpoint-10000, fixed noise seed 2026)
Source: `outputs/eval_libero_plus/gr00t_libero_plus_flare_suite_corrected_gt_replay_54_10_checkpoint_10000_fixed_noise_seed2026/four_model_comparison.json`

| Model | successes/episodes | success_rate | vs gt_only |
|---|---|---|---|
| **gt_replay_54_10** (84% GT) | 903/1120 | **0.8063** | +0.98pt |
| **gt_only** (= GR00T baseline) | 892/1120 | **0.7964** | — |
| image_aug | 858/1120 | 0.7661 | −3.04pt |
| flare (epoch50, mix 0.12) | 845/1120 | 0.7545 | −4.20pt |

Paired McNemar (episode-level): `gt_replay_54_10 − gt_only` net +11, p=0.422 (not significant); `gt_only − flare` net +47, p=3.4e-4 (flare significantly worse); `gt_replay_54_10 − flare` net +58, p=2.5e-5.

> ⚠️ **The "0.8063 vs 0.7812" discrepancy — RESOLVED (diagnosed 2026-09-30, not noise).** These two numbers are **two different training runs**, not the same model re-evaluated:
>
> | | **0.8063** (cite this one) | **0.7812** (stale) |
> |---|---|---|
> | training run | `..._suite_corrected_gt_replay_54_10` | `..._image_aug_gt_replay_54_10_rerun_20260911` |
> | **teacher data** | `distillation_dataset_v2` ← `teacher_scoring_v2` (epoch-9) | `distillation_dataset` ← `teacher_scoring` (epoch-9, **pre-v2 selection**) |
> | learning_rate | 1e-4 | 1e-5 |
> | mix_ratio (GT/teacher) | 0.84375 / 0.15625 | 0.85 / 0.15 |
> | eval dir | `...suite_corrected_gt_replay_54_10...` | `...image_aug_gt_replay_54_10_rerun_20260911...` |
>
> Both evals are checkpoint-10000, fixed-noise seed 2026, 112 tasks × 10. The 2.5pt gap reflects the different (corrected) teacher set + 10× higher LR in the canonical run. **Rule: cite only the `suite_corrected` 0.8063; the `rerun_20260911` 0.7812 is a stale/older variant.** The earlier audit's worry ("the 0.8063 is stale, use 0.7812") had it backwards.

### 5.2 GT:teacher replay-ratio sweep (the one active experiment axis)
`gt_replay_44_20` (68.75% GT, teacher_mix_ratio 0.3125) vs `gt_replay_54_10` baseline.
Source: `..._gt_replay_44_20_checkpoint_10000_fixed_noise_seed2026/gt_replay_44_20_vs_54_10_comparison.json`

- 44_20 overall **0.8125** (910/1120) vs 54_10 **0.8063** (903/1120) → **+0.62pt**, paired-task outcome 20 better / 21 worse / 71 tied (essentially a wash, not significant).
- 44_20 per-category: Robot Initial States 0.300, appearance 0.9271.
- **Interpretation**: raising teacher share from 10→20 per batch did not help beyond noise; consistent with the ceiling diagnosis (more sub-GT teacher ≈ more noise).

### 5.3 Variance estimate (random diffusion noise, gt_replay_44_20)
Two independent random-diffusion-noise runs (no fixed seed):
- run1: **0.8214** (920/1120)
- run2: **0.8179** (916/1120)
- Fixed-seed run: 0.8125. Spread ≈ ±0.5pt across seeds → **any single-number downstream claim carries ≈ ±0.5–1pt noise**; treat sub-1pt differences as inconclusive.

### 5.4 Where the policy is weakest (target for any real improvement)
From the four-model breakdown, **Robot Initial States** is the hardest category across all models (gt_only 0.350, flare 0.369, image_aug 0.388, gt_replay_54_10 0.269, gt_replay_44_20 0.300). Difficulty-5 tasks: gt_only 0.61 vs flare 0.525. This is where a *correctly aligned* teacher signal would have the most headroom.

### 5.5 Sensitivity analysis — scorer convergence (epoch-9 deployed vs epoch-50)
The headline C1/C4 numbers above are from the **deployed epoch-9** scorer. Re-running the same diagnosis on the more-converged **epoch-50** scorer (`teacher_scoring_epoch50_v2`, N=489,632) shows the misalignment **sharpens**, not disappears — so the finding is not an artifact of an undertrained scorer:

| Metric | epoch-9 (deployed) | epoch-50 (sensitivity) |
|---|---|---|
| Selected rate | 65.2% | 53.4% |
| GT beats all 4 candidates (both) | 44.4% | **61.0%** |
| GT vs best-candidate latent MSE | 0.01127 vs 0.01155 (**1.03×**) | 0.00415 vs 0.01386 (**3.3×**) |
| Beneficial diversity kept at D≥P90 | 0% | 0% |

Interpretation: as the scorer converges it separates GT from the open-loop candidates *more* cleanly, making the ceiling (C5) and the discarded-diversity (C4) starker. The qualitative conclusion — the filter selects self-consistent/redundant teachers and cannot exceed the candidate pool — is invariant to scorer epoch; only the magnitude grows.
**Source**: `teacher_scoring_epoch50/validation_summary.json`, `teacher_scoring_epoch50_v2/teacher_quality_summary.json`.

---

## 6. Claims table (for /result-to-claim and /paper-claim-audit)

| ID | Claim | Evidence | Status (adjudicated 2026-09-30, Codex gpt-5.5) |
|---|---|---|---|
| C1 | Filter selects self-consistent near-demo teachers with little new info beyond GT (65.2% pass, 44.4% GT-better, GT≈1.03×, agreement 0.9435) | teacher_scoring_v2 audit (deployed, epoch-9); epoch-50 sensitivity 61%/3.3× in §5.5 | **yes (narrowed)** (med-high) — not "selects sub-GT teachers" |
| C2 | 138 features → few interpretable ~orthogonal axes; diversity ⊥ oracle-gap | teacher_scoring_v2 PCA + Spearman (this session) | **yes** — top-4 PCs = 70.1% var, cleanly blocked; corr(div, oracle-gap)=0.056 |
| C3 | Per-task downstream Δ explained by a small feature subset; current filter under-weights oracle gap | task_summary × 44_20 per-task eval (this session) | **no (evidenced negative)** — no block significant (all p>0.39, n=35); oracle_gap ρ=−0.03 |
| C4 | Beneficial-vs-harmful decomposition → filter is structurally misaligned (D≥P90: 0% of beneficial kept; selected = 35.4% redundant + 6.8% harmful-jitter) | measured on deployed teacher_scoring_v2 (N=487,880) | **partial** (medium) — descriptive misalignment solid; causal leg untested |
| C5 | Teacher ceiling-bounded by open-loop same-capability generator (44% GT-better deployed; 61% epoch-50) | teacher_scoring_v2 + generator code | **yes/partial** (med-high) — selector can only pick among generated candidates |
| D1 | FLARE main recipe (mix 0.12) underperforms GT-only by −4.2pt (significant) | four_model_comparison.json | **yes** (high) — McNemar p=3.4e-4 |
| D2 | gt_replay_54_10 (+0.98pt) and 44_20 (+0.62pt over 54_10) beat gt_only | four_model + 44_20 comparison + 2 random-seed runs | **partial** (medium) — 54_10 vs gt_only p=0.422 (NS), 44_20 vs 54_10 a wash, ±0.5pt seed noise. The 0.8063-vs-0.7812 discrepancy is **RESOLVED** (§5.1): two different runs — cite only the canonical `suite_corrected` 0.8063. |

> **Adjudication source**: `CLAIMS_FROM_RESULTS.md` (verdict ADJUDICATED, integrity pass / design WARN). Overall: paper writable only under **narrowed framing** — "we diagnose a structural mismatch consistent with the observed underperformance", **not** "we proved the causal mechanism". **Update 2026-09-30**: C2 and C3 are now *computed* (not pending) — C2 supports the factor story; **C3 is an evidenced negative** (per-task selection features do not linearly predict per-task downstream success at n=35). The only remaining gate is **C4-causal** (the re-filtered retrain). C3's negative result also means Fig 4 / Table 2 must report "no per-task signal" rather than imply a predictive subset.

---

## 7. Figure plan

| ID | Type | Description | Data source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Hero (2-panel) | LEFT: pipeline-as-instrument (K candidates → LeWM score → filter) with the 4 feature blocks annotated. RIGHT: the misalignment in one plot — histogram of oracle_gap over 489k windows, 61% shaded "GT beats all candidates", overlaid with the 53% that PASS → shows the filter keeps many sub-GT trajectories. | teacher_scoring_epoch50{,_v2} | HIGH |
| Fig 2 | Heatmap / PCA | Factor structure of 138 features → ~4 interpretable axes; highlight diversity ⊥ oracle-gap (C2). | audit.parquet | HIGH |
| Fig 3 | Scatter + quadrant | Beneficial-vs-harmful score: x = D (action divergence), y = G (oracle gap); quadrants labeled; point color = per-task downstream Δ (C3/C4). | audit.parquet × per-task eval | HIGH |
| Fig 4 | Bar | Per-feature-block correlation with downstream success → which blocks carry signal (C3). | task_summary × eval | MEDIUM |
| Table 1 | Filter anatomy | Quality-level counts, rejection-reason counts, pass-rate (C1). | teacher_quality_summary.json | HIGH |
| Table 2 | Score validation | Correlation/AUC of beneficial-vs-harmful score vs downstream Δ; (if run) re-filtered retrain vs original (C4). | to-compute | HIGH |

**Hero requirement (Fig 1 right panel)**: a skim reader must see, in one panel, that (a) most windows' best candidate is worse than GT and (b) the filter nonetheless keeps over half of them. That juxtaposition is the paper's hook.

---

## 8. Open items / evidence to close BEFORE writing (priority order)

1. ~~**Re-adjudicate claims**~~ — **DONE 2026-09-30**: `/result-to-claim` run, Codex adjudication recorded in `CLAIMS_FROM_RESULTS.md`; `[unadjudicated]` tags replaced by per-claim verdicts.
2. ~~**C2 (compute-only)**~~ — **DONE 2026-09-30**: PCA + Spearman over `teacher_scoring_v2` (487,880 × selected features). Top-4 PCs = 70.1% variance, cleanly blocked onto oracle-gap / divergence+dynamics / diversity / action-divergence; diversity ⊥ oracle-gap (corr 0.056). Verdict → **yes**. Outputs: `c2_c3_analysis.json`, `c2_feature_spearman.csv`, `c2_pca_loadings.csv`.
3. ~~**C3 (compute-only)**~~ — **DONE 2026-09-30**: joined 40-task selection summary to 44_20 per-task downstream success (107/112 conf≥0.6 → 35 train tasks). **Evidenced negative**: no feature block significant (all p>0.39; oracle_gap ρ=−0.03). Verdict → **no**. Outputs: `c3_eval_to_train_join.csv`, `c3_train_task_features_with_downstream.csv`.
4. **C4 causal leg (the one real experiment)** — implement the §3.1 beneficial-vs-harmful score, re-filter the distillation set, retrain (same recipe, no new baseline), paired eval.
5. ~~**Resolve the 0.8063 vs 0.7812 gt_replay_54_10 discrepancy**~~ — **RESOLVED 2026-09-30** (§5.1): two distinct training runs (different teacher set `distillation_dataset_v2` vs `distillation_dataset`, LR 1e-4 vs 1e-5), not noise. Cite only the canonical `suite_corrected` **0.8063**. The random-seed variance bound (§5.3) still applies to any single downstream number.
6. Then `/paper-figure` → `/paper-write` → `/paper-compile`.

---

## 9. Limitations (to state honestly in the paper)

- Single benchmark (LIBERO-Plus); per-task attribution is correlational across ≤35 joined train tasks — and at that granularity **no monotonic feature→downstream signal was found** (C3 negative).
- Headline C1/C4 numbers are from the **deployed epoch-9** scorer; the epoch-50 sensitivity (§5.5) shows the same diagnosis sharpened. The absolute "how much better is GT" magnitude is scorer-dependent (1.03× at epoch-9 → 3.3× at epoch-50), so we report the deployed-scorer value as primary and do not over-claim the larger epoch-50 gap.
- C4's causal claim is not yet validated (needs the re-filtered retrain).
- The beneficial-vs-harmful score is LeWM-specific; generalization to other scorers is untested.
- The epoch-50 downstream GR00T model was interrupted at 5031/10000 steps (no eval) — "epoch-50 teacher → higher downstream success" is **unverified**.
- Downstream differences under ~1pt are within the observed seed noise.

---

## 10. Source-file index (for claim audit)

- `FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50/validation_summary.json` — oracle-better rates, agreement, MSE distributions (N=489,632)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_v2/teacher_quality_summary.json` — selection rate, quality levels, rejection reasons, thresholds
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_v2/teacher_quality_audit.parquet` — 489,632 × 138 per-trajectory features (epoch-50 scorer; headline C1/C4)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/teacher_quality_audit.parquet` — 487,880 × 140 features (epoch-9 scorer; **the set distilled into gt_replay_44_20**; basis for C2/C3)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/c2_c3_analysis.json` — C2/C3 results (this session)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/c2_feature_spearman.csv`, `c2_pca_loadings.csv` — C2 correlation matrix + PCA loadings
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/c3_eval_to_train_join.csv`, `c3_train_task_features_with_downstream.csv` — C3 join + per-task features
- `FLARE-VLA/scripts/libero_plus/10_c2_c3_analysis.py` — the C2/C3 analysis script
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_v2/teacher_quality_task_summary.parquet` — 40-task summary (epoch-50)
- `outputs/eval_libero_plus/..._gt_replay_54_10_.../four_model_comparison.json` — 4-model downstream (0.8063/0.7964/0.7661/0.7545)
- `outputs/eval_libero_plus/..._gt_replay_44_20_.../gt_replay_44_20_vs_54_10_comparison.json` — 44_20 vs 54_10
- `outputs/eval_libero_plus/..._44_20_..._random_diffusion_noise_run{1,2}/summary_all.json` — variance (0.8214, 0.8179)
- `gr00t/eval/sim/LIBERO/libero_env.py` — real-success provenance (`check_success`)
- `EXPERIMENT_AUDIT.md` / `EXPERIMENT_AUDIT.json` — integrity audit (verdict WARN; integrity PASS)
- `PAPER_PLAN_20260927_v3.1_revised_divergence_metric.md` — full outline this report condenses
