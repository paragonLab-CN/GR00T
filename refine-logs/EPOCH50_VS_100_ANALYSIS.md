# Epoch 50 vs Epoch 100+: FLARE-VLA Teacher Audit Quality Analysis

**Problem**: Whether to continue training LeWM from epoch 50 to epoch 100 (or beyond) to improve the quality of teacher trajectories used in FLARE-VLA distillation.
**Date**: 2026-10-12.
**Status**: Analysis report (not a research-refine re-run).
**Author basis**: existing `FINAL_PROPOSAL.md` (Round-4 READY) method thesis is preserved; this report focuses on whether the upstream `LeWM` teacher scoring input should be refreshed.

---

## 1. FLARE-VLA Pipeline End-to-End (What Epoch Actually Affects)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE A: LeWM training                                                  │
│    02_train_lewm.sh → weights_epoch_{1..50}.pt                           │
│    Affected by: epoch count, batch size, dataset (libero_plus_lerobot__)  │
│    Output: world-model + latent predictor (the "teacher latent")         │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE B: GT latent extraction                                           │
│    03_extract_gt_latent_libero_plus.py                                   │
│    Reads weights_epoch_N.pt, encodes all GT demo windows to latent bank   │
│    Output: gt_latents/libero_plus_epoch{N}_h8_framebank_trainpipe.h5     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE C: Candidate scoring (audit inputs)                               │
│    05_score_libero_plus.py                                               │
│    Reads candidate teacher actions + GT latent bank, scores each window  │
│    by best_mse, oracle_mse, oracle_gap, teacher_agreement, late_mse, ... │
│    Output: validation_scores.csv, validation_teacher_records.pt,         │
│            validation_stepwise_metrics.npz, validation_summary.json      │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE D: Teacher audit / filter                                         │
│    06_audit_teacher_quality.py  (Pol_base — current)                     │
│    06c_refilter_teacher_quality_v3.py (Pol_full / Pol_z — proposed)      │
│    Output: teacher_quality_direct_records.pt, teacher_quality_summary.json│
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE E: Distillation dataset build                                     │
│    08_build_distillation_dataset.py (byte-identical for both policies)   │
│    Output: distillation_dataset_epoch50_fixed/                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE F: GR00T-N1.7-3B student training                                 │
│    08c_train_gr00t_flare_44_20.sh (44:20 GT:teacher ratio, 20k steps)    │
│    Input: distillation index from Stage E                                │
│    Output: GR00T student checkpoint                                      │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE G: Evaluation                                                     │
│    LIBERO-Plus evaluator, 40 tasks × 50 rollouts × 3 seeds                │
│    Output: per-task success rate → the paper's headline metric            │
└──────────────────────────────────────────────────────────────────────────┘
```

**Where epoch matters**: only Stages B–C read `weights_epoch_N.pt`. The audit (Stage D) reads Stage C outputs. Stages E–G are *byte-identical* between Pol_base and Pol_full — they consume the same distillation index. So the question reduces to: **does a later-epoch LeWM produce a cleaner teacher scoring signal (Stage C), and therefore better-selected teacher trajectories in Stage D?**

---

## 2. Current State at Epoch 50 (Quantitative Audit)

### 2.1 LeWM Training Curve (validation_loss)

| epoch | val_loss | val_pred | val_sigreg | Notes |
|-------|----------|----------|------------|-------|
| 1     | 2.284    | 0.08122  | 24.500     | initial |
| 2     | 0.136    | 0.00475  | 1.461      | converged |
| 5     | 0.117    | 0.00251  | 1.274      | min so far |
| 10    | 0.115    | 0.00127  | 1.269      | best epoch |
| 20    | 0.128    | 0.00073  | 1.376      | regressing |
| 30    | 0.136    | 0.00058  | 1.509      | + sigreg drift |
| 40    | 0.144    | 0.00041  | 1.596      | |
| **50**| **0.146**| **0.00038** | **1.616** | **final** |

Loss decomposition (verified): `val_loss ≈ val_pred + 0.09 × val_sigreg`.

**Key reading**: `val_pred` is the predictor accuracy we actually care about for distillation. It **kept improving** through epoch 50 (0.00127 → 0.00038 = 3.3× reduction epoch 10→50, **70% reduction** epoch 10→50). `val_sigreg` is the sigma regularizer that is growing — this is what makes `val_loss` *look* like it's getting worse, but it is a regularization term, not prediction quality. The total loss is misleading at face value; the predictor is still improving.

### 2.2 Teacher Scoring Quality at Epoch 50

| metric (subset 5000 windows)        | epoch 20 | epoch 30 | epoch 50 |
|-------------------------------------|----------|----------|----------|
| teacher_agreement_rate              | 0.9174   | 0.9030   | 0.9136   |
| oracle_better_mse_rate              | 0.6182   | 0.6728   | 0.6452   |
| oracle_mse_mean                     | 0.006312 | 0.004926 | **0.004168** |
| best_mse_mean                       | 0.016223 | 0.015235 | **0.014514** |
| oracle_gap_mse_mean                 | 0.009911 | 0.010309 | 0.010346 |
| teacher_mse_step_slope              | 0.005248 | 0.004940 | 0.004843 |
| oracle_mse_step_slope               | 0.001950 | 0.001518 | 0.001373 |

**Key reading**: from epoch 30 → 50, `oracle_mse_mean` dropped **15%** (0.004926 → 0.004168) and `oracle_mse_step_slope` dropped **9.6%** (0.001518 → 0.001373). The predictor is **still improving**. The flattening of `oracle_better_mse_rate` (epoch 30: 0.6728 → epoch 50: 0.6452) is expected — once oracle gets better, fewer candidates beat it, so the rate goes down even though absolute quality is up.

### 2.3 Per-Task Direct-Aug Rate Distribution at Epoch 50

```
min: 0.452, p10: 0.484, p25: 0.549, median: 0.610, p75: 0.701, max: 0.773
```

**8 lowest-rate tasks** (target candidates for improvement):

| suite        | task | rate | samples | xyz_mse | rot_mse | oracle_gap | jerk |
|--------------|------|------|---------|---------|---------|------------|------|
| libero_spatial | 32  | 0.452 | 9864   | 0.0261 | 0.00111 | 0.001538 | 0.260 |
| libero_goal    | 24  | 0.468 | 8130   | 0.0443 | 0.00121 | 0.002370 | 0.177 |
| libero_goal    | 36  | 0.483 | 6194   | 0.0499 | 0.00109 | 0.001007 | 0.063 |
| libero_spatial | 2   | 0.484 | 12766  | 0.0174 | 0.00124 | 0.000262 | 0.196 |
| libero_goal    | 7   | 0.488 | 9081   | 0.0386 | 0.00161 | 0.001123 | 0.063 |
| libero_goal    | 19  | 0.509 | 11336  | 0.0288 | 0.00080 | 0.000910 | 0.240 |
| libero_spatial | 23  | 0.539 | 8440   | 0.0207 | 0.00098 | 0.000453 | 0.245 |
| libero_goal    | 9   | 0.540 | 11872  | 0.0169 | 0.00052 | 0.001080 | 0.228 |

These are exactly the **drawer/cabinet tasks** (high xyz error → high baseline-quantile cut) and **"open and place" goals** (high jerk + high gripper variability). The bottleneck here is **task-natural scale**, not LeWM quality — `xyz_mse` is genuinely 5–6× larger on these tasks than on `libero_object` pick-up tasks.

### 2.4 Step-by-Step Oracle MSE (8-step rollout, full 487880 samples)

| step | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      |
|------|--------|--------|--------|--------|--------|--------|--------|--------|
| oracle_cosine | 0.999995 | 0.999977 | 0.999935 | 0.999857 | 0.999735 | 0.999554 | 0.999299 | 0.998942 |
| oracle_mse    | 0.000330 | 0.000912 | 0.001737 | 0.002812 | 0.004159 | 0.005771 | 0.007691 | **0.009970** |
| teacher_mse   | 0.000349 | 0.000983 | 0.001887 | 0.003064 | 0.004529 | 0.006276 | 0.008347 | **0.010788** |

**Teacher-vs-oracle gap at step 8: +8.2% MSE.** This gap *is* the `oracle_mse_gap` signal the audit uses. Smaller gap → fewer `large_oracle_gap` rejections (current count: 48788 / 487880 = 10% reject). A tighter LeWM would shrink this gap and put more candidates into the safety_r pass.

---

## 3. Marginal Cost of Epoch 50 → 100

| Step | Wall time | GPU hours (1× A100/4090) |
|------|-----------|--------------------------|
| Continue LeWM 50→100 (30867 steps × 0.11 s/step) | ~1.0 h | 1.0 |
| Re-extract GT latents (libero_plus framebank, 313 fps × 2.24M frames) | ~2.4 h | 2.4 |
| Re-score 487880 candidates (~23 ms/sample, batch=32) | ~3.1 h | 3.1 |
| Re-audit (06_audit_teacher_quality.py) | ~0.5 h | 0.5 |
| Re-build distillation dataset (08_build_distillation_dataset.py) | ~0.2 h | 0.2 |
| Re-train GR00T student (Pol_full × 3 seeds × 20k steps, ~5.5 s/step) | ~91 h | 91 |
| Re-eval LIBERO-Plus (40 × 50 × 3 seeds × 2 systems) | ~30 h | 30 |
| **Total** | **~128 h** | **~128** |

Compare to **keep epoch 50 + just run Pol_full refilter**: ~110 GPU-hours — **the marginal cost of bumping epoch is +18 hours, mostly the LeWM/extraction/scoring re-do** (the GR00T training is shared). The "research-grade incremental cost" is ~6.5 GPU-hours (Stages A–E).

---

## 4. Marginal Benefit Forecast (Epoch 50 → 100)

### 4.1 Empirical anchor: epoch 30 → 50 gave +15% oracle_mse reduction

From the sweep data, `oracle_mse_mean` decreased monotonically:
- epoch 20 → 30: 0.006312 → 0.004926 = **−22%**
- epoch 30 → 50: 0.004926 → 0.004168 = **−15%**

Extrapolating the log-linear decay, **epoch 50 → 100** likely gives **−6% to −10%** oracle_mse. This is *not* nothing — it is on the order of the difference between Pol_base and Pol_full's headline number.

### 4.2 What specifically would improve?

| Audit signal | Current (ep50) | Forecast (ep100) | Direction |
|--------------|---------------|------------------|-----------|
| teacher_agreement_rate | 0.9136 | ~0.92 | marginal ↑ |
| oracle_better_mse_rate | 0.6452 | ~0.62 (lower = better oracle) | ↓ by definition |
| oracle_mse_step_last_over_first | 31.13 | ~27–29 | teacher step-quality ↑ |
| late_mse (per-task p90 cutoff) | 0.0147 | slightly tighter | more passes on borderline |
| oracle_gap_mse_p95 | 0.04949 | ~0.045 | more passes at boundary |

The mechanism that *directly* improves the audit is: **smaller teacher-vs-oracle gap at step 8 → more candidates pass `large_oracle_gap` filter → more candidates eligible for top-K**. So the easy-decile tasks (rate > 0.7) stay at 0.7; the hard-decile tasks (rate 0.45–0.55) might rise to 0.55–0.62, closing the gap by 5–8 pp.

### 4.3 What will *not* improve?

- **Task-natural xyz_mse scale** — these are physical units (mm/cm). The LeWM predictor accuracy is already very high. Epoch 100 won't shrink a drawer's xyz_mse from 0.04 to 0.005. So the per-task scale-conflation problem that Pol_full addresses via per-task empirical-CDF normalization **remains unsolved by epoch alone**.
- **Gripper match rate** (already ≥ 0.93 on average; 1.0 on the hardest task 36) — no LeWM can fix a gripper mismatch in the demonstration itself.
- **Jerk smoothness on natural motion** — the audit's `teacher_jerk_outlier` flag (31023 samples = 6.4%) is about whether the teacher action trajectory is jittery. This is partly an artifact of the noise injection in `05_score_libero_plus.py` and partly about the latent's decoding fidelity. Epoch 100 might improve marginally.

### 4.4 Where epoch 100 *would* help Pol_full

In the Pol_full proposal:
- **`safety_z`** uses per-task empirical CDF over `{xyz_mse, rot_mse, jerk_xyz, jerk_rot, late_mse}`. **All five** of these are LeWM-derivative measurements. A 6–10% shrinkage in raw values shifts the *upper tail* of the task-conditional distribution. The CDF rank at the top of the task stays at 1.0 regardless, so **Pol_full's `safety_z` is rank-invariant to absolute scale** — only the relative ordering matters.
- **`safety_r`** uses log-ratio of `best_mse / oracle_mse`. Both shrink with epoch, but the *ratio* is more sensitive to whether oracle shrinks faster than best_mse. Looking at the sweep, `oracle_mse_mean` shrinks faster than `best_mse_mean`, so the ratio decreases — meaning **safety_r threshold τ_r becomes tighter, fewer candidates pass**, but those that pass are *genuinely* closer to oracle. Quality up, quantity down → for Pol_full this is a wash for budget but a win for student training signal.

### 4.5 Net Effect Estimate

| Outcome | Likely Change (epoch 50 → 100) |
|---------|--------------------------------|
| Pol_base overall success rate | **+0.5 to +1.5 pp** (mostly via fewer `large_oracle_gap` rejections on borderline hard tasks) |
| Pol_full overall success rate | **+0.5 to +1.0 pp** (Pol_full's safety_z is already robust to scale; safety_r tightens slightly) |
| Worst-decile task improvement | **+3 to +5 pp** (where the gap-to-oracle is binding) |
| Audit acceptance delta | +8000 to +15000 samples (from 309128) |
| Direct-aug rate on task 7 (drawer) | 0.488 → maybe 0.52 |
| Direct-aug rate on task 32 (spatial drawer) | 0.452 → maybe 0.49 |

**Forecast headline**: epoch 100 likely moves the needle by **~1 pp overall**, with **most of the gain concentrated in the bottom-decile tasks** that are already the diagnostic targets of Pol_full. This is a real but **small** effect compared to the Pol_base → Pol_full jump (forecast ~3–5 pp).

---

## 5. Decision

### 5.1 Primary recommendation

**Do NOT train LeWM to epoch 100 (yet). Keep epoch 50 and ship Pol_full.**

Reasoning:
1. **The audit filter change (Pol_base → Pol_full) is the dominant intervention** that addresses the task-scale-conflation bottleneck. Continuing training the upstream model addresses a *different* bottleneck (teacher-vs-oracle gap), and the Pol_full proposal explicitly *uses* the gap as one of two safety signals.
2. **Diminishing returns on LeWM training**. Empirical sweep shows oracle_mse reduction rate halved from epoch 30→50 vs epoch 20→30. Continued training is sub-linear.
3. **The 6.5 GPU-hour incremental cost (Stages A–E)** is small, but the **128 GPU-hour full re-pipeline cost** (Stages A–G) is the actual budget — and that budget is better spent on running Pol_full's 3 paired seeds for the paper claim.
4. **`val_pred` is still decreasing at epoch 50** (3.3× epoch 10→50) — true — but `val_loss` is increasing due to `val_sigreg`. Without a sweep at epoch 70 or 80, we don't know when val_pred truly saturates. **Spending another ~50 min on a 70-epoch checkpoint and re-scoring the 5000-window subset is the cheapest, highest-information experiment.**
5. **The paper claim must be C1 (Pol_full > Pol_base) on epoch 50.** If that claim holds (forecast strong, given Pol_full's design), the paper has its anchor. A separate ablation "does Pol_full's gain survive an epoch-100 teacher refresh?" is appendix material, not the headline.

### 5.2 What to do instead (sequenced, ~110 GPU-hours total)

| Step | Action | GPU-hours | Risk |
|------|--------|-----------|------|
| 1 | Run Pol_full refilter on epoch-50 scoring inputs (`06c_refilter_teacher_quality_v3.py`) | 0.5 | data-level abort if `|E|<K` |
| 2 | Validate inputs (`09_validate_training_inputs.py`) | <0.1 | downstream contract |
| 3 | Pol_base seed 0 anchor reproduce (`08c_train_gr00t_flare_44_20.sh`) | 18 | training variance |
| 4 | Pol_full × {0, 1, 2} paired fine-tunes | 54 | OOM (rare) |
| 5 | Pol_z × {1} ablation | 18 | — |
| 6 | Evaluation: 40 × 50 × (Pol_base seed0 + Pol_full × {0,1,2} + Pol_z ×1) | 24 | eval noise |
| 7 | Analysis: bootstrap CI, sign test, rescueability moderator | 0.1 | — |
| **Total** | | **~115** | |

This sequence delivers the paper's anchor claim at the *minimum* total compute budget, and **leaves the door open** for an epoch-100 refresh as an appendix ablation.

### 5.3 When epoch 100+ *is* worth doing

Only after Pol_full's headline claim is established and *if* one of these holds:
- Pol_full's gain on hard-decile tasks is < 1 pp → epoch 100 might rescue borderline samples.
- Reviewer demands a "robustness across teacher-quality levels" ablation.
- Submission is for a top venue (NeurIPS / CoRL / RSS) where the cost is justified.

In that case, run the **minimum-cost experiment first** to decide:
1. Continue LeWM to epoch 70 only (~30 min GPU) and re-score the **5000-window sweep subset** (not all 487880).
2. Compare sweep metrics epoch 50 vs 70: `oracle_mse_mean` and `oracle_mse_step_last_over_first`.
3. If epoch 70 shows >5% improvement over epoch 50 → train to 100, re-extract GT, re-pipeline Stages A–G. ~128 GPU-hours.
4. If epoch 70 shows <2% improvement → abandon the refresh, report epoch 50 in paper.

### 5.4 Predeclared decision rule

> **Epoch bump budget = 0 if Pol_full's C1 claim on epoch 50 passes the bootstrap CI; epoch bump budget = full 128 GPU-hours only if C1 fails AND the 5K-window epoch-70 sweep shows >5% oracle_mse improvement over epoch 50.**

---

## 6. Pre-Mortem on "What if epoch 100 *would* have helped?"

The risk of *not* training longer is small but worth bounding. Worst case:
- Pol_full with epoch-50 teachers lifts LIBERO-Plus overall by ~3 pp (forecast).
- Pol_full with epoch-100 teachers *would* have lifted by ~4 pp.
- Lost gain: ~1 pp on the paper headline.

Mitigation: the appendix ablation "Pol_full × epoch-100" can be added post-acceptance as a 1-page note. This is standard reviewer-driven revision and does not require re-running the headline experiment.

---

## 7. Compatibility with Existing Round-4 Proposal

The Pol_full / Pol_z method thesis in `FINAL_PROPOSAL.md` is **independent of which epoch's scoring inputs we use**. The two predicates `safety_z` (per-task empirical-CDF) and `safety_r` (log-ratio) are both *scale-relative* — Pol_full is robust to epoch choice by construction.

**No changes required** to:
- `FINAL_PROPOSAL.md`
- `EXPERIMENT_PLAN.md`
- `EXPERIMENT_TRACKER.md`
- The 4-policy claim map, the sign-test primary unit, the rescueability moderator

**Optional appendix ablation** (no impact on main paper):
- Sweep checkpoint epoch ∈ {50, 70, 100} × Pol_full; report how Pol_full's C1 hold rate changes.
- Cost: ~250 GPU-hours (3 checkpoints × 3 paired seeds × 18 hr training). Only run if reviewer requests or if headline under-delivers.

---

## 8. Appendix: Quantitative Summary

| Question | Answer |
|----------|--------|
| Is the LeWM predictor still improving at epoch 50? | **Yes** — `val_pred` 0.00127 (ep10) → 0.00038 (ep50). |
| Is the total `val_loss` still improving? | **No** — `val_sigreg` regularization drift inflates it. The model is *not* overfitting; the loss curve is misleading. |
| How much did epoch 30→50 improve oracle_mse? | **−15%** (0.004926 → 0.004168). |
| Forecast for epoch 50→100? | **−6% to −10%** oracle_mse (sub-linear extrapolation). |
| Marginal cost of epoch refresh? | **+18 GPU-hours** incremental on top of 110-hour Pol_full pipeline. |
| Marginal benefit forecast? | **+0.5 to +1.5 pp** overall LIBERO-Plus success, **+3–5 pp on worst-decile**. |
| Recommend? | **No** — keep epoch 50, ship Pol_full. Only revisit if C1 fails. |
| Cheapest signal to revisit? | **Train LeWM to epoch 70 only (~30 min), re-score 5000-window subset**, compare to epoch 50 sweep metrics. |

---

## 9. Memory / Follow-up Pointers

- Update `safety-z-and-safety-r.md` memory: append a note that Pol_full is **epoch-agnostic by construction** (both predicates are scale-relative).
- Add a new memory `epoch-selection-decision-2026-10-12.md` recording the epoch-50-vs-100 decision and the predeclared rule for revisiting.
- Save the per-task direct_aug_rate table (Q1 vs Q4 quartiles) as `task-quartile-distribution-2026-10-12.md` for future reference — these are the diagnostic targets of Pol_full.