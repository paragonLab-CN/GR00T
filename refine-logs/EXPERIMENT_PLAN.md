# Experiment Plan

**Problem**: optimize the FLARE-VLA teacher filter on LIBERO-Plus to raise student success rate, in service of a paper-grade core claim.
**Method Thesis**: budget-constrained, top-K teacher selection under per-task component-badness (`safety_z`) and one shared cross-task log-ratio latent-regret constraint (`safety_r`), calibrated on a small calibration subset, evaluated under paired training seeds.
**Date**: 2026-10-12.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| **C1 (Main)** | Validates the dominant contribution: a budget-controlled teacher filter with two eligibility predicates beats the global-quantile baseline on overall LIBERO-Plus success rate. | (i) Overall 10030 success rate > Pol_base's prior 0.7921 anchor at step 10000; (ii) 10030-task sign test favoring Pol_full on >50% of tasks (one-tailed); (iii) easy-decile non-inferiority lower bound > −1.5 pp. | B1, B2, B3, B5 |
| **C2 (Supporting)** | Validates the refilter is *small*: no measurable regression on previously-easy tasks or per-suite means. | Easy-decile (per-task success rate ≥ 0.8 in Pol_base) sign test non-negative in ≥ 80% of tasks; per-suite paired t-test Δ ≥ 0 in all four suites (predeclared). | B1, B4 |

## Paper Storyline
- **Main paper must prove**:
  - Pol_full > Pol_base on LIBERO-Plus overall success under fixed accepted-sample budget.
  - Pol_z vs Pol_full isolates the safety-r contribution (rescueability-driven).
  - Easy-decile non-inferiority.
- **Appendix can support**:
  - Per-(suite, task) diagnostic table with rescueability as moderator.
  - Spearman calibration check at multiple ε values.
  - max-CDF intra-task correlation diagnostic.
- **Experiments intentionally cut**:
  - Any ablation that does not isolate one of the three mechanism differences (per-task, log-ratio, top-K).
  - Multiple frontier-model baselines (VLM judges, reward models, diffusion teachers) — reviewer explicitly rejected this route.
  - λ / τ_r sensitivity sweeps across more than 3 ε values — predeclared as M-1.

## Experiment Blocks

### Block 1: Anchor result — Pol_full vs Pol_base on LIBERO-Plus
- **Claim tested**: C1.
- **Why this block exists**: this is the dominant contribution; without it the paper has no anchor.
- **Dataset / split / task**: **Full LIBERO-Plus benchmark — 10030 tasks × 1 trial/task** (user's prior evaluation protocol; see `outputs/eval_libero_plus/gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20_checkpoint_10000_full10030/summary_all.json`, which records Pol_base at step 10000 = 79.21% overall). 10030 = libero_spatial (2402) + libero_object (2518) + libero_goal (2591) + libero_10 (2519).
- **Compared systems**: `Pol_base` (existing 08c_train_gr00t_flare_44_20 teacher input), `Pol_full` (newly refiltered teacher input).
- **Metrics**: overall success rate over 10030 trials; per-suite success rate (the 4 suites × 1 trial); per-task success (0/1 per task); per-category and per-difficulty breakdowns (already produced by the user's prior eval script).
- **Setup details**: identical training recipe (`08c_train_gr00t_flare_44_20.sh`), except `teacher_quality_direct_records.pt`; paired seeds; same 20k steps; same checkpoints. **Evaluation invocation is byte-identical to the user's prior eval runs** — same `gr00t/eval/run_gr00t_server.py` invocation, same `libero_plus` benchmark, same `10030` task list (the prior `summary_all.json` shape is the output contract).
- **Success criterion** (predeclared): Pol_full > Pol_base on overall 10030 success rate at the same checkpoint step; **the prior Pol_base at step 10000 = 0.7921** is the anchor.
- **Failure interpretation**: if Pol_full does not improve, we report honestly as a negative ablation. If paired seeds show |Δ| < 1 pp, the contribution may be too small to publish.
- **Table / figure target**: main paper Table 1 (overall + per-suite) and Figure 1 (per-task Δ scatter).
- **Priority**: MUST-RUN.

### Block 2: Mechanism isolation — Pol_z vs Pol_full (rescueability moderates)
- **Claim tested**: C1 (sub-mechanism).
- **Why this block exists**: isolates the contribution of `safety_r`.
- **Dataset / split / task**: same 10030 × 1 LIBERO-Plus evaluation; Pol_z runs seed 1 only (ablation column).
- **Compared systems**: `Pol_z` (no safety_r, same K=309128, same per-task `safety_z`), `Pol_full`.
- **Metrics**: per-task Δ; linear-regression coefficient of per-task Δ on `rescueability`; rescueability-by-suite breakdown.
- **Setup details**: Pol_z = `TopK_{z_s}{prereq}` with K=309128; one seed; same recipe.
- **Success criterion**: regression coefficient of Δ on rescueability is positive with 95% CI excluding 0; on the low-rescueability tasks (≤ 0.30), Pol_full > Pol_z in ≥ 60% of tasks.
- **Failure interpretation**: if Δ does not correlate with rescueability, the safety_r is not the active mechanism — but the paper still has Block 1 as the dominant claim.
- **Table / figure target**: Figure 2 (rescueability vs per-task Δ scatter).
- **Priority**: MUST-RUN (single-seed ablation).

### Block 3: Frontier-necessity check (replaced by an architectural minimalism defense)
- **Claim tested**: "no VLM judge / reward model / diffusion / RL stage is necessary".
- **Why this block exists**: the cross-model reviewer explicitly defended the no-frontier-primitive stance; the paper must *show* the dominance of the no-new-model option.
- **Dataset / split / task**: same LIBERO-Plus.
- **Compared systems**: `Pol_full` (proposed); `Pol_full + VLM-judge rerank` is NOT run (intentional); `Pol_full + diffusion teacher` is NOT run (intentional).
- **Metrics**: discussion-table format, no numeric runs.
- **Setup details**: cite prior literature to show that VLM-judge / reward-model additions in this setting typically add inference compute and require new label sources not available in this scenario.
- **Success criterion**: a written argument, in §5 of the paper, that any frontier primitive here would not change the C1 outcome within ±1 pp while strictly worsening either compute or causal attribution.
- **Failure interpretation**: if a reviewer demands a VLM-judge baseline, the paper can decline on cost-attribution grounds.
- **Table / figure target**: paper §5 (Discussion), no figure.
- **Priority**: WRITTEN DEFENSE — no new run.

### Block 4: Simplicity check — deletion study on Pol_full
- **Claim tested**: "the two-predicates-and-top-K is minimal".
- **Why this block exists**: defend elegance against reviewer-conjecture "you added an extra scoring model".
- **Dataset / split / task**: same calibration subset.
- **Compared systems**: `Pol_full` (proposed); a hypothetical "Pol_full_plus_minus" (with extra learned weights or a second auxiliary metric) is NOT run.
- **Metrics**: comparison of (i) total mechanism lines of Python, (ii) number of new forward passes, (iii) # hyperparameters fit on the calibration subset. Reported as a textual table.
- **Setup details**: static analysis from `06c_refilter_teacher_quality_v3.py` (count of: predicates, threshold parameters, downstream modifications).
- **Success criterion**: Pol_full ends with ≤ 2 predicates and ≤ 2 calibrated thresholds, and zero new forward passes.
- **Failure interpretation**: if either predicate turns out to be removable without losing Block 1's success criterion, that ablation is reported.
- **Table / figure target**: paper §5, no separate figure.
- **Priority**: WRITTEN DEFENSE — no new run.

### Block 5: Failure analysis — `safety_z` vs `safety_r` mechanism diagnostic
- **Claim tested**: "the two predicates do not collapse into one another".
- **Why this block exists**: prevents future readers from collapsing them.
- **Dataset / split / task**: same calibration subset.
- **Compared systems**: distribution of `safety_z` in selected vs rejected (`use_for_direct_augmentation=0`); distribution of `safety_r` by `Pol_base` reject cause; correlation matrix among the five component metrics per (suite, task_index) cluster.
- **Metrics**: report per-task CDF of `safety_z`; per-cause rejection breakdown; per-suite Spearman correlation between component metrics.
- **Setup details**: diagnostic computation only; no GPU.
- **Success criterion**: documentation that no single component completely dominates `safety_z` and that `safety_r` rejects with characteristic reasons different from `safety_z` rejections.
- **Failure interpretation**: if `safety_z` is monotonic in one component, the paper still works but the discussion notes it.
- **Table / figure target**: paper Appendix A1.
- **Priority**: MUST-RUN (diagnostic, <1 GPU-hour).

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Data-level audit + Pol_full/Pol_z materialize | Materialize Pol_full, Pol_z; verify |E|≥K, hashes, deterministic reruns | pass | abort & redeploy calibration if |E|<K | 0.5 GPU-hr | Spearman failure |
| M1 | Pol_base seed 0 (anchor reproduce) | Reuse `08c_train_gr00t_flare_44_20.sh` with current Pol_base teacher input, seed 0 | matches prior log within ±2 pp | 16-20 GPU-hr | training variance |
| M2 | Pol_full × {0, 1, 2} | Three paired fine-tunes | per-seed loss curves converge, no NaN | 48-60 GPU-hr | OOM |
| M3 | Pol_z × {1} | One ablation fine-tune | converges in same wall-clock | 16-20 GPU-hr | not run if M2 fails |
| M4 | Evaluation × **10030 tasks × 1 trial/task** × 3 seeds × 2 systems (Pol_base, Pol_full) + 1 ablation seed × Pol_z | `gr00t/eval/run_gr00t_server.py` (same invocation as user's prior `gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20_checkpoint_10000_full10030` run) | means computed; sign test p <0.05 | ≈ 1-2 GPU-hr per model×seed = 5 × 1.5 = 7-8 GPU-hr total | eval-rollout noise is real (1 trial = binary); a 1-trial win on a 10030-set requires Δ ≳ 0.5 pp to be meaningful |
| M5 | Hierarchical bootstrap + per-decile Δ + rescueability moderator | — | primary claim passes | 0.1 GPU-hr | — |

Total: ≈ 105-130 GPU-hours on a single 4090/A100.

## Compute and Data Budget
- **Total estimated GPU-hours**: ≈ 80-160 GPU-hours (7 fine-tunes × 10k-20k steps at ~10 sec/step, plus evaluation: trial count × 40 tasks × 5 models at user's prior trial setting).
- **Data preparation needs**: zero new labels; reuses existing `teacher_scoring_epoch50_fixed/` artifacts.
- **Human evaluation needs**: none.
- **Biggest bottleneck**: training-wall-clock under a single-GPU constraint (acceptable for one paper).

## Risks and Mitigations
- **Risk**: `|E| < K=309128` (data-level failure). *Mitigation*: predeclared abort; fallback is Pol_z published as the primary, and the paper's contribution is reframed as "Pol_z vs Pol_base".
- **Risk**: `|ρ_Spearman(r_s, oracle_mse_s)| > 0.10` at both ε values. *Mitigation*: predeclared fallback to Pol_z and a written "cross-task-regret claim invalid" diagnostic. The Pol_z claim must still hold.
- **Risk**: training-variance across seeds larger than expected. *Mitigation*: 3 paired seeds, hierarchical bootstrap, predeclared CI for sign test; if CI doesn't exclude 0, paper reports honestly.
- **Risk**: easy-decile non-inferiority fails. *Mitigation*: easy-decile lower bound reported honestly; if lower bound ≤ −1.5 pp, paper interprets as over-aggressive rejection on easy tasks.
- **Risk**: max-CDF correlation sensitivity skews `safety_z`. *Mitigation*: per-task calibration CDF reported in Appendix A1; document the within-task metric correlations.

## Final Checklist
- [x] Main paper tables are covered (Block 1 → Table 1; Block 2 → Figure 2; Block 5 → Appendix A1).
- [x] Novelty is isolated (Block 2 isolates `safety_r`; Block 5 isolates `safety_z` vs `safety_r`).
- [x] Simplicity is defended (Block 4 written defense).
- [x] Frontier contribution is justified or explicitly not claimed (Block 3 written defense).
- [x] Nice-to-have runs are separated from must-run runs (Pol_z ablation is MUST-RUN because C1 mechanism claim requires it; Block 3 and Block 4 are written defenses without new runs).
