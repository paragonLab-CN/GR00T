# Paper Plan

**Title** (working): *When Does Latent-Scored Teacher Distillation Help? A Controlled Study of Open-Loop Self-Distillation for Vision-Language-Action Models*
**One-sentence contribution**: We provide a controlled, fixed-noise paired evaluation showing that distilling open-loop candidates from a VLA model—ranked by a latent world model—does not improve over behavior-cloning on ground truth, and we characterize the structural conditions under which such "stronger-teacher" distillation can or cannot help.
**Venue**: ICLR (MAX_PAGES = 9, main body to Conclusion end, excluding references & appendix)
**Type**: empirical / diagnostic (NOT a method-wins paper — see honesty note below)
**Date**: 2026-09-25
**Section count**: 6

> ⚠️ **HONESTY / GATE NOTE (read first)**
> - `CLAIMS_FROM_RESULTS.md` = `verdict: REVIEW_UNAVAILABLE` → claims were **never adjudicated** by the cross-model jury. Under `assurance: draft` I proceed, but **every claim below is tagged `[unadjudicated]`**. Do NOT submit until `/result-to-claim` is re-run with a working Codex/manual-review backend.
> - **The current fixed-noise paired results show FLARE *underperforming* the GT-only baseline** (gt_only 0.7964 > gt_replay 0.7812 > image_aug 0.7661 > flare 0.7545). Therefore this plan is framed as a **diagnostic/analysis paper**, not a "our method beats baseline" paper. If future closed-loop/planning experiments flip the result, the framing can be upgraded to a method paper.
> - No Codex MCP this session → Step-6 cross-review not performed; the outline has **no independent reviewer pass**.

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|---|---|---|---|
| C1. Open-loop self-distillation (teacher = same-model open-loop candidates, LeWM-ranked) does **not** improve over GT-only BC on LIBERO-Plus | four_model_comparison.json (paired, fixed noise seed 2026): flare 0.7545 vs gt_only 0.7964 | `[unadjudicated]` — evidence verified to exist; direction supported by data | §4.1 |
| C2. The failure is structural: in 61% of windows the GT trajectory's future-latent error is lower than **all** K=4 open-loop candidates (oracle_mse 0.00415 vs best-candidate 0.01386, ~3.3×) | teacher_scoring_epoch50/validation_summary.json (N=489,632) | `[unadjudicated]` — verified | §4.2 |
| C3. A higher-quality scorer (LeWM epoch-50 vs epoch-9) sharpens the GT-vs-candidate gap (oracle_better_both_rate 0.444→0.610) rather than closing it — i.e. the ceiling is the candidate generator, not the scorer | validation_summary.json (epoch-9) vs (epoch-50); checkpoint_sweep/comparison.json | `[unadjudicated]` — verified (sweep subset is first-N biased; full-run numbers primary) | §4.2 |
| C4. A high GT:teacher ratio (54:10) + image augmentation recovers most of the gap (0.7812) — teacher acts as light perturbation, not stronger supervision | four_model_comparison.json | `[unadjudicated]` — verified, single seed | §4.3 |
| C5. Gains/losses are category-dependent: appearance buckets strong, Robot Initial States weak (0.369) | three/four_model_comparison per_category | `[unadjudicated]` — verified | §4.4 |

> Claims NOT made (insufficient evidence, do not add): "FLARE beats baseline", "teacher rivals GT", "results robust across seeds" (single diffusion seed for most variants), "not overfit" (no val loss logged).

## Structure

### §0 Abstract
- **What we achieve**: a controlled study isolating *why* latent-scored open-loop self-distillation fails to beat BC-on-GT for VLA models.
- **Why it matters / is hard**: self-distillation / RLAIF-style teacher selection is increasingly used to bootstrap robot policies without extra demos; its failure modes are under-characterized.
- **How we do it**: paired, fixed-diffusion-noise evaluation (112 tasks × 10 episodes) across four training recipes, plus a large-scale (489k-window) latent-scoring diagnostic of teacher vs GT.
- **Evidence**: C1–C4.
- **Most remarkable result**: in 61% of windows the GT demonstration beats *all* open-loop candidates under the very scorer used to select teachers — the "stronger teacher" largely does not exist in the data.
- **Estimated length**: 200–240 words.
- **Self-contained check**: reader learns the question, the negative result, and the diagnostic without the body.

### §1 Introduction
- **Opening hook**: Robot policies are data-hungry; a tempting remedy is to distill a model's own generated trajectories, keeping only the ones a world model scores as good.
- **Gap / challenge**: prior work assumes the selected teacher is at least as good as demonstrations; this assumption is rarely tested under controlled conditions.
- **One-sentence contribution**: (see title block).
- **Approach overview**: we (i) generate K open-loop candidates per window from a VLA model, (ii) rank by a latent world model, (iii) distill, and (iv) evaluate under paired fixed-noise protocol; we then audit the scorer's own oracle gap.
- **Key questions**: (Q1) Does latent-scored self-distillation beat BC-on-GT? (Q2) If not, is the bottleneck the scorer or the candidate generator? (Q3) Under what data-mixing does it stop hurting?
- **Contributions** (falsifiable, map to matrix):
  1. A controlled paired evaluation showing open-loop self-distillation underperforms GT-only BC on LIBERO-Plus (C1).
  2. A diagnostic identifying the candidate generator—not the scorer—as the ceiling (C2, C3).
  3. A data-mixing study showing teacher acts as light perturbation at high GT ratio (C4).
- **Results preview**: 61% oracle-better-than-all-candidates; −4.2pt for the main recipe.
- **Hero figure**: (see Figure Plan, Fig 1).
- **Estimated length**: 1.5 pages.
- **Key citations**: VLA (GR00T / OpenVLA / π0), latent world models (LeWM / PLDM), self-distillation / RLAIF, DAgger-style aggregation. `[VERIFY]` all.
- **Front-loading check**: skim reader gets the negative result + the 61% diagnostic before the method.

### §2 Related Work
- **Subtopics** (by methodological family, not paper-by-paper):
  1. Vision-language-action models (GR00T, OpenVLA, RT-2, π0).
  2. World-model / latent-model scoring & planning (LeWM, PLDM, MPC-with-latent).
  3. Self-distillation & teacher selection (RLAIF, self-training, best-of-n distillation).
  4. Imitation learning & covariate shift (BC vs DAgger).
- **Positioning**: we are the first (to our knowledge) to *quantify*, under a controlled paired protocol, when latent-scored open-loop self-distillation fails to beat BC — and to attribute the failure to the generator, not the scorer.
- **Minimum length**: 1 page, 4 paragraphs of synthesis.
- **Organization rule**: by assumption ("teacher ≥ demo"), not by paper.

### §3 Method / Setup
- **Notation**: observation o_t, GT action chunk a*_t:t+H, candidate set {a^(k)}_{k=1..K}, latent encoder φ, scorer S(·), horizon H.
- **Problem formulation**: given demonstrations D, generate K open-loop candidates per window, select argmin_k S(φ(rollout(a^(k))) vs φ(GT future)), distill selected teachers into the policy.
- **Method description**: (1) LeWM latent world model training (epoch-9 vs epoch-50); (2) GT latent framebank extraction; (3) K=4 open-loop candidate generation (denoise=4, independent noise, horizon=8); (4) latent scoring + teacher-quality audit; (5) four training recipes (flare mix 0.12 / image_aug / gt_only / gt_replay_54_10).
- **Formal statements**: define oracle_gap = S(GT) − min_k S(candidate_k); define selection policy.
- **Estimated length**: 1.5–2 pages.
- (Note: document the horizon mismatch teacher-8 vs processor-16 as a known design parameter.)

### §4 Experiments / Main Results
- **Setup**: LIBERO-Plus (112 tasks, 4 suites), paired fixed-diffusion-noise protocol (master seed 2026, per-request seed formula), 10 episodes/task → 1120/model.
- **§4.1 Main comparison** (C1): Table 1 — four recipes, overall + per-suite success rate. Result: gt_only best.
- **§4.2 Teacher-quality diagnostic** (C2, C3): Fig 2 — oracle vs best-candidate latent MSE distribution; 61% oracle-better; epoch-9 vs epoch-50 scorer comparison.
- **§4.3 Data-mixing** (C4): gt_replay_54_10 vs flare mix; GT:teacher ratio sweep (if run).
- **§4.4 Category breakdown** (C5): per-category bar (appearance strong, Robot Initial States weak).
- **Data source**: `outputs/eval_libero_plus/*/four_model_comparison.json`, `three_model_comparison.json`, `FLARE-VLA/outputs/libero_plus/teacher_scoring*/`.
- **Estimated length**: 3 pages.

### §5 Analysis / Discussion
- Why the candidate generator is the ceiling (open-loop, same-capability model, denoise=4).
- Why a better scorer doesn't help (C3) — and what *would*: closed-loop / MPC replanning, per-window gating to candidate<oracle windows, planner-on-GT.
- Limitations (honest): single diffusion seed for most variants; no held-out val loss; horizon mismatch; epoch-50 downstream model interrupted mid-training (5031/10000) so epoch-50-teacher downstream benefit is unverified.
- **Estimated length**: 1 page.

### §6 Conclusion
- **Restatement**: latent-scored open-loop self-distillation, as commonly configured, does not beat BC-on-GT; the bottleneck is the candidate generator, not the scorer.
- **Limitations**: single benchmark, single seed, scorer-specific.
- **Future work**: closed-loop/MPC teachers; cross-seed and cross-benchmark replication.
- **Estimated length**: 0.5 pages.

**Page budget check**: 1.5+1+2+3+1+0.5 = 9.0 = MAX_PAGES ✓

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Hero (2-panel) | LEFT: pipeline diagram (LeWM scoring of K open-loop candidates → distill). RIGHT: headline result — bar chart of 4 recipes' success_rate with gt_only tallest, annotated "open-loop self-distillation < BC-on-GT". Caption states the negative result + the 61% diagnostic in one sentence. | four_model_comparison.json | HIGH |
| Fig 2 | Distribution | oracle_mse vs best-candidate_mse histogram/scatter (489k windows), mark 61% region where GT beats all candidates; overlay epoch-9 vs epoch-50 scorer. | teacher_scoring_epoch50/validation_summary.json | HIGH |
| Fig 3 | Bar chart | Per-category success (appearance vs Robot Initial States) for the 4 recipes. | per_category | MEDIUM |
| Fig 4 | Line/bar | Scorer quality sweep: oracle_mse at LeWM epoch 20/30/50 (monotonic improvement) — supports "scorer not the bottleneck". | checkpoint_sweep/comparison.json | MEDIUM (note first-N subset caveat) |
| Table 1 | Main results | 4 recipes × (overall + 4 suites) success rate, paired fixed-noise. | four_model_comparison.json | HIGH |
| Table 2 | Teacher audit | selection rate, high/medium/low/reject counts, teacher-GT action MSE thresholds, epoch-9 vs epoch-50. | teacher_scoring*/teacher_quality_summary.json | MEDIUM |

**Hero figure detail (Fig 1)**: a skim reader should see (a) what the method does and (b) that it does NOT beat the baseline, plus the one-line "why" (61% of the time the demo already beats every candidate). The right panel's gt_only bar must be visually tallest.

## Citation Plan
- §1 Intro: GR00T N1.7 `[VERIFY]`, OpenVLA `[VERIFY]`, π0 `[VERIFY]`, LeWM `[VERIFY]`, RLAIF `[VERIFY]`.
- §2 Related: RT-2, PLDM, DAgger, best-of-n / self-training, world-model planning `[VERIFY]` each.
- §3 Method: LeWM (scorer, built upon), GR00T (base policy), LIBERO / LIBERO-Plus benchmark `[VERIFY]`.
- **Rule**: never generate BibTeX from memory; verify authors/year/venue via search before submission. All current entries are `[VERIFY]`.

## Reviewer Feedback
- **Not performed** — Codex MCP unavailable this session. This outline has had no independent cross-model review. Before submission, run Step-6 review with a working backend.

## Next Steps / Evidence gaps to close BEFORE writing
- [ ] Re-run `/result-to-claim` with a working Codex/manual-review backend to adjudicate claims (remove `[unadjudicated]` tags).
- [ ] Finish the interrupted epoch-50 downstream model + eval, or drop the epoch-50-teacher claim.
- [ ] Add a 2nd diffusion seed (random_noise) for gt_only and the strongest variant to bound single-seed variance.
- [ ] (Optional, strengthens C3) closed-loop or MPC teacher pilot to test whether the ceiling can be lifted.
- [ ] /paper-figure to generate Fig 1–4 and Table 1–2 from the JSON sources above.
- [ ] /paper-write to draft LaTeX (only after claims adjudicated).
- [ ] /paper-compile to build PDF.
