# Paper Plan (v2 — reframed around action-head × multipath distillation)

**Title** (working): *Not All Distilled Paths Help Equally: How Action-Head Parameterization Mediates Multipath Demonstration Distillation in Robot Policies*
**One-sentence contribution**: We show that distilling multipath (multi-solution) teacher trajectories into a robot policy has *axis-dependent* effects on generalization — helping some generalization axes while harming others — and that the sign and magnitude of the effect depend on the action head's parameterization (flow/DiT vs CVAE vs regression), because these heads differ in how they represent and resolve trajectory ambiguity; we provide metrics that separate "beneficial multipath diversity" from "harmful ambiguity" in teacher data.
**Venue**: ICLR (MAX_PAGES = 9, main body to Conclusion end, excluding references & appendix)
**Type**: empirical / diagnostic (analysis + controlled comparison across action heads)
**Date**: 2026-09-25
**Section count**: 7

> ⚠️ **GATE / HONESTY NOTES (read first)**
> 1. `CLAIMS_FROM_RESULTS.md` = `verdict: REVIEW_UNAVAILABLE` → no claim has been adjudicated by the cross-model jury. Under draft mode I proceed; **every claim is tagged `[unadjudicated]`**. Re-run `/result-to-claim` with a working Codex/manual-review backend before submission.
> 2. **Premise fragility — must resolve before writing.** The claim "a model beats gt-only" rests on ONE checkpoint: `suite_corrected_gt_replay_54_10` = **0.8063** vs gt_only 0.7964 (+1.0pt). A *different* 54:10 checkpoint (`image_aug_gt_replay_54_10_rerun_20260911`) scored **0.7812** (−1.5pt) against the *identical* baselines. The +1.0pt is **single-seed, single-checkpoint, and not replicated** — treat it as a motivating signal, not a result. (See Next Steps.)
> 3. **All three sub-hypotheses currently have NO direct evidence.** The existing data (4 GR00T-DiT recipes + LeWM teacher-quality audit) can *motivate* the hypotheses but cannot test them: there are no CVAE/ACT or regression-head runs, no multi-seed scaling curves, and no validated "ambiguity" metric. This plan therefore marks most experimental cells as `needs experiment`.
> 4. No Codex MCP this session → Step-6 cross-review not performed.

## The pivot (old → new framing)

- **Old (engineering)**: "distillation improves data generalization" — a yes/no performance claim. Low novelty; current evidence is mixed/negative.
- **New (scientific)**: the *interaction* between (a) multipath nature of distilled teacher data and (b) the action head's parameterization. This reframes FLARE from "a method that must win" into "a controlled probe of how action heads handle multimodal supervision." A negative or axis-dependent result is still publishable and informative.

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|---|---|---|---|
| C1. Multipath distillation's effect is **axis-dependent**: it improves some generalization axes while degrading others (sub-hypothesis 1) | per_category in four_model_comparison (e.g. appearance strong, Robot Initial States 0.369) shows *axes differ*, but does NOT isolate multipath as the cause | `[unadjudicated]` **needs experiment** (controlled per-axis multipath on/off) | §4.2 |
| C2. Action-head parameterization changes the *scaling behavior* on multipath data (DiT/flow vs CVAE vs regression) (sub-hypothesis 2) | none — only GR00T-DiT head run so far | `[unadjudicated]` **needs experiment** (train ACT/CVAE + regression heads) | §4.3 |
| C3. Teacher trajectories can be decomposed into "beneficial multipath diversity" vs "harmful ambiguity", quantifiable by a proposed metric (sub-hypothesis 3, core) | LeWM teacher-quality audit has *ingredients* (oracle_gap, candidate diversity, teacher-GT action MSE, jerk, gripper mismatch) but no validated beneficial-vs-harmful split | `[unadjudicated]` **needs experiment** (define metric, validate it predicts downstream Δ) | §3.3, §4.4 |
| C4. A high GT:teacher replay ratio (54:10, suite-corrected) can match/exceed GT-only on LIBERO-Plus | suite_corrected comparison: 0.8063 vs 0.7964 (+1.0pt) — **single seed/checkpoint, contradicted by rerun 0.7812** | `[unadjudicated]` **needs replication** (multi-seed) | §4.1 (motivation only) |
| C5. The candidate generator (open-loop, same-capability model) bounds teacher quality: GT beats all K candidates in 61% of windows | teacher_scoring_epoch50/validation_summary.json (verified) | `[unadjudicated]` — evidence verified | §5 (analysis) |

## Structure

### §0 Abstract
- **What we achieve**: a controlled study of how multipath distilled trajectories affect different action heads, plus a metric separating beneficial diversity from harmful ambiguity.
- **Why it matters / is hard**: robot demos are inherently multimodal (many valid solutions); how a policy's action head absorbs this multimodality determines whether extra (distilled) data helps or hurts — yet this interaction is rarely isolated.
- **How we do it**: fix the distilled dataset and the backbone; vary only the action head (flow/DiT, CVAE/ACT, regression); measure per-axis generalization and correlate with a proposed teacher-ambiguity metric.
- **Evidence**: C1–C4 (mostly pending).
- **Most remarkable result** (target): the same multipath data raises success on some axes and lowers it on others, with the split predicted by head type.
- **Length**: 200–240 words. **Self-contained**: yes.

### §1 Introduction
- **Opening hook**: A robot reaching for an object has many valid paths; when we distill extra "teacher" trajectories to augment demonstrations, we assume more valid paths = better generalization.
- **Gap / challenge**: this assumption ignores that (a) some distilled paths are *ambiguous* rather than *diversely valid*, and (b) different action heads (diffusion/flow vs CVAE vs regression) represent multimodality very differently — regression averages modes, diffusion can model them, CVAE relies on the latent.
- **One-sentence contribution**: (title block).
- **Approach**: hold backbone + data fixed, sweep the action head, decompose teacher quality.
- **Key questions**: (Q1=sub-hyp1) is multipath distillation uniformly beneficial across generalization axes? (Q2=sub-hyp2) does head parameterization change the scaling on multipath data? (Q3=sub-hyp3) can we predict, per teacher trajectory, whether it will help or harm?
- **Contributions** (map to matrix):
  1. Evidence that multipath distillation is axis-dependent, not uniformly beneficial (C1).
  2. A controlled head-sweep showing parameterization-dependent scaling on multipath data (C2).
  3. A teacher-ambiguity metric that separates beneficial diversity from harmful ambiguity and predicts downstream effect (C3).
- **Results preview**: axis-dependent sign flips; head-dependent scaling curves.
- **Hero figure**: Fig 1 (below).
- **Length**: 1.5 pages.
- **Key citations**: GR00T/OpenVLA/π0 (VLA), Diffusion Policy, ACT (CVAE), multimodality-in-BC / mode-averaging literature. `[VERIFY]` all.
- **Front-loading check**: skim reader gets the 3 questions + the "axis-dependent" headline before the method.

### §2 Related Work
- **Subtopics** (by family/assumption, not paper-by-paper):
  1. Action-head parameterizations: regression / tokenized / CVAE (ACT) / diffusion & flow (Diffusion Policy, GR00T, π0) — and their known mode-averaging vs multimodal behavior.
  2. Multimodality & ambiguity in imitation learning (mode averaging, compounding error, action chunking).
  3. Data augmentation / distillation for robot policies (self-distillation, best-of-n, world-model scoring).
  4. Scaling behavior of robot policies (data-scaling laws).
- **Positioning**: we isolate the *head × multipath-data* interaction explicitly, rather than reporting a single method's score.
- **Length**: 1 page, 4 synthesis paragraphs.

### §3 Method / Setup
- **§3.1 Notation & problem**: observation o, GT action chunk a*, teacher set {τ_i} per scene (multipath), head h_θ ∈ {flow/DiT, CVAE, regression}; objective per head.
- **§3.2 Controlled setup**: fixed backbone + fixed distilled dataset; only the action head varies. Define the paired fixed-noise eval protocol (112 tasks × 10, seed 2026) and per-axis buckets (LIBERO-Plus categories).
- **§3.3 Teacher-ambiguity metric (core, C3)**: define a per-trajectory score combining (i) oracle gap (LeWM latent error vs GT), (ii) inter-candidate diversity (spread across the K candidates), (iii) teacher-GT action distance, (iv) dynamics plausibility (jerk/terminal error). Propose a split: *beneficial diversity* = high diversity + low oracle gap; *harmful ambiguity* = high diversity + high oracle gap (or high teacher-GT distance). State as formal definitions.
- **§3.4 Heads compared**: GR00T flow/DiT head (existing); ACT-style CVAE head (baseline); regression (MSE) head. Specify training parity (same steps, batch, data).
- **Length**: 2 pages.

### §4 Experiments / Main Results
- **§4.1 Motivating result** (C4): 54:10 suite-corrected vs gt_only (+1.0pt) — *presented as motivation with the replication caveat*, not a claim.
- **§4.2 Axis-dependence** (C1): per-axis success under multipath on/off, per head. `needs experiment`.
- **§4.3 Head scaling** (C2): success vs amount of multipath teacher data (scaling curve) for each head; test whether diffusion/flow scales better than CVAE/regression on multipath. `needs experiment`.
- **§4.4 Metric validation** (C3): does the §3.3 score predict the *sign* of per-axis downstream Δ? Correlation / AUC of "predicted-harmful" vs measured harm. `needs experiment`.
- **§4.5 Teacher-quality ceiling** (C5): oracle-better-61% diagnostic (existing data) — bounds how good multipath teachers can be when generated open-loop.
- **Data sources**: existing — `outputs/eval_libero_plus/*four_model_comparison.json`, `FLARE-VLA/outputs/libero_plus/teacher_scoring*/`; to-generate — ACT/CVAE + regression runs, multi-seed, scaling sweep.
- **Length**: 3 pages.

### §5 Analysis / Discussion
- Why open-loop same-model teachers cap quality (C5); how head choice interacts.
- When would multipath help: axes where GT is suboptimal / long-horizon; heads that model multimodality.
- Limitations (honest): single benchmark; C4 not replicated; metric is correlational until validated; heads share a backbone (VLA-specific) so ACT/regression parity is approximate.

### §6 Conclusion
- Restate: multipath distillation is not uniformly beneficial; its effect is axis- and head-dependent; a teacher-ambiguity metric can predict the sign.
- Limitations + future: closed-loop/MPC teachers, cross-benchmark, real-robot.

**Page budget**: 1.5+1+2+3+1+0.5 = 9.0 = MAX_PAGES ✓ (7 sections)

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Hero (2-panel) | LEFT: schematic — one scene, K multipath teacher paths, colored by predicted "beneficial diversity" vs "harmful ambiguity". RIGHT: per-axis success deltas (multipath-on minus multipath-off) across heads, showing sign flips. Caption: the same distilled data helps some axes and hurts others, and the split depends on the action head. | existing per_category + new head runs | HIGH |
| Fig 2 | Line (scaling) | success vs # multipath teacher samples, one curve per head (DiT/CVAE/regression) → head-dependent scaling (C2). | to-generate | HIGH |
| Fig 3 | Scatter | teacher-ambiguity score (x) vs measured per-axis downstream Δ (y); beneficial vs harmful quadrants labeled (C3). | to-generate + LeWM audit | HIGH |
| Fig 4 | Distribution | oracle_mse vs best-candidate_mse, 61% region shaded (C5). | teacher_scoring_epoch50 | MEDIUM |
| Table 1 | Head × recipe | success (overall + per-axis) for {DiT, CVAE, regression} × {GT-only, multipath}. | to-generate + existing | HIGH |
| Table 2 | Metric ablation | correlation/AUC of each metric component vs downstream harm. | to-generate | MEDIUM |

**Hero figure detail (Fig 1)**: a skim reader should grasp (a) multipath teacher data is a mix of "good diversity" and "bad ambiguity," and (b) the action head decides which dominates. The right panel must show at least one axis going up and another going down for the same data, differing by head.

## Citation Plan
- §1: GR00T N1.7 `[VERIFY]`, OpenVLA `[VERIFY]`, π0 `[VERIFY]`, Diffusion Policy `[VERIFY]`, ACT `[VERIFY]`.
- §2: mode-averaging / multimodality-in-BC `[VERIFY]`, action chunking `[VERIFY]`, world-model scoring (LeWM) `[VERIFY]`, robot data-scaling `[VERIFY]`.
- §3: LeWM (scorer), GR00T flow head, ACT CVAE head, LIBERO/LIBERO-Plus `[VERIFY]`.
- **Rule**: never generate BibTeX from memory; verify authors/year/venue. All entries `[VERIFY]` until checked.

## Reviewer Feedback
- **Not performed** — Codex MCP unavailable this session; no independent cross-model review. Run Step-6 with a working backend before submission.

## Next Steps (evidence to close BEFORE writing — in priority order)
- [ ] **Resolve the C4 contradiction**: re-eval `suite_corrected` AND `image_aug` 54:10 checkpoints on a 2nd (random) diffusion seed; document exactly what "suite_corrected" changed vs the rerun. +1.0pt single-seed is not a result.
- [ ] **Re-adjudicate claims**: re-run `/result-to-claim` with a working Codex/manual-review backend (remove `[unadjudicated]`).
- [ ] **C2 (head sweep)**: train an ACT/CVAE head and a regression head on the SAME backbone + distilled data; fixed-noise paired eval. (Largest missing piece.)
- [ ] **C1 (axis-dependence)**: design controlled per-axis multipath on/off so the cause is isolated, not just observed.
- [ ] **C3 (metric)**: implement §3.3 score from the existing LeWM audit fields; validate it predicts per-axis downstream sign.
- [ ] **C2 scaling**: vary multipath teacher count per head → scaling curves.
- [ ] Then: /paper-figure → /paper-write (only after claims adjudicated) → /paper-compile.
