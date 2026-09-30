verdict: ADJUDICATED
date: 2026-09-30
judge: codex (gpt-5.5, xhigh) — gpt-6-astra+ultra unavailable (capability error), fell back per reviewer-routing
integrity_status: pass  [INTEGRITY: WARN] — audit flagged design issues (teacher systematically < GT), which IS this paper's finding
evidence_precheck: 10/10 cited numbers verified to exist in source files
threadId: 01a0f021-dbf9-7380-b9d2-b8fe6d1bde1d

# Per-claim verdicts

| Claim | claim_supported | confidence | one-line |
|---|---|---|---|
| C1 self-consistency not improvement-over-GT | **yes (narrowed)** | med-high | deployed epoch-9: 65.2% pass, 44.4% GT-better, GT≈1.03×, agreement 0.9435. Phrase as "selects self-consistent near-demo teachers, little NEW info beyond GT" — NOT "selects sub-GT teachers" (epoch-50 only) |
| C2 138 features → few interpretable ~orthogonal axes | **yes** | 0.8 | PCA top-4=70.1% var, cleanly blocked; corr(div,oracle-gap)=0.056. Use "approximately orthogonal / decoupled", not "independent" (no ICA/MI test) |
| C3 subset of features predicts downstream | **no (evidenced negative)** | 0.85 | measured: no block significant (all p>0.39, n=35); oracle_gap ρ=−0.03; exact-match flips sign. Revise to a NULL finding |
| C4 beneficial-vs-harmful → structural misalignment | **partial** | medium / med-low | epoch-9: keeps 64.5%/43.3% of beneficial at P50/P75 (NOT "broadly discards"), but **0/9,173 at D≥P90**; selected = 35.4% redundant + 6.8% jitter. Narrow claim: "fails at the high-divergence tail, over-admits redundant near-GT teachers" |

# Epoch-9 re-adjudication (2026-09-30, threadId 01a0f057)
- **C1 → yes, narrowed (med-high)**: epoch-9 does not support "selects sub-GT teachers" (44% GT-better = minority; MSE ~tie 1.03×). The load-bearing pillar is now teacher_agreement=0.9435 (self-consistency), not "GT crushes candidates".
- **C4 → partial (medium/med-low)**: NOT "broadly discards beneficial diversity" (keeps ~half at moderate thresholds). Narrow: fails at high-divergence tail (0 kept at D≥P90) + over-admits redundant near-GT copies.
- **Central hook survives but reframed**: NOT "the filter is misaligned / selects worse teachers". Defensible thesis = "the deployed filter is a *conservative self-consistency selector*: it mostly preserves near-GT redundant teachers, admits some jitter, and loses the most-divergent valid alternatives where exploration value would be highest — adding little useful signal beyond GT." Epoch-50 is sensitivity evidence (sharper same failure mode), not the headline.
| C5 ceiling bounded by open-loop same-model pool | **yes/partial** | med-high | selector can only pick among generated candidates (44% GT-better deployed / 61% epoch-50) |
| D1 FLARE mix0.12 < GT-only | **yes** | high | 0.7545 vs 0.7964, −4.20pt, McNemar p=3.4e-4 |
| D2 gt_replay 54_10/44_20 > gt_only | **partial** | medium | small gains, 54_10 vs gt_only p=0.422 (NS), 44_20 vs 54_10 a wash, seed spread ±0.5pt. 0.8063/0.7812 **RESOLVED** (§5.1): two different runs, cite only canonical `suite_corrected` 0.8063 |

# Scorer-convention correction (2026-09-30, user decision)
All headline numbers unified to the **DEPLOYED filter** = `teacher_scoring_v2` (LeWM **epoch-9**,
N=487,880, selected 65.2%=317,941) — confirmed via `distillation_dataset_v2/distillation_summary.json`
that this is the teacher set actually fed to `gt_replay_44_20`. The epoch-50 audit
(`teacher_scoring_epoch50_v2`, N=489,632) is demoted to a **sensitivity analysis** (§5.5). Epoch-9
headline values recomputed from `teacher_scoring_v2/teacher_quality_audit.parquet` and spot-verified:
oracle_better_both=0.4439, oracle_mse=0.01127, best_mse=0.01155, agreement=0.9435, sel_rate=0.6517.
C4 on epoch-9: beneficial diversity D≥P50/P75/P90 = 12.28%/5.51%/1.88%; kept-by-filter 64.5%/43.3%/0%;
selected = 35.4% redundant + 6.8% jitter.
**Hook impact**: the "GT 3.3× better / 61%" figure is epoch-50-specific; on the deployed filter GT is
only ~1.03× better and beats all candidates in 44%. The misalignment conclusion still holds (filter
admits mostly redundant copies; discards the most-divergent valid paths) but is quantitatively weaker.

# Re-adjudication 2026-09-30 (C2/C3 now computed)
Data correction applied: C2/C3 run on `teacher_scoring_v2` (LeWM epoch-9 scorer, N=487,880,
selected 65.2%) — the teacher set ACTUALLY distilled into `gt_replay_44_20`. Headline C1/C4 are
now ALSO on this deployed filter (see scorer-convention correction above); epoch-50 is sensitivity only.
- **C2 → yes (0.8)**: factor structure + diversity-decoupling now measured. Wording: "approximately orthogonal / decoupled", not "independent".
- **C3 → no / evidenced negative (0.85)**: at task-level aggregation, downstream success is NOT linearly predicted by any selection-feature block. Revised claim: "per-example filter diagnostics do not transfer straightforwardly to per-task downstream generalization; any misalignment signal is likely conditional, nonlinear, interaction-based, or obscured by task aggregation."
- **Framing impact**: the C3 null WEAKENS the specific thesis "filter is misaligned because it under-weights oracle gap". The safer paper thesis shifts from "we identified the missing feature" to "the filter's feature geometry is interpretable (C2), but downstream utility is not captured by simple per-task aggregates (C3) — and the descriptive misalignment (C1/C4/C5) + downstream underperformance (D1) are the load-bearing findings."

# Overall verdict
Paper is writable IF the central framing is narrowed. "Measured + supported": the filter
selects self-consistent/redundant teachers from a weak open-loop pool and discards the rare
beneficial-diverse ones; FLARE significantly underperforms GT-only. "Plausible but unproven":
that this misalignment is the CAUSE of the downstream failure.

# Honest framing to use
"We diagnose a structural mismatch between the latent teacher-selection signal and useful
supervision, and this mismatch is consistent with the observed downstream underperformance."
Do NOT yet claim the full causal mechanism as proven.

# Remaining work before writing (gates)
1. ~~C2 PCA/factor~~ — DONE: supported (yes, 0.8).
2. ~~C3 task-join~~ — DONE: evidenced negative (no). Report as a null finding, not a predictive subset.
3. Causal re-filter ablation: keep beneficial-diverse, drop jitter/redundant, retrain + paired eval (C4-causal) — the ONE remaining real experiment.
4. ~~Resolve gt_replay_54_10 0.8063 vs 0.7812 discrepancy~~ — RESOLVED 2026-09-30: two distinct training runs (different teacher set + LR), not noise. Cite only the canonical `suite_corrected` 0.8063.
