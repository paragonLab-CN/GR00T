# Research Findings

## 2026-09-30 — result-to-claim adjudication (FLARE-VLA teacher-selection anatomy)

**Verdict: ADJUDICATED** via Codex (gpt-5.5, xhigh; gpt-6-astra unavailable → fallback).
Integrity: pass (design WARN — the "teacher < GT" design flaw is the paper's finding, not a defect).
Evidence pre-check: 10/10 cited numbers verified present in source files. See `CLAIMS_FROM_RESULTS.md`.

| Claim | Verdict | Confidence |
|---|---|---|
| C1 filter selects self-consistency not improvement | yes | high |
| C2 138 features → few interpretable ~orthogonal axes | **yes** | 0.8 |
| C3 feature subset predicts downstream | **no (evidenced negative)** | 0.85 |
| C4 structural misalignment (beneficial-vs-harmful) | partial | medium |
| C5 ceiling bounded by open-loop pool | yes/partial | med-high |
| D1 FLARE < GT-only | yes | high |
| D2 gt_replay > gt_only | partial | medium |

## 2026-09-30 (update) — C2/C3 computed

Ran the two compute-only analyses on `teacher_scoring_v2` (LeWM epoch-9 scorer, N=487,880 — the
teacher set actually distilled into `gt_replay_44_20`), per user correction. Script
`FLARE-VLA/scripts/libero_plus/10_c2_c3_analysis.py`; outputs `c2_c3_analysis.{json,csv}`.

- **C2 → supported**: top-4 PCs = 70.1% variance, cleanly blocked (oracle-quality / divergence+dynamics /
  diversity / action-divergence); corr(candidate diversity, oracle gap)=0.056 ≈ 0. Wording softened to
  "approximately orthogonal / decoupled" (no ICA/MI test run).
- **C3 → evidenced negative**: joining 40-task selection summary to 44_20 per-task downstream success
  (107/112 eval tasks joined, 35 train tasks), NO feature block correlates significantly (all p>0.39;
  oracle_gap ρ=−0.03; exact-match subset flips oracle_gap positive). Not a join artifact (real downstream
  variance 0.45–1.0). This is a genuine null.

**Framing consequence (Codex)**: the C3 null weakens "the filter is misaligned *because it under-weights
oracle gap*". The defensible thesis shifts to: **the filter's feature geometry is interpretable (C2), but
its per-task aggregates don't predict downstream utility (C3 null) — while the descriptive misalignment
(C1/C4/C5) and the significant downstream underperformance (D1) remain the load-bearing findings.**

**Updated constraints**: do NOT claim a predictive feature subset or an oracle-gap reweighting prescription;
report C3 as a null. Only remaining gate before writing: C4-causal re-filter retrain + resolve the
0.8063/0.7812 discrepancy for D2.

## 2026-09-30 (update 2) — scorer-convention correction (user decision)

Confirmed via `distillation_dataset_v2/distillation_summary.json` that the teacher set fed to
`gt_replay_44_20` is `teacher_scoring_v2` = **LeWM epoch-9** scorer. **All headline C1/C4 numbers are
now unified to this deployed filter**; epoch-50 (`teacher_scoring_epoch50_v2`) demoted to sensitivity
analysis (§5.5). Recomputed + spot-verified from `teacher_scoring_v2/teacher_quality_audit.parquet`:

| Metric | epoch-9 (DEPLOYED, primary) | epoch-50 (sensitivity) |
|---|---|---|
| selected rate | 65.2% | 53.4% |
| GT beats all 4 (both) | 44.4% | 61.0% |
| GT vs best-cand MSE | 0.01127 vs 0.01155 (1.03×) | 0.00415 vs 0.01386 (3.3×) |
| beneficial diversity D≥P50/P75/P90 | 12.28% / 5.51% / 1.88% | — |
| beneficial kept by filter | 64.5% / 43.3% / **0%** | — |
| selected = redundant / jitter | 35.4% / 6.8% | — |

**Hook weakens but survives**: the punchy "GT 3.3× / 61%" is epoch-50-only; on the deployed filter GT is
~1.03× and beats all in 44%. The misalignment conclusion (filter admits mostly redundant copies, discards
the most-divergent valid paths, significant downstream underperformance D1) still holds. Constraint: never
cite the 3.3×/61% figure without labeling it epoch-50 sensitivity; primary numbers are epoch-9.

## 2026-09-30 (update 3) — D2 0.8063/0.7812 discrepancy RESOLVED

The two gt_replay_54_10 numbers are **two different training runs**, not the same model re-evaluated:
- **0.8063** = canonical `suite_corrected_gt_replay_54_10`: teacher data `distillation_dataset_v2`
  (from `teacher_scoring_v2`, epoch-9), LR 1e-4, mix 0.84375/0.15625.
- **0.7812** = stale `image_aug_gt_replay_54_10_rerun_20260911`: teacher data `distillation_dataset`
  (from `teacher_scoring`, pre-v2 selection), LR 1e-5, mix 0.85/0.15.
Both evals checkpoint-10000 fixed-noise seed 2026. The 2.5pt gap is real (different corrected teacher set +
10× LR). **Cite only the canonical `suite_corrected` 0.8063; the rerun_20260911 0.7812 is an older variant.**
The earlier audit note ("use 0.7812") had it backwards. D2 remains partial (p=0.422 NS, ±0.5pt seed noise).

**What is measured + supported**: the filter selects self-consistent/redundant teachers from a
weak open-loop same-model candidate pool; GT beats all 4 candidates in 61% of windows; selected
teachers are 41% redundant + 15,574 jitter + 0 beneficial-diverse; FLARE mix-0.12 significantly
underperforms GT-only (−4.2pt, p=3.4e-4).

**What is plausible but unproven**: that this misalignment is the *cause* of downstream failure.

**Routing**: mixed verdict (yes/partial/no across claims). Paper can proceed ONLY under narrowed
framing — "diagnose a structural mismatch consistent with underperformance", not "we proved the
mechanism". Gates to close: C2 (PCA compute), C3 (task-join compute), C4-causal (re-filter retrain),
D2 (resolve 0.8063/0.7812 discrepancy).

**Constraints for future attempts**: do NOT claim causal mechanism before the C4 re-filter ablation;
do NOT cite D2 without a variance bound; do NOT use the stale 0.7812 gt_replay_54_10 figure.
