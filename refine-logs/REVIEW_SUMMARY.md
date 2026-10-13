# Review Summary

**Problem**: optimize the FLARE-VLA teacher filter on LIBERO-Plus to raise student success rate.
**Initial Approach**: (vague) improve the global-quantile outlier filter in `06_audit_teacher_quality.py`.
**Date**: 2026-10-12.
**Rounds**: 4 / 5.
**Final Score**: 9.01 / 10.
**Final Verdict**: READY.

## Problem Anchor (Verbatim, Preserved Across All Rounds)
*(See `FINAL_PROPOSAL.md` "Problem Anchor" section — preserved verbatim from round 0 through round 4, with no semantic edits.)*

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|-------|-------------------------|------------------------------------------|---------|----------------|
| 1     | (a) Oracle gate duplicates baseline behavior; (b) per-task oracle quantile can't identify unreliable tasks; (c) 1.5×IQR + rank-OR gate is arbitrary/unstable; (d) clamp backwards; (e) novelty framing weak; (f) ablation matrix inconsistent with compute; (g) stat plan underpowered; (h) target subset (0.20 vs 0.30) inconsistent. | (Round 2 introduced task-relative component badness + cross-task latent-regret; fixed-budget bisection; 4-policy ablation; sign test on 40 dichotomies.) | Partial | τ_r-bisection collapses into latent-regret ranking; "first combination" still weak; rescueability exclusion non-falsifiable. |
| 2     | (a) τ_r definition collapses mechanism to latent-regret ranking; (b) equation ambiguous + r=0 edge unsafe; (c) Spearman one-sided; (d) Pol_z under-specified; (e) compute underestimated. | (Round 3 tightened: deterministic top-K, clip[] formula, two-sided Spearman, Pol_z definition, restated compute ~100-120 GPU-hours.) | Partial | τ_r definition still coupled to K (cannot be fit on 47k calibration subset while hitting K=309128); equation formatting still ambiguous; r=0 special case unsafe. |
| 3     | (a) τ_r collapes mechanism; (b) equation wrong + r=0 unsafe; (c) Spearman one-sided; (d) Pol_z under-specified. | (Round 4 fixed: τ_r fit by matching Pol_base marginal `large_oracle_gap` reject rate on calibration subset; exact clip[] equation with `max(best,oracle)` removing special case; two-sided Spearman with terminal-failure policy; Pol_z = TopK_{z_s}{prereq} exact definition.) | **YES** | (No CRITICAL remains; IMPORTANT items below are implementation hygiene, not blocking.) |
| 4     | (No CRITICAL raised.) | — | — | IMPORTANT-I1: data-level audit before launching 7 fine-tunes. IMPORTANT-I2: keep Spearman interpretation narrow. MINOR-M1/M2/M3: quantile conventions, calibration reuse, max-CDF dependence. |

## Overall Evolution
- **Concrete-ness**: from "improve the filter" (vague) → exact predicates with closed-formula equations; explicit calibration step; failure modes with terminal behavior.
- **Focus**: from three conjunctive gates (some redundant with the baseline) → two predicates with distinct, independent statistical roles.
- **Complexity removed**: the 1.5×IQR and rank-OR clauses; the un-falsifiable rescueability exclusion; the HOLDOUT-not-in-set terminology collision; the duplicated oracle-quantile.
- **Modern leverage**: explicitly defended as **statistical calibration of the existing latent signal**, not architectural expansion; reviewer concurred.
- **Drift correction**: the *safety-not-rescueable exclusion* in round 3 was the only drift risk; round 4 removed it and made rescueability an explanatory variable.

## Final Status
- **Anchor status**: preserved verbatim and substantively across all 4 rounds.
- **Focus status**: tight; one dominant contribution; one supporting diagnostic.
- **Modernity status**: appropriately frontier-aware (uses existing LeWM as a curation primitive, not a new architecture).
- **Strongest parts of final method**:
  - Independence of `safety_r` and `safety_z` roles.
  - Top-K deterministic budget control.
  - Per-task empirical-CDF rank normalization with a transparent max-aggregation.
  - Paired training seeds for honest causal attribution.
  - Predeclared non-inferiority margin on easy-decile tasks.
- **Remaining weaknesses** (carried into experiment plan as risks):
  - max-CDF score's dependence on within-task metric correlation is intentionally accepted but should be reported.
  - Calibration-subset reuse (in-sample vs out-of-sample CDF values) is acknowledged but may need empirical quantification.
  - The Spearman check is a sanity guard, not a proof — must be coupled with controlled Pol_z vs Pol_full comparison.

## Score Trajectory

| Round | Overall | Verdict |
|-------|---------|---------|
| 1     | 4.7     | RETHINK |
| 2     | 7.5     | REVISE  |
| 3     | 8.3     | REVISE  |
| 4     | **9.01**| **READY** |

## Raw Reviewer Responses (preserved)

- Round 1 review: see `refine-logs/round-1-review.md`.
- Round 2 review: see `refine-logs/raw-round-2-review.txt` (transcript).
- Round 3 review: see `refine-logs/raw-round-3-review.txt` (transcript).
- Round 4 review: see `refine-logs/round-4-review.md` and `refine-logs/raw-round-4-review.txt` (transcript).
