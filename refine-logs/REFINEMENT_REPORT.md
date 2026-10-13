# Refinement Report

**Problem**: optimize the FLARE-VLA teacher filter on LIBERO-Plus to raise student success rate, in service of a paper-grade core claim.
**Initial Approach**: (vague) replace `06_audit_teacher_quality.py`'s global-quantile outlier policy.
**Date**: 2026-10-12.
**Rounds**: 4 / 5.
**Final Score**: 9.01 / 10.
**Final Verdict**: READY.

## Problem Anchor
*(Verbatim across all rounds; see `FINAL_PROPOSAL.md`.)*

## Output Files
- Review summary: `refine-logs/REVIEW_SUMMARY.md`.
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`.
- Round-by-round:
  - `refine-logs/round-0-initial-proposal.md` — initial proposal (vague novelty, duplicated oracle gate, three conjunctive gates, four-policy ablation with two fine-tunes).
  - `refine-logs/round-1-review.md` — first cross-model review.
  - `refine-logs/round-2-refinement.md` — round-2 revision (task-relative + cross-task log-ratio, bisection λ).
  - `refine-logs/raw-round-2-review.txt` — round-2 review transcript.
  - `refine-logs/round-3-refinement.md` — round-3 revision (deterministic top-K, clip[] formula, two-sided Spearman, removed rescueability exclusion).
  - `refine-logs/raw-round-3-review.txt` — round-3 review transcript.
  - `refine-logs/round-4-refinement.md` — round-4 revision (independent τ_r by Pol_base reject-rate matching; max(best, oracle) in clip[] formula; two-sided Spearman with terminal failure; Pol_z exact spec).
  - `refine-logs/round-4-review.md` — round-4 review summary.
  - `refine-logs/raw-round-4-review.txt` — round-4 review transcript.

## Score Evolution

| Round | Prob. Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|----------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 7.0            | 3.0                | 3.0                  | 9.0               | —           | 4.0              | —               | 4.7     | RETHINK |
| 2     | 8.0            | 7.5                | 7.0                  | 9.0               | 6.5         | 7.5              | 6.0             | 7.5     | REVISE  |
| 3     | 9.5            | 7.5                | 8.0                  | 9.5               | 8.0         | 9.0              | 7.0             | 8.3     | REVISE  |
| 4     | 9.8            | 9.2                | 8.2                  | 9.5               | 9.0         | 9.3              | 8.0             | 9.01    | READY   |

Weights: Problem Fidelity 15%, Method Specificity 25%, Contribution Quality 25%, Frontier Leverage 15%, Feasibility 10%, Validation Focus 5%, Venue Readiness 5%.

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|-------------------------|------------------|--------|
| 1     | Three CRITICAL: (a) oracle gate duplicates baseline; (b) per-task oracle quantile unreliable; (c) 1.5×IQR + rank-OR arbitrary. Plus four IMPORTANT. | Replaced quantile-by-task with task-relative empirical-CDF; replaced duplicated oracle gate with single global cross-task log-ratio latent regret; dropped four-policy ablation to three; added paired seeds {0,1,2}; sign-test + hierarchical bootstrap; rescueability diagnostic. | Partial — fixed some, but bisection τ_r still coupled to budget; equation ambiguity remained; "first combination" framing still weak. |
| 2     | Two CRITICAL: (a) τ_r definition collapses method; (b) equation ambiguous + r=0 edge unsafe. Plus three IMPORTANT. | Switched to deterministic top-K (τ_r only gates eligibility); adopted clip[] formula with explicit edge policy; two-sided Spearman; explicit Pol_z definition; restated compute. | Partial — fixed some, but Round 3 still tied τ_r to K feasibility, equation formatting still ambiguous, near-zero-oracle special case still present. |
| 3     | Two CRITICAL (same areas) and two IMPORTANT. | τ_r now independent (matches Pol_base reject rate on calibration); max(best,oracle) instead of best (no r=0 special case); two-sided Spearman with terminal failure; max-CDF as conjunctive "worst component" criterion. | **Resolved** — overall 9.01, READY. No CRITICAL remains. |
| 4     | (None CRITICAL.) IMPORTANT-I1: pre-flight data audit; IMPORTANT-I2: keep Spearman interpretation narrow. MINOR quantile/reuse/correlation conventions. | (None, since CRITICAL closed.) | READY. |

## Final Proposal Snapshot
- Canonical clean version: `refine-logs/FINAL_PROPOSAL.md`.
- Dominant contribution in one sentence: *fixed-budget task-relative teacher selection with a single shared latent-regret safety constraint, evaluated end-to-end on LIBERO-Plus under paired training seeds*.
- Distilled in 5 bullets:
  1. Per-task empirical-CDF rank normalization over `M = {xyz_mse, rot_mse, jerk_xyz, jerk_rot, late_mse}`, aggregated by `max`.
  2. Cross-task log-ratio latent regret `r_s = clip[log((max(best, oracle) + ε)/(oracle + ε)), 0, log(1001)]` with `ε = 10^-3`.
  3. τ_r fit on a calibration subset (1-of-10 episode-hashed partitions) to match Pol_base's marginal `large_oracle_gap` reject rate.
  4. Deterministic top-K = 309128 selection on safety_z (eligible set for Pol_full; prerequisite set for Pol_z).
  5. Pol_base × Pol_full paired over seeds {0,1,2}; 40-task sign test (primary); per-suite paired t-test (secondary); easy-decile non-inferiority vs −1.5 pp; rescueability reported as continuous moderator.

## Method Evolution Highlights
1. **Most important simplification (round 2 → 4)**: collapse three conjunctive gates into two predicates, each with independent statistical role; remove duplicated oracle-quantile from L1.
2. **Most important mechanism upgrade (round 3 → 4)**: deterministic top-K replaces bisection-bounded λ; the budget is now guaranteed *by construction* and `safety_r`/`safety_z` roles are no longer muddied.
3. **Most important modernization framing**: defended the method as **disciplined offline calibration of the existing latent signal**, not architectural expansion — explicitly corroborated by the cross-model reviewer.

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 2 | "safety-not-rescueable exclusion lets method remove its own likely failures." | Removed exclusion in round 3; rescueability became a moderator; all 40 tasks back in primary test. | Accepted. |
| 3 | "τ_r fit on 47k calibration subset cannot itself contain K=309128 eligible samples." | Restated τ_r as an *independent* statistical fit (matching Pol_base reject rate), not a budget-enforcing mechanism. | Accepted (round 4 fix). |
| 3 | "near-zero oracle ⇒ r=0 makes an arbitrarily bad teacher eligible." | Replaced with `max(best, oracle)` in clip[] numerator — no special case needed. | Accepted (round 4 fix). |
| 3 | "Spearman one-sided permits ρ=-0.8." | Tightened to two-sided `|ρ| ≤ 0.10` with terminal failure. | Accepted (round 4 fix). |

## Remaining Weaknesses (Honest)
- **max-CDF dependence on within-task metric correlation**: conjunctive "worst component" criterion; defensible but not uniformly distributed — should be reported.
- **Calibration subset reuse**: calibration episodes are pooled back into the selection population with in-sample CDF values; acknowledged but may need empirical quantification.
- **Spearman check is a sanity guard, not proof**: evidence for the latent constraint must come from Pol_z vs Pol_full comparison and rescueability moderation.

## Next Steps
- If READY (current state): proceed to `EXPERIMENT_PLAN.md` (already integrated below as part of this pipeline run).
- If REVISE: address remaining mechanism weaknesses; re-run `research-refine`.
- If RETHINK: revisit core mechanism; consider `/idea-creator` for parallel framings.

## Raw Reviewer Responses

<details>
<summary>Round 1 Review</summary>
See `refine-logs/round-1-review.md`.
</details>

<details>
<summary>Round 2 Review</summary>
See `refine-logs/raw-round-2-review.txt`.
</details>

<details>
<summary>Round 3 Review</summary>
See `refine-logs/raw-round-3-review.txt`.
</details>

<details>
<summary>Round 4 Review</summary>
See `refine-logs/round-4-review.md` and `refine-logs/raw-round-4-review.txt`.
</details>
