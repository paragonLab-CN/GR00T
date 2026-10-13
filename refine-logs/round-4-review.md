# Round 4 Review (GPT-6-Astra via Codex MCP, effort xhigh)

**Date**: 2026-10-12
**Thread**: 01a12fe3-84d6-7b83-998d-c1ffdb7fd95b
**Format**: 7-dimension rubric with priority-tagged action items

## 7 Scores

| Dimension | Weight | Score |
|-----------|-------:|------:|
| Problem Fidelity | 15% | 9.8/10 |
| Method Specificity | 25% | 9.2/10 |
| Contribution Quality | 25% | 8.2/10 |
| Frontier Leverage | 15% | 9.5/10 |
| Feasibility | 10% | 9.0/10 |
| Validation Focus | 5% | 9.3/10 |
| Venue Readiness | 5% | 8.0/10 |
| **Overall** | — | **9.01/10** |

**Verdict: READY.**

## Drift Warning
NONE. Problem anchor preserved verbatim and substantively. The dominant contribution is sharp, not broad. The method is minimal; do not add per-task learned thresholds, a second scorer, or another acceptance gate.

## Simplification Opportunities
NONE further. Mechanism has distinct roles for each component; abort conditions are preferable to heuristic fallbacks.

## Modernization Opportunities
NONE. Frontier leverage is appropriate. No VLM judge / reward model / diffusion / RL is warranted.

## Remaining Action Items

### CRITICAL
None.

### IMPORTANT

**I-1. Materialize the data-level audit before launching 7 fine-tunes.**
- Verify `|E| ≥ 309128`.
- Verify exactly 309,128 unique records per policy.
- Verify all selected records satisfy their respective prerequisites.
- Verify Pol_full contains no `safety_r > τ_r` records.
- Verify deterministic reruns produce identical sample-ID hashes.
- Verify every task retains nonzero coverage.
- Checksum the calibration subset, τ_r, ε, and policy outputs.
- Failure of `|E| ≥ K` or the two-ε Spearman condition remains failure of the proposed full method; falling back to Pol_z is informative but not successful validation of the dominant claim.

**I-2. Interpret the Spearman check narrowly.**
- `|ρ| ≤ 0.10` is a sanity check against obvious denominator-driven behavior, not proof of cross-task calibration.
- Evidence for the latent constraint comes from Pol_z vs Pol_full comparison and the rescueability moderation analysis.

### MINOR

**M-1. Quantile computation conventions** — record whether τ_r is estimated over all calibration records or only prerequisite-qualified records; interpolation convention; baseline rejection rate uses matching calibration population. Prefer computing baseline rejection fraction and r_s quantile on the same prerequisite-qualified calibration population.

**M-2. Calibration-subset reuse** — state explicitly that calibration episodes are pooled back into the selection population; their CDF values are in-sample while the others are out-of-sample.

**M-3. max-CDF score** — its distribution depends on within-task metric correlations; this is intentional but should be acknowledged.

## Score Evolution

| Round | Prob. Fidelity | Tech. Soundness / Method Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Overall | Verdict |
|-------|-----------------|--------------------------------------|--------------|----------|-------------|------------|-------|---------|---------|
| 1     | 7               | 3                                    | 3            | 9        | —           | 4          | —     | 4.7     | RETHINK |
| 2     | 8.0             | 7.5                                  | 7.0          | 9.0      | 6.5         | 7.5        | 6.0   | 7.5     | REVISE  |
| 3     | 9.5             | 7.5                                  | 8.0          | 9.5      | 8.0         | 9.0        | 7.0   | 8.3     | REVISE  |
| 4     | 9.8             | 9.2                                  | 8.2          | 9.5      | 9.0         | 9.3        | 8.0   | **9.01**| **READY** |

## Raw reviewer response

<details>
<summary>Verbatim Round 4 review (truncated for brevity; full text archived)</summary>

See `refine-logs/raw-round-4-review.txt` for the verbatim transcript from Codex MCP thread `01a12fe3-84d6-7b83-998d-c1ffdb7fd95b`.
</details>
