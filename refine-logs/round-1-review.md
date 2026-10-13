# Round 1 Review (GPT-6-Astra via Codex MCP, effort xhigh)

**Date**: 2026-10-12
**Thread**: 01a12fe3-84d6-7b83-998d-c1ffdb7fd95b
**Format**: 7-dimension rubric with CRITICAL/IMPORTANT/MINOR priority action items

## Dimension Scores

| Dimension | Score |
|-----------|-------|
| Problem-anchor fidelity | 7/10 |
| Technical soundness | 3/10 |
| Implementation concreteness and feasibility | 5/10 |
| Contribution focus and elegance | 6/10 |
| Novelty and significance | 3/10 |
| Validation and causal identifiability | 4/10 |
| Foundation-model-era appropriateness | 9/10 |
| **Overall** | **4.7/10** |

**Verdict: RETHINK** — Keep the problem and the no-new-model constraint; reformulate the filter around task-relative normalization + a single global latent-safety constraint.

## Drift Warning
NONE — fixes remain in scope. Adding a learned judge, retraining LeWM, modifying GR00T, or using downstream evaluation to learn thresholds would be drift.

## Modernization Opportunities
NONE — no architectural frontier primitive fits. Useful modernization is **statistical**: calibrated excess latent error, fixed-budget selection, uncertainty-aware paired evaluation.

## Simplification Opportunities (verbatim from reviewer)
1. Replace the three-part conjunction with one task-relative component score + one global latent-regret gate.
2. Remove the minimum-plus-IQR rule and rank OR entirely.
3. Match the baseline selection count so the paper studies curation quality rather than dataset volume.
4. Drop `B-nosuite-nooracle`; it is mislabeled or irrelevant.
5. Treat the per-task table as analysis, not a separate contribution.
6. Describe `(suite, task_index)` simply as a task key.

## Weak-Dimension Critiques (priority-tagged, expanded)

### CRITICAL — Oracle layer duplicates baseline
- The current `06_audit_teacher_quality.py` already thresholds `oracle_gap_mse` (line 831).
- The proposed Layer-1 + Layer-2 both re-introduce it → twice-filtered, novelty evaporates.
- **Fix**: drop oracle-gap from the per-group thresholds (it is one shared quantity across the dataset); make it a single global latent-safety check; state explicitly what is *new*.

### CRITICAL — Per-task oracle quantile cannot identify unreliable tasks
- A task-local p90 of `oracle_gap_mse` always accepts ~90% of that task even if every candidate is poor.
- **Fix**: replace with a **cross-task excess-latent-error** statistic
  \[
  r_i = \log \frac{\text{best\_mse}_i + \epsilon}{\text{oracle\_mse}_i + \epsilon}
  \]
  and a single **globally calibrated** threshold on \(r_i\). This lets difficult-but-valid tasks retain their support while globally untrustworthy tasks are still penalized.

### CRITICAL — `1.5 × IQR` and rank-OR are arbitrary and unstable
- Anchoring on the per-group minimum makes the threshold shrink with group size.
- The rank-≤-0.5 OR effectively shrinks the dataset.
- **Fix**: delete `best_group_relative` entirely. Collapse the per-component metrics into one task-relative **badness score**
  \[
  z_i = \max_k \widehat F_{g,k}(m_{ik})
  \]
  where \(\widehat F_{g,k}\) is the empirical CDF of metric \(k\) restricted to group \(g\), computed on the *training* audit split, with explicit ties / NaN / direction conventions. Accept iff \(z_i \le \lambda\) **and** the global latent-safety gate passes. Choose \(\lambda\) by bisection to **match the baseline's selection count** — fixed-budget comparison.

### IMPORTANT — Acceptance-floor safeguard is backwards
- `q ≤ 0.95` is an upper clamp on permissiveness, not a floor on acceptance.
- **Fix**: redefine as a *coverage* mechanism: require at least \(N_{\min}\) accepted samples per task among safety-qualified candidates, with a documented report of tasks where this is unattainable.

### IMPORTANT — Novelty framing is currently too weak
- "First combination of two heuristics" is not a venue-grade claim.
- **Fix**: reframe as **budget-constrained conditional calibration under a global latent-risk constraint** — the insight is *task conditioning removes nuisance-scale variance while a shared latent-regret constraint preserves cross-task trust calibration*. Make fixed-budget comparison *part of the method*.

### IMPORTANT — Validation/causal plan and compute estimate are inconsistent
- `B-nosuite-nooracle` is not the current baseline (which already uses oracle gap).
- Four policies with only two fine-tunes does not support causal claims.
- **Fix**: collapse to **three policies** — (P0) current global baseline; (P1) task-conditional component filter without latent-regret gate; (P2) task-conditional + one global latent-regret gate. Each evaluated with **paired training seeds**. Hierarchical bootstrap over training seed × task × eval rollout.

### IMPORTANT — Statistical plan underpowered
- A sign test over 40 task rates ignores eval-rollout variance and only *one* training seed → not a top-venue causal claim.
- **Fix**: hierarchical bootstrap (seed → task → rollout); report Wilcoxon / paired t-test on suite means as secondary summaries; use a **predeclared non-inferiority margin** on the easy decile.

### IMPORTANT — Target subset inconsistent / non-falsifiable clause
- Anchor says `<0.20`; validation says `<0.30` (12 tasks); "or document why no filter could rescue" is not falsifiable.
- **Fix**: freeze the diagnostic task set from the baseline pre-training; use a single threshold; define a **rescueability statistic** = count of candidates that *would pass* the global safety gate for that task. Low coverage is a legitimate diagnostic *but does not satisfy the success criterion*.

### MINOR — Implementation details
- Specify filtering *unit*: per window (current), per trajectory, or per candidate.
- Define rank semantics, NaN policy, metric direction, tie handling, and whether thresholds fit on a held-out split. (No fitting from LIBERO-Plus eval outcomes.)

## Score History Entry

| Round | Prob. Fidelity | Tech. Soundness | Impl. | Focus | Novelty | Validation | Frontier | Overall | Verdict |
|-------|----------------|------------------|-------|-------|---------|------------|----------|---------|---------|
| 1     | 7              | 3                | 5     | 6     | 3       | 4          | 9        | 4.7     | RETHINK |

## Raw reviewer response

<details>
<summary>Verbatim GPT-6-Astra response</summary>

```
[see Round 1 review output above in this conversation]
```

</details>
