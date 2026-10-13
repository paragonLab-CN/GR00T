# Reviewer-Notice Analysis: LeWM Epoch 50 vs Default 100

**Question**: Will a senior ML reviewer (NeurIPS / ICML / CoRL level) notice that LeWM was trained to 50 of its 100 default epochs, and how serious a concern is this?
**Date**: 2026-10-12.
**Reviewer**: Codex MCP `gpt-6-astra` with reasoning effort `ultra` (deep-audit tier). Thread ID: `01a1303c-9180-78e3-b512-77bb95be4bfd`. Round 2 reply read successfully.
**Verdict**: 1-2 point concern (3 if undisclosed); not a rejection-level flaw; but the candidate's "epoch-agnostic" claim is *overclaimed* and must be tightened.

---

## TL;DR

- **Severity**: 1-2 NeurIPS score impact normally; 2-3 if undisclosed and unsupported.
- **Discoverability**: ~10-25% casual / 25-40% adversarial / >50% artifact-tracing reviewer.
- **Defensibility**: "do not retrain" is *defensible*; "Pol_full is scale-relative so epoch-agnostic" is *overclaim*.
- **Action**: add one cheap checkpoint-sensitivity cell + tighten wording.

---

## 1. Severity (Codex verdict)

> "max_epochs: 100 is a configured maximum, not proof that epoch 100 is necessary or optimal. Selecting an earlier checkpoint is normal if it was chosen independently of downstream results and performance has stabilized."

This becomes serious *only if* any of:
- epoch 50 was chosen after inspecting downstream outcomes,
- validation or teacher-quality metrics were still improving materially,
- later checkpoints substantially reorder selected windows,
- the paper implies training was completed or converged when it was not,
- the conclusions depend on an accidentally favorable checkpoint.

**None of these apply** in our case:
- epoch 50 was a fixed training-budget choice (50 epochs in `02_train_lewm.sh`), not post-hoc cherry-picking.
- The empirical sweep (epoch 20/30/50, 5000-window subset) shows sub-linear improvement.
- The Round-4 READY claim (Pol_full > Pol_base on LIBERO-Plus) is structural — it is a teacher-filter intervention, not a teacher-quality story.

→ Severity = **1-2 points**. Not a 5+ point rejection-level flaw. A senior reviewer would *request* a sensitivity result, not invalidate the paper.

---

## 2. Discoverability

- Casual reviewer reading only §3 of the paper: **~10%** chance of noticing (most won't trace into the appendix).
- Adversarial reproducibility reviewer: **25-40%**.
- Dedicated artifact evaluator tracing commands + checkpoints: **>50%**.

The paper must *proactively disclose* epoch 50 in §3, otherwise the discoverability rises (and so does the score impact, per the "undisclosed" clause in Codex's severity table).

---

## 3. Defensibility — "Scale-relative, therefore epoch-agnostic" is OVERCLAIMED

This is the key correction. Codex's reasoning:

> Pol_full effectively uses a task-conditional percentile $\nu_e(x) = \hat F_{e,t}(s_e(x))$, which is invariant only to monotone task-specific transformations of LeWM scores. **It is not invariant when further training reorders examples.**

> Two checkpoints can preserve the *count* of rejections (because $\tau_r$ is fit to match Pol_base's global reject rate) and the *count* of selections (because $K=309128$ is fixed) while substantially disagreeing on **which** windows are rejected/selected. **Counts are nearly useless evidence of robustness** because they are forced to agree by construction.

What this means for the candidate's analysis:
- "Pol_full is epoch-agnostic by construction" — **REMOVE this phrase** from the proposal.
- Correct claim: "Pol_full is invariant to within-task monotone rescaling of LeWM scores and empirically stable across the evaluated checkpoints."
- The Pol_full proposal's robustness argument must be **empirical** (the 20/30/50 sweep + a new post-50 cell), not theoretical.

This is a real semantic correction. The two predicates are scale-invariant *by construction* (per-task empirical-CDF and log-ratio), but they are *not* rank-invariant under arbitrary training perturbations — and ranking matters because that's what `safety_z`'s top-K uses.

---

## 4. Minimum-Cost Mitigation

Codex's recommendation: a **paired checkpoint-sensitivity ablation on the existing 20/30/40/50 sweep + ideally one post-50 checkpoint**. The 5K-window subset is sufficient subject to four conditions:

1. **Same windows** at each checkpoint (not random resamples).
2. **Compare retained fraction** $q = 309128 / 487880$, not literal $K$ on 5K examples.
3. **Report identity stability**, not just fixed-by-design counts:
   - within-task rank correlation / rank displacement
   - reject-set overlap / flip rate
   - top-$q$ overlap or Jaccard
   - per-task selection changes
   - audit-metric distribution (mean, p95, etc.)
4. **Show epoch-50-to-70 validation trajectory**, not just standalone `val_pred=0.00038`.

If 50 → 70 shows material movement, then the experiment has *failed* and training should continue. If the existing 20/30/40/50 sweep shows stable rank/set/audit, the arbitrary `max_epochs: 100` default stops being scientifically relevant.

**Cost correction from Codex**: producing an epoch-70 checkpoint requires ~22 hours of LeWM training (10 epochs ≈ 10-12 hours), then 30 minutes for the 5K-window inference. **The existing 20/30/40/50 checkpoints already permit an immediate inference-only sweep** (no new training needed). The new post-50 checkpoint is *more convincing*, not strictly necessary.

**Practical recommendation** (added on top of Codex's):

| Step | Action | GPU-hours | What it buys |
|------|--------|-----------|---------------|
| A | Re-score the 5K-window subset at **epoch 40** (existing checkpoint, no new training) and compare to epoch 50 sweep | ~30 min | mid-epoch sensitivity check |
| B | Train LeWM to **epoch 70** (~22 hours) | 22 h | post-50 evidence |
| C | Re-score 5K-window subset at epoch 70 | ~30 min | post-50 robustness |
| D | **Only if B+C shows material movement**: train to 100, full re-pipeline Stages A–G | ~128 h | final ship |

Step A is essentially free and immediately informative. Steps B-C are the *cheap test* that Codex recommends. Step D is the contingency.

---

## 5. Paper-Writing Guidance (Codex recommendation)

> Use one transparent sentence in §3, not only a footnote:
>
> *"We train LeWM for 50 epochs and use the epoch-50 checkpoint; the released upstream configuration specifies a maximum of 100 epochs."*

Then delimit the theoretical claim:

> *"Task-conditional empirical-CDF normalization is invariant to within-task monotone transformations of LeWM scores, but not to checkpoint-induced changes in window ordering."*

Add a compact checkpoint-sensitivity row in §5 with task-level details in appendix. **Report rank and membership agreement, not rejection rate or selected count alone.** A footnote can identify the exact checkpoint filename.

### Concrete paper-edit checklist (for our Round-4 proposal)

1. **`FINAL_PROPOSAL.md`** — find the phrase "epoch-agnostic by construction" or "scale-invariant" if present, replace with "invariant to within-task monotone rescaling, empirically stable across the evaluated checkpoints."
2. **§3 paper text** — add the disclosure sentence above.
3. **§5 paper text** — add a checkpoint-sensitivity row showing top-$q$ overlap between epoch 20 / 30 / 50 (using the existing sweep data).
4. **Appendix A1** — task-level rank correlation / flip rate detail.

---

## 6. Final Verdict

> **The work is not broken merely because training ended at epoch 50. Full retraining to epoch 100 is not presently justified. However, "ship Pol_full because it is epoch-agnostic" overclaims the theory. Ship after a paired checkpoint-sensitivity cell — or, if the existing 20/30/50 sweep already contains strong rank, set-overlap, and quality stability, present that evidence explicitly and add one post-50 checkpoint if feasible.**

What changes in our pipeline:

| Layer | Before | After |
|-------|--------|-------|
| `EPOCH50_VS_100_ANALYSIS.md` | "Pol_full is epoch-agnostic by construction" | "Pol_full is invariant to within-task monotone rescaling; empirical stability needs a sensitivity ablation" |
| `EXPERIMENT_PLAN.md` | No checkpoint-sensitivity block | Add Block 6: "Checkpoint-sensitivity ablation (epoch 20/30/40/50 + epoch 70 if produced)" |
| `EXPERIMENT_TRACKER.md` | No R020+ for sensitivity | Add R020–R023: 5K-window sweep comparison at epoch 40, 50, 70 |
| Paper §3 | (to be added) | Disclosure sentence |
| Paper §5 | (to be added) | Sensitivity row |
| Memory `safety-z-and-safety-r.md` | "epoch-agnostic" | "invariant to monotone rescaling; **NOT** rank-invariant" |

---

## 7. Cost-Adjusted Action Plan

| Item | Status | Compute | Owner |
|------|--------|---------|-------|
| Edit `FINAL_PROPOSAL.md`: replace "epoch-agnostic" with "scale-relative; empirically stable" | TODO | 0 | research-refine |
| Add R020: re-score 5K subset at epoch 40 (no new training) | TODO | 0.5 GPU-hr | run-experiment |
| Add R021: report rank/set overlap at {20, 30, 40, 50} from existing sweep | TODO | <0.1 GPU-hr | analysis |
| Train LeWM to epoch 70 | OPTIONAL (only if R020-R021 are insufficient evidence) | 22 GPU-hr | run-experiment |
| Add R023: 5K-window sweep at epoch 70 | OPTIONAL | 0.5 GPU-hr | run-experiment |
| Edit paper §3: disclosure sentence | TODO | 0 | paper-writing |
| Edit paper §5: sensitivity row | TODO | 0 | paper-writing |

**Net incremental cost**: ~1 GPU-hour for the *minimum* disclosure + sensitivity ablation; ~23 GPU-hours if a fresh epoch-70 cell is added.

---

## 8. Memory Pointer

This analysis updates the prior memory `epoch-selection-decision-2026-10-12.md`:

- The decision rule is unchanged: **keep epoch 50 by default, only bump if sensitivity ablation shows material movement.**
- Add to that memory the *theoretical correction*: Pol_full is **scale-invariant by construction** but **NOT rank-invariant**; the "epoch-agnostic" framing is overclaim.