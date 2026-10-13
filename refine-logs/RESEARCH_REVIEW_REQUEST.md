# Research Review Brief: Will Reviewers Notice LeWM Half-Training (Epoch 50 vs Default 100)?

## 1. Context

This is a question on a *paper-grade* artifact (`FINAL_PROPOSAL.md` Round-4 READY in `refine-logs/`) about whether reviewers will raise the question:

> "Why didn't you train LeWM to its default 100 epochs? You stopped at 50."

The candidate's analysis (in `refine-logs/EPOCH50_VS_100_ANALYSIS.md`) recommends **keeping epoch 50** and not retraining. The user wants an adversarial reviewer-perspective check on whether that decision will be flagged as a weakness.

## 2. Hard Evidence (Verified in Repo)

- **LeWM official default**: `/root/autodl-tmp/le-wm/config/train/lewm.yaml` declares `trainer.max_epochs: 100`. This is the *out-of-the-box* default.
- **What we actually trained**: `/root/autodl-tmp/le-wm/logs/libero_plus_bs64_epoch50.log` runs `trainer.max_epochs=50` (overridden in `FLARE-VLA/commands/libero_plus/02_train_lewm.sh:40`).
- **The loss curve** (`EPOCH50_VS_100_ANALYSIS.md` Section 2.1):
  - epoch 1: 2.284, epoch 10: 0.115, epoch 50: 0.146
  - Decomposed: `val_loss ≈ val_pred + 0.09 × val_sigreg`
  - `val_pred` (predictor accuracy we care about) at epoch 10: 0.00127, at epoch 50: 0.00038 — still decreasing.
  - `val_sigreg` (sigma regularization) is what's driving total `val_loss` up.
- **The empirical sweep** at 5000-window subset:
  - epoch 20: `oracle_mse_mean = 0.006312`
  - epoch 30: `oracle_mse_mean = 0.004926` (−22%)
  - epoch 50: `oracle_mse_mean = 0.004168` (−15%)
  - The "improvement curve" is sub-linear; epoch 30 → 50 is roughly half of epoch 20 → 30.
- **The audit policy at epoch 50** (`FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_fixed_v2/teacher_quality_summary.json`):
  - 487880 candidates → 309128 selected (rate 0.6336)
  - Rejections: `cosine_mse_teacher_disagree: 47971`, `large_oracle_gap: 48788`, `xyz_distance_outlier: 48788`, `rot_distance_outlier: 48788`, `teacher_jerk_outlier: 31023`, `late_mse_outlier: 48788`, `gripper_mismatch_outlier: 9185`.

## 3. The Proposed Method (Round-4 READY)

The current paper proposal is a *teacher-data filter* (not a new model). It runs `06c_refilter_teacher_quality_v3.py` on epoch-50 scoring inputs and selects 309128 candidates via:
- **`safety_z`**: per-task empirical-CDF rank normalization over 5 components (`{xyz_mse, rot_mse, jerk_xyz, jerk_rot, late_mse}`) with `max` aggregation.
- **`safety_r`**: cross-task log-ratio latent-regret `clip[log((max(best_mse, oracle_mse)+ε)/(oracle_mse+ε)), 0, log(1001)]`.
- Top-K deterministic selection (K = 309128).

The Pol_full claim is **scale-relative** — Pol_full operates on rank orderings and ratios, not absolute values. The proposal claims it is robust to which LeWM epoch produced the inputs.

## 4. The Specific Reviewer-Concern Question

**Will a senior reviewer (NeurIPS / ICML / CoRL)** notice that LeWM was trained to 50 of 100 default epochs, and how seriously will they mark it?

The candidate's answer in `EPOCH50_VS_100_ANALYSIS.md` is "no, because the cost-benefit math says don't retrain, and the paper's intervention is the filter, not the upstream teacher." The user wants an adversarial check on that conclusion.

## 5. Specific Questions for the Reviewer

1. **Severity**: Is "LeWM stopped at half its default training" a NeurIPS-level concern (5+ score impact) or a minor stylistic issue (1-2 score impact)?

2. **Discoverability**: How likely is a reviewer to *notice* this? Would they read the appendix carefully enough to find `weights_epoch_50.pt` references and cross-check against `config/train/lewm.yaml: max_epochs: 100`?

3. **Defensibility**: The candidate's argument is "Pol_full is scale-relative, so it's epoch-agnostic." Is that argument strong enough to deflect, or does it invite "but did you verify on the recommended checkpoint?"

4. **Mitigation**: What is the *minimum-cost* experiment that, if added, would silence this concern?

5. **Paper-writing**: How should the paper acknowledge this without making it a *bigger* deal than it is? Should it be a sentence in §3, an ablation in §5, or a footnote?

## 6. What I Am Asking

Please act as a senior ML reviewer (NeurIPS / ICML / CoRL level). Take the candidate's analysis seriously but adversarially:

- **Find the path by which a reviewer would raise this issue.** Walk through what a reviewer reads, when, and what triggers the question.
- **Estimate the severity if raised.** Score impact on a NeurIPS 1-10 scale.
- **If you think the candidate's "scale-relative" defense is weak, say why.** Specifically attack the claim that Pol_full is epoch-agnostic by construction.
- **If you think the cost-benefit math is right, propose the *cheapest* experiment that would close the concern** (not necessarily epoch 100 — could be epoch 70, could be a 5000-window sweep, could be a single ablation cell).
- **Be brutally honest about whether this is a real concern or whether the candidate is being overcautious.**

Read the relevant files to verify, especially:
- `/root/autodl-tmp/le-wm/config/train/lewm.yaml` — the `max_epochs: 100` default.
- `/root/autodl-tmp/Isaac-GR00T/refine-logs/EPOCH50_VS_100_ANALYSIS.md` — the candidate's analysis.
- `/root/autodl-tmp/Isaac-GR00T/refine-logs/FINAL_PROPOSAL.md` — the paper proposal.
- `/root/autodl-tmp/Isaac-GR00T/FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_fixed_v2/teacher_quality_summary.json` — the audit outputs at epoch 50.
- `/root/autodl-tmp/le-wm/logs/libero_plus_bs64_epoch50.log` — the actual training log.

Report anything that is genuinely wrong, including in the candidate's analysis. If the work holds up, say so. Do not manufacture findings.