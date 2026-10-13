# Pipeline Summary

**Problem**: optimize the FLARE-VLA teacher filter on LIBERO-Plus to raise student success rate, in service of a paper-grade core claim.
**Final Method Thesis**: replace the global-quantile outlier filter with a **fixed-budget top-K** teacher selection under two eligibility predicates — per-task component-badness (`safety_z`) and one shared cross-task log-ratio latent-regret constraint (`safety_r`) — calibrated independently on a small calibration subset, evaluated end-to-end on LIBERO-Plus under paired training seeds.
**Final Verdict**: READY (round 4, weighted overall 9.01/10).
**Date**: 2026-10-12.

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`.
- Review summary: `refine-logs/REVIEW_SUMMARY.md`.
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`.
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`.
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`.
- Round-by-round: `round-0-initial-proposal.md`, `round-1-review.md`, `round-2-refinement.md`, `round-3-refinement.md`, `round-4-refinement.md`, `round-4-review.md`, `score-history.md`.
- Round 2-3 raw reviewer transcripts: `raw-round-2-review.txt`, `raw-round-3-review.txt`, `raw-round-4-review.txt`.

## Contribution Snapshot
- **Dominant contribution**: fixed-budget task-relative teacher selection with a single shared latent-regret safety constraint, evaluated end-to-end on LIBERO-Plus under paired training seeds.
- **Optional supporting contribution**: per-(suite, task) diagnostic table with rescueability as a continuous moderator.
- **Explicitly rejected complexity**: no learned reward model / VLM judge; no RL fine-tuning; no per-task learnable threshold; no new forward pass; no architectural change to the student, scorer, or training budget.

## Must-Prove Claims
- **C1 (Main)**: Pol_full > Pol_base on LIBERO-Plus overall success rate, same accepted-sample budget.
  - 40-task sign test favoring Pol_full in ≥ 25 of 40 (predeclared).
  - 95% bootstrap CI excludes 0.
  - Easy-decile non-inferiority lower 95% bound > −1.5 pp.
- **C2 (Supporting)**: Pol_z ablation isolates `safety_r`'s contribution and rescueability moderates Pol_full vs Pol_z.

## First Runs to Launch
1. **R001** — Materialize Pol_full via `06c_refilter_teacher_quality_v3.py`; data-level audit must show `|E| ≥ K=309128`, deterministic reruns identical by hash, every task retains nonzero coverage. If `|E| < K`, invoke the predeclared Pol_z fallback immediately.
2. **R019** — Validate training inputs via `09_validate_training_inputs.py` to confirm Pol_full `teacher_quality_direct_records.pt` is consumed correctly by `08_build_distillation_dataset.py`.
3. **R004** — Reproduce Pol_base seed 0 (`08c_train_gr00t_flare_44_20.sh`) within ±2 pp of the master log to anchor baseline variance.

## Main Risks
- **`|E| < K`**: data-level failure (not enough latent-regret-eligible candidates). *Mitigation*: predeclared fallback to Pol_z as the primary method.
- **Spearman stability failure** at both ε values: cross-task-regret claim invalid. *Mitigation*: predeclared fallback to Pol_z as the primary method.
- **Training-variance > ±2 pp across seeds**: bootstrap CI may include 0. *Mitigation*: 3 paired seeds; report honestly as a negative ablation if it does.
- **Easy-decile non-inferiority fails**: report honestly as over-aggressive rejection on easy tasks.

## Next Action
- **Proceed to `/run-experiment`** with the first three runs (R001, R019, R004) as the immediate next steps. The full experiment tracker (`EXPERIMENT_TRACKER.md`) defines the order through R019.
