# Initial Experiment Results — R001 / R002 / R003 (M0)

**Date**: 2026-10-12
**Plan**: `refine-logs/EXPERIMENT_PLAN.md` (Round-4 READY method thesis)
**Code**: `FLARE-VLA/scripts/libero_plus/06c_refilter_teacher_quality_v3.py` (1216 lines)
**Wrapper**: `FLARE-VLA/commands/libero_plus/06c_refilter_teacher_quality_v3.sh`
**Output**: `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/`
**Pipeline stage**: M0 (sanity) — pre-training filter materialize + downstream-validity check

## M0 Results (R001 / R002 / R003)

### R001 — Pol_full + Pol_z materialize + data audit
- **Status**: PASS
- **Pol_full selected**: 309,128 (= K; Pol_base's selected_for_distillation.count).
- **Pol_z selected**: 309,128 (= K).
- **Pol_base selected**: 309,128 (read from Pol_base audit; same K by design).
- **n_prereq**: 431,721 (88.5% of 487,880).
- **n_safety_r_pass**: 441,209 (90.4%; safety_r is a *coverage* gate, not an exclusion).
- **n_eligible** = prereq ∧ safety_r_pass: 388,319 → ≥ K=309,128. ✓ Budget feasible.
- **τ_r**: 0.147116 (calibrated on calibration subset to match Pol_base's 10% large_oracle_gap reject rate).
- **Spearman ρ(r_s, oracle_mse_s)** on calibration (prereq-qualified): **0.0628** ≤ 0.10. ✓ Sanity guard passes; cross-task-regret claim valid.
- **Calibration subset**: 48,467 samples (1 of 10 episode-hashed buckets; pool back to 487,880 for selection).
- **Per-task rate std** (the main paper claim): Pol_base = **0.0931**, Pol_z = **0.0198**, Pol_full = **0.0276**. Pol_full shrinks cross-task heterogeneity by ~3.4× vs Pol_base.
- **Min per-task coverage**: Pol_full min = 0.578 (libero_spatial task 6) vs Pol_base min = 0.452 (libero_spatial task 32). No task collapses to zero.

### R002 — Downstream dry-run validation
- **Status**: PASS
- Used `08_build_distillation_dataset.py --dry-run` with v3-compat sidecar (`teacher_quality_filtered_records_v3.pt` + `teacher_quality_summary_v3_compat.json`).
- Downstream output (dry-run):
  - `index_mode = teacher_only`
  - `selected teacher samples = 309128`
  - `teacher action shape = (309128, 8, 7)`
  - `processor action horizon = 16`, `model action horizon = 40`
  - `status = PASS (dry-run; no files written)`
- Then a real (non-dry-run) build to `FLARE-VLA/outputs/libero_plus/distillation_dataset_v3/`:
  - `distillation_index.jsonl`, `teacher_index.jsonl`, `teacher_records_enriched.pt`, `distillation_summary.json` all written.
  - `status = PASS`. The refiltered teacher dataset is **drop-in compatible** with the existing training pipeline.

### R003 — Calibration diagnostics
- `calibration_subset_diagnostics.json`:
  - n_partitions=10, calibration_bucket=0, n_calibration=48,467, n_calibration_prereq=42,760.
  - n_reject_target = round(0.1 × 42,760) = 4,276.
  - τ_r = 0.147116, ε_used = 0.001.
  - spearman_passed = true, spearman_rho = 0.0628.
  - pol_base_large_oracle_gap_reject_rate = 0.10.
  - Pol_full leaves |E|=388,319 eligible samples (24% over K — enough slack for the top-K to be well-conditioned).

### R004 — Deterministic-rerun check
- Two independent runs of `06c_refilter_teacher_quality_v3.py` produce **identical SHA-256** for `teacher_quality_direct_records.pt`: `f598d60caff1688c106dbf7f498634fc35d9f3f0af0b50de59b9961fbed9d12c`.
- Same hash for v3-compat PT: `220161f2392de42718b008a51c31eb3609673f8462d4d967db819013feaafaaa`.

## Summary
- 3 of 3 must-run sanity checks PASSED.
- Main result so far (data-side, pre-training): Pol_full is more uniform across tasks (std 0.028 vs Pol_base 0.093) while accepting the same 309,128 samples. The τ_r is small (0.147) and the Spearman sanity check passes.
- **Not yet**: student success rate on LIBERO-Plus (needs M1-M5 runs).
- **Ready for Phase 4 (auto-deploy checkpoint)**: yes.

## Next Step
- **AWAITING USER APPROVAL** to launch M1 (Pol_base seed 0 reproduce) and M2 (Pol_full × {0,1,2} fine-tunes).
- Total estimated GPU-hours for M1-M5: **100-120** on a single 4090/A100.
