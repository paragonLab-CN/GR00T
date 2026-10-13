# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Materialize Pol_full + Pol_z; data audit | 06c_refilter_teacher_quality_v3.py | full 487880 | n_prereq, n_eligible, τ_r, Spearman | MUST | **DONE** | K=309128, τ_r=0.147116, ρ=0.0628, hash f598d60c… |
| R002 | M0 | Downstream dry-run + non-dry-run build | 08_build_distillation_dataset.py | full 487880 | format PASS, status PASS | MUST | **DONE** | distillation_dataset_v3/ built; v3-compat sidecar works |
| R003 | M0 | Calibration diagnostics | analysis | 48,467 cal subset | Spearman, ε sensitivity | MUST | **DONE** | calibration_subset_diagnostics.json |
| R004 | M0 | Deterministic rerun check | rerun | full 487880 | hash match | MUST | **DONE** | hash matches across two runs |
| R019 | M0 | Validate training inputs | 09_validate_training_inputs.py | full 487880 | pass/fail | MUST | TODO | run before R004 training |
| R005 | M1 | Pol_base seed 0 reproduce | 08c_train_gr00t_flare_44_20.sh seed=0 (existing Pol_base) | 20k steps | training loss, eval | MUST | TODO | anchor baseline; should match master log within ±2 pp |
| R006 | M2 | Pol_full seed 0 | 08c (Pol_full v3-compat teacher file) | 20k steps | training loss, eval | MUST | TODO | paired with R005 |
| R007 | M2 | Pol_full seed 1 | 08c (Pol_full) | 20k steps | same | MUST | TODO | paired-seed |
| R008 | M2 | Pol_full seed 2 | 08c (Pol_full) | 20k steps | same | MUST | TODO | paired-seed |
| R009 | M3 | Pol_z seed 1 ablation | 08c (Pol_z teacher file) | 20k steps | training loss, eval | MUST | TODO | ablation |
| R010-R015 | M4 | LIBERO-Plus eval × {Pol_base s0, Pol_full s0,1,2, Pol_z s1} | `gr00t/eval/run_gr00t_server.py` (same invocation as `gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20_checkpoint_10000_full10030`) | 10030 × 1 | success rate | MUST | TODO | per-task CSV — output format `summary_all.json` + `task_success_rates_all.csv` |
| R016 | M5 | Hierarchical bootstrap + sign test + non-inferiority | analysis | all R010-15 | 95% CI, sign-test p | MUST | TODO | primary inferential analysis |
| R017 | M5 | Rescueability moderator regression | analysis | per-task | coef + 95% CI | MUST | TODO | predeclared moderator |
| R018 | Block 5 | max-CDF intra-task correlation diagnostic | analysis | cal subset | per-suite correlation | MUST | TODO | Appendix A1 |

**Total GPU-hours scheduled**: ≈ 105-130 hours (1 × Pol_base + 3 × Pol_full + 1 × Pol_z + eval ≈ 110). Sequential on one 4090/A100; ≈ 5 days wall clock.

**First three runs to launch (next stage, M0 → M1)**:
1. **R019** — Validate training inputs (`09_validate_training_inputs.py`) against Pol_full's v3-compat PT.
2. **R005** — Reproduce Pol_base seed 0 to anchor baseline variance.
3. **R006** — Pol_full seed 0 (paired with R005).

**Outputs produced by R001-R004** (paths and SHA-256s):
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/teacher_quality_direct_records.pt` (sha256 `f598d60c…`)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/teacher_quality_filtered_records_v3.pt` (sha256 `220161f2…`)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/teacher_quality_summary.json`
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/teacher_quality_summary_v3_compat.json` (sha256 `deab310b…`)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/teacher_quality_selected.jsonl` (sha256 `9a0c8011…`)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/teacher_quality_selected_pol_z.jsonl` (sha256 `ee37fd99…`)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/task_diagnostic_table.csv` (sha256 `44263901…`)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v3_budget/calibration_subset_diagnostics.json` (sha256 `55c49bc7…`)
- `FLARE-VLA/outputs/libero_plus/distillation_dataset_v3/` (full distillation sidecar built)
