# Planning Gate Check (Phase 2)

| Question | Answer |
|----------|--------|
| What is the final method thesis? | Replace the global-quantile outlier filter with a **fixed-budget top-K** teacher selection rule on (a) per-task component-badness `safety_z` and (b) one shared cross-task log-ratio latent-regret constraint `safety_r`, calibrated independently on a small calibration subset, evaluated end-to-end on LIBERO-Plus under paired training seeds. |
| What is the dominant contribution? | A fixed-budget, paired-seed, end-to-end LIBERO-Plus evaluation of a teacher-filter that combines per-task empirical-CDF rank normalization with a cross-task log-ratio latent-regret gate. |
| What complexity was intentionally rejected? | No learned reward model / VLM judge; no RL on filtered teachers; no per-task learnable threshold; no new forward pass; no architectural change to the student, scorer, or training budget. |
| Which reviewer concerns still matter for validation? | **I-1**: pre-flight data-level audit (`|E| ≥ K`, deterministic reruns, task coverage, checksums). **I-2**: Spearman check is a *sanity guard*, not proof — evidence must come from `Pol_z` vs `Pol_full` and rescueability moderation. **M-1..M-3**: minor quantile conventions, calibration-subset reuse acknowledgment, max-CDF correlation note. |
| Is a frontier primitive central, optional, or absent? | **Absent** — explicitly so. The latent world model is already trained and serves as an offline curation signal only. Adding a VLM judge / RL fine-tuning / diffusion would obscure the contribution. |

All five answers are crisp. Proceeding to Phase 3 (experiment-plan).
