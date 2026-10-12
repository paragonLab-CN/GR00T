verdict: ADJUDICATED
date: 2026-10-09
judge: codex (gpt-5.5, xhigh) — gpt-6-astra unavailable (capability error: requires newer Codex CLI), fell back per reviewer-routing
integrity_status: pass  (EXPERIMENT_AUDIT.json integrity_status=pass; design WARN is the paper's finding)
evidence_precheck: 12/12 cited numbers verified to exist in source files (full-precision) — .aris/evidence_precheck.json
threadId: 01a11ffa-4279-73f3-9f87-88b242a02f1b
scope: FOUR NEW downstream claims only (C4-causal, D2-overturn, D3, D4). Prior audit-side claims C1–C5, D1 adjudicated 2026-09-30 (see below, unchanged).

# Per-claim verdicts (2026-10-09, controlled four-model comparison)

| Claim | claim_supported | confidence | one-line |
|---|---|---|---|
| C4-causal (remove xyz_distance_outlier → improves downstream) | **partial** | medium | v3_no_xyz 0.8295 vs deployed 0.8125 = +1.7pt, McNemar p=0.084, single seed. Directional, NOT statistically conclusive. Teacher-set comparison is clean (only the filter rule differs). |
| D2-overturn ("teacher-augmented beats GT-only") | **yes (overturn the positive claim)** | high | Controlled: all three teacher models within ±1.1pt of recipe-matched GT-only, none significant, only v3_no_xyz numerically above. "Not supported as improving over GT-only." BUT teacher-vs-control boundary is aug-confounded. |
| D3 (remove xyz → best model + lifts Robot Initial States) | **partial** | medium | Descriptively true: v3_no_xyz best overall (0.8295), best RIS (0.3438 vs 0.294–0.300). But single-run ranking, not robust; overall pairwise NS; RIS needs paired/category-level significance. |
| D4 (recipe lr=1e-4+no-aug is a confound > any teacher effect) | **yes (descriptive)** | medium-high | gt_only 0.7964→0.8232 = +2.68pt (two levers changed together) vs largest teacher contrast +1.7pt (NS). Individual LR vs aug contributions NOT separable. |

# Codex wording (verbatim, per-claim)

## C4-causal → partial (medium)
- what_results_support: v3_no_xyz numerically better than deployed_44_20: 0.8295 vs 0.8125 (+1.7pt). McNemar 64 vs 45, net +19, p=0.084 — meaningful directional evidence, especially because the teacher-set comparison is clean: same epoch-9 scorer, same launcher, same mix, same augmentation, same eval manifest/seeds, differing only by removal of xyz_distance_outlier.
- what_results_dont_support: does NOT support a settled causal claim. p=0.084 not conventionally significant, one fixed diffusion-noise seed, effect size within the ~0.5–1pt seed-noise band plus only modestly larger.
- missing_evidence: repeated eval seeds for both v3_no_xyz and deployed; ideally repeated training seeds; category-level paired stats for Robot Initial States; CIs or hierarchical model over task/episode pairing.
- suggested_claim_revision: WEAKEN. "Removing xyz_distance_outlier produced a directional downstream improvement over the deployed filter in the controlled comparison, but the evidence is not yet statistically conclusive."
- next_experiments_needed: 3–5 diffusion seeds for v3_no_xyz and deployed; preferably 2–3 training seeds; report paired McNemar aggregated across seeds and task-level mixed-effects or bootstrap intervals.

## D2-overturn → yes (high)
- what_results_support: controlled results do NOT support "teacher-augmented models beat GT-only". All three teacher models within ±1.1pt of gt_only control, none significant, only v3_no_xyz numerically above control (0.8295 vs 0.8232, McNemar p=0.595). deployed and epoch50 both numerically below control.
- what_results_dont_support: does NOT prove teacher augmentation useless/harmful. GT-only control differs from teacher runs in BOTH teacher presence AND appearance augmentation (teacher models have blur aug, GT-only does not) → teacher-vs-no-teacher boundary partly confounded.
- missing_evidence: a truly recipe-identical GT-only control with the SAME augmentation as teacher models; teacher models without augmentation; repeated training/eval seeds.
- suggested_claim_revision: OVERTURN the positive claim. "Under the controlled four-model comparison, teacher augmentation is not supported as improving over GT-only; any advantage is absent or inconclusive."
- next_experiments_needed: run GT-only with the exact same augmentation and recipe as teacher models; optionally all four variants across multiple training seeds.

## D3 → partial (medium)
- what_results_support: descriptively yes — v3_no_xyz best overall (0.8295) and best RIS (0.3438 vs 0.2938–0.3000). RIS lift aligned with the hypothesis that retaining more divergent-but-valid teachers helps the hardest/headroom category.
- what_results_dont_support: "yields the best controlled model" supported only as a descriptive single-run ranking, not a robust performance conclusion. Overall pairwise NS; task-level sign tests do not separate; RIS needs paired/category-level significance and seed replication.
- missing_evidence: RIS episode counts, paired RIS McNemar or bootstrap CIs, repeated eval seeds, repeated training seeds, task-level breakdown showing whether gains are broad or concentrated.
- suggested_claim_revision: REFRAME. "In this single controlled run, v3_no_xyz ranked best overall and showed the largest Robot Initial States score, but the advantage is not yet statistically established."
- next_experiments_needed: repeat eval seeds for all models; compute per-category paired stats; inspect whether RIS gains persist across seeds and tasks.

## D4 → yes (medium-high)
- what_results_support: as a descriptive finding, the combined recipe change is larger than any measured teacher effect: GT-only 0.7964→0.8232 (+2.68pt) vs largest teacher contrast v3_no_xyz vs deployed +1.7pt (NS); all teacher-control contrasts NS. Supports "training recipe is a major confound relative to the observed teacher effects".
- what_results_dont_support: does NOT identify which recipe lever caused the gain. LR changed 1e-5→1e-4 AND appearance augmentation changed on→off; individual effects not separable from the provided evidence.
- missing_evidence: factorial controls: old LR/no aug, new LR/aug, new LR/no aug, old LR/aug; ideally with GT-only and at least deployed/v3 teacher conditions.
- suggested_claim_revision: KEEP, phrase carefully. "The combined training-recipe change is a larger descriptive confound than any measured teacher effect; the separate effects of LR and augmentation remain unresolved."
- next_experiments_needed: run a 2×2 LR × augmentation ablation on GT-only first, then repeat the best recipe against deployed and v3_no_xyz teachers.

# Consequence for the paper
- The C4-causal claim is NOT yet a proven causal mechanism. The honest framing remains "diagnose a structural mismatch consistent with downstream underperformance", now with a DIRECTIONAL (not significant) causal datapoint from the v3_no_xyz retrain.
- D2 must be reframed: teacher augmentation does NOT beat the recipe-matched GT-only control under the controlled comparison.
- D4 is a new, load-bearing caveat: the training recipe (lr + augmentation) is a confound larger than any teacher effect — earlier cross-recipe comparisons overstated the teacher benefit.
- The aug asymmetry between the GT-only control (no aug) and the three teacher models (aug) is a real limitation on the "teacher vs no-teacher" boundary.

# Remaining gates before writing
1. Confirm C4-causal: ≥2 diffusion seeds (3–5 recommended) for v3_no_xyz AND deployed_44_20; aggregate McNemar across seeds. Single-seed p=0.084 is not enough.
2. Disentangle D4 recipe confound: 2×2 LR × augmentation ablation (at least on GT-only).
3. Resolve the GT-only aug asymmetry: a GT-only run with the SAME augmentation as the teacher models, so "teacher vs no-teacher" is aug-matched.
4. Per-category (Robot Initial States) paired stats for D3.
