verdict: ADJUDICATED
date: 2026-10-09 (run02 — RE-ADJUDICATION with gpt-6-astra after Codex CLI 0.152.1 → 0.162.0 upgrade)
judge: codex (gpt-6-astra, reasoning effort ultra) — invoked directly via upgraded CLI (`codex exec --model gpt-6-astra -c model_reasoning_effort='"ultra"'`), NOT via the MCP stdio channel
session_id: 01a12051-be23-7c53-8ac6-161a4d659d29
integrity_status: pass  (EXPERIMENT_AUDIT.json integrity_status=pass; design WARN is the paper's finding)
evidence_precheck: 12/12 cited numbers verified to exist in source files (full-precision) — .aris/evidence_precheck.json
scope: FOUR downstream claims (C4-causal, D2-overturn, D3, D4) from the controlled four-model comparison. Audit-side claims C1–C5, D1 adjudicated 2026-09-30 (gpt-5.5), unchanged.
supersedes: 20261009_run01 (gpt-5.5 xhigh). Verdicts are CONSISTENT in direction; astra is STRICTER on D3/D4 wording (descriptive-only) and adds two new audit points (v3 teacher-count reconciliation; seed-noise is NOT a calibrated CI).

# Per-claim verdicts (gpt-6-astra, ultra)

| Claim | claim_supported | confidence | one-line |
|---|---|---|---|
| C4-causal | **partial** | high | +1.70pt (v3 0.8295 vs deployed 0.8125), McNemar net +19 p=0.084, single seed → directional evidence favoring rule-removal, but NOT a reliable causal improvement. Also flags: why does removing a rejection rule yield FEWER teachers (284,320 vs 317,941)? Needs reconciliation. |
| D2-overturn | **yes** | high | Overturn to "teacher superiority is NOT supported under this comparison." Only v3 numerically exceeds GT-only (+0.63pt, p=0.595); none demonstrates superiority. BUT does NOT establish equivalence/harm, and teacher-vs-GT is aug-confounded (GT-only aug off, teachers aug on). |
| D3 | **yes — strictly as a description of the reported evaluation** | high | v3 has highest OBSERVED overall (0.8295) and RIS (0.3438) success. Reporting observed rankings needs no significance. But "best" ≠ reliably superior across runs/seeds; "lifts" ≠ established category-level causal benefit. |
| D4 | **yes — as a descriptive historical comparison** | medium | GT-only 0.7964→0.8232 = +2.68pt exceeds the largest observed teacher gap (1.70pt) and every teacher-vs-control gap (≤1.07pt). Does NOT separate LR vs aug, NOT a causal recipe effect, NOT proof the true recipe effect exceeds the true teacher effect. Confidence medium because older-baseline comparability is incompletely specified. |

# Codex (gpt-6-astra) wording — verbatim

## C4-causal → partial (high)
- support: within the identically trained+augmented teacher models, v3 exceeds deployed by 19/1120 = 1.70pt — directional evidence favoring rule-removal.
- dont_support: reliable causal improvement NOT established by single eval seed + two-sided McNemar p=0.084. Being the observed winner and leading RIS do not independently establish that. Conversely, non-significance does NOT establish no effect.
- missing: replicated training AND eval seeds, uncertainty estimates, confirmation of the rule-only teacher-generation change. **Explain why removing a rejection rule produces FEWER teachers (284,320 vs 317,941) — selection-order interactions could explain; the counts warrant reconciliation, not automatic design rejection.**
- revision (WEAKEN): "Removing xyz_distance_outlier produced a directional downstream improvement of 1.70 points in the seed-2026 evaluation, but a reproducible benefit remains unconfirmed."
- next: repeat deployed vs no-xyz with independently trained replicas + matched diffusion-seed evals; prespecify this primary contrast; report paired CIs; audit candidate-pool provenance and teacher-set differences.

## D2-overturn → yes (high)
- support: only v3 numerically exceeds GT-only (+0.63pt, p=0.595); other teacher models trail; none demonstrates superiority.
- dont_support: does NOT establish equivalence, teacher ineffectiveness, or harm. Does NOT isolate the effect of adding teachers (GT-only aug OFF, all teacher models aug ON) — calling this control fully "recipe-matched" is inaccurate.
- missing: an augmentation-matched GT-only control, replicated seeds, precise enough intervals.
- revision (REFRAME): "No teacher-augmented model demonstrated superior success to the unaugmented GT-only control. Teacher-versus-GT contrasts remain confounded by augmentation."
- next: train GT-only at lr=1e-4 with the SAME augmentation+launcher as teacher models; repeat matched evals across seeds; cross teacher presence × aug status if the interaction matters.

## D3 → yes, strictly descriptive (high)
- support: v3 has highest OBSERVED overall (0.8295) and RIS (0.3438) success — ~4.4pt above GT-only/deployed and ~5.0pt above epoch50 on RIS. Statistical significance not required to report observed rankings.
- dont_support: "best" cannot mean reliably superior across training runs/seeds; "lifts" cannot imply established category-level causal benefit; overall McNemar does not establish RIS significance.
- missing: RIS episode counts, paired discordant outcomes, category-level uncertainty, replication; account for how the category was selected / how many examined.
- revision (REFRAME): "In the seed-2026 evaluation, v3_no_xyz achieved the highest observed overall and Robot Initial States success rates among the four models."
- next: none needed for that scoped observation; for a general superiority claim, replicate seeds + prespecify the RIS analysis with paired uncertainty.

## D4 → yes, as a descriptive historical comparison (medium)
- support: GT-only 0.7964→0.8232 = +2.68pt exceeds the largest observed teacher-variant gap (1.70pt) and every teacher-vs-control gap (≤1.07pt). Recipe comparability materially affects interpretation.
- dont_support: does NOT separate LR vs aug; does NOT establish a causal recipe effect of exactly 2.68pt; does NOT prove the true recipe effect exceeds the true teacher effect (a larger point estimate + NS teacher comparisons ≠ a significant difference between effects).
- missing: confirmation the older GT-only used comparable eval/training settings; replicated estimates + uncertainty for the historical contrast.
- revision (REFRAME): "The historical GT-only recipe change was associated with a 2.68-point increase, larger than any observed model gap in the current teacher comparison. LR and augmentation contributions remain unresolved, and augmentation confounds current teacher-versus-GT comparisons."
- next: GT-only 2×2 factorial lr={1e-5,1e-4} × aug={on,off}, replicate seeds; use augmentation-matched controls for teacher comparisons.

# Closing note (verbatim)
"The historical seed runs demonstrate sensitivity to evaluation randomness; they do NOT establish a calibrated ±0.5–1-point confidence interval."

# Delta vs run01 (gpt-5.5 xhigh)
- Directionally CONSISTENT: C4-causal partial, D2-overturn yes, D4 yes-descriptive. astra RAISES D3 from "partial" to "yes (strictly descriptive)".
- astra is STRICTER: downgrades D3/D4 to "descriptive-only / observed", and D4 confidence high→medium (older-baseline comparability).
- NEW audit points from astra: (a) reconcile why removing the xyz rejection rule yields FEWER teachers (284,320 < 317,941); (b) the "±0.5–1pt seed noise" is NOT a calibrated CI — just observed sensitivity.
- Recommended claim wording in the paper should follow astra's reframed (descriptive / non-causal) phrasing.
