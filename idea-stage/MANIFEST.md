# Idea Discovery Output Manifest — FLARE-VLA

**Date**: 2026-10-12
**Pipeline**: /idea-discovery-robot (research-lit → idea-creator → novelty-check → research-review)
**Direction**: 根据仓库中现有的FLARE-VLA流程,总结出可写的论文idea

## Outputs

| File | Description | Status |
|------|-------------|--------|
| `idea-stage/IDEA_REPORT.md` | Main report with robotics problem frame, landscape matrix, 3 ranked ideas + 9 sub-analyses, CoRL reviewer feedback, novelty check | ✅ written |

## Phase Results

| Phase | Result |
|-------|--------|
| 0 — Robotics Problem Frame | Franka 7-DoF tabletop, LIBERO-Plus sim, GR00T-N1.7 + LeWM, fixed 44:20 GT:teacher distillation |
| 1 — Literature Survey | 11 papers in landscape matrix; key gaps: no WM-rank calibration, no first-principles ratio, no LeWM ablation |
| 2 — Idea Generation | 12 candidates → 3 strong (1/2/3) + 9 sub-analyses |
| 3 — Pilot Design | Sim-first pilots for all 3 ideas; compute costs: 0 / 1 week / 15-20 A100-day |
| 4 — Novelty Check | Idea 1 strongest (new framing: SRPO+167% vs FLARE ≈0 same primitive sign disagreement); Idea 3 moderate; Idea 2 weakest |
| 5 — CoRL Reviewer-2 Critique | 8 key issues; central recommendation: 5 seeds + random/oracle baselines + conditional reframe |
| 6 — Final Report | 455-line IDEA_REPORT.md |

## Key Findings (already measured, not guessed)

1. **All 14 teacher-quality features** in `c2_c3_analysis.json` have |Spearman ρ| < 0.3 vs downstream VLA success (n=35 train tasks, none significant).
2. **Controlled four-model comparison** (fixed-noise seed=2026, 112 tasks × 10 episodes):
   - GT-only baseline: **82.3%**
   - FLARE w/ xyz filter: **81.5%** (−0.8pp)
   - FLARE deployed: **81.25%** (−1.0pp)
   - FLARE w/o xyz filter: **82.95%** (+0.7pp)
3. **All four models collapse on "Robot Initial States" distribution-shift category** (30.0% / 29.4% / 30.0% / 34.4% vs 94% on "Background Textures").
4. **Per-difficulty success d1=93.6% → d5=61.5%** (steady decline).
5. **SRPO (arXiv:2511.15605)** — same latent-rollout primitive, reports **+167% on LIBERO-Plus** from 48.9% baseline — *opposite sign* from FLARE-VLA. The sign-disagreement is the most novel finding.

## Top 3 Ranked Ideas (final)

| Rank | Idea | Novelty | Compute | Recommended framing |
|------|------|---------|---------|---------------------|
| #1 ★ | Honest negative result / conditional finding | ★★★★★ | ~0 A100-day | "Why does SRPO+167% vs FLARE≈0 on same primitive? — characterize scope conditions" |
| #2 | GT:teacher ratio sweep with CIs | ★★★ | ~1 H100-week | "Frozen-manifest 7-ratio sweep with bootstrap CIs and failure-mode stratification" |
| #3 | Calibrated audit thresholds | ★★ | ~15-20 A100-day | Reframe as "Auditing the audit" or deprioritize |

## Critical Recommendations

1. **必改名**: "FLARE" 撞名三处 (NVIDIA 2505.15659, CVPR 2026 2608.26645, FLARE-VLA)。推荐 **FLARe** / **LRT-VLA** / **LeWM-Augmented GR00T**。
2. **必加 random-teacher + oracle-teacher baseline** (CoRL reviewer + novelty check 都要求): 隔离 ranking vs audit vs self-distillation 各自的贡献。
3. **必做 5 seeds per config**: 现有 1 seed, SE of ρ ≈ 0.17, 不能区分 "ρ=0" vs "ρ=0.3"。
4. **必做 K × horizon × audit-strictness sweep**: K=4 与 horizon=8 都是未论证默认。
5. **必加 LAC-WM / SRPO / NVIDIA-FLARE head-to-head**: SRPO 是最直接的方法学竞品,**sign disagreement** 是 Idea 1 真正卖点。
6. **必做 per-suite / per-difficulty / per-category 失败诊断**: 现有 4 模型全在 "Robot Initial States" 类别崩塌到 30% — 这是真正的 research 问题。

## Sources / Citations

- NVIDIA FLARE: arXiv:2505.15659
- SRPO: arXiv:2511.15605 (robot variant — same primitive, opposite sign)
- LAC-WM: arXiv:2610.10846 (cross-embodiment latent-action WMs, NOT a K-candidate-rank primitive per novelty-check)
- FRAPPE: arXiv:2602.17259
- DreamGen: arXiv:2505.12705 (CoRL 2025)
- VideoAgent: arXiv:2410.10076 (ICLR 2025)
- RIPT-VLA: arXiv:2509.19752
- World-VLA-Loop: arXiv:2602.06508
- ROBOMETER (CoRL 2025)
- RoboRetry (2026)
- ReMix (Hejna et al., CoRL 2025)
- FT-NCFM (AAAI 2026)
- A Mechanistic Analysis of Sim-and-Real Co-Training (Lei et al., 2026; arXiv:2604.13645)
- ASU DIME-LAB λ-sweep (Yadav et al., 2026)
- CVPR 2026 "FLARE" (Failure-Leveraged Autonomous Recovery, arXiv:2608.26645) — newly surfaced collision

## In-Repo Evidence Files Used

- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/c2_c3_analysis.json` (Spearman correlations)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/teacher_quality_summary.json` (audit thresholds + pass rates)
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_v2/teacher_quality_suite_summary.csv` (per-suite selection rates)
- `outputs/eval_libero_plus/paired_stats_controlled_four.json` (controlled four-model eval)
- `outputs/eval_libero_plus/gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20_checkpoint_10000_fixed_noise_seed2026/summary_all.json` (per-category, per-difficulty breakdowns)
- `FLARE-VLA/scripts/libero_plus/06_audit_teacher_quality.py` (audit implementation, 7 filters at 0.9 quantile)

## Audit-Pivot Iteration

On 2026-10-12 the user asked a follow-up pivot question: "对于 idea-1 来说,现在实验得出负结果,有不有可能通过优化教师筛选和审计条件,来使得最后的成功率提升,作为论文的卖点". A focused re-run of the idea-discovery pipeline produced `idea-stage/IDEA_PIVOT_REPORT.md` (361 lines). Key new findings:

1. **v3_no_xyz already proves the negative result is flippable**: 82.95% (+1.43pp over default, +0.7pp over GT-only baseline)
2. **7 ranked sub-ideas** for audit optimization; top-3 are per-filter LOO, audit-strictness Pareto, per-suite threshold calibration
3. **Only Sub-Idea 3 (per-suite calibration) is genuinely novel** per novelty check
4. **Sub-Idea 4 (MLP replacement) is already published** as Demo-SCORE (RSS 2025, arXiv:2503.03707) — must head-to-head to redeem
5. **Citation integrity issues**: 4 unverified/wrong arXiv IDs flagged (RIPT-VLA/FIRE-VLA/SCIZOR/Hi-ORS)
6. **Compute cost**: 7-cell LOO × 5 seeds + per-suite 24-cell = ~25 A100-day; full evidence package with ROBOMETER-lite head-to-head ≈ 67 A100-day
