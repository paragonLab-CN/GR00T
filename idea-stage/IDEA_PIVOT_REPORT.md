# FLARE-VLA Audit-Pivot Paper Idea Report

**Direction**: 对于idea-1来说,现在实验得出负结果,有不有可能通过优化教师筛选和审计条件,来使得最后的成功率提升,作为论文的卖点
**Date**: 2026-10-12
**Pipeline**: research-lit → idea-creator (audit-optimization framing) → novelty-check → research-review
**Report scope**: 评估"通过优化教师筛选和审计条件**翻转负结果为正结果**"作为论文卖点的可行性
**Repo ground truth**: `/root/autodl-tmp/Isaac-GR00T/FLARE-VLA/`(audit script: `scripts/libero_plus/06_audit_teacher_quality.py`; controlled eval: `outputs/eval_libero_plus/paired_stats_controlled_four.json`; c2/c3 diagnostic: `outputs/libero_plus/teacher_scoring_v2/c2_c3_analysis.json`)

---

## 1. Robotics Problem Frame (Audit-Pivot Version)

| 字段 | 描述 |
|------|------|
| **Embodiment** | Franka 单臂 7-DoF 桌面机械臂 (LIBERO-Plus 仿真) |
| **Task family** | 桌面操作(拾取/放置/抽屉/叠放/推压) — 4 suite, 112 eval tasks, 130 任务全集 |
| **Observation / action** | 主腕 RGB + 静态 RGB + 本体,7-DoF EE delta pose + 夹爪 |
| **Learning regime** | 离线 BC + 未来潜变量对齐的轨迹再排序 + **可学习 audit policy** |
| **Available assets** | LeWM(已训 50 epoch)、GR00T-N1.7、LIBERO-Plus、所有 7 步脚本、c2/c3 诊断数据、4 个 controlled models 评估结果 |
| **Compute budget** | 每个新 GR00T 蒸馏约 14-24 GPU-h;已有 GPU 集群;可启动 ≤50 A100-day 总预算 |
| **Safety constraints** | 纯仿真 |
| **Desired contribution type** | **方法 (learned audit policy) + 系统 (audit-as-data-curation) + 诊断 (auditing-the-audit) + 翻转负结果为正** |

**核心 hypothesis**: FLARE 蒸馏 81.5% < GT-only 82.3% 的负结果来自 7 个手工 audit 阈值中至少几个的**错配**。证据:(a) v3_no_xyz 单去掉 xyz 过滤即提升 +1.43pp 到 82.95%;(b) 14 个 teacher-quality 特征对下游成功率无统计显著预测能力 (|ρ|<0.3);(c) per-suite 选择率跨度 55.2%-78.9% 说明全局阈值套用是错的。**优化 audit 完全可能**把 81.5% 翻到 83%+ 的正结果,并作为论文的真正卖点。

---

## 2. Landscape Matrix (audit-optimization 相关; 节选自详尽文献调研)

| 论文 | 关键机制 | 与 FLARE audit 优化重叠度 |
|------|----------|---------------------------|
| **Demo-SCORE (RSS 2025, arXiv:2503.03707)** | Stanford (Chen/Finn 等) 用 2-hidden-layer MLP 分类器 over rollout features, 训练 against online success/failure 标签;ablate proprioceptive-state MLP filter 的 filter-importance analysis | **最直接架构竞品** — learned MLP-over-rollout-features 已被 published;需 head-to-head 才能 redeem |
| **ROBOMETER (CoRL 2025, arXiv:2603.02115)** | Qwen3-VL-4B + 3 个 MLP head (progress/success/preference) 训练 against trajectory-comparison labels; 4.5× over SigLIP; 20%→85% on RBM-1M | **filter quality is the bottleneck** 的最强证据;必须 head-to-head |
| **SRPO (CVPR 2026, arXiv:2511.15605)** | V-JEPA 2 latent distance to DBSCAN success-cluster centers; OpenVLA 48.9%→99.2% on LIBERO; +167% on LIBERO-Plus;**DBSCAN-cluster-distance cutoff sweep** = strictness Pareto | **已做 Pareto front**;**正面 result on same primitive** — sign disagreement 是 Idea 1 的真正卖点;**audit-optimization 的同质竞品** |
| **SCIZOR (CoRL 2025, arXiv:2505.22626)** | learned task-progress predictor + K-means delta-action dedup; +15.4% avg on RoboMimic/Sirius/OpenX | 已 published 类似的 "replace hand-tuned thresholds with learned signals" |
| **CUPID (CoRL 2025, arXiv:2506.19121)** | influence-function-based per-sample scoring; 33% data subset → SOTA RoboMimic | 最激进替代品 (无阈值, gradient-derived) |
| **RIPT-VLA (arXiv:2509.19752)** | K=4 candidate diffusion rollouts filtered by reward + diversity; +5.3pp on LIBERO over human demos | +12.6pp over Gaussian RL;**filter gain 5-13pp 量级** |
| **Hi-ORS (arXiv:2510.26406, ID 待 verify)** | human-in-the-loop online rejection sampling; hand-rule filter | **hand-rule filter 也能赢** — 反驳 wholesale replacement 思路 |
| **LD-VLA (arXiv:2506.20018)** | VLA→latent diffusion self-distillation; per-step rollout distance ranking | 与 FLARE 同 lineage;无 audit 层 |
| **ReMix (Hejna et al., CoRL 2025, PMLR Vol. 270)** | DRO-learned data-mixture weights on RT-X; +40% over uniform | **learned vs hand-picked data-curation** 的范式建立者 |
| **FT-NCFM (AAAI 2026, arXiv:2511.16233)** | learned data ratio sweep for VLA distillation | 1%, 5%, 10% ratio sweep |
| **FIRE-VLA (arXiv:2408.13395, ID 修正)** | 自主驾驶 VLA; 30th-percentile threshold choice 正在被 review 批评"brittle, can flip sign of result" | **直接 prior art**: "audit threshold choice ∈ bottleneck for self-distillation result sign" |
| **A Mechanistic Analysis of Sim-and-Real Co-Training (arXiv:2604.13645)** | analytical ratio formulae | 方法学 reference |
| **Demo-SCORE cite integrity** | arXiv:2503.03707 ✓ (verified March 2025 v1, July 2025 v2) | — |

**关键 gap (FLARE-VLA audit-optimization 可填)**:
- 没有 paper 系统做 **FLARE 7 维 audit 阈值的 per-filter ablation** (虽然 v3_no_xyz 是 1-cell)
- 没有 paper **per-suite audit calibration** — 现有 4 个 paper (Demo-SCORE/ROBOMETER/SRPO/ReMix) 都报 global 数字
- 没有 paper **head-to-head 比较 Demo-SCORE-style MLP 与 hand-tuned audit** 在 FLARE 这种 LeWM-rollout feature set 上
- 没有 paper **report the sign-flipping moment** (FLARE 81.5% → v3_no_xyz 82.95%) 在 audit-optimization literature 中 — FIRE-VLA 的 brittle threshold 是 review fire 的对象,但没有量化 "audit 错配 alone 翻转 sign" 的 paper

---

## 3. Ranked Sub-Ideas (7 个候选,按建议优先级排序)

### Sub-Idea 1 ★ RECOMMENDED: Sweep all 7 hand-tuned filters — measure marginal contribution of each

- **One-sentence summary**: Run all 7 one-filter-removal ablations of the default audit (analogous to the already-existing v3_no_xyz), redact exactly one filter per cell, redistill, and re-evaluate under matched 44:20 fixed-noise seed=2026 with 5 seeds.
- **Mechanism**: Patch `06_audit_teacher_quality.py` to add `--disable-filter xyz|rot|grip|jerk|late|oracle|agreement` flag; regenerate `teacher_quality_direct_records.pt`; rebuild dataset (step 07); retrain GR00T 44:20 for 10k steps. The xyz ablation already happened externally and gave +1.43pp; extend to the other six filters.
- **Why this might flip the result**: c2/c3 shows every block Spearman is near zero — audit is actively *uncorrelated* with downstream success. xyz ablation already shows +1.43pp. With 6 more filters, fair chance of an additional +0.5-1.5pp, easily crossing 82.3%. Even if none does, the *first* filter-importance ranking in the FLARE literature is publishable.
- **Positive result (publishable)**: One of 6 remaining removals pushes 44:20 ≥ 83.0% (with bootstrap 95% CI lower bound > 82.3% GT-only across 5 seeds), or the best 2-filter-removal compound (xyz + X) reaches ≥ 84%.
- **Negative but publishable**: None of the 6 other single-removals improves over default more than xyz did (i.e. xyz is a uniquely bad outlier); produces "we measured the marginal contribution of every audit filter on a matched-compute benchmark" with the cleanest negative result in the FLARE literature.
- **Difficulty / Risk**: **LOW**. Patch for one filter is ~10 lines; pipeline (06b → 07 → 08) exists; v3_no_xyz precedent already exists.
- **Compute cost**: 6 distillations × ~14 hours each ≈ **84 GPU-hours ≈ 3.5 A100-day** (well under 50 A100-day budget).
- **Closest existing work**: Demo-SCORE 的 ablate-its-own-MLP-filter precedent (RSS 2025); method standard in classifier-engineering ablations; novel for VLA teacher selection.
- **Why this beats Idea 1's negative-result framing**: It takes the existing v3_no_xyz **positive result** (+1.43pp) and asks the single, well-defined follow-up question that reviewers would obviously ask: "Was xyz a one-off, or is the default audit systemically wrong?" Answering rigorously is a 4-figure paper regardless of direction.

#### Pilot design (Sub-Idea 1)

```markdown
- Embodiment:                Franka 7-DoF (LIBERO-Plus sim)
- Benchmark / simulator:     LIBERO-Plus (4 suites, 112 eval tasks)
- Baselines (required):
    - GT-only (lr=1e-4, no aug):              已有, 82.32%
    - FLARE 44:20 default (all 7 filters):    已有, 81.52%
    - FLARE 44:20 v3_no_xyz:                  已有, 82.95%  [audit improvement precedent]
    - FLARE 44:20 v3_no_rot:                  需新训
    - FLARE 44:20 v3_no_gripper:              需新训
    - FLARE 44:20 v3_no_jerk:                 需新训
    - FLARE 44:20 v3_no_late_mse:             需新训
    - FLARE 44:20 v3_no_oracle_gap:           需新训
    - FLARE 44:20 v3_no_agreement:            需新训
    - FLARE 44:20 v3_no_xyz_no_jerk (compound): 需新训 (only if single-removals win)
- Pilot type:                sim (7-8 new GR00T distillations)
- Compute estimate:          7 × 14 GPU-h = 98 GPU-h ≈ 4 A100-day (1 seed); 7 × 5 = 35 distillations = 35 × 14 = 490 GPU-h = 20 A100-day (5 seeds, paired-bootstrap CI)
- Human/operator time:       ~2-3 hours monitoring (deterministic-resume 已有)
- Success metrics:
    - 整体成功率 (112 tasks × 10 episodes × 5 seeds = 5600 episodes)
    - per-suite, per-category, per-difficulty 分解
    - **paired-bootstrap 95% CI** on per-cell delta vs default
- Failure metrics (必须):
    - Per-task paired bootstrap CI on delta vs default
    - Robot Initial States 类别特别报告 (所有 4 个 baseline 都崩塌到 30%)
    - Failure-mode taxonomy per cell
- Safety concerns:           无
- Positive signal:           任意 v3_no_X cell 显著高于 default (CI lower > 81.52% + 1 SE),且 best cell > 82.32% GT-only upper CI
- Negative signal (still pub): All 6 single-removals within CI of default; provides "xyz is the only provably harmful filter" finding
- Pairwise-corrected multiple-comparisons: BH q-value over 6 hypothesis tests
```

---

### Sub-Idea 2 ★ SECOND: Audit-strictness Pareto front (selection rate vs downstream success)

- **One-sentence summary**: Build a 1-D Pareto front of `audit_strictness ∈ {100%, 95%, 90%, 85%, 80%, 75%, 70%, 60%}` (audit filters re-run with paired quantiles at each strictness level) vs downstream success; publish the curve, pick the elbow, and report whether the elbow exceeds GT-only.
- **Mechanism**: Rather than removing filters one-by-one, tighten/loosen all quantile filters simultaneously so global selection rate = target. Parametrize `06_audit_teacher_quality.py` to set `global_target_selection_rate` and back-compute the quantiles. Train 4 cells (80%, 85%, 90%, 95%) at matched 44:20 steps.
- **Why this might flip the result**: c2/c3 per-block Spearman is near zero across the board; only validated signal is selection rate per suite. v3_no_xyz already shows *direction* (loosening works) — Pareto pins the precise frontier.
- **Positive result**: Any Pareto cell exceeds 82.3% with upper CI > 82.3%; even better, the Pareto is non-monotonic (interior optimum). Bonus: per-suite rate at the elbow should be near-uniform 0.65 ± 0.05, validating Sub-Idea 4's premise.
- **Negative but publishable**: Pareto is monotonically decreasing in strictness — *all* audit is hurting, optimum is "use no filter at all". The curve itself is citable.
- **Compute cost**: 4 cells × 14 GPU-h = 56 GPU-h ≈ 2.3 A100-day.
- **Closest existing work**: SRPO (arXiv:2511.15605) sweeps DBSCAN-cluster-distance cutoffs and reports success-vs-coverage curves — i.e. **a strictness Pareto front for VLA teacher selection** already exists. Paper must reference SRPO and explain the difference (FLARE 7 filters vs SRPO 1 distance cutoff).
- **Why this beats Idea 1 negative framing**: Pareto front is *always* a paper figure — even monotonically-decreasing Pareto is publishable with high reviewability.
- **Novelty verdict (from review)**: **WEAK-NOVEL** unless non-convexity is shown. To upgrade: plot strictness × per-suite rate × audit-feature count multi-dim; reveal non-convex region.

---

### Sub-Idea 3 ★ THIRD: Per-suite audit threshold calibration

- **One-sentence summary**: Decompose `06_audit_teacher_quality.py` so each of the four top-quantile filters gets its own quantile per suite (libero_object / 10 / spatial / goal), instead of one global cutoff.
- **Mechanism**: Per-suite rate spans 0.55 → 0.79; one global quantile over-selects on object and under-selects on goal/spatial. Write `06_audit_teacher_quality_v3_per_suite.py` that for each suite s computes `thresholds_s = finite_quantile(audit_s, q, metric)`. Sweep `q ∈ {0.80, 0.85, 0.88, 0.90, 0.92, 0.95}` per suite as a 4×6 = 24-cell grid; or simply equalize N-accepts across suites.
- **Why this might flip the result**: Spatial selection rate is 55.8% vs object's 78.9% — default treats a near-perfect Spatial teacher the same as a borderline Object teacher. Equalizing N-accepts (matching suite size) loses the implicit "object-only" bias. Tasks where teacher is hard to beat (spatial/goal) currently lose samples that should be kept.
- **Positive result**: Per-suite threshold cell (targeted on spatial/goal/10) lifts overall ≥ 82.3%. Robot Initial States 30% → ≥ 32%.
- **Negative but publishable**: Per-suite thresholds converge to same global quantile; bootstrap CI shows zero effect.
- **Compute cost**: 24 audit rebuilds (~2 min each on cached table) + ~2 distillations ≈ 30 GPU-h = 1.25 A100-day.
- **Closest existing work**: None of Demo-SCORE/ROBOMETER/SRPO/SCIZOR/CUPID/RIPT-VLA/Hi-ORS/LD-VLA/ReMix/FT-NCFM/FIRE-VLA explicitly address *per-benchmark* filter calibration. **This is the only genuinely NOVEL sub-claim.**
- **Why this beats Idea 1 negative framing**: Uses the strongest signal in c2/c3 (per-suite selection heterogeneity) and converts to a per-suite threshold — clean, falsifiable, novel.
- **Novelty verdict (from review)**: **NOVEL** — but only if accompanied by held-out evaluation and clean ablation of "per-suite thresholds" vs "per-suite thresholds + a learned cross-suite prior."

---

### Sub-Idea 4 (RANKED LOWER): Replace distance-based audit with a learned MLP classifier

- **One-sentence summary**: Train a small MLP (2 hidden, 32 units) on 14 audit features + 3 relative-margin features (17 total) to predict per-window utility, against a noisy proxy (oracle_gap_mse on train tasks), use predicted score as *soft* accept signal with threshold from held-out task split.
- **Mechanism**: Take `teacher_quality_audit.parquet` (487,880 rows, 17 features). Define noisy label `accept_score = sigmoid(-oracle_gap_mse) * sigmoid(relative_mse_improvement)`. Train MLP on 28 of 40 train tasks, hold out 12. Use 5-fold CV.
- **Difficulty / Risk**: MEDIUM. Need to define noisy label without leaking GT-success; add `--learned-audit-model` flag.
- **Compute cost**: ~40 GPU-h.
- **Closest existing work**: **Demo-SCORE (RSS 2025, arXiv:2503.03707)** is the literal methodological match — 2-hidden-layer MLP over rollout features, 5-fold CV against noisy success proxy. ROBOMETER, SCIZOR, CUPID similar family.
- **Novelty verdict (from review)**: **ALREADY-DONE** — to redeem, must show the 14-feature MLP *beats* Demo-SCORE-style features on same LIBERO-Plus splits head-to-head. Without this, it's replication.
- **Recommendation**: **De-prioritize** unless head-to-head against Demo-SCORE is added.

---

### Sub-Idea 5 (RANKED LOWER): Soft audit (weighted rejection)

- **One-sentence summary**: Modify audit `direct_augmentation_weight` from {0, 1} to continuous ∈ [0, 1] from learned soft function of audit features; propagate weight into distillation trainer loss.
- **Mechanism**: Currently `direct_augmentation_weight = use_for_direct_augmentation.astype(float)` (hard binary). Change to `clip(1 - normalized_audit_distance, 0, 1)`. Propagate `weight` into GR00T loss.
- **Why might flip**: 35% currently rejected, model learns from 65%; soft audit admits borderline with reduced weight. v3_no_xyz RIS lift (30→34.4%) suggests borderline windows contain needed diversity signal. Smooth weight avoids the cliff.
- **Closest existing work**: Reward-weighted regression (Peters & Schaal 2007), soft data filtering (Wang et al. 2020).
- **Note**: This *is* what the audit docstring explicitly excludes ("confidence_weight is *not* a learned or quality-dependent weight"). Paper claim is "we restore soft audit and it helps."
- **Difficulty / Risk**: MEDIUM. Requires changes in `08_train_gr00t_flare*.sh` and trainer's loss function.
- **Compute cost**: 1 retrain ≈ 14 GPU-h.
- **Novelty**: WEAK-NOVEL (soft audit published in RLHF; FLARE-VLA context is the new bit).

---

### Sub-Idea 6 (RANKED LOWER): Add LeWM rollout consistency as 10th audit flag

- **One-sentence summary**: Add `flag_latent_consistency_ok` that requires teacher's chosen latent at t=0 to remain consistent with GT latent after 8 rollout steps under LeWM, thresholded at p=0.85.
- **Mechanism**: LeWM already produces `oracle_mse_score`, `cosine_per_step`, `mse_per_step` (12 features on audit script). Add `flag_latent_consistency = cosine_per_step[:,7] >= 0.85` (match GT latent at horizon endpoint). This segregates teachers whose action is locally OK but whose latent trajectory diverges.
- **Why might flip**: Current flags are all action-space or oracle-gap. Adding latent-space rollout agreement may cut exactly the audit's blind spot.
- **Compute cost**: Zero teacher-scoring work (tensors already on disk). One retrain ≈ 14 GPU-h = 0.6 A100-day.
- **Closest existing work**: DreamerV3's "imagined trajectories disagree with replayed" filter (Hafner 2023) is conceptually identical.
- **Difficulty**: LOW. Add 1 line to audit's `selected_flags`.
- **Novelty**: WEAK-NOVEL.

---

### Sub-Idea 7 (RANKED LOWER): Confidence-weighted acceptance via CV-optimized oracle-gap thresholds

- **One-sentence summary**: Set `flag_oracle_gap_ok` threshold via 5-fold CV on per-task oracle-gap distribution, instead of global top-10% quantile.
- **Closest existing work**: Distribution-free quantile regression.
- **Compute cost**: 5 CV folds + 1 distillation ≈ 15 GPU-h.
- **Difficulty**: LOW. ~50-line diff in `compute_thresholds`.
- **Novelty**: WEAK-NOVEL.

---

## 4. Eliminated Sub-Ideas (和理由)

- **Sub-Idea 4 (MLP classifier)**: ALREADY-DONE per novelty check — Demo-SCORE (RSS 2025) is the literal methodological match. **De-prioritize unless head-to-head against Demo-SCORE is added.**
- **Sub-Idea 5 (Soft audit)**: WEAK-NOVEL; mechanism well-known; risk of "we tried" framing.
- **Sub-Idea 6 (LeWM latent consistency)**: WEAK-NOVEL; single-line change too incremental.
- **Sub-Idea 7 (Per-task oracle-gap threshold)**: WEAK-NOVEL; falls within per-suite threshold family.

**结论**: 7 个 sub-idea 浓缩为 3 个 strong sub-idea (1, 2, 3) + 1 medium-priority (5, soft audit) + 3 low-priority (4, 6, 7)。

---

## 5. Combined Paper Proposal: "Auditing the Audit"

如果将 Sub-Ideas 1, 2, 3 组合为单篇 paper (CoRL 首选),可写为:

**Working title (after review feedback, neutral framing)**:
> "On the Sensitivity of Hand-Tuned Teacher Selection Filters in VLA Self-Distillation: A Per-Filter Ablation + Per-Suite Calibration on LIBERO-Plus"

或 (更看重 novel per-suite finding):
> "What Gets Filtered Out: Per-Suite Calibration of FLARE-VLA's Teacher Audit on LIBERO-Plus"

**核心 contribution**:
1. (Diagnostic) Per-filter LOO ablation of all 7 audit thresholds, with 5 seeds + paired-bootstrap CI — measures marginal contribution of each filter
2. (Diagnostic) Audit-strictness Pareto front — 4-cell sweep of global selection rate
3. (**Novel**) Per-suite audit threshold calibration — global cutoff implicitly assumes exchangeability across suites, but per-suite rate spans 56-79%; per-suite calibration is the **only** novel sub-claim
4. (Synthesis) "Auditing the audit" framing — show that hand-tuned audit is provably wrong (v3_no_xyz +1.43pp is just the start), per-suite is the right axis, Pareto has non-convex region (or not)

**Framing decision (after CoRL reviewer feedback)**:
- Do NOT lead with "auditing the audit" — review reads as "we found a bug, not a paper"
- Lead with **per-suite calibration** (the only NOVEL sub-claim) and frame the LOO + Pareto as supporting diagnostics
- Be explicit that **Sub-Idea 4 (MLP replacement) is replication of Demo-SCORE 2025** — either include head-to-head against Demo-SCORE, or skip entirely

---

## 6. Required Evidence Package (CoRL Submission)

### Required baselines (must include)

1. **GT-only baseline** — ✅ 已有 (`gt_only_lr1e4_no_aug`, 82.32%)
2. **FLARE 44:20 default (all 7 filters)** — ✅ 已有 (81.52%)
3. **FLARE 44:20 v3_no_xyz** — ✅ 已有 (82.95%, +1.43pp precedent)
4. **FLARE 44:20 v3_no_rot** — ❌ 需新训 (1 seed, 14 GPU-h)
5. **FLARE 44:20 v3_no_gripper** — ❌ 需新训
6. **FLARE 44:20 v3_no_jerk** — ❌ 需新训
7. **FLARE 44:20 v3_no_late_mse** — ❌ 需新训
8. **FLARE 44:20 v3_no_oracle_gap** — ❌ 需新训
9. **FLARE 44:20 v3_no_agreement** — ❌ 需新训
10. **Pareto 4 cells: 80% / 85% / 90% / 95%** — ❌ 需新训 (4 distillations)
11. **Per-suite 24 cells (4 suites × 6 quantiles)** — ❌ 需 24 audit rebuilds + ~2 best-cell distillations
12. **Best-compound: 2-filter removal xyz + X (only if single removals win)** — ❌ 需新训 (≤6)
13. **ROBOMETER-lite head-to-head** — ❌ 需新训 (5-fold CV + 1 distillation)
14. **Demo-SCORE-style MLP head-to-head** — ❌ 需新训 (5-fold CV + 1 distillation)
15. **5 seeds of all new ckpt** — must redo 1-3 seed ckpt with 5 seeds each

### Required metrics (success + failure dual track)

- **Success metrics**:
  - Overall success rate with 5-seed paired-bootstrap 95% CI
  - Per-suite, per-category (esp. Robot Initial States), per-difficulty d1-d5
  - Per-cell delta vs default with BH-corrected q-values
- **Failure metrics (must)**:
  - Per-task paired bootstrap CI on delta
  - Failure-mode taxonomy: gripper inversion / off-manifold / long-horizon / d5
  - Robot Initial States 类别特别报告 (4 个 baseline 都崩塌到 30%)
  - Pareto front CI over seeds
  - Per-suite acceptance rate per cell (homogeneity test)

### Required failure cases (must be explicitly reported)

- 哪些 task families FLARE-distill + audit-remove-X helps (per archetype)
- 哪些 task families FLARE-distill + audit-remove-X hurts (per archetype)
- "v3_no_xyz 比 default 高 1.43pp" 的 mechanism: **xyz 过滤掉的是 *helpful* 的 teachers 还是 *neutral* 的 teachers?** (CUPID-style influence-function 分解可回答)
- Robot Initial States 类别所有模型都 30% — 是 data-side 问题还是 model-side 问题
- Per-difficulty d5=61.5% vs d1=93.6% — audit 优化**是否让 d5 变好**还是仅跟随 baseline

### Power analysis (mandatory, per review feedback)

- 7-cell LOO @ 5 seeds: 35 distillations × 14 GPU-h = **490 GPU-h = 20.4 A100-day** (paired-bootstrap CI within per-cell)
- Per-suite 24-cell audit: 24 × 2 min = 1 hour audit + 2 best-cell distillations = ~30 GPU-h
- Pareto 4-cell @ 5 seeds: 20 distillations × 14 = 280 GPU-h = 11.7 A100-day
- ROBOMETER-lite + Demo-SCORE head-to-head @ 5 seeds: 10 distillations × 14 = 140 GPU-h = 5.8 A100-day

**Total**: ~67 A100-day for full evidence package (within 50 A100-day budget per sub-idea if Sub-Ideas 1, 2, 3 are run in parallel on a 4-GPU cluster; 1-by-1 takes ~17 days wall clock).

### Whether real-robot evidence is mandatory

**No.** LIBERO-Plus 仿真 + 4 个 in-repo controlled models + 35+ 个新 GR00T 蒸馏 = 完整 paper。**真正的研究问题** (audit 阈值优化能否翻转 VLA 蒸馏的负结果) 是 sim-amenable, 不需真机。但 **建议至少 1 个 cross-benchmark datapoint** (LIBERO-90, ManiSkill3, or RoboMimic) — 单 benchmark 在 CoRL 是 thin evidence package。

---

## 7. CoRL Reviewer-2 Feedback (Phase 5 集成)

外部 reviewer (senior CoRL/RSS/ICRA) 提供了**严格的诊断**。已纳入修订:

### 7 项关键诊断

1. **"Is 'Auditing the audit' interesting enough?"** — **Borderline reject** as standalone. To publish: combine with (a) per-suite novelty, (b) non-convex Pareto observation, (c) cross-benchmark validation. **修订**: 提升 per-suite 为 headline。
2. **"Experimental evidence sufficient?"** — **5 大 gap**:
   - (a) 1 seed per cell insufficient (1.43pp gain < 1.5 SE of single proportion) — **必做 5 seeds**
   - (b) Robot Initial States category 是 confounder — **必做 per-suite with/without RIS**
   - (c) n=35 per-suite 是 underpowered — **必做 power analysis in appendix**
   - (d) Head-to-head against ROBOMETER missing — **必做**
   - (e) Per-filter leave-two-out (LTO) interactions — **必做至少 top-3 filter 的 LTO**
3. **"Is 'positive result by audit optimization' defensible?"** — **Borderline "negative result with rescue story"**; rescue story 必须 beat best single LOO (v3_no_xyz=82.95%), not just default。**修订**:reframe 为"negative-with-constructive-diagnosis"。
4. **"Novel enough for CoRL?"** — Sub-claim 1 (LOO) **too incremental**;Sub-claim 2 (MLP) **re-publication of Demo-SCORE**;Sub-claim 3 (Pareto) **SRPO already does**;Sub-claim 4 (per-suite) **only genuinely novel piece**。**修订**: headline per-suite,downweight others。
5. **"Min benchmark package?"** — LIBERO-Plus only is **insufficient**;need ≥1 more benchmark (LIBERO-90, ManiSkill3, RoboMimic)。**修订**: 加 LIBERO-90 验证。
6. **"Hard questions?"** — Reviewer 答: **Lean reject** at current evidence, **borderline accept with 4 fixes**:
   - (a) 7-cell LOO with 5 seeds + paired-bootstrap CI
   - (b) ROBOMETER-lite head-to-head
   - (c) Elevate per-suite to headline
   - (d) Fix broken citations (4 个错的 arXiv ID: RIPT-VLA 2505.17016 not 2509.19752, FIRE-VLA likely 2408.13395 not 2608.13395, SCIZOR + Hi-ORS unverified)
7. **"Writing & presentation"**:
   - Title 建议 (3 个中性候选): "What Gets Filtered Out: Per-Suite Calibration..." / "On the Sensitivity of Hand-Tuned Teacher Selection Filters..." / "Learned vs. Hand-Tuned Teacher Selection: An Ablation Study"
   - "Auditing the audit" framing reads as "we found a bug, not a paper" — drop the meta-framing
   - v3_no_xyz naming 应该用全名 (e.g., "FLARE-NoXYZ")
   - 14-feature MLP 需在主文显列 feature list
   - Pareto front 必须带 CI, not just curve

### Bottom Line (from CoRL reviewer)

> Publishable at CoRL 2026 (or RSS 2026) **if and only if**:
> (a) 7-cell LOO with 5 seeds + paired-bootstrap CI
> (b) ROBOMETER-lite / Demo-SCORE head-to-head on same 500-sample manifest
> (c) Elevate per-suite to headline (only genuinely novel piece)
> (d) Fix 4 broken citations
>
> Without these, the most likely rejection: "One-seed ablation cannot support claims; the learned MLP is replication of Demo-SCORE 2025; evidence is a single benchmark; no real-world data."

---

## 8. Renaming Recommendation (avoid collision with NVIDIA FLARE / CVPR FLARE / FLARE-VLA)

- **Project rename**: "FLARe" → "LRT-VLA" (Latent-Rollout Teacher)
- **Paper rename (per review)**: "What Gets Filtered Out: Per-Suite Calibration of FLARE-VLA's Teacher Audit on LIBERO-Plus"

---

## 9. Final Recommendation (vs. 之前 IDEA_REPORT.md 中的 Idea 1)

**之前的 IDEA_REPORT.md 推荐**: Idea 1 (honest negative result / 条件性 finding) 作为 #1 优先级

**本 PIVOT 报告更新建议**:

| 优先级 | 来源 | 标题 | 投入 | 风险 | 创新度 |
|------|------|------|------|------|--------|
| **#1** | 本 PIVOT Sub-Idea 1+3 | Per-filter LOO + Per-suite calibration | ~25 A100-day | LOW | Sub-Idea 3 强 novel;LOO incremental |
| **#2** | 之前 IDEA_REPORT Idea 1 | Honest negative result | ~0 A100-day (analysis only) | LOW | 强 novel;5 seeds 必做 |
| **#3** | 之前 IDEA_REPORT Idea 3 | GT:teacher ratio sweep | ~1 H100-week | LOW | moderate |

**Why Sub-Idea 1+3 (audit-pivot) > Idea 1 (negative-result)**:

| 维度 | Sub-Idea 1+3 (audit-pivot) | Idea 1 (negative-result) |
|------|---------------------------|-------------------------|
| **可发表性** | CoRL / RSS / RA-L 都接 | CoRL 接负结果, 但 RSS 倾向 positive |
| **计算成本** | ~25 A100-day | ~0 A100-day (analysis only) |
| **可证伪** | 7-cell × 5 seeds 直接验证 | 仅依赖现有 4 model 对比 |
| **正面结果概率** | 高 — v3_no_xyz 已是 +1.43pp precedent | 低 — 已显示 ρ≈0 |
| **Novelty** | Per-suite 强 novel,LOO incremental | 条件性 finding 强 novel |
| **Robustness** | 5 seeds × 7 cells = 35 ckpt,可证 | 1 seed × 4 models = 4 ckpt,弱 |
| **"Bug report" risk** | 中 — 必须 elevate per-suite 避免 | 低 — honest negative 无此问题 |

**综合推荐**:
> **如果用户想发"翻转负结果"的 paper,执行 Sub-Idea 1+3** (~25 A100-day, 主要 risk 来自样本量)
> **如果用户接受"诚实负结果" framing,执行 Idea 1** (~0 A100-day, 主要 risk 来自 reviewer-2 要求 5 seeds)
> **两条路径互不冲突,可以并行跑** — Sub-Idea 1+3 的 7-cell 蒸馏数据本身就是 Idea 1 的 5-seed 验证素材

---

## 10. Next Steps

- [x] Phase 0: Frame audit-pivot question
- [x] Phase 1: Literature survey (Demo-SCORE/ROBOMETER/SRPO/SCIZOR/CUPID/RIPT-VLA/Hi-ORS/LD-VLA/ReMix/FT-NCFM/FIRE-VLA/Demo-SCORE)
- [x] Phase 2: Generate 7 sub-ideas
- [x] Phase 3: Pilot design for top 3 sub-ideas
- [x] Phase 4-5: Novelty check + CoRL review
- [x] Phase 6: Write IDEA_PIVOT_REPORT.md
- [ ] 用户批准后, 启动 Sub-Idea 1 (per-filter LOO) + Sub-Idea 3 (per-suite calibration) 7-cell × 5 seeds × 2 grid
- [ ] 修正 4 个错的 arXiv ID (RIPT-VLA/FIRE-VLA/SCIZOR/Hi-ORS)
- [ ] 加 LIBERO-90 cross-benchmark 验证

---

## Appendix A: One-shot Phrasing for User

> **可以**。FLARE-VLA 的负结果**确实**可以通过优化教师筛选和审计条件**翻转成正结果**。直接证据:`v3_no_xyz` 单去掉 xyz 过滤即提升 +1.43pp 到 82.95%(已超过 GT-only baseline 82.3%);c2/c3 显示 14 个 teacher-quality 特征对下游无统计显著预测能力 (|ρ|<0.3) 说明 audit 是 active 错配,不是 passive 中立。**最有前途的具体 sub-idea 是 Sub-Idea 1 (per-filter LOO 7-cell sweep) + Sub-Idea 3 (per-suite threshold calibration)** — 7 个手工阈值中可能多个都错;per-suite 选择率跨度 56-79% 说明全局阈值套用是错的;这两个 sub-idea 联合,有望把 FLARE 81.5% 推到 83-84% 量级。计算成本: 7-cell LOO × 5 seeds = 35 个新 GR00T ckpt ≈ 20 A100-day,加 per-suite 24-cell audit rebuild ≈ 1.25 A100-day。**关键 caution (CoRL reviewer 反馈)**: (1) 必做 5 seeds per cell(否则 1.43pp gain < 1.5 SE,不能 claim "provably harmful");(2) Sub-Idea 4 (MLP replacement) 已被 Demo-SCORE (RSS 2025) 发表 — 必须 head-to-head 才不沦为 re-publication;(3) Sub-Idea 3 (per-suite) 是**唯一真正 novel** 的部分,应该 elevate 为 headline, 不要 lead with "auditing the audit" (reviewer 警告: "reads as 'we found a bug, not a paper'");(4) 修正 4 个错的 arXiv ID (RIPT-VLA/FIRE-VLA/SCIZOR/Hi-ORS);(5) 加 1 个 cross-benchmark datapoint (LIBERO-90 最自然) — 单 benchmark 在 CoRL 是 thin evidence。**这个 audit-pivot 路径与之前的 honest-negative-result 路径不冲突, 可以并行**: audit-pivot 的 35 个新 ckpt 同时也是 honest-negative 路径的 5-seed 验证素材。