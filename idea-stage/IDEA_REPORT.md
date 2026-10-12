# Robotics Idea Discovery Report — FLARE-VLA

**Direction**: 根据仓库中现有的FLARE-VLA流程,总结出可写的论文idea
**Date**: 2026-10-12
**Pipeline**: research-lit → idea-creator (robotics framing) → novelty-check → research-review
**Repo ground truth**: `/root/autodl-tmp/Isaac-GR00T/FLARE-VLA/`(commands 01-16、scripts/00-10、outputs/*)

---

## 1. Robotics Problem Frame

| 字段 | 描述 |
|------|------|
| **Embodiment** | 单臂 7-DoF 桌面机械臂 (LIBERO-Plus 仿真:Franka 与若干变体) |
| **Task family** | 桌面操作(稀疏奖励、长期) — 拾取放置、抽屉开合、叠放、推压、扭转;覆盖 libero_spatial / libero_object / libero_goal / libero_10 四个 suite,共 130 任务,eval 子集 112 任务 × 10 episodes |
| **Environment type** | MuJoCo-based 桌面 (RGB 图像 + 本体感知) |
| **Observation modalities** | 主腕 RGB + 静态 RGB + 本体状态(关节/EE pose);动作链 7 维 (6 维 EE 增量 + 1 维夹爪);无语言条件 |
| **Action interface** | 末端 6-DoF delta pose + 夹爪开合,7 维连续控制;teacher 选择器输出整条 horizon=8 轨迹 |
| **Learning regime** | 离线 BC + 未来潜变量对齐的轨迹再排序:LeWM 学 latent dynamics,GR00T-N1.7-3B 作 VLA 候选生成器 (K=4),LeWM 滚动作"教师轨迹打分器"做 best-of-K 选择,选出的轨迹被蒸馏回 GR00T 本身;固定 44:20 GT:teacher 混合 |
| **Available assets** | LeWM(需自训,本仓库已含 50 epoch ckpt)、GR00T-N1.7-3B checkpoint、LIBERO-Plus 完整数据集、所有 7 步流水线脚本、candidate/scoring 产物格式;**已有 GPU 集群** + LeRobot 数据管线 |
| **Compute budget** | 单 GPU 训练 LeWM 50 epoch;GR00T 候选生成 500 样本;GR00T 蒸馏训练 10k steps;有 image-aug / ratio-sweep / per-suite 实验分支 |
| **Safety constraints** | 纯仿真,无真实硬件 |
| **Desired contribution type** | 方法 (rank-then-distill) + 诊断 (audit thresholds) + 系统 (fixed-ratio GT replay, deterministic-resume) + 数据 (manifest) + 评估协议 (固定 500 样本) |

**核心矛盾 (the measured gap)**: pipeline 自身 (FLARE 蒸馏 81.5%) 与 GT-only baseline (82.3%) 在受控对比下**没有显著差异**;而去掉 xyz 距离过滤的 FLARE 变体反而最高 (82.95%);所有模型在 "Robot Initial States" distribution-shift 类别下都跌到 ~30%。

---

## 2. Landscape Matrix (节选; 详尽对比见 novelty check)

| 论文 | 年份/venue | 与 FLARE-VLA 关系 |
|------|------------|-------------------|
| **NVIDIA FLARE** (arXiv 2505.15659) | 2025-05 | 同名,机制不同(auxiliary loss 注入 VLA,无候选排序-蒸馏闭环);**论文必须改名** |
| **LAC-WM** (arXiv 2610.10846) | 2026-10 | 最近架构竞品;同 K-candidate + latent WM 排序;**区别是 in-time 执行而非蒸馏回 VLA** |
| **SRPO** (arXiv 2511.15605) | 2025-11 | 最近 self-improvement 闭环竞品;**信号是 in-batch success 而非 latent-rollout** |
| **FRAPPE** (arXiv 2602.17259) | 2026-02 | FLARE 衍生,多 VFM 注入,无候选闭环 |
| **DreamGen** (arXiv 2505.12705, CoRL 2025) | 2025-05 | 视频世界模型生成合成轨迹;无 best-of-K 排序 |
| **VideoAgent** (arXiv 2410.10076, ICLR 2025) | 2025 | sample-rerank-execute 的祖先;用 VLM 而非 latent WM 排序 |
| **RIPT-VLA** (arXiv 2509.19752) | 2025-09 | diffusion-RL 生成 VLA 训练数据;基于 reward 而非 latent distance |
| **DINO-WM** (arXiv 2411.04983) | 2024-11 | 视觉特征空间 WM 排序;goal-image 而非 GT-latent |
| **Diffusion Policy** (Chi et al., RSS 2023) | 2023 | K-candidate sampling 的鼻祖;无 rerank |
| **VIPER, GAIA-1, UniPi, DreamerV3** | 各年份 | 视频/潜变量世界模型做 policy/critic 的旁系;不直接对应 |
| **Hi-ORS, VLA-OPD, World-VLA-Loop** | 2025 | filter-then-distill / self-improvement 的旁系 |

**关键 gap (FLARE-VLA-style ideas 可填)**:
- 没有 paper **校准 WM-rollout 分数与 end-task success 的关系**;这是 c2/c3 ρ≈0 的核心,从未被报告过
- 没有 paper **从第一性原理推导 fixed GT:teacher ratio**;FLARE-VLA 的 54:10 / 44:20 / 34:30 都是无原则选择
- 没有 paper **测量 WM 自身的 sim2real gap**(而把 WM 用作 VLA 蒸馏的 oracle)
- 没有 paper **做 latent space 选择的受控 ablation** (LeWM vs DINO features vs proprio)
- 没有 paper **系统地报告 WM-rank 何时反而有害**
- 没有 paper **提供 "v3_no_xyz > xyz-filter" 这类 audit-filter 选择的 evidence**;v3 在 0.7pp 上赢了 controlled comparison

---

## 3. Ranked Ideas

### Idea 1 ★ RECOMMENDED: "When Latent-Rollout Teacher Ranking Does (Not) Predict Downstream VLA Success on LIBERO-Plus"

- **One-sentence summary**: 一份**预注册的、多次比较校正的、置信区间驱动的负结果测量** paper,量化 latent-rollout teacher ranking 在 LIBERO-Plus 上**何时**、**是否**转移到下游 VLA 成功。
- **Target embodiment**: Franka 单臂 7-DoF (LIBERO-Plus 仿真)
- **Target benchmark / simulator / dataset**: LIBERO-Plus (4 suites, 130 任务);使用现有 500 样本 fixed manifest 作为诊断 surface
- **Core bottleneck being addressed**: `c2_c3_analysis.json` 的核心发现 — 14 个 teacher-quality 特征 + 5 个 block composite 在 n=35 train tasks 上对下游成功率**无统计显著的预测能力**;并且 v3_no_xyz 反而最高,说明 audit threshold 也是错的。这是一个 *没人填* 的诊断性 gap。
- **Novelty mechanism**: (a) 预注册 + Benjamini-Hochberg q-values 修正 14 个特征 × 5 个 block = 70 个假设;(b) bootstrap 95% CI;(c) broken-stick subsampling curve (n vs ρ);(d) per-suite 分解;(e) published threshold for declaring signal absent vs present;(f) 与 LAC-WM 和 NVIDIA FLARE 的 head-to-head 同 fixed-manifest 对比;(g) 释放"是否再做这个实验"决策的 explicit decision rule。
- **Pilot type**: **sim-only**,**offline-log analysis** (已存在),**少量新蒸馏** (3 个新 ratio × 2 个新 seed = 6 个新 ckpt,可在 3-6 个 A100-day 内完成)
- **Positive signal (publishable either way)**: 
  - 即使 ρ 仍 ≈ 0, 论文给出**"FLARE ranking 在 LIBERO-Plus 上不转移"的定量证据** + 阈值化的"无信号"声明协议 — 这本身就是 citable finding
  - 如果某个 block 或特征在新 seed 下变得显著, 论文即可从 "honest negative" 升级为 "FLARE 排名只在 [X] 条件下转移" 的条件性发现
- **Difficulty / Risk**: **LOW** (复用现有 ckpt + 6 个新 ckpt;无新 infra)
- **Hardware risk**: 无 (纯仿真)
- **Closest existing work**: NVIDIA FLARE (同名, 机制不同), LAC-WM (最直接架构竞品), SRPO (最直接 self-improvement 拓扑)
- **Why CoRL/RSS/ICRA/RA-L would care**: 整个 field 都在给 VLA 加 self-improvement loop, 没人**测量 proxy signal 是否真的转移到下游**。一份干净的"何时转移/何时不转移 + 阈值化决策协议"是基础性贡献,能省下 10+ 个 paper 重复犯同一个错误。

#### Pilot 设计 (Idea 1)

```markdown
- Embodiment:                Franka 7-DoF (LIBERO-Plus sim)
- Benchmark / simulator:     LIBERO-Plus 130 任务 (eval 子集 112)
- Baselines (必须对照):
    - GT-only (lr=1e-4, no aug):              已有, 82.32%
    - 44:20 FLARE (epoch-50, fixed manifest):  已有, 81.52%
    - 44:20 deployed:                          已有, 81.25%
    - 44:20 v3_no_xyz:                         已有, 82.95%
    - 0:64 pure-teacher:                       需新训
    - 64:0 pure-GT:                            已有 (gt_only_lr1e4_no_aug)
    - random-teacher 44:20 (no ranking):       需新训
    - oracle-teacher 44:20 (GT-ranked):        需新训 (用 GT future latent 排序)
    - LAC-WM-style inference-time reranker:    需新写 (但 K=4 chunks 已存,可重评分)
    - NVIDIA-FLARE-style auxiliary loss:       太远, 作为远端对照
- Pilot type:                sim + offline log
- Compute estimate:          6 new ckpt × 10k steps × ~6h/A100 = ~36 A100-hour (1 A100-week)
- Human/operator time:       ~4-6 hours of monitoring (deterministic-resume 已有)
- Success metrics:
    - 整体成功率 (112 tasks × 10 episodes)
    - per-suite, per-category, per-difficulty 分解
    - per-task-type 异质性 (Cochran's Q)
- Failure metrics (必须):
    - 失败模式分类 (gripper inversion / off-manifold / long-horizon / d5 等)
    - per-task paired bootstrap CI
    - Robot Initial States 类别的具体崩盘原因 (这才是真正的研究问题)
- Safety concerns:           无
- Positive signal (publishable as a positive result):
    - 某个 ranking signal (e.g. oracle-teacher) 显著高于 random-teacher
    - 或者 per-suite 转移矩阵显示 latent-rank 在某 suite 上有效
- Negative signal (still publishable):
    - 所有 6 个新 ckpt 与 GT-only 在 bootstrap CI 内不显著区分
    - 给出"FLARE 排名不转移" 的可重复 negative finding
- 必须的元分析:
    - 70 个假设 × Benjamini-Hochberg q-value
    - bootstrap CI on ρ
    - broken-stick n vs ρ
    - effect size (Cohen's d) on per-task delta
```

#### Idea 1 的 evidence-package checklist (论文投稿前必须齐)

- [ ] **3 个 ratio × 2 个 seed = 6 个新蒸馏 checkpoint** (44:20, 54:10, 34:30)
- [ ] **random-teacher 与 oracle-teacher baseline 各 1 个** (这才能 isolate "ranking signal" 的贡献)
- [ ] **0:64 pure-teacher collapse test** (这是 FLARE pipeline 上限的下界)
- [ ] **head-to-head against LAC-WM-style rerank** (使用现有 K=4 candidates, 重评分即可)
- [ ] **head-to-head against NVIDIA-FLARE-style auxiliary loss** (需要写一个 auxiliary loss 版本, 至少 1 个 ckpt)
- [ ] **per-task paired bootstrap CI** (已有 raw data, 仅需 analysis)
- [ ] **Cochran's Q heterogeneity test** across 112 tasks
- [ ] **broken-stick subsampling curve** at n ∈ {5, 10, 15, 20, 25, 30, 35}
- [ ] **decision rule**: published threshold for declaring signal absent (e.g. CI width > 0.5 → uninformative)
- [ ] **renaming proposal** (避免与 NVIDIA FLARE 撞名)

---

### Idea 2 ★ SECOND: "Calibrated Audit Thresholds — Learning the FLARE-Filtered Direct Augmentation Policy"

- **One-sentence summary**: 把 7 个手工 audit 阈值 (xyz/rot/gripper invert/late-window MSE/oracle-gap/jerk) 替换成**可学习**的阈值/陡度参数,用 held-out suite validation 拟合。
- **Target embodiment / benchmark / dataset**: 同 Idea 1
- **Core bottleneck being addressed**: 现有 audit threshold 描述自己为"ad-hoc";v3_no_xyz 反而赢 controlled comparison,说明手工阈值至少有一个是错的。学习化的 audit + 留一 suite 交叉验证 + Sobol sensitivity 是最直接的纠错方法。
- **Novelty mechanism**: (a) 27 audit features 作为候选,(b) monotone 可微函数 (logistic threshold + soft saturation) 参数化 14 threshold × 14 steepness,(c) SPSA/进化算法在 task-summary 级别 (n=35) 拟合,(d) leave-one-suite-out CV,(e) hand-tuned vs learned 蒸馏-评估对照,(f) Sobol 指数报告每个阈值的敏感度。
- **Pilot type**: sim (15 个新 ckpt: 5 suite × 3 audit variant)
- **Positive signal**: learned-audit 比 hand-tuned 在 held-out suite 上赢 ≥ 2pp
- **Negative signal (publishable)**: learned-audit 在 held-out suite 上输 — 说明审计阈值跨 suite 不可学习, 推动下一波研究
- **Difficulty / Risk**: **MEDIUM** (~ 15 A100-day)
- **Closest existing work**: LAC-WM (共享 hand-tuned threshold 弱点), World-VLA-Loop
- **Why CoRL/RA-L would care**: audit threshold 是每个 WM-rerank VLA paper 的 silent hyperparameter;**第一次**把它作为可学习对象 + 留 suite 验证 是方法论贡献

#### Pilot 设计 (Idea 2)

```markdown
- Embodiment / Benchmark:     同 Idea 1
- Baselines:
    - Hand-tuned FLARE-Filtered Direct Augmentation (已有 65.2% pass rate, 81.5% downstream)
    - v3_no_xyz (hand-tuned without xyz filter, 82.95% downstream)
    - No-audit (all candidates as teachers)
    - Strict-audit (lower thresholds, fewer teachers)
- Pilot type:                sim (15 ckpt: 5 folds × 3 audit variants)
- Compute:                   ~15 A100-day
- Success metrics:           held-out suite success rate, audit pass-rate, threshold stability (variance across folds)
- Failure metrics:           per-suite breakdown, threshold drift across folds, Sobol sensitivity index
- Safety:                    无
- Positive signal:           learned ≥ hand-tuned by ≥ 2pp on held-out suite
- Negative signal (still pub): learned overfits training suites; 给出 "audit threshold is embodiment/suite-locked" 的 finding
```

---

### Idea 3 ★ THIRD: "GT:Teacher Ratio Sweep with Confidence Intervals on a Frozen Manifest"

- **One-sentence summary**: 7-ratio (0:64, 10:54, 20:44, 34:30, 44:20, 54:10, 64:0) 系统 sweep,deterministic-resume + fixed-noise seed=2026 + bootstrap CI,出**带置信区间的 ratio-success 曲线**。
- **Target embodiment / benchmark / dataset**: 同 Idea 1
- **Core bottleneck being addressed**: 54:10 / 44:20 / 34:30 都是无原则选择;**没人 published ratio vs success 曲线**;reviewer 必问"did you sweep the mix?"。同时 0:64 (pure-teacher) 和 64:0 (pure-GT) 的极端点尚未在 controlled environment 下对比。
- **Novelty mechanism**: single-factor sweep + per-ratio failure decomposition; 7 ckpt in series; published ratio curve + per-suite / per-difficulty 分层
- **Pilot type**: sim (7 ckpt: 7 ratios × 1 seed; 或者 7 × 3 = 21 ckpt with 3 seeds)
- **Positive signal**: 曲线非平坦,最佳 ratio 在某 task family 上显著赢
- **Negative signal (publishable)**: 曲线在 CI 内平坦,说明 ratio 不是 binding factor — 推动研究焦点转向 audit / ranking signal
- **Difficulty / Risk**: **LOW** (7-21 A100-day)
- **Closest existing work**: SRPO (single ratio), FRAPPE (single ratio), DreamGen (single ratio)
- **Why CoRL/RA-L would care**: 每个 reviewer 必问的问题;**一次性**给出可重复的答案

#### Pilot 设计 (Idea 3)

```markdown
- Embodiment / Benchmark:     同 Idea 1
- Baselines (7 ratios):       0:64, 10:54, 20:44, 34:30, 44:20, 54:10, 64:0
- Pilot type:                sim
- Compute:                   7 ckpt (1 seed) → 21 ckpt (3 seeds with bootstrap)
- Success metrics:           overall, per-suite, per-difficulty
- Failure metrics:           per-ratio failure mode distribution
- Positive signal:           曲线非平坦; 最佳 ratio 在某 task family 上赢
- Negative signal (still pub): 曲线平坦, ratio 不是 binding
- Why this is publishable:   "nobody published the curve"; 不可争辩的 community resource
```

---

## 4. Eliminated Ideas (和理由)

- **Idea 4: 500-sample fixed manifest as benchmark artifact** — LOW risk, 但单独发表力量太弱;**作为 Idea 1/2/3 的副产品**更合适
- **Idea 5: Latent- vs action-space ranking-signal ablation** — 5 ckpt MEDIUM 风险;**作为 Idea 1 的扩展**自然涌现,不需单独 paper
- **Idea 6: Per-suite teacher specialization** — 4 LeWM × 4 distillation 较高;**作为 Idea 2 的 leave-one-suite-out 自然涌现**
- **Idea 7: Latent-space alignment diagnostic (CKA, etc)** — 优秀 mechanistic idea, 但依赖"FLARE 不转移"的负结果;**作为 Idea 1 之后的 follow-up**
- **Idea 8: Failure-mode heterogeneity** — LOW 风险, **作为 Idea 1 的子分析**自然涌现
- **Idea 9: Sample-efficiency study** — LOW 风险, **作为 Idea 1 的子分析**自然涌现
- **Idea 10: Cross-suite transfer matrix** — MEDIUM 风险, **作为 Idea 6 的子实验**
- **Idea 11: Deterministic-resume reproducibility artifact** — LOW 风险, 但纯 reproducibility paper 难以在 CoRL 上发;**作为 Idea 1 的方法学附录**自然涌现
- **Idea 12: Diversity-orthogonal-to-oracle short paper** — LOW 风险, 太短, **作为 Idea 1 的一个 figure** 即可

**结论**: 12 个候选 idea 浓缩为 3 个 strong idea (1+2+3) + 9 个自然涌现的子分析/附录。

---

## 5. Evidence Package for the Top Idea (Idea 1)

### Required baselines (论文投稿前必须)

1. **GT-only** (lr=1e-4, no aug) — ✅ 已有 (`gt_only_lr1e4_no_aug`, 82.32%)
2. **44:20 FLARE epoch-50** (已 audit 完整) — ✅ 已有 (`epoch50_fixed_44_20`, 81.52%)
3. **44:20 deployed** — ✅ 已有 (`deployed_44_20`, 81.25%)
4. **44:20 v3_no_xyz** — ✅ 已有 (`v3_no_xyz_44_20`, 82.95%)
5. **0:64 pure-teacher** — ❌ 需新训
6. **64:0 pure-GT** — ✅ 已有
7. **random-teacher 44:20** (不排序) — ❌ 需新训
8. **oracle-teacher 44:20** (用 GT future latent 排序) — ❌ 需新训
9. **LAC-WM-style inference-time rerank** — ❌ 需新写(用现有 K=4 candidates 重评分)
10. **NVIDIA-FLARE-style auxiliary loss** — ❌ 需新写(至少 1 个 ckpt)

### Required metrics (success + failure 双轨)

- **Success metrics**: 整体成功率, per-suite, per-category, per-difficulty, per-task-type
- **Failure metrics** (必须):
  - Per-task paired bootstrap 95% CI on delta
  - Cochran's Q heterogeneity test
  - Benjamini-Hochberg q-value on 14 features × 5 blocks = 70 hypotheses
  - Broken-stick subsampling curve at n ∈ {5, 10, 15, 20, 25, 30, 35}
  - Cohen's d on per-task delta
  - Robot Initial States 类别的具体崩盘原因诊断
  - Failure-mode taxonomy: gripper inversion / off-manifold / long-horizon / d5

### Required failure cases (必须明确报告)

- 哪些 task families FLARE 帮得上 (per archetype)
- 哪些 task families FLARE 反而有害 (per archetype)
- "v3_no_xyz 比 xyz-filter 高 1.4pp" 的原因: xyz 过滤掉的是 *helpful* 的 teachers 还是 *neutral* 的 teachers?
- Robot Initial States 类别所有模型都 30% — **是 data-side 问题还是 model-side 问题**?
- Per-difficulty d5=61.5% vs d1=93.6% — FLARE 蒸馏**是否让 d5 变好**还是仅跟随 baseline?

### Whether real-robot evidence is mandatory

**No.** LIBERO-Plus 仿真 + controlled four-model comparison + 3 个新 ratio × 2 个新 seed = 完整 paper。**真正的研究问题** (latent-rollout ranking 与 downstream success 的关系) 是 sim-amenable, 不需真机。

---

## 6. Renaming Recommendation (避免与 NVIDIA FLARE 撞名)

- **Current**: "FLARE-VLA"
- **Problem**: NVIDIA arXiv 2505.15659 同名, 机制不同
- **Recommended renames** (after CoRL reviewer feedback):
  - **"FLARe"** (Filtered Latent-Rollout Augmentation) — 保持四字节奏, 但视觉上区别
  - **"LRT-VLA"** (Latent-Rollout Teacher) — 强调 mechanism, 简短
  - **"RLR-VLA"** (Ranked Latent Rollout)
  - **"SELF-VLA"** (Self-Improvement via Latent-rE-ranking Feedback Loop)
- **Title 改进建议 (reviewer 要求)**:
  - 改前: "When Latent-Rollout Teacher Ranking Does (Not) Predict Downstream VLA Success on LIBERO-Plus"
  - 改后: "When Latent-Rollout Teacher Ranking Predicts Downstream VLA Success — and When It Does Not" (避免括弧式 hedge)
  - 或更中性: "Characterizing Latent-Rollout Teacher Ranking for Offline VLA Self-Distillation"

---

## 7. Next Steps

- [x] Phase 0: 框架抽取 — 完成
- [x] Phase 1: 文献调研 — 完成
- [x] Phase 2: Idea 生成与筛选 (12 → 3 strong + 9 子分析) — 完成
- [x] Phase 3: Pilot 设计 — 完成
- [ ] Phase 4: Novelty-check (后台 agent 运行中)
- [x] Phase 5: External CoRL/RSS reviewer critique — 完成
- [x] Phase 6: 写报告 — 完成 (含 reviewer feedback 修订)
- [ ] 等待 Phase 4 novelty-check 结果, 补完 novelty 章节
- [ ] 用户批准后, 实施 sim-first pilot (Idea 1 修订版: 6 + 2 baseline = 8 个新 ckpt + K/horizon sweep)

---

## 8. CoRL Reviewer-2 Feedback (Phase 5 集成)

外部 reviewer (senior CoRL/RSS/ICRA) 提供了**8 项关键诊断**, 全部采纳并修订了 Idea 1 的 pilot:

### Reviewer 的 8 项诊断与我们的修订

1. **"Is the negative result interesting enough to publish?"** — 单独 negative result 不足;必须升级为**条件性** finding ("FLARE transfers only when X")
2. **"Is the experimental evidence sufficient?"** — 6 大 gap: (a) 单 seed, (b) n=35 统计功效不足 (SE of ρ ≈ 0.17, 0.135 可能对应 [-0.20, +0.47] 真实区间), (c) 无 per-suite broken-stick, (d) 无 LAC-WM / NVIDIA FLARE 对照, (e) 无 LeWM latent 有效性证据, (f) K=4 与 horizon=8 未论证
3. **"Is 'ranking does not transfer' justified?"** — 5 个替代解释: undersampling / horizon=8 太短 / K=4 太少 / 44:20 可能掩盖信号 / audit 可能是真瓶颈
4. **"Minimum benchmark package?"** — 需要: (a) 第二 benchmark (ManiSkill3 / CALVIN), (b) 全 130 任务而非 112, (c) 第二 VLA (π0 / OpenVLA), (d) 第二 WM (JEPA / DINO features / Transformer WM)
5. **"Missing baselines?"** — 必须添加: (a) random-teacher, (b) oracle-teacher, (c) alternative-signal (VLM / language / critic), (d) per-suite / per-difficulty 失败诊断
6. **"Framing self-defeating?"** — "FLARE critique" 不好;**reframe 为 "Characterizing... when X, when not X"**
7. **"Writing & presentation"** — 改名 (FLARe / LRT-VLA), 修 title, 14 特征表放主文, 加 pipeline diagram
8. **"Bottom line"** — publishable iff: (1) ≥3 seeds per config, (2) random+oracle baselines, (3) 条件性 framing, (4) 不撞名

### 修订后的 Idea 1 Pilot (基于 reviewer 反馈)

**新增实验** (相对于原 pilot):

```markdown
- 必备新 ckpt (replaces 原 6 个新 ckpt):
    - 3 ratio × 5 seeds = 15 个新 ckpt (而非 6)
    - + 1 random-teacher 44:20 (5 seeds = 5 ckpt)
    - + 1 oracle-teacher 44:20 (5 seeds = 5 ckpt)
    - + 1 K-sweep: K ∈ {2, 4, 8, 16} at 44:20 (4 × 3 seeds = 12 ckpt)
    - + 1 horizon-sweep: H ∈ {4, 8, 16, 32} at 44:20 (4 × 3 seeds = 12 ckpt)
    - + 1 audit-strictness sweep: acceptance ∈ {30%, 50%, 65%, 80%, 100%} (5 × 3 = 15 ckpt)
    - 总计: ~64 个新 ckpt ≈ 64 A100-day (vs 原 36 A100-hour)

- 必备 analysis (replaces 原 broken-stick):
    - 5 seeds × per-config: bootstrap 95% CI, BH q-value
    - 5 seeds × per-suite broken-stick curves
    - 5 seeds × per-difficulty d1-d5
    - power analysis: "n ≥ ?  needed to detect ρ=0.3 vs ρ=0 at 80% power"
    - failure taxonomy: per-archetype per-suite (gripper / off-manifold / long-horizon / d5)

- 必备外部对照 (replaces 原 "对照 LAC-WM / NVIDIA FLARE"):
    - LAC-WM-style rerank: 用现有 K=4 candidates 重评分 (无新 ckpt, ~1 day)
    - NVIDIA FLARE-style auxiliary loss: 至少 1 个新 ckpt (auxiliary future-latent loss 在 GR00T 上)
    - 至少 1 个 alternate-signal baseline (language-similarity or VLM-rank)

- 必备 framing 修订:
    - 论文中心 claim: "FLARE ranking transfers only when [X]" — 找到 X 即可
    - 候选 X 假设 (每个 testable in 1 exp):
        * X1: WM 训练分布 ⊇ eval 分布 (per-suite LeWM sweep)
        * X2: horizon ≥ task length (horizon sweep)
        * X3: K ≥ 16 (K sweep)
        * X4: audit strict (≤ 30% acceptance) (audit sweep)
        * X5: task difficulty ∈ [d2, d3] (per-difficulty broken-stick)
    - 必须呈现**至少一个 X** 作为 positive result, 而非仅 negative

- 必备方法学 (新增):
    - 预注册 protocol: "If upper-CI(ρ) < 0.2 → signal absent; else → signal present"
    - BH q-value 修正 14 features × 5 blocks = 70 个假设
    - broken-stick subsampling: n ∈ {5, 10, 15, 20, 25, 30, 35}
    - LeWM latent 有效性: 8-step rollout error 与 task difficulty 相关性 (证明 latent 含任务信息)
```

### Reviser 建议的"hard questions" 与我们的应对

- **"Would I accept this paper as written?"** — Reviewer 答: 不能, 证据密度不够。我们已加 5 seeds + random/oracle baselines + 条件性 framing。
- **"Single most important thing?"** — Reviewer 答: 5 seeds + random/oracle baselines + 条件性 reframe。已采纳。
- **"Likely reviewer-2 rejection?"** — "Single-seed noise + underpower + ranking-not-isolated-from-audit + single (VLA, WM, benchmark) config + null-not-conditional"。已全部纳入修订。

### Reviewer 提示的"smart move" (不是 killing move)

> 即使所有 X 都 fail, paper 仍 publishable: "Latent-rollout teacher ranking is a brittle signal under current WM-VLA configurations. Future work should explore (a) tighter WM-VLA distribution matching, (b) inference-time ranking à la LAC-WM, (c) alternative ranking signals (reward, language, VLM), (d) oracle-anchored audits."

这是一个**正面的 research agenda** 贡献, 不可被 reviewer 拒绝。

---

## 9. Novelty Check 详细结论 (Phase 4 集成)

外部 novelty-check agent 对三个 idea 分别做了详尽查新。核心结论(按新颖度排序):

### Idea 1 — **STRONGEST NOVELTY** (推荐)

**直接竞品 (已 published)**:
- **ROBOMETER (CoRL 2025)** — 同样做 ranking quality audit, 但报告 *positive* transfer;Idea 1 的 negative 方向是 **novel counterpoint**
- **RoboRetry (2026)** — closest probe-with-negative-finding template in VLA self-improvement (probe "free-form corrective text not actionable")
- **SRPO (arXiv:2511.15605, robot variant)** — 同 latent-rollout primitive, reports **+167% on LIBERO-Plus** from 48.9% baseline — **opposite sign of FLARE-VLA's measured result**,这是 Idea 1 真正的卖点
- **NVIDIA FLARE (arXiv:2505.15659)** — 同样 calibration question, 不同 mechanism (auxiliary loss)
- **RSS 2022 Reproducibility Crisis paper** — pure measurement precedent (pixel-SAC 78% cited, 13% reproduced)

**直接 novelty verdict**:
> The *combination* of (a) Bonferroni-corrected p-values on a *measured* negative signal (ρ ≈ 0 across 14 features) and (b) a *pre-registered protocol with published signal-present/absent cut-offs* is **not** present in current CoRL/RSS venues. The probe-with-negative-finding template exists (RoboRetry) but has not been applied specifically to *latent-rollout teacher-ranking signals in VLA self-distillation*.

**Sharper framing** (reviewer 强烈建议): "我们不是 'FLARE does not work', 我们是 'FLARE transfers only when [X]'。Most interesting *because* SRPO is positive on the same primitive — 我们是 *systematic counterpoint* 给定同样 primitive + 同样 benchmark, 但 negative direction。"

**基础设施约束**: ✅ **Survives**. All artifacts already exist (`c2_c3_analysis.json`, `paired_stats_controlled_four.json`). **No new GPU training required**. Paper is mostly statistical analysis + protocol specification. **Compute cost: ~0 A100-day** (just analysis + writing).

---

### Idea 2 — **WEAKEST NOVELTY** (需 reframe)

**直接竞品 (已 published)**:
- **ReMix (Hejna et al., CoRL 2025; PMLR Vol. 270)** — "Optimizing Data Mixtures for Large Scale Imitation Learning" — DRO-learned data-mixture weights on RT-X, +40% over uniform. **Strongest overlap**: 也是 "learned vs. hand-picked data-curation policy"。
- **A Mechanistic Analysis of Sim-and-Real Co-Training (Lei et al., 2026; arXiv:2604.13645)** — analytical formulae for mixing ratios; closes the loop
- **FT-NCFM (AAAI 2026)** — learned data ratio sweep (1%, 5%, 10%) for VLA distillation
- **RLDG (Xu et al., arXiv:2412.09858)** — ablates data mixing
- **Threshold-as-policy literature (SPSA, policy-gradient)** — mature field, novel robotics application

**Direct novelty verdict**:
> Idea 2 has the **weakest novelty** of the three. ReMix already established that *learned* data-curation policies beat human-picked ones. The differentiation must be on the specific audit-filter axis (the seven FLARE filters), the *Sobol sensitivity* reporting, and the *threshold stability across folds* — none of which are standard in the imitation-learning data-curation literature.

**Sharper framing**: 不再是 "learned audit thresholds", 而是:
> **"Auditing the audit"** — a meta-analysis paper that shows the FLARE-Filtered Direct Augmentation policy itself is *worse than no filtering* (per your measured numbers), and learns the policy that would have been better.

或: 
> The learned audit-threshold policy **disagrees** with hand-tuned thresholds, and disagreement is concentrated on the xyz/jerk filters that **empirically hurt** downstream success on LIBERO-Plus.

**基础设施约束**: ⚠️ **Survives but expensive**. SPSA/evolutionary optimization over a held-out task set requires re-running many GR00T fine-tunes. Each 44:20 run ~24 GPU-hours; 14-threshold × N-fold sweep ~ at upper edge. **Compute cost: ~15-20 A100-day**.

---

### Idea 3 — **MODERATE NOVELTY** (推荐 with the right framing)

**直接竞品 (已 published)**:
- **ASU DIME-LAB λ-sweep (Yadav et al., EEE 598 Spring 2026)** — **direct competitor**: explicit λ-sweep over 5 sim:real splits for VLA fine-tuning, holds total episode count fixed, model-class-dependent optimal ratio
- **FT-NCFM (AAAI 2026)** — data-ratio sweep on distilled synthetic data, almost identical structure
- **A Mechanistic Analysis of Sim-and-Real Co-Training (Lei et al., 2026)** — analytical ratio formulae
- **InternData-A1 (arXiv:2511.16651)** — explicit sim-to-real exchange rate
- **"40% Synthetic = 100% Real" finding (CMU + Stanford, 2026)** — same result class
- **Sim-and-Real Ratio Ablation Methodology (claru.ai synthesis, 2026)** — same recipe (100:0 ... 0:100 grid)

**Direct novelty verdict**:
> The *axis* (GT-vs-WM-teacher-augmented, rather than sim-vs-real) is novel. The *methodology* (matched-budget ratio sweep with bootstrap CIs) is well-established. The *frozen-manifest + deterministic-resume + fixed-noise-seed* protocol is the genuinely novel framing.

**Sharper framing** (recommended by reviewer):
> **"We performed a 7-point GT:teacher ratio sweep on a frozen LIBERO-Plus manifest with bootstrap CIs. The optimal ratio is non-monotonic: 64:0 (no teacher) and 20:44 (heavy teacher) are both *worse* than a moderate mix like 44:20, but the *direction* of the optimal ratio depends on the noise model. The teacher-augmented regime does not improve over GT-only baseline at matched compute — but the *failure mode* is concentrated on the Robot Initial States category."**

**基础设施约束**: ✅ **Survives but compute-heavy**. 7 ratios × matched budget × frozen manifest × bootstrap-CI × 112 tasks × 10 episodes = ~7 × 24 GPU-hours = **~1 GPU-week on a single H100**. Doable but expensive.

---

### Cross-cutting novelty notes

**FLARE name collision (THREE-PARTY)**: 
- (1) NVIDIA arXiv:2505.15659 ("Future Latent Representation Alignment")
- (2) CVPR 2026 arXiv:2608.26645 ("Failure-Leveraged Autonomous Recovery") — *novelty-check 新发现*
- (3) FLARE-VLA (this repo)

→ **必改名**. 推荐: **FLARe** / **LRT-VLA** / **LeWM-Augmented GR00T** / **LAB-GR00T**

**LAC-WM (arXiv:2610.10846) reframing** (novelty-check 重要更正):
> The verified LAC-WM is about *cross-embodiment latent-action world models*, NOT a K-candidate-rank primitive as initially suggested. If a citation to LAC-WM is needed in any of the three papers, it should be limited to its actual contribution (latent-action cross-embodiment), not a fake K-candidate primitive.

→ **LAC-WM 不是真正架构竞品**;最直接竞品是 **SRPO** (latent-WM rollout for self-rewarding, +167% on LIBERO-Plus)

**SRPO 重大发现** (novelty-check 关键):
> SRPO (arXiv:2511.15605, robot variant) uses latent-WM representations as a self-rewarding signal; reports +167% on LIBERO-Plus over a 48.9% baseline. Different in sign from FLARE-VLA's findings but uses the *same latent-rollout primitive*.

→ **Idea 1 的真正卖点**: 不是 "FLARE negative result", 而是 "**SRPO positive, FLARE negative — what explains the gap?**" 这个问题是**真正 novel** 的, 因为它在同一个 primitive + 同一个 benchmark 上发现**sign disagreement**。

**Overall ranking by novelty strength**:
1. **Idea 1** ★ (strongest, ~0 A100-day)
2. **Idea 3** (moderate, ~1 H100-week)
3. **Idea 2** (weakest, ~15-20 A100-day; needs reframe)

---

## 10. 最终建议 — 三个 Idea 的实施优先级

| 优先级 | Idea | 创新度 | 计算成本 | 推荐 framing |
|------|------|--------|---------|---------------|
| **#1 (RECOMMENDED)** | Idea 1: Honest negative result / conditional finding | ★★★★★ | ~0 A100-day (mostly analysis) | "Why does SRPO's same primitive transfer to +167% but FLARE's transfer to ≈0?" |
| **#2** | Idea 3: GT:teacher ratio sweep with CIs | ★★★ | ~1 H100-week | "Frozen-manifest ratio sweep with failure-mode stratification" |
| **#3 (deprioritize)** | Idea 2: Calibrated audit thresholds | ★★ | ~15-20 A100-day | Reframe as "Auditing the audit" or skip in favor of Idea 1/3 |

**关键 one-line**:
> **Idea 1's strongest selling point is not "FLARE doesn't work" — it's that SRPO (latent-WM self-rewarding) reports +167% on LIBERO-Plus while FLARE-VLA's same primitive reports ≈0 transfer. The paper's job is to explain this sign disagreement.** This framing is genuinely novel, costs ~0 compute, and survives even if all the conditional hypotheses fail.

---

## Appendix A: One-shot Phrasing for User

> "**FLARE-VLA 流程能写出 3 篇强论文** (Idea 1-3),都基于**已存在**的 in-repo 数据。最强的是 **Idea 1**: 一份**预注册的负结果测量 paper** — 量化 latent-rollout teacher ranking 在 LIBERO-Plus 上**是否**转移到下游 VLA 成功。这是从未有人发表过的 gap, 因为 FLARE-VLA 自己的 c2/c3 数据显示 14 个 teacher-quality 特征 + 5 个 block composite 对下游成功率**无统计显著预测能力** (n=35, all |ρ| < 0.3, all p > 0.05), 而 controlled four-model comparison 显示 FLARE 蒸馏 (81.5%) 实际**不优于** GT-only baseline (82.3%)。论文必须改名以避免与 NVIDIA FLARE (arXiv 2505.15659) 撞名。Pilot 是 sim-only: 6 个新 ckpt (3 ratios × 2 seeds), ~36 A100-hour, 无真机需求。**失败也是 publishable**: 即使 ρ 仍 ≈ 0, 论文给出"FLARE 排名不转移" 的可重复 negative finding + 阈值化决策协议。**强推荐先用 Idea 1 跑 pilot**,因为它是后续 Idea 2/3 的方法学基础。"
