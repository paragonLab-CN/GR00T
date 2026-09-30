# Experiment Audit Report — FLARE-VLA vs Isaac-GR00T GT-only

**Date**: 2026-09-23
**Project**: Isaac-GR00T / FLARE-VLA (LIBERO-Plus 蒸馏对比)
**Auditor**: ⚠️ **Executor-side analysis (Claude) — 跨模型评审后端不可用**

> 本会话未挂载 `mcp__codex__codex` / `mcp__manual_review__review` 任一评审后端，
> 因此**未能执行本 skill 要求的跨模型（cross-model）完整性评审**。下方结论由执行器
> （Claude）直接阅读代码与结果文件得出，并已明确标注；它**不构成**跨模型独立裁决。
> 若需合规裁决，请补装 `manual-review` MCP 后重跑（见文末）。

## 核心结论（一句话）

**这主要不是「实验造假/度量失真」问题，而是「方法前提不成立」的设计问题。**

## Overall Verdict: WARN（设计层面；完整性层面基本 PASS）

## 完整性（Integrity）检查 — 执行器侧

### A. Ground Truth Provenance: ✅ PASS
- 最终评估指标 `success_rate = successes/episodes` 来自 **真实 LIBERO 仿真**：
  `gr00t/eval/sim/LIBERO/libero_env.py:164` `info = {"success": self._env.check_success()}`。
- 教师选择所用的「未来 latent GT」来自真实演示数据的 latent 帧库（
  `gt_latents/libero_plus_epoch50_h8_framebank_trainpipe.h5`），非模型输出伪造。
- 「teacher」本身是 GR00T 开环候选经 LeWM 评分筛选的产物，README 已明确标注为
  proxy（future-latent alignment），非伪装成真值。

### B. Score Normalization: ✅ PASS
- 未发现除以模型自身 max/min/mean 的自归一化。成功率是二值成功计数占比。
- `four_model_comparison.json` 中 success_rate 均为真实计数比（如 845/1120=0.7545）。

### C. Result File Existence: ✅ PASS
- 所有被引用的 summary/comparison JSON 均存在且数字一致。
- 四模型对比文件存在：`...suite_corrected_gt_replay_54_10.../four_model_comparison.json`。
- 无「幽灵结果」（引用了不存在的文件/从未计算的函数）。

### D. Dead Code: ⚠️ WARN（轻微）
- `scripts/flare/` 下有大量一次性 `check_*` / `inspect_*` / `summarize_*` 诊断脚本，
  属开发期工具，不影响主链路，但导致「哪个脚本是当前生效版本」难以一眼辨识。

### E. Scope Assessment: ✅ PASS（对设计问题而言反而是加分项）
- 112 任务 × 10 episode × 4 模型 = 1120 episode/模型，固定扩散噪声 seed=2026，
  逐任务配对（paired），分 suite / category / difficulty 汇报。
- 这个评估**控制得很好**——问题不在评估充分性，而在方法本身。

### F. Evaluation Type: `real_gt`（仿真成功率）
- 最终指标为 `real_gt`（LIBERO 仿真成功判定）；教师选择用 `synthetic_proxy`
  （LeWM 未来 latent 评分），二者分离清晰。

---

## 真正的问题：设计层面（为什么 FLARE 打不过 GT-only）

### 关键证据（`teacher_scoring_epoch50/validation_summary.json`）

| 指标 | 数值 | 含义 |
|---|---|---|
| `oracle_better_than_all_candidates.both_rate` | **0.6103** | 61% 样本里，**GT 动作轨迹比全部 4 条候选都好** |
| `oracle_better_than_all_candidates.mse_rate` | 0.6366 | 按 latent MSE，63.7% 样本 GT 最优 |
| `best_mse_score` (候选) mean | **0.01386** | 最优候选的 latent MSE |
| `oracle_mse_score` (GT) mean | **0.00415** | GT 的 latent MSE（**好 3.3×**） |
| `teacher_agreement.rate` | 0.913 | 教师「自一致」≠「优于 GT」 |

**这就是根本矛盾**：FLARE 用 LeWM 未来 latent 评分来选「比 GT 更好的教师轨迹」，
但 LeWM 自己给出的评分显示——**61% 的情况下，直接用 GT（BC oracle）就比任何一条
GR00T 开环候选更优**。也就是说，FLARE 想蒸馏的那个「比 GT 更强的教师」在数据里
**基本不存在**。

### 教师质量分布（`teacher_quality_summary.json`, epoch50_v2）

- 总样本 489,632，**仅选中 53.4%（261,380）** 作为教师。
- 选中者中 **high 仅 26,561（10%）**，其余 234,819 是 **medium**（90%）。
- 即便「selected」教师，其与 GT 的动作 MSE 上限放宽到 `teacher_gt_action_mse_max=0.1955`
  （high 级也到 0.146）——即教师动作与真值动作**显著偏离**。

### 由此导致的三个具体设计缺陷

1. **教师不优于学生，是「同能力模型的开环自蒸馏」**
   教师候选由 `GR00T-N1.7-LIBERO` **开环**去噪生成（`generate_k_candidates_openloop.py`），
   无环境闭环修正。开环轨迹在 latent 上系统性劣于 GT（61%），把它们当监督目标蒸馏，
   等于用「噪声更重的伪目标」稀释干净的真值监督。

2. **目标冲突 + horizon 不匹配**
   教师 horizon=8、processor horizon=16（`--allow-padding`，mask[8:16]=0），而 GT 是
   16 步全监督。混入 12–15% 的 8 步截断目标，与 GT 目标在动作空间又偏离（MSE 0.15+），
   造成互相拉扯的监督信号，最终拖累收敛。

3. **主 recipe（mix-ratio 0.12）把教师当「占比很小的额外数据」，但收益为负**
   四模型固定噪声对比（checkpoint-10000）实测：
   - gt_only: **0.7964**
   - flare (epoch50, mix 0.12): **0.7545**（−4.2pt）
   - flare + image_aug: **0.7661**（−3.0pt）
   - gt_replay_54_10（54 GT + 10 教师/batch，含图像增广）: **0.8063**（+1.0pt）

   `gt_replay_54_10` 能反超，恰恰印证了上面的诊断：它把 GT:教师压到 **54:10（84% GT）**
   并叠加图像增广，教师只作「轻量扰动」，GT 信号占主导——本质更接近「GT-only + 增广」。

---

## Action Items（建议）

1. **先做「教师是否值得存在」的判据**：把 `oracle_better_than_all_candidates` 作为
   硬门槛。只有当某任务的 oracle 落后于候选时才引入教师；否则该任务教师 = 纯噪声。
2. **教师生成改成闭环 / 更高能力教师**：当前开环候选的 latent 上限被 GT 压制。改用
   闭环 rollout，或让 LeWM 做多步 latent 规划（model-based planning）而非单步挑选。
3. **分离两个被混在一起的变量**：`gt_replay_54_10` 同时改了 (a) 教师占比、(b) 图像增广、
   (c) GT replay。要归因就必须做 2×2 消融：{教师 on/off} × {增广 on/off}。
4. **重新表述方法卖点**：如果教师的 latent 评分与 GT 几乎一致（cosine>0.99），那 FLARE
   的「对齐」信号在 LIBERO-Plus 上几乎没有信息量。要么换任务/基准（更难的 long-horizon），
   要么承认当前设置下「distill a stronger teacher」这个前提不成立。

## Claim Impact

- 「FLARE 优于 GT-only」：**不支持**（当前主 recipe −4.2pt；仅 54:10+增广变体 +1.0pt）。
- 「LeWM 能选出比 GT 更优的轨迹」：**不支持**（61% 样本 oracle 优于全部候选）。
- 「评估流程可靠」：**支持**（固定噪声、配对、checkpoint 对齐，度量真实仿真成功率）。

---

## 评审后端说明（合规性）

本 skill 要求跨模型评审。本会话可用工具中无 `codex` / `manual-review` MCP，
**未伪造跨模型裁决**。如需合规完整审计，安装 manual-review 后端后重跑，例如：

```bash
# manual-review MCP 未安装时
claude mcp add manual-review -- <manual-review-server-command>
```

本文所有结论均来自执行器（Claude）直接阅读以下文件得出：
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50/validation_summary.json`
- `FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50_v2/teacher_quality_summary.json`
- `FLARE-VLA/outputs/libero_plus/distillation_dataset_epoch50/distillation_summary.json`
- `outputs/eval_libero_plus/...suite_corrected.../four_model_comparison.json`
- `FLARE-VLA/commands/libero_plus/{08b,09,09b,10,14}_train_*.sh`
- `scripts/flare/generate_k_candidates_openloop.py`、`train_libero_plus_flare_image_aug_gt_replay_54_10.sh`
- `gr00t/eval/sim/LIBERO/libero_env.py`

---

# 附：LeWM 打分 checkpoint 选择（epoch-9 vs epoch-50 vs epoch-100）— 2026-09-23

**问题**：teacher 筛选目前用 LeWM **epoch-9** 打分，是否应改用 **epoch-50**，或把 LeWM 训到 **epoch-100** 以提高 teacher 质量？

**审计状态**：REVIEW_UNAVAILABLE（跨模型后端未配置）。以下为执行器直接读取 sha256 校验过的产物得出。

## 直接回答

**应该改成 epoch-50；不需要、也不建议训到 epoch-100。** 这个结论不是推测——你自己的 `11_checkpoint_sweep` 已经做过 20/30/50 的对比，而且全量 epoch-50 评分已跑完。

## 证据 1：全量评分（N=489,632，同一批 K=4 候选，唯一变量是 LeWM checkpoint）

来源 `teacher_scoring/validation_summary.json`(epoch-9) vs `teacher_scoring_epoch50/validation_summary.json`(epoch-50)：

| 指标 | epoch-9 | epoch-50 | 解读 |
|---|---|---|---|
| `oracle_mse_score` mean | 0.01127 | **0.00415** | GT 轨迹自身在 latent 上的「自评分」好 **2.7×** → 表征质量显著提升 |
| `oracle_mse` step1→step8 | 0.00117→0.02555 | **0.00033→0.00991** | 每个预测步都更准，尤其早期步 |
| `oracle_better.both_rate` | 0.444 | **0.610** | 世界模型更能分辨「GT 优于开环候选」→ 打分更有判别力 |
| `best_mse_score`(候选) mean | 0.01155 | 0.01386 | 候选是**同一批**；差异来自打分模型，不是候选变了 |
| `teacher_agreement.rate` | 0.943 | 0.913 | 略降，但 agreement 只是「自一致」，不是「更准」 |

> ⚠️ 注意：`oracle_mse`(epoch-50 更优) 与 `best_mse`(epoch-9 更低) 看似矛盾，其实不矛盾——epoch-50 的 latent 空间把 GT 和候选**分得更开**（GT 分更低、相对排序更清晰），这正是「打分模型更可信」的含义，不是「候选变差」。

## 证据 2：checkpoint 扫描 20/30/50（`checkpoint_sweep/comparison.json`，5k 子集）

`oracle_mse_mean`：epoch20=0.00631 → epoch30=0.00493 → epoch50=0.00417，单调下降 → **LeWM 表征随训练继续变好**，epoch-9 远未收敛。
（Caveat：该扫描用 `--max-samples 5000` 且 `05_score_libero_plus.py:1209` 取 `all_rows[:5000]` 前 N 个，非随机子集，可能与 manifest 排序相关；但结论与全量 epoch-50 一致，方向可信。）

## 为什么不建议 epoch-100

1. **收益递减**：20→30 提升 −22%，30→50 提升 −15%；曲线在趋于平缓。epoch-100 的边际增益很可能 <10%。
2. **早期步不稳定风险**：epoch-50 的 `oracle_mse step_last_over_first = 31×`（epoch20 是 21.7×）——模型对长期 rollout 的相对误差在累积。继续训练会进一步拉大首尾步差距，反而可能损害对长 horizon 候选的一致评分。
3. **成本**：`02_train_lewm.sh` 是 `max_epochs=50`。再训 50 epoch 约需 3 天（按现有 ckpt 时间戳 ≈75min/epoch），还要重抽 GT latents（~2h/epoch）+ 全量重打分。
4. **更关键的瓶颈不在打分精度**：本审计上文已指出，真正的约束是 **61% 样本 GT 已优于全部开环候选**——换一个更准的打分器只会让这个结论更硬，不会「筛出更多好 teacher」。把算力投在 epoch-100 上，解决不了「teacher 不优于 GT」这个设计问题。

## 关于下游

- 全量 epoch-50 评分 → `teacher_scoring_epoch50_v2` 审计 → `distillation_dataset_epoch50` 均 PASS。
- 但对应的下游模型 `gr00t_libero_plus_flare_epoch50`（脚本 14）**在 5031/10000 步被 KeyboardInterrupt 中断**（见 `epoch50_pipeline_master.log`），且**没有对应 eval**。因此「epoch-50 teacher 带来更高下游成功率」目前**未被验证**——需要先补训 + 补 eval 才能下这个结论。

## 建议（按优先级）

1. **打分 checkpoint 固定用 epoch-50**（已收敛、已全量验证）。把 `05_score_libero_plus.py` / `03_extract_gt_latent_libero_plus.py` 里硬编码的 `weights_epoch_9.pt` 默认值改为 epoch-50，避免后续误用。
2. **先补完 epoch-50 下游训练 + eval**，验证「更准的打分 → 更高成功率」是否成立，再决定要不要继续加码。
3. **不要训 epoch-100**；若想再压 oracle_mse，优先考虑闭环候选生成或 model-based planning（针对 61% oracle 更优这个真瓶颈），而非堆打分器 epoch。
4. 若未来还要做 checkpoint 子集扫描，用 seed-shuffle 的随机子集替代 first-N。

---

# 附 2：FLARE 实验设计审查 — 如何让 teacher 匹敌 GT、让 FLARE 超过 GR00T 基线 — 2026-09-24

**问题**：审查当前实验设计的问题；怎么做才能让 teacher 质量与 GT 匹敌，使 FLARE-VLA 效果超过 GR00T 基线？

**审计状态**：REVIEW_UNAVAILABLE（跨模型后端未配置）。以下为执行器直接读取代码与结果得出，非跨模型独立裁决。

## 当前实测排名（固定扩散噪声、配对、ckpt-10000,112 任务 ×10）

| 模型 | success_rate | 相对 gt_only |
|---|---|---|
| **gt_only（= GR00T 基线）** | **0.7964** | — |
| gt_replay_54_10 (54 GT + 10 teacher/batch + 图像增广） | 0.7812 | −1.5pt |
| image_aug | 0.7661 | −3.0pt |
| flare (mix 0.12) | 0.7545 | −4.2pt |

> 结论先行：**基线 gt_only 是目前最强的。FLARE 各变体全部低于基线。** 注意上一轮记忆里 0.8063 那个数字来自旧目录，本次 rerun 的 `four_model_comparison.json` 是 0.7812——以本次为准。

## 为什么 teacher 追不上 GT：三个结构性原因（按影响排序）

### 1. teacher 是「同能力模型的开环采样」，先天低于 GT（根因）
`scripts/flare/generate_k_candidates_openloop.py`:
- `denoising_steps=4`（`1058` 行默认）、`action_horizon=8`、`candidate_k=4`、`noise_mode=independent`、`noise_scale=1.0`。
- 它从**同一个 GR00T-N1.7-LIBERO**（就是学生/基线本身）开环采样 4 条 8 步动作，无环境闭环修正。
- epoch-50 LeWM 评分（`teacher_scoring_epoch50/validation_summary.json`）:**oracle(GT) 在 61% 样本里优于全部 4 条候选**;`oracle_mse=0.00415` vs 最优候选 `best_mse=0.01386`(GT 好 3.3×)。
- **含义**：你在用「和学生同源、且开环更噪声」的轨迹当监督目标。这类 teacher 的期望质量上限就被 GT 封死，蒸馏它 = 用更脏的伪标签稀释干净 GT。

### 2. horizon / 目标不匹配
teacher 是 8 步，processor 是 16 步（`--allow-padding`,mask[8:16]=0)。混入 12–15% 的截断目标，与 GT 的 16 步全监督在动作空间又偏离（`teacher_gt_action_mse_max≈0.196`)，监督信号互相拉扯。

### 3. 评分器只「排序」不「纠正」
LeWM 只能从这 4 条开环候选里**挑一条最好的**，不能把候选**改得更好**。当 4 条全都差于 GT 时（61% 的情况），挑出「最差中较好者」仍然差于 GT。这是 selection-based FLARE 的天花板。

## 怎么做才可能让 FLARE 超过基线（按优先级/可行性排序）

**核心判据：只有当一个 teacher 样本在其窗口内 LeWM 评分优于 oracle(GT) 时，它才可能提供 GT 之外的信息。当前只有 ~39% 窗口满足（both_rate 反向），其中真正显著优于 GT 的更少。**

### A. 先补验证，再谈改进（必做）
- epoch-50 下游模型在 5031/10000 步被中断、无 eval。先把任一改进跑完 + 固定噪声配对 eval，否则所有结论都是空中楼阁。

### B. 让 teacher 真正能 ≥ GT 的三条路（选一条主攻）
1. **闭环 / 重规划 teacher（最有希望）**：不要开环采样。让 GR00T 在 LIBERO 仿真里**闭环 rollout**，每步用 LeWM 对下一状态 latent 打分做 **MPC/重规划**(model-predictive control)，或执行后用真实仿真 `check_success` 过滤，只保留**成功且** latent 一致的轨迹。这样 teacher 才可能超过开环 GT（尤其在 GT 演示本身次优的长 horizon 任务）。
2. **只用「打败 GT」的子集**：把 distillation 的硬门槛从「agreement + 非离群」改成 **`candidate_mse < oracle_mse`（按窗口）**。当前 53% 选中率里混着大量差于 GT 的样本；只蒸馏那部分真正优于 GT 的窗口，teacher 信号才是「增量信息」而非噪声。
3. **Model-based 增广而非替换**：用 LeWM 对 GT 做 latent 空间的多步 rollout 规划，生成**优于演示 GT** 的动作（planner-in-the-loop)，而不是从学生采样里挑。这把 LeWM 从「裁判」变成「教练」。

### C. 训练配方层面（B 选了之后的调参，不是解药）
- **提高 GT:teacher 比**:gt_replay_54_10(84% GT）已比 flare(0.12 mix）好，方向正确。继续往 90%+ GT 压，teacher 只作轻量扰动。
- **horizon 对齐**：把 teacher 生成也提到 16 步，或把监督 mask 收紧到有效 8 步，消除目标拉扯。
- **2×2 消融归因**:gt_replay_54_10 同时改了 teacher 占比 + 图像增广 + GT replay，无法归因。补 {teacher on/off}×{aug on/off} 才能知道是哪个在起作用。

### D. 重新框定卖点（如果 B 都做不到）
如果闭环/planning 后 teacher 仍不超 GT，那在 LIBERO-Plus 上「distill a stronger teacher」这个前提不成立——因为基线本身就是产生候选的那个模型，开环自蒸馏没有信息增量。此时 FLARE 的正确定位是 **GT 增广/正则化**(A/B 测试它是否提升鲁棒性，例如 Robot Initial States 这类基线最弱的分桶，当前 flare 在该桶 0.369 极低），而不是「更强教师蒸馏」。换更难的 long-horizon 基准可能让前提成立。

## Claim Impact
- 「FLARE 能超过 GR00T 基线」:**当前不支持**（全部变体低于 gt_only)。
- 「teacher 质量可与 GT 匹敌」:**当前不支持**(61% GT 更优，候选 MSE 差 3.3×)；需闭环/重规划或按窗筛选才有望。
- 「评估协议可靠」:**支持**（固定噪声、配对、真实仿真成功率）。
