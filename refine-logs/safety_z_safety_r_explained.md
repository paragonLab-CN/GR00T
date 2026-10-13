# 安全_z 与 安全_r:直观解释

> 这两个术语来自 `refine-logs/FINAL_PROPOSAL.md`(Round-4 已就绪版)。
> 配合代码 `FLARE-VLA/scripts/libero_plus/06c_refilter_teacher_quality_v3.py`(待实现)的语境理解。

## 背景:候选教师样本的"质量"是怎么算出来的

FLARE-VLA 这条管线里有两组共八个数字:

| 字段 | 含义 |
|------|------|
| `best_mse_score` | 这个样本 4 条候选教师动作里,在 LeWM 潜变量评分下 MSE 最小的那个(教师测得) |
| `oracle_mse_score` | GR00T 给出的"真值/标签"动作,在同一个 LeWM 评分下的 MSE(理论上界) |
| `teacher_gt_xyz_mse` | 选出来的教师的 xyz 平移分量与真实动作的 MSE |
| `teacher_gt_rot_mse` | 同上,旋转分量 |
| `teacher_jerk_xyz_l2_mean` | 教师动作平移通道的 jerk(平滑度) |
| `teacher_jerk_rot_l2_mean` | 同上,旋转通道 |
| `late_mse` | 这个教师在 future 后期几步里 MSE 的平均 |
| `teacher_gt_gripper_match_rate` | 教师夹爪的开/关与真值一致率 |
| `teacher_agreement` | cosine 准则下的最佳教师与 MSE 准则下的最佳教师是否一致 |

前两个数字(best_mse 与 oracle_mse)是这个样本**整体上**的"好或不好";后面几个是把"好不好"**拆解**到各零件维度。

baseline(`Pol_base`)目前是这样处理的:**整批 487880 个样本**用同一组阈值——比如 `teacher_gt_xyz_mse ≤ 整批的 p90`。这就是 "global quantile outlier filter"。

问题:这条 p90 切断对所有任务是同一条线。容易任务的 xyz 误差天然就小(毫米级),相对阈值"宽松";难任务的 xyz 误差天然就大(开抽屉 1 cm),同样阈值就把它们大体砍掉了。结果:`direct_aug_rate` 从 0.045(task 7 开抽屉)一路滑到 0.776(task 6 抓黑碗)——见 `teacher_quality_task_summary.csv`。

我们的解决方案是给"接受不接受"加两个分数:`safety_z` 和 `safety_r`,含义如下。

## safety_z — per-task component-badness(任务内零件排名)

**白话说:"这个样本在这个任务里、这 5 个零件指标里最差的一项,在任务内部的相对排名。"**

### 公式

`M = {teacher_gt_xyz_mse, teacher_gt_rot_mse, teacher_jerk_xyz_l2_mean, teacher_jerk_rot_l2_mean, late_mse}`

$$
z_s = \max_{k \in M} \hat F_{g(s),k}(m_{s,k})
$$

其中 $g(s) = (\text{suite}, \text{task\_index})$ 表示样本 s 所属的任务;$\hat F_{g,k}$ 是在 calibration 子集上对任务 g、指标 k 算的经验 CDF(右尾),即"任务 g 内指标 k ≤ 这个值的占比"。

### 怎么算

举一个具体例子:任务里有 10000 个样本,其中教师 `xyz_mse` 的数值从小到大排列:

| rank | xyz_mse | 排名比 = F̂(.) |
|------|---------|------------|
| 1    | 0.001   | 0.0001      |
| 100  | 0.005   | 0.01        |
| 5000 | 0.020   | 0.50        |
| 9000 | 0.080   | 0.90        |
| 9999 | 0.500   | 0.9999      |

如果你这个样本 xyz_mse=0.080,那在这个任务里它的相对排名是 0.90(差)。如果它旋 rot_mse 排名是 0.30(还可以)——z_s = max(0.90, 0.30, 其他三项) = 0.90。

### 直觉

- z 是 0~1 之间的数字。**z 越小,说明这 5 项里最差的一项,在任务里也属于前百分比。**
- 不同任务 z 的可比性很好(都是 0~1 的相对位置);原始值不可比(0.020 vs 0.080 是同一难度还是不同难度?没法说)。
- 取 `max` 是合取式判据——任意一项非常离谱就该拒。

### 为什么不用原始阈值

如果用"teacher_gt_xyz_mse ≤ 整批 p90",这个 p90 是所有任务的混合。抓杯子任务天然 xyz≈0.01,p90 可能在 0.05——很松;开抽屉任务天然 xyz≈0.15,p90=0.05 砍掉了任务里 70% 的样本。这就是 baseline 现在的问题。任务内的相对排名把这种尺度混淆消掉了。

## safety_r — cross-task log-ratio latent-regret(跨任务的对数比遗憾)

**白话说:"这个教师动作比真值动作在 LeWM 评分下差了多少倍"——取对数。**

### 公式

$$
r_s = \operatorname{clip}\!\left[\log\!\left(\frac{\max(\text{best\_mse}_s,\, \text{oracle\_mse}_s) + \epsilon}{\text{oracle\_mse}_s + \epsilon}\right),\, 0,\, \log(1001)\right],\ \epsilon = 10^{-3}
$$

### 怎么读

| 情形 | best_mse | oracle_mse | r_s |
|------|----------|------------|-----|
| 教师和 oracle 一样好 | 0.05 | 0.05 | 0 |
| 教师比 oracle 差 2 倍 | 0.10 | 0.05 | log(2) ≈ 0.69 |
| 教师比 oracle 差 10 倍 | 0.50 | 0.05 | log(10) ≈ 2.30 |
| 教师更好 | 0.03 | 0.05 | ≤ 0(被 max 截断到 0,视为"无遗憾") |
| oracle 接近 0(best=0.5, oracle=0) | 0.50 | 0.0001 | log(0.5001/0.0001) ≈ log(5001)=8.52 → 截断到 log(1001)≈6.91 |

### 直觉

- r 越大,遗憾越大,**最大的 r 是 log(1001)≈6.91**。
- 用对数比,而不是 `best_mse - oracle_mse` 这种**绝对差**:`baseline` 用的 `oracle_gap_mse = best_mse − oracle_mse` 是加性差距。在抓杯子任务里 0.01 的差距已经很大了,在开抽屉任务里 0.01 的差距可以忽略。对数比把这种尺度的偏差消掉了,所以 r 在**跨任务之间是可比的**——这就是"cross-task"的含义。
- 非有限数值 → ineligible,r = +∞。

### 阈值 τ_r 怎么定

在 calibration 子集(全部样本的 1/10)上,把 `large_oracle_gap` 在 baseline 里被拒的样本比例算出来,记作 `reject_rate_baseline`。然后:

$$
\tau_r = \text{quantile}(r_{\text{calibration}},\ 1 - \text{reject\_rate\_baseline})
$$

意思是让"被 r ≤ τ_r 接受的比例"等于 baseline 因 `large_oracle_gap` 拒掉的总比例。这是一个**独立于数据预算 K 的统计意义**——和 budget 是解耦的。

## 它们怎么合起来用

### Pol_full(我们提出的方法)

1. 所有候选 → 必须先过 prerequisite(有限 + 教师准则一致 + 夹爪匹配率高)。
2. 再过 `safety_r(s) ≤ τ_r` —— 把"教师比 oracle 差太多"的样本拦掉。
3. 过后的"合格集 E"按 `safety_z` 从小到大排序,前 K=309128 名留下。
4. 同分按 `sample_id ASC` 断。

### Pol_z(消融实验:去掉 safety_r)

只看 prerequisite 和 safety_z 的 top-K。**没有 safety_r**。

### 对比 Pol_base vs Pol_full vs Pol_z 的解释

- Pol_base 与 Pol_full 都接受 309128 个样本(budget 一样)。
  - Pol_base 用"全局 p90 + global oracle gap";
  - Pol_full 用"任务内 top-K + cross-task 对数比遗憾"。
- Pol_z 与 Pol_full 都接受 309128 个、都不用 safety_z vs safety_r 的二分,所以两者只差一个 safety_r。
  - 优:Pol_z 与 Pol_full 的差 = safety_r 的边际贡献。

## 数据流一览

> **重要:** "calibration 子集"只在过滤阶段用来**估计阈值**;**下游评测协议完全没动**。你之前的评测是 `10030 任务 × 1 trial/task`(参考 `outputs/eval_libero_plus/gr00t_libero_plus_flare_suite_corrected_gt_replay_44_20_checkpoint_10000_full10030/summary_all.json` 里 Pol_base @ step 10000 = 0.7921)。我们继续沿用这个协议——`08c_train_gr00t_flare_44_20.sh` 和 `gr00t/eval/run_gr00t_server.py` 都不需要修改。Calibration 桶里的样本估完阈值后**会和其他 9 个桶一起**(pool back)进入 top-K 选择。

```
                              [评测前:训练数据准备(本研究改的就是这一段)]
                              ==============================================================
[LeWM 打分]      [过滤策略选哪些教师样本]              [训练学生 GR00T]            [评测]
487880 样本  →  Pol_base(全局 p90)/Pol_full(任务内)  →  distill_dataset_v{2,3}  →  08c_train  →  你之前的评测脚本
                                                                                              ▲
                                                                                              │
                                                                            (评测协议完全没有变)
```

具体到 `06c` 内部:

```
CSV/PT/NPZ (487880 windows,全部 40 任务)
  │
  ▼
按 episode 分 10 桶,挑 1 桶作 calibration(约 47k windows,这里只用来估阈值)
  │
  ▼
calibration 子集上:
   ├─ 给 M = {xyz, rot, jerk_xyz, jerk_rot, late_mse} 每个 (suite, task) 算经验 CDF
   ├─ 给所有候选算 r_s,取 quantile 使被 τ_r 拒的比例 = Pol_base 的 large_oracle_gap reject rate
   ├─ 验证 |Spearman(r_s, oracle_mse)| ≤ 0.10(无关校准)
   │
  ▼
所有 487880 候选(包括 calibration 那 1 桶一起,pool back):
   ├─ prerequisite(4 个二元 flag)
   ├─ safety_z(s) ≤ ?(只算 CDF,没有阈值;用 top-K)
   ├─ safety_r(s) ≤ τ_r
   │
  ▼
Pol_full:从 E 取 top-K by ascending safety_z
Pol_z:   从 prereq 取 top-K by ascending safety_z
  │
  ▼
写 teacher_quality_direct_records.pt,下游 08c 训练不动一个字
```

## 为什么能提升 LIBERO-Plus 成功率

- baseline 在容易任务"过松"(留下了噪声)— Pol_z 任务内 top-K 把这部分切掉了。
- baseline 在难任务"过严"(砍掉了真实的好教师)— Pol_z 在每个任务内取 top-K,难的也不会被砍光。
- baseline 用 additive oracle gap,在容易任务偏严、在难任务偏松(尺度混淆)— Pol_full 的 safety_r 用对数比,统一了。
- safety_r 把真正"教师动作与 oracle 差距离谱"的少数样本再剔掉一道。
- 三个数字加起来,劣质训练样本减少 → 学生学得更好。

## 一句话总结

- **`safety_z`**:在这个任务里,这 5 项指标相对排名最差的一项(0~1,越小越好)。
- **`safety_r`**:这个教师动作在 LeWM 评分下比 oracle 差多少倍,取 log(越大越糟)。
- **Pol_full** = 通过 safety_r 的合格集 + 按 safety_z 取前 K 个。
