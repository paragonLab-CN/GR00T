# FLARE-VLA

FLARE-VLA 是一套离线未来潜变量对齐（future-latent alignment）实验流水线：先用 LIBERO-90 训练 LeWM 潜变量世界模型，再由 Isaac-GR00T 为同一观测生成多条候选动作轨迹，最后比较各候选轨迹预测出的未来潜变量与真实未来潜变量，选出得分最佳的教师轨迹。

> 本仓库是实验代码、上游补丁和命令记录的集合，不是可直接 `pip install` 的独立 Python 包。运行时需要分别准备 [LeWM](https://github.com/kylemin/latent-energy-world-model)（或实验所用的对应版本）、NVIDIA Isaac-GR00T、LIBERO-90 与 LeRobot 格式数据，并将本仓库文件部署到两个上游工程的相应目录。

## 流程概览

```text
LIBERO-90 原始 HDF5
        │
        ▼
转换为 LeWM 单文件 H5 ──► 训练 LeWM ──► 提取真实未来 latent framebank
                                              │
LeRobot LIBERO-90 ──► 对齐 episode 顺序 ──────┤
        │                                     │
        └──► 构建验证集 manifest              │
                    │                         │
                    ▼                         ▼
          GR00T 生成 K 条候选动作 ──► LeWM rollout 与真实 latent 对齐评分
                                              │
                                              ▼
                                      选择最佳教师轨迹
```

默认验证协议使用 100 个 episode、每个 episode 5 个窗口、历史长度 3、预测 horizon 8、每个样本 4 条候选轨迹（共 500 个样本）。这些参数可在命令中修改。

## 仓库结构

```text
FLARE-VLA/
├── commands/                 # 01～07 的完整流水线命令示例
├── configs/
│   └── lewm/                 # LeWM 主配置和 LIBERO-90 数据配置
├── convert/                  # LeRobot v3→v2、LIBERO H5 等格式转换工具
├── scripts/
│   ├── 00_check/             # 配置、数据顺序及动作链路检查
│   ├── 01_data/              # LIBERO-90 → LeWM H5
│   ├── 02_gt_latent/         # GT latent 提取、校验和顺序重排
│   ├── 03_gr00t_candidates/  # 验证 manifest 与 GR00T 候选生成
│   ├── 04_scoring/           # 候选分布检查、LeWM 评分与最优轨迹选择
│   └── dev/                  # 开发期诊断脚本
├── patches/
│   ├── le_wm/                # 需覆盖/合并到 LeWM 的活动补丁
│   └── isaac_gr00t/          # GR00T 活动补丁及历史实验变体
├── archive/                  # 已归档的旧版本
└── docs/、examples/          # 预留的文档与示例目录
```

### 版本目录约定

- `patches/*/current_active/`：当前实验使用的上游代码补丁。
- `patches/isaac_gr00t/variants/`：候选生成的历史或对照实现，不是默认入口。
- `archive/old_versions/`：仅供追溯，不建议用于新实验。
- 同名脚本存在 `v2`、`v3` 时，当前命令链采用 `extract_gt_latent_all_v3.py` 和 `verify_gt_latent_all_v3.py`。

## 环境与数据

建议在 Linux、NVIDIA GPU 和 CUDA 环境下运行。代码涉及两个独立环境，依赖应以各自上游工程锁定的版本为准。

### LeWM 侧

- LeWM 及其 `conda` 环境
```text
cd /root/autodl-tmp/le-wm
conda activate LeWorldModel
```
- Python、PyTorch、Lightning、Hydra/OmegaConf
- `stable_pretraining`、`stable_worldmodel`
- `h5py`、NumPy、Pillow、pandas、PyArrow、tqdm、einops
- LIBERO-90 原始 HDF5
- 训练得到的 LeWM checkpoint

### Isaac-GR00T 侧

- Isaac-GR00T 及其 `uv` 环境
```text
cd /root/autodl-tmp/Isaac-GR00T
source .venv/bin/activate
```
- PyTorch、Transformers、LeRobot 数据读取依赖
- GR00T-N1.7 LIBERO checkpoint
- LeRobot 格式的 LIBERO-90 数据集

仓库不包含数据集、模型权重、H5、候选张量、日志或实验输出；这些文件也已在 `.gitignore` 中排除。

## 部署

下面沿用命令文件中的默认目录：

```text
/root/autodl-tmp/le-wm
/root/autodl-tmp/Isaac-GR00T
/root/autodl-tmp/lerobot_data/libero_90_lerobot
```

运行前需要：

1. 将 `configs/lewm/` 合并到 LeWM 的 `configs/`。
2. 将 `patches/le_wm/current_active/` 中的文件合并到 LeWM 根目录对应位置。
3. 将数据转换、GT latent 和评分脚本复制到 LeWM；保持 `commands/` 中调用的文件名或同步修改命令路径。
4. 将 `patches/isaac_gr00t/current_active/gr00t/` 合并到 GR00T 的 `gr00t/`。
5. 将 `patches/isaac_gr00t/current_active_scripts/scripts/flare/` 以及候选生成脚本放到 GR00T 的 `scripts/flare/`。
6. 根据实际机器统一修改绝对路径。部分 Python 脚本也包含 `/root/autodl-tmp/...` 常量，不能只修改 shell 命令。

建议先搜索全部硬编码路径：

```bash
grep -RIn '/root/autodl-tmp' commands scripts
```

## 完整运行顺序

所有示例均位于 `commands/`，应先检查其中的路径、checkpoint 名称和 GPU 设置。

### 1. 转换 LIBERO-90 数据

```bash
bash commands/01_convert_libero90_to_lewm_h5.sh
```

输出默认为 `stablewm_data/libero90_image_train_v2.h5`，包含 LeWM 训练所需的图像、动作、状态及 episode 索引信息。

### 2. 训练 LeWM

```bash
bash commands/02_train_lewm_libero90.sh
```

示例使用单 GPU、50 epochs、batch size 64，并关闭 W&B。显存不足时优先减小 `loader.batch_size`。

### 3. 提取并校验真实未来潜变量

```bash
bash commands/03_extract_gt_latent.sh
```

当前入口为 v3 脚本，生成 horizon=8 的 framebank H5，并执行抽样数值校验。脚本中的 checkpoint、输出路径和数据根目录需与训练结果一致。

### 4. 对齐 LeRobot episode 顺序

```bash
bash commands/04_reorder_gt_latent.sh
```

该步骤根据 episode 映射重排 latent H5，使其与 LeRobot 数据中的 `episode_index` 和局部时间步一致。后续 manifest 和评分均依赖这一顺序，不能跳过。

### 5. 构建固定验证集

```bash
bash commands/05_build_validation_manifest.sh
```

脚本采用近似均匀的任务分层 episode 抽样，并在时间分箱中按固定 seed 选取窗口；同时检查 GT H5 的窗口映射。主要输出：

```text
outputs/flare_validation_500/validation_manifest.jsonl
outputs/flare_validation_500/validation_manifest_summary.json
```

### 6. 生成 K 条 GR00T 候选轨迹

```bash
bash commands/06_generate_candidates.sh
```

生成器使用共享 backbone、向量化 action head 和独立初始噪声，为每个 manifest 样本保存候选动作。任务支持断点续跑：已有且校验通过的候选会被跳过，并记录事件、失败项和汇总信息。

### 7. 使用 LeWM 评分并选择轨迹

```bash
bash commands/07_score_candidates.sh
```

评分脚本将候选动作转换到 LeWM 动作约定，从当前 latent rollout 未来状态，与 GT future latent 比较并汇总最佳候选。其默认路径和预期 checkpoint SHA-256 写在脚本中；更换 checkpoint 时应同步确认或修改校验值。

建议首次运行时显式传参，避免依赖脚本内的机器路径：

```bash
python scripts/flare/05_score_validation_set.py \
  --manifest /path/to/validation_manifest.jsonl \
  --candidate-dir /path/to/candidates \
  --candidate-summary /path/to/candidate_generation_summary.json \
  --gt-h5 /path/to/reordered_gt_latent.h5 \
  --checkpoint /path/to/weights_epoch_50.pt \
  --output-dir /path/to/scoring_output \
  --batch-size 32
```

## 一致性检查

在正式评分前，建议执行 `scripts/00_check/verify_libero90_chain_alignment.py`。它检查：

- LeRobot `episodes.jsonl` 与 parquet episode 数量、长度和顺序；
- LeWM 训练 H5 与 latent H5 的 `ep_len`、`ep_offset`、`episode_idx`；
- 训练动作与 `frame_action` 的逐块数值一致性；
- future window 的行索引映射；
- LeRobot 与 LeWM 动作维度顺序及夹爪符号约定。

这里的关键映射是：

```text
global_row = ep_offset[episode_index] + local_step
```

数据对齐检查失败时不要继续评分，否则 latent 距离对应的可能是错误 episode 或时间步。

## 主要产物


| 阶段      | 典型产物                         | 用途                       |
| ------- | ---------------------------- | ------------------------ |
| 数据转换    | `libero90_image_train_v2.h5` | LeWM 训练数据                |
| LeWM 训练 | `weights_epoch_50.pt`        | latent 编码与未来 rollout     |
| GT 提取   | `*_h8_framebank_*.h5`        | 真实帧 latent、动作和窗口映射       |
| 顺序重排    | `*_lerobot_order.h5`         | 与 GR00T/LeRobot 样本对齐的 GT |
| 验证抽样    | `validation_manifest.jsonl`  | 固定、可复现的样本协议              |
| 候选生成    | `candidates/*.pt`            | 每个样本的 K 条候选动作            |
| 候选评分    | JSON/JSONL/Parquet/PT 汇总     | 分数、最佳候选及教师轨迹             |


## 注意事项

- `commands/*.sh` 是特定实验机器的可复现记录，不是跨机器开箱即用脚本。
- 数据转换、latent 提取、候选生成和评分对 episode 顺序、动作维度顺序、history size 与 horizon 有严格共同约束。
- 当前协议的动作维度为 7，history size 为 3，future horizon 为 8；修改任一项时必须同步检查两侧模型、manifest 与 H5 元数据。
- 候选生成命令启用了确定性相关环境变量，但不同 CUDA、PyTorch 或算子版本仍可能产生轻微数值差异。
- `convert/` 中的 LeRobot v3→v2 工具依赖 LeRobot 与 Hugging Face 组件，仅在数据版本不匹配时使用。

## 开发建议

新增实验实现时保留 `current_active/` 为唯一默认版本，将对照方案放入 `variants/`；过期脚本移入 `archive/`。提交代码前不要加入 checkpoint、数据集或生成结果。