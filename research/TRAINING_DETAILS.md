# VLA 科研版训练细节

更新时间：2026-09-23

本文记录当前 `vla-experience-v1` 科研版决策模型的训练、评测和部署配置。命令均以仓库目录 `/data/ssd/xyt/data/jev` 为工作目录。

## 1. 实验目标和边界

本实验训练的是一个中文 VLA 科研任务决策器。输入是文字化的项目状态，输出是带概率的结构化决策，包括研究阶段、数据契约、数据划分、归一化、微调方式、评估协议、证据、失败归因、下一步动作、阻塞状态和风险等级。

它使用 Kev 的 typed decision 结构和 pointer readout，不生成自然语言答案，也不是视觉动作策略。当前数据没有图像、机器人状态序列或动作轨迹，因此本实验不能替代 OpenVLA、Pi0 等模型的 policy training，也不能把本文准确率解释为机器人任务成功率。

## 2. 数据构建

数据生成脚本为 [`build_experience_dataset.py`](build_experience_dataset.py)，知识审计记录为 [`vla_github_knowledge.json`](vla_github_knowledge.json)，生成版本为 `vla-experience-v1`，知识版本为 `vla-github-review-2026-09-23`。

场景经验来自以下公开项目：

- [OpenVLA](https://github.com/openvla/openvla)：RLDS、动作反归一化、LoRA/全量微调和推理服务
- [LeRobot](https://github.com/huggingface/lerobot)：LeRobotDataset、视频/状态/动作/task 数据契约
- [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)：clean/random 仿真任务和双臂数据
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)：Spatial/Object/Goal/10 和 90/10 迁移划分
- [Octo](https://github.com/octo-models/octo)：多机器人轨迹和 head/head-MLP/full 适配方式
- [openpi / Pi0](https://github.com/Physical-Intelligence/openpi)：LoRA、显存预算和 normalization 统计量
- [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment)：RLDS episode、机器人元数据和跨 embodiment 经验

数据由 15 个 case family 程序化生成，每条记录的状态、问题数量和问题措辞会随场景变化，不再对所有记录固定询问同一组问题。主要覆盖：

`lerobot_missing_contract`、`openvla_rlds_mix`、`robotwin_clean_random`、`libero_transfer_leak`、`octo_target_adaptation`、`openpi_memory_budget`、`openpi_normalization_drift`、`openvla_unnorm_key`、`openx_metadata_gap`、`real_rollout_small_n`、`robotwin_language_shift`、`deployment_watchdog`、`failure_camera_shift`、`cross_embodiment_action`、`benchmark_reporting`。

固定划分如下，划分单位是记录，不是单独的问题：

| 分区 | 记录数 | 问题数 |
|---|---:|---:|
| train | 1,600 | 12,375 |
| dev | 240 | 1,856 |
| test | 480 | 3,712 |

数据文件位于 [`data/vla-experience-v1/`](data/vla-experience-v1/)，其中 [`manifest.json`](data/vla-experience-v1/manifest.json) 保存生成版本、问题覆盖和来源仓库。每条记录包含 `_meta.source_urls`、项目 id、case family 和标签来源，便于追溯。

### 2.1 生成流程

数据不是把项目 README 直接复制到训练集，而是先把公开项目经验整理成可审计的决策知识，再用固定模板生成带标签记录。生成流程是：

1. 读取 [`vla_github_knowledge.json`](vla_github_knowledge.json)，建立项目 id、仓库 URL 和知识版本的映射。
2. 在 `QUESTION_BANK` 中定义问题类型、问题措辞和候选答案。候选答案是稳定的机器键，说明文字作为 criteria 展示给模型。
3. 在 `CASES` 中定义 case family。每个 case 指定涉及的项目、要问哪些问题、每个问题的标准答案和若干状态模板。
4. `make_state()` 随机组合状态前缀、场景模板、机器人形态、资源约束、随机种子和结论要求，形成当前项目状态。
5. `build_record()` 根据 case 的问题列表生成 7–9 道问题。问题的 instruction 从问题库中随机选择，criteria 深拷贝后写入记录，label 使用 case 的标准答案。
6. `build()` 按 case family 轮转生成记录，再用固定随机种子打乱记录顺序，避免文件顺序直接暴露 case 类型。
7. 最后写出三个 JSONL 分区和 `manifest.json`。manifest 记录记录数、问题覆盖、项目仓库和 case family。

当前生成入口：

```bash
cd /data/ssd/xyt/data/jev
python research/build_experience_dataset.py
```

生成器使用固定分区种子：train=`101`、dev=`103`、test=`107`。因此在代码、知识文件和 Python 随机逻辑不变时可以重建同一批记录。重新生成前应先保存旧版 manifest，避免覆盖已经用于论文或比较实验的数据。

### 2.2 标签如何得到

每个标签由 case 的决策规则显式给出，不是让另一个语言模型自动打标签：

- `choice`：label 是候选答案的字符串 key，例如 `"lerobot"`、`"episode_disjoint"`。
- `noul`：label 是布尔值。例如数据缺字段、存在泄漏或没有安全门槛时为 `true`。
- `score`：label 是有序 criteria 的整数下标。`risk` 的 `低/中/高/阻断` 对应 `0/1/2/3`。

例如，LeRobot 数据缺少 state 和 task 字段时，规则同时给出：研究阶段=`data_contract`、数据契约=`lerobot`、划分=`episode_disjoint`、归一化=`audit`、阻塞=`true`、风险=`3`。这样一条状态会监督多个相互关联的科研判断，而不是只监督一个分类标签。

标签的依据是项目公开文档和仓库实践的结构化提炼。它们是**来源约束的程序化标签**，不是经过机器人专家逐条盲审的人类金标准；因此后续真实 VLA 研究仍需要用实际 episode 和人工复核校验这些规则。

### 2.3 JSONL 记录结构

每一行是一个独立 JSON 对象，核心字段如下：

```json
{
  "state": "项目状态：...",
  "questions": {
    "data_format": {
      "type": "choice",
      "instructions": "当前数据应该采用什么数据契约？",
      "criteria": {
        "rlds": "统一为 RLDS episode 格式并保留来源元数据",
        "lerobot": "采用 LeRobotDataset 的视频、状态、动作和 task 契约"
      },
      "label": "lerobot"
    },
    "blocked": {
      "type": "noul",
      "instructions": "当前是否存在必须先解决的阻塞？",
      "criteria": {"true": "存在阻塞", "false": "可以继续"},
      "label": true
    },
    "risk": {
      "type": "score",
      "instructions": "当前科研风险是多少？",
      "criteria": ["低", "中", "高", "阻断"],
      "label": 3
    }
  },
  "_meta": {
    "source": "vla_github_experience_v1",
    "case": "lerobot_missing_contract",
    "projects": ["lerobot"],
    "source_urls": ["https://github.com/huggingface/lerobot"],
    "knowledge_version": "vla-github-review-2026-09-23",
    "question_count": 8
  }
}
```

`criteria` 同时承担候选空间和可读解释两个作用；训练时模型读到完整说明，评测时只比较 label 对应的 key/index。`_meta` 不作为答案标签，但用于追溯来源、按 case 分析和检查数据泄漏。

### 2.4 问题覆盖和不均匀分布

不是每条记录都包含全部问题。以下五项是所有 case 都包含的主干问题：`research_stage`、`evidence`、`next_action`、`blocked`、`risk`。其他问题按场景条件加入，例如只有涉及数据格式时才加入 `data_format`，只有涉及跨机器人或动作接口时才加入 `action_space`。

| qid | train | dev | test | 说明 |
|---|---:|---:|---:|---|
| `research_stage` | 1,600 | 240 | 480 | 主研究阶段 |
| `evidence` | 1,600 | 240 | 480 | 可复现证据 |
| `next_action` | 1,600 | 240 | 480 | 可执行下一步 |
| `blocked` | 1,600 | 240 | 480 | 是否阻塞 |
| `risk` | 1,600 | 240 | 480 | 有序风险等级 |
| `data_format` | 428 | 64 | 128 | RLDS/LeRobot/原生格式 |
| `split_protocol` | 747 | 112 | 224 | 泄漏和泛化划分 |
| `benchmark` | 746 | 112 | 224 | LIBERO/RoboTwin 等基准 |
| `action_space` | 640 | 96 | 192 | action/state 对齐 |
| `failure_mode` | 532 | 80 | 160 | 失败归因 |

其余 qid（`eval_protocol`、`finetune_mode`、`normalization`、`gpu_plan`）也按 case 条件出现。这样的不均匀覆盖模拟真实科研记录：一个数据契约问题不一定需要立刻讨论真机 rollout，一个部署问题则应重点询问延迟和 watchdog。

### 2.5 实际样本示例

下面示例取自 `train.jsonl` 的真实记录，做了可读化摘要：省略随机附加的机器人/seed 字段和完整 criteria，只保留状态主线与正确标签。原始文件保留完整 JSON。

**示例 A：LeRobot 数据契约不完整。**

```text
状态：准备把示教数据接入 LeRobot 的 train/eval/rollout 流程；视频能播放，但动作长度不一致，状态字段和语言任务字段没有完整覆盖。当前只有仿真，没有真机。

判断：
- research_stage = data_contract
- data_format = lerobot
- split_protocol = episode_disjoint
- normalization = audit
- evidence = manifest
- next_action = schema_audit
- blocked = true
- risk = 阻断

含义：先审计字段、时间戳和 episode manifest，不能因为视频能播放就直接启动训练。
```

**示例 B：RoboTwin 只测了 clean 场景。**

```text
状态：已有大量 RoboTwin 双臂轨迹，但训练集和评估任务名相同，初始状态未核对；目前只有一张 24GB GPU，下一轮要比较单任务、多任务和跨 embodiment。

判断：
- research_stage = benchmark
- split_protocol = clean_random
- benchmark = robotwin
- eval_protocol = clean_random
- evidence = per_task
- next_action = run_benchmark
- blocked = false
- risk = 高

含义：结果必须拆成 clean/random 和逐任务统计，不能只报告一个平均成功率。
```

**示例 C：OpenVLA 真机动作尺度异常。**

```text
状态：模型 checkpoint、图像预处理和语言 prompt 都能运行，但 Panda 真机失败集中在动作尺度和控制周期，尚未检查 dataset key 与控制接口单位。

判断：
- research_stage = real_robot
- action_space = unnorm_key
- failure_mode = action_scale
- evidence = video
- next_action = calibrate_control
- blocked = true
- risk = 阻断

含义：先确认对应数据集的 unnormalization key、动作单位、控制频率和安全边界，再继续真机测试。
```

**示例 D：远程部署缺少 watchdog。**

```text
状态：策略通过服务器远程推理，网络抖动可能超过控制周期；当前没有本地 fallback、动作限幅和急停策略。

判断：
- research_stage = deployment
- action_space = action_token
- evidence = latency
- failure_mode = control_loop
- next_action = serve_policy
- blocked = true
- risk = 阻断

含义：先保存 P50/P95/P99 延迟并加入 watchdog 和 fallback，不能把仿真成功直接接入机器人控制回路。
```

### 2.6 数据检查清单

生成新版本后至少检查以下项目：

```bash
cd /data/ssd/xyt/data/jev
python research/build_experience_dataset.py

# 查看分区规模和问题覆盖
python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("research/data/vla-experience-v1/manifest.json").read_text())
print(json.dumps(m["counts"], ensure_ascii=False, indent=2))
print(m["case_families"])
PY
```

还应确认：

- train/dev/test 的 `group_id` 没有交集；
- 每个问题的 label 都存在于 choice criteria 或 score levels 中；
- `noul` label 是布尔值，score label 是合法整数下标；
- `_meta.source_urls` 与项目 id 一致，知识版本没有丢失；
- 所有记录能在 `MAX_STATE`、`MAX_BRANCH`、`MAX_PACKED` 限制内编码；
- 训练、开发、测试文件没有混用，测试集只在最终评测时读取。

## 3. 输入编码和模型结构

每条记录先通过 `kev.data.materialize()` 转成与服务端相同的内部格式，然后编码为：

```text
<state> 状态文本
<question> 问题文本 <option>选项...</option> <decide>
<question> 问题文本 <option>选项...</option> <decide>
...
```

模型由三部分组成：

1. Qwen causal LM backbone，只取隐藏状态，不使用词表生成文本。
2. LoRA 适配层，训练时冻结大部分基座参数。
3. 256 维 pointer head，将 `<decide>` 隐藏状态与各选项边界隐藏状态匹配，输出选项 logits。

问题类型为：

- `choice`：从命名选项中选择一个
- `noul`：输出 `p(true)`，用于是否题
- `score`：从有序等级中选择一个，并额外计算等级误差

编码约束为 `MAX_STATE=384`、`MAX_BRANCH=1024`、`MAX_PACKED=2048` token。训练和测试都使用严格长度检查；本次训练和完整测试的 rejected/truncated 数量均为 0。Qwen3.5-4B 是 hybrid backbone，评测时使用逐问题 causal row 路径；Qwen2.5-0.5B 使用 packed block-causal 路径。

## 4. 共同训练配置

- Python：`>=3.12`
- GPU：CUDA，训练记录中的 dtype 为 `bf16`
- 优化器：AdamW
- 学习率调度：OneCycleLR，`pct_start=0.1`
- weight decay：`0.01`
- LoRA rank：`8`
- LoRA alpha：`16`（由训练脚本按 `2 * rank` 设置）
- LoRA dropout：`0.05`
- LoRA target：`all`
- pointer head dimension：`256`
- head learning rate：未单独指定，跟随 LoRA 学习率
- epochs：`1`
- seed：`0`
- gradient checkpointing：开启
- special delimiter embeddings：不单独训练
- option isolation：关闭
- permutation KL：`0`
- ranked probability score 训练权重：`0`
- label smoothing、Brier loss、focal loss：均为 `0`

因此当前训练目标是每道问题的 hard-label cross entropy。`score` 的 ranked probability score 在评测时记录，但没有作为本轮训练损失加权。

## 5. 0.5B 训练配置

基座为本地 `models/Qwen2.5-0.5B`，输出目录为 [`runs/vla-exp-qwen05b`](runs/vla-exp-qwen05b)。完整参数保存在 [`training_config.json`](runs/vla-exp-qwen05b/training_config.json)。

| 参数 | 值 |
|---|---:|
| batch | 2 |
| gradient accumulation | 4 |
| 有效 batch（按记录） | 8 |
| learning rate | `1e-4` |
| dtype | `bf16` |
| weights dtype | `bf16` |
| 训练记录 | 1,600 |
| optimizer steps | 200 |
| 训练耗时 | 270.63 秒 |
| forward tokens | 1,334,466 |

复现命令：

```bash
cd /data/ssd/xyt/data/jev
CUDA_VISIBLE_DEVICES=0 uv run python -m kev.train \
  --base models/Qwen2.5-0.5B \
  --n_per_source 1000 \
  --epochs 1 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --lora 8 \
  --accum 4 \
  --batch 2 \
  --device cuda \
  --dtype bf16 \
  --weights_dtype bf16 \
  --checkpointing 1 \
  --head_dim 256 \
  --lora_targets all \
  --out research/runs/vla-exp-qwen05b \
  --data research/data/vla-experience-v1/train.jsonl \
  --seed 0
```

训练记录：[`training_metrics.json`](runs/vla-exp-qwen05b/training_metrics.json)。

## 6. 4B 训练配置

基座为本地 `models/Qwen3.5-4B-Base`，输出目录为 [`runs/vla-exp-qwen35-4b-v2`](runs/vla-exp-qwen35-4b-v2)。完整参数保存在 [`training_config.json`](runs/vla-exp-qwen35-4b-v2/training_config.json)。

| 参数 | 值 |
|---|---:|
| batch | 1 |
| gradient accumulation | 8 |
| 有效 batch（按记录） | 8 |
| learning rate | `5e-5` |
| dtype | `bf16` |
| weights dtype | `bf16` |
| 训练记录 | 1,600 |
| optimizer steps | 200 |
| 训练耗时 | 1,515.43 秒 |
| forward tokens | 1,243,073 |

复现命令：

```bash
cd /data/ssd/xyt/data/jev
CUDA_VISIBLE_DEVICES=0 uv run python -m kev.train \
  --base models/Qwen3.5-4B-Base \
  --n_per_source 1000 \
  --epochs 1 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --lora 8 \
  --accum 8 \
  --batch 1 \
  --device cuda \
  --dtype bf16 \
  --weights_dtype bf16 \
  --checkpointing 1 \
  --head_dim 256 \
  --lora_targets all \
  --out research/runs/vla-exp-qwen35-4b-v2 \
  --data research/data/vla-experience-v1/train.jsonl \
  --seed 0
```

训练记录：[`training_metrics.json`](runs/vla-exp-qwen35-4b-v2/training_metrics.json)。

## 7. 完整测试

两个模型使用同一个 `test.jsonl` 和同一个 `kev.benchmark` 评测命令，指标按问题级 Top-1 精确匹配统计。`choice` 和 `score` 取最大概率选项；`noul` 使用 `p(true) >= 0.5` 判为 True。测试不使用远程 API，直接加载本地 checkpoint，以便记录 logits 和保持评测口径一致。

0.5B：

```bash
cd /data/ssd/xyt/data/jev
CUDA_VISIBLE_DEVICES=0 uv run python -m kev.benchmark \
  --run research/runs/vla-exp-qwen05b \
  --data research/data/vla-experience-v1/test.jsonl \
  --out research/runs/vla-exp-qwen05b-test \
  --device cuda
```

4B：

```bash
cd /data/ssd/xyt/data/jev
CUDA_VISIBLE_DEVICES=0 uv run python -m kev.benchmark \
  --run research/runs/vla-exp-qwen35-4b-v2 \
  --data research/data/vla-experience-v1/test.jsonl \
  --out research/runs/vla-exp-qwen35-4b-full \
  --device cuda
```

结果对比：

| 指标 | Qwen2.5-0.5B | Qwen3.5-4B |
|---|---:|---:|
| 问题级准确率 | 94.881% | **99.973%** |
| Choice 准确率 | 96.185% | 99.964% |
| Noul 准确率 | 93.125% | 100% |
| Score 准确率 | 89.167% | 100% |
| Score MAE | 0.2134 | **0.0150** |
| ECE | 3.995% | **0.471%** |
| 置信度 ≥ 0.9 时准确率 | 99.925% | 100% |
| 置信度覆盖率 | 72.306% | 98.734% |
| median latency | 49.3 ms | 166.5 ms |
| P95 latency | 53.2 ms | 190.8 ms |
| rejected/truncated | 0/0 | 0/0 |

报告文件：

- [0.5B report.json](runs/vla-exp-qwen05b-test/report.json)
- [4B report.json](runs/vla-exp-qwen35-4b-full/report.json)
- [4B rows.json](runs/vla-exp-qwen35-4b-full/rows.json)
- [4B predictions.jsonl](runs/vla-exp-qwen35-4b-full/predictions.jsonl)

4B 完整测试为 3,711/3,712 道问题正确，剩余 1 道错误属于 Choice。4B 之前的 16 条记录冒烟测试报告 [`vla-exp-qwen35-4b-smoke/report.json`](runs/vla-exp-qwen35-4b-smoke/report.json) 仅用于快速检查，不能替代这里的完整测试。

## 8. 服务部署

当前用户级服务如下：

| 服务 | 模型 | 地址 | GPU |
|---|---|---|---:|
| `kev-vla-api.service` | Qwen3.5-4B | `127.0.0.1:8014` | 2 |
| `kev-vla-05b.service` | Qwen2.5-0.5B | `127.0.0.1:8013` | 0 |
| `kev-web.service` | Next.js 前端 | `0.0.0.0:3001` | - |

前端通过 Next rewrite 隔离 API 端口：`/vla-api/4b/*` 转到 8014，`/vla-api/05b/*` 转到 8013。网页模型切换不会把 8013/8014 暴露给手机浏览器。

```bash
systemctl --user enable --now kev-vla-api.service
systemctl --user enable --now kev-vla-05b.service
systemctl --user restart kev-web.service
```

网页入口：`http://121.48.164.183:3001/vla`

## 9. 结果解释和限制

当前 4B 的 99.973% 是在程序化、来源约束、文字化的科研决策测试集上的结果。它说明模型能较好地完成当前的科研任务路由，不等于真实机器人抓取或导航成功率。

下一阶段若要形成真正的 VLA policy 结果，需要接入带图像、状态、动作和 episode 元数据的 LeRobot/RLDS 数据，并在 LIBERO、RoboTwin 或真机固定 episode 协议上报告：

- 每任务 episode success rate，而不只是问题准确率
- clean/random 或 task/language/embodiment 留出结果
- 碰撞、超时、动作越界和 watchdog 触发率
- P50/P95/P99 控制延迟和失败视频
- 训练、开发、测试以及 calibration split 的独立结果


## 10. 训练机器与运行环境

训练和本地完整评测使用同一台服务器，硬件和软件记录如下（2026-09-23）：

### 硬件

| 项目 | 配置 |
|---|---|
| GPU | 5 × NVIDIA A100-SXM4-80GB |
| GPU 显存 | 每卡 80,920 MiB（约 80 GB） |
| NVIDIA 驱动 | 580.173.02 |
| CPU | 2 × AMD EPYC 7763 64-Core Processor |
| CPU 核心/线程 | 128 物理核心 / 256 线程 |
| 内存 | 377 GiB |
| 训练磁盘 | `/data/ssd`，3.5 TB，总使用 2.9 TB，可用约 437 GB（87%） |

当前服务的 GPU 分配：4B API 使用 GPU 2，0.5B API 使用 GPU 0。完整 4B 测试使用 GPU 0，与在线 4B 服务隔离。

### 软件

| 软件 | 版本 |
|---|---|
| OS | Ubuntu 24.04 系列，Linux kernel 7.0.0-31-generic |
| Python | 3.13.13 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 5.17.0 |
| PEFT | 0.21.0 |
| Accelerate | 1.15.0 |
| Python 约束 | `>=3.12` |

训练使用 `bf16` autocast 和 bf16 backbone 权重；优化器的可训练参数为 LoRA 和 pointer head。当前磁盘使用率已经达到 87%，继续保存多个完整 checkpoint 或 prediction dump 前应清理缓存和旧实验文件。

## 11. 手机公网访问排查

当前网络状态：

- Next.js 前端监听 `0.0.0.0:3001`，不是只监听 localhost。
- 4B API `127.0.0.1:8014` 和 0.5B API `127.0.0.1:8013` 只监听本机，这是有意的，手机请求应通过前端代理访问。
- 在服务器上访问 `http://121.48.164.183:3001/vla` 返回 HTTP 200。
- 从外部网络探测 `121.48.164.183:3001` 超时，因此目前阻断点在主机防火墙、云安全组、校园网边界或 NAT，而不是 Next.js 绑定配置。

手机访问地址仍然是：

```text
http://121.48.164.183:3001/vla
```

需要管理员完成以下任一方案：

1. 在云安全组或边界防火墙放行入站 TCP `3001`，来源可先限制为需要访问的公网网段。
2. 如果主机使用 UFW，在主机上执行：

   ```bash
   sudo ufw allow 3001/tcp
   sudo ufw reload
   ```

3. 更适合长期使用的方式是由 Nginx/Caddy 监听 80/443，反向代理到 `127.0.0.1:3001`，并配置 HTTPS；8013/8014 不需要对公网开放。

当前用户没有 sudo 密码，无法代替管理员修改主机防火墙或云安全组。放行后可用手机流量重新访问上述 URL；不应把 8013 或 8014 直接作为手机访问地址。
