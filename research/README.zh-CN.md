# VLA 科研版决策模型

完整训练、评测和部署参数见 [`TRAINING_DETAILS.md`](TRAINING_DETAILS.md)。

这个实验训练的是一个小型中文科研任务决策器，不是完整的视觉语言动作策略。它把 VLA 项目状态文字化后，输出适合实验调度的结构化判断：

- 当前总任务：数据管线、视觉语言对齐、动作策略训练、仿真评估、真机评估、部署控制或失败分析
- 优先程度：低、中、高、紧急
- 主模块：轨迹数据、视觉编码器、动作策略、仿真、真机评估、部署控制或失败分析
- 下一步动作：依据总任务给出可执行的下一步
- 是否阻塞：判断是否缺数据、设备、评估环境或其他前置条件

数据是程序化生成的中文 VLA 研究场景，使用 RoboTwin 2.0、LIBERO、Open X-Embodiment、CALVIN、RLBench、ALOHA、BridgeData V2 和 LeRobot 等领域语境。它不包含图像和动作轨迹，因此不能替代真正的 VLA policy training；它训练的是科研任务路由与实验管理能力。

训练、开发和测试数据分别是 `data/train.jsonl`、`data/dev.jsonl` 和 `data/test.jsonl`。每条记录采用 Kev 的带标签 System One 格式，解析后仍通过 `kev.data.materialize()` 进入和服务端一致的编码路径。

## 基于 GitHub 项目经验的数据集

更适合科研决策训练的是 `data/vla-experience-v1/`。它基于 OpenVLA、LeRobot、RoboTwin 2.0、LIBERO、Octo、openpi/Pi0 和 Open X-Embodiment 的公开仓库经验构造场景，每条样本有 7–9 个随场景变化的问题，覆盖数据契约、RLDS/LeRobot 格式、任务/语言/embodiment 留出、动作归一化、LoRA 资源预算、评估 episode 协议、失败归因、证据和部署安全门槛。

- `data/vla-experience-v1/train.jsonl`：1600 条记录，12375 个问题
- `data/vla-experience-v1/dev.jsonl`：240 条记录，1856 个问题
- `data/vla-experience-v1/test.jsonl`：480 条记录，3712 个问题
- `data/vla-experience-v1/manifest.json`：问题覆盖、来源仓库和生成版本
- `VLA_GITHUB_REVIEW.md`：经验提炼和数据边界说明

生成命令：`python research/build_experience_dataset.py`。这些仍是文字化科研决策样本，不是视觉图像和动作轨迹；真正的 VLA policy 训练需要接入对应项目的 episode 数据管线。

当前经验版 checkpoint：`runs/vla-exp-qwen05b`（轻量 0.5B）和 `runs/vla-exp-qwen35-4b-v2`（Qwen3.5-4B Base）。4B API 由 `kev-vla-api.service` 常驻在 `127.0.0.1:8014`，0.5B API 由 `kev-vla-05b.service` 常驻在 `127.0.0.1:8013`，网页可以切换两个模型。
