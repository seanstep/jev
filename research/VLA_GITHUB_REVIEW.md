# 从公开 VLA 项目提炼的训练经验

原来的数据把每条状态都压成“总任务、优先级、模块、下一步、阻塞”五个固定问题。这适合演示 API，但不能训练科研决策：它没有要求模型检查数据契约、动作空间、归一化、留出策略、评估协议、证据和部署风险。

`build_experience_dataset.py` 改成了基于公开仓库实践的场景监督。每条样本有一段具体项目状态，问题集合随场景变化，通常包含 6–9 个决策维度。问题不是固定五项，而是从当前状态推导：数据格式、划分方式、微调策略、显存计划、动作适配、基准和 episode 协议、证据、失败归因、下一步和是否阻塞。

## 经验来源

| 项目 | 训练数据中保留的经验 |
| --- | --- |
| OpenVLA | RLDS 混合、Open X-Embodiment、动作反归一化 key、LoRA/全量微调、REST 服务 |
| LeRobot | 视频 + action + state + task 的数据契约；统一 train/eval/rollout；报告 episode 数和硬件 |
| RoboTwin 2.0 | 10 万级轨迹、任务/embodiment 配置、clean/random 评估、视觉/语言/跨本体鲁棒性 |
| LIBERO | 130 个任务、知识迁移轴、LIBERO-90/10 留出、固定任务顺序和 seed |
| Octo | 80 万轨迹预训练、多相机/语言/动作空间适配、head-only/head+MLP/full 对照 |
| openpi / Pi0 | 24GB 单卡优先 LoRA、全量微调显存门槛、target normalization 对照 |
| Open X-Embodiment | RLDS episode、RGB + task、robot/embodiment/语言字段元数据和跨本体留出 |

每条样本 `_meta.source_urls` 保留对应仓库地址，便于审计；模型输入只看到状态和问题，不看到答案或来源字段。

## 数据边界

这仍然是科研流程决策数据，不是视觉图像、机器人状态和动作序列本身。因此它可以训练“下一步怎么设计实验、先检查什么、何时阻止错误结论”，不能替代 OpenVLA、Octo 或 Pi0 的视觉动作策略训练。真正的 policy 训练还需要把 LeRobot/RLDS/RoboTwin/LIBERO 的 episode 轨迹接入对应模型的数据管线。

生成命令：

```bash
python research/build_experience_dataset.py
```

输出目录：`research/data/vla-experience-v1/`。
