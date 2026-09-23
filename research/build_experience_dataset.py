"""Build a source-grounded VLA research decision dataset.

The old research dataset asked the same five questions for every record. This generator
uses public project practices as supervision: each scenario has a concrete state,
different questions, evidence requirements, resource constraints and a next action.
It still trains Kev's typed decision head; it is not image/action trajectory data.
"""
from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "vla-experience-v1"
KNOWLEDGE = json.loads((ROOT / "vla_github_knowledge.json").read_text(encoding="utf-8"))
PROJECTS = {p["id"]: p for p in KNOWLEDGE["projects"]}


def choice(instructions, criteria, label):
    return {"type": "choice", "instructions": instructions, "criteria": criteria, "label": label}


def score(instructions, criteria, label):
    return {"type": "score", "instructions": instructions, "criteria": criteria, "label": criteria.index(label)}


def noul(instructions, label, true_text="存在会影响实验可信度或执行的数据、资源或安全问题", false_text="前置条件基本齐全，可以执行下一步"):
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true_text, "false": false_text}, "label": label}


QUESTION_BANK = {
    "research_stage": {
        "type": "choice",
        "criteria": {
            "data_contract": "数据契约与轨迹整理",
            "finetune": "策略微调与适配",
            "benchmark": "仿真基准评估",
            "real_robot": "真机验证",
            "deployment": "策略部署与控制",
            "failure_analysis": "失败归因与复盘",
            "cross_embodiment": "跨机器人泛化",
        },
        "instructions": ["当前最应该先解决哪一类科研问题？", "从项目状态看，主研究阶段是什么？"],
    },
    "data_format": {
        "type": "choice",
        "criteria": {
            "rlds": "统一为 RLDS episode 格式并保留来源元数据",
            "lerobot": "采用 LeRobotDataset 的视频、状态、动作和 task 契约",
            "native": "保留原生轨迹格式，只写适配读取器",
            "manifest": "先建立 episode manifest 和字段覆盖审计",
        },
        "instructions": ["当前数据应该采用什么数据契约？", "哪种数据整理方案最能减少后续训练歧义？"],
    },
    "split_protocol": {
        "type": "choice",
        "criteria": {
            "episode_disjoint": "按 episode 去重后划分，避免同轨迹泄漏",
            "task_disjoint": "按任务或语言目标留出，测组合泛化",
            "language_disjoint": "按语言模板或指令表达留出，测语言鲁棒性",
            "embodiment_disjoint": "按机器人和传感器形态留出，测跨本体泛化",
            "libero_90_10": "按 LIBERO-90 预训练、LIBERO-10 下游测试",
            "clean_random": "保留 clean 与 randomized 两个评估阶段",
        },
        "instructions": ["训练和测试应该怎样划分？", "哪个留出策略能支持当前论文结论？"],
    },
    "normalization": {
        "type": "choice",
        "criteria": {
            "recompute": "在目标数据上重新计算并保存统计量",
            "reuse": "直接复用源 checkpoint 的统计量",
            "compare": "同时跑复用与重算两条对照实验",
            "audit": "先审计量纲、范围和缺失值，再决定统计量",
        },
        "instructions": ["动作和状态归一化应该怎么处理？", "目标域的 normalization 方案应该选哪一个？"],
    },
    "finetune_mode": {
        "type": "choice",
        "criteria": {
            "lora": "LoRA 微调，冻结大部分基座参数",
            "head_only": "先只训练动作头或输出头",
            "head_mlp": "训练输出头和小型 MLP 适配层",
            "full": "全量微调视觉、语言和动作模块",
            "scratch": "从头训练并作为容量基线",
        },
        "instructions": ["在当前数据量和 GPU 约束下先选哪种微调方式？", "适配新机器人时，第一条训练试验应该怎么配？"],
    },
    "gpu_plan": {
        "type": "choice",
        "criteria": {
            "single_lora": "单卡运行 LoRA，并把 batch/累积写入记录",
            "multi_lora": "多卡 LoRA，固定全局 batch 和通信配置",
            "a100_full": "使用 80GB 级 GPU 才考虑全量微调",
            "cpu_debug": "仅用 CPU 做数据和接口 smoke test，不报告训练效果",
        },
        "instructions": ["当前硬件下哪种训练计划可行？", "资源预算应该如何约束本轮实验？"],
    },
    "action_space": {
        "type": "choice",
        "criteria": {
            "match_schema": "先对齐目标机器人 action/state schema",
            "unnorm_key": "推理时使用与数据集对应的 unnormalization key",
            "ee_delta": "明确使用末端位姿增量和夹爪维度",
            "joint_position": "明确使用关节位置控制并记录频率",
            "action_token": "固定动作 tokenizer、chunk 长度和控制周期",
        },
        "instructions": ["动作输出在执行前最需要确认什么？", "当前 action 适配的首要检查项是什么？"],
    },
    "benchmark": {
        "type": "choice",
        "criteria": {
            "libero": "LIBERO suite，按知识迁移轴选择 Spatial/Object/Goal/10",
            "robotwin": "RoboTwin clean/random 双阶段任务评估",
            "bridge": "BridgeData V2 WidowX 真实环境评估",
            "custom_gym": "封装成 Gym 接口后做可复现实验",
            "real_rollout": "真机固定任务和固定 episode 数 rollout",
        },
        "instructions": ["哪种评估基准最匹配当前主张？", "下一轮应该在哪个环境中验证？"],
    },
    "eval_protocol": {
        "type": "choice",
        "criteria": {
            "task_100": "每个任务至少 100 个 episode 并按任务报告",
            "suite_50": "每个 suite 至少 50 个 episode 并报告硬件",
            "real_10": "真机至少 10 次 rollout 并保存失败视频",
            "fixed_seed": "固定 seed、任务顺序和 checkpoint 后再比较",
            "clean_random": "分别报告 clean 与 randomized 成功率",
        },
        "instructions": ["本轮结果应采用什么评估协议？", "怎样的 episode 设计才能支撑结果比较？"],
    },
    "evidence": {
        "type": "choice",
        "criteria": {
            "manifest": "数据 manifest、字段覆盖和去重报告",
            "per_task": "逐任务成功率、失败类型和置信区间",
            "learning_curve": "按 epoch/step 保存 loss 与 success 曲线",
            "normalization": "源统计量与目标统计量的对照结果",
            "latency": "控制周期、P50/P95/P99 延迟和 watchdog 日志",
            "video": "失败轨迹视频、传感器同步和动作回放",
        },
        "instructions": ["现在最缺哪类证据？", "为了让结论可复现，下一步应该保存什么？"],
    },
    "failure_mode": {
        "type": "choice",
        "criteria": {
            "data_quality": "轨迹质量、重复、缺失字段或标签问题",
            "language_shift": "语言模板或任务指令分布发生变化",
            "camera_shift": "视角、分辨率、光照或相机同步变化",
            "action_scale": "动作量纲、归一化或控制频率错误",
            "task_shift": "物体、布局、任务组合或 embodiment 分布变化",
            "control_loop": "推理延迟、chunk 执行或安全控制问题",
        },
        "instructions": ["当前失败现象最应该先归因到哪里？", "哪类假设最值得先验证？"],
    },
    "next_action": {
        "type": "choice",
        "criteria": {
            "schema_audit": "审计字段、单位、时间戳和 episode manifest",
            "make_split": "生成去重且与论文主张一致的 train/dev/test split",
            "compute_stats": "计算目标域 normalization 并做复用/重算对照",
            "lora_smoke": "先做短步数 LoRA smoke test，确认 loss 和 checkpoint 可加载",
            "run_benchmark": "运行固定任务、seed 和 episode 数的基准评估",
            "collect_failures": "保存失败视频、状态和动作并建立失败标签",
            "calibrate_control": "校准动作反归一化、控制周期、延迟和安全边界",
            "serve_policy": "通过独立推理服务接入机器人控制回路并加 watchdog",
        },
        "instructions": ["下一步最应该执行哪一个可验证动作？", "如果只做一件事，下一步应该是什么？"],
    },
    "blocked": {
        "type": "noul",
        "criteria": {"true": "存在缺失字段、数据泄漏、资源或安全阻塞", "false": "前置条件齐全，可以开始该实验"},
        "instructions": ["当前是否应该阻止训练或对外报告？", "现在是否存在必须先解决的阻塞？"],
    },
    "risk": {
        "type": "score",
        "criteria": ["低", "中", "高", "阻断"],
        "instructions": ["如果直接执行当前计划，科研风险有多高？", "当前结论被误读或不可复现的风险是多少？"],
    },
}


COMMON = {
    "robot": ["双臂机械臂", "WidowX", "Panda", "ALOHA 双臂", "移动操作平台"],
    "seed": ["固定 seed=0/1/2", "三个独立随机种子", "训练和评估使用不同 seed"],
    "constraint": ["只有一张 24GB GPU", "暂时没有真机，只有仿真", "控制周期要求不超过 100ms", "数据来自两个不同机器人"],
}


CASES = [
    {
        "id": "lerobot_missing_contract",
        "projects": ["lerobot"],
        "questions": ["research_stage", "data_format", "split_protocol", "normalization", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "data_contract", "data_format": "lerobot", "split_protocol": "episode_disjoint", "normalization": "audit", "evidence": "manifest", "next_action": "schema_audit", "blocked": True, "risk": "阻断"},
        "templates": [
            "LeRobot 记录目录里有同步 RGB 视频和 action，但缺少 observation.state，部分 episode 也没有 task 字段。团队想直接调用统一 train 命令微调策略。",
            "准备把示教数据接入 LeRobot 的 train/eval/rollout 流程；目前视频能播放，动作长度不一致，状态字段和语言任务字段没有完整覆盖。",
            "项目声称使用 LeRobotDataset，但 manifest 没有记录相机、状态、动作维度和 task 覆盖率；已有轨迹不能证明每个 episode 都可被同一个 processor 读取。",
        ],
    },
    {
        "id": "openvla_rlds_mix",
        "projects": ["openvla", "openx"],
        "questions": ["research_stage", "data_format", "split_protocol", "action_space", "finetune_mode", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "data_contract", "data_format": "rlds", "split_protocol": "task_disjoint", "action_space": "match_schema", "finetune_mode": "lora", "evidence": "manifest", "next_action": "make_split", "blocked": True, "risk": "高"},
        "templates": [
            "要把 BridgeData V2 和 Open X-Embodiment 的多个来源混给 OpenVLA。当前有的 episode 带语言任务，有的只有图像和动作；不同机器人 action 维度与单位也没有对齐。",
            "OpenVLA 的训练脚本可以读 RLDS mixture，但本次 mixture 只按文件夹随机切分，来自同一任务的 paraphrase 同时出现在 train 和 test，且没有保留 embodiment 元数据。",
            "团队把两个机器人数据集拼成一个 OpenVLA 数据目录，发现 action spec、语言字段和相机命名不一致，却准备直接开始 LoRA。",
        ],
    },
    {
        "id": "robotwin_clean_random",
        "projects": ["robotwin"],
        "questions": ["research_stage", "split_protocol", "benchmark", "eval_protocol", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "benchmark", "split_protocol": "clean_random", "benchmark": "robotwin", "eval_protocol": "clean_random", "evidence": "per_task", "next_action": "run_benchmark", "blocked": False, "risk": "高"},
        "templates": [
            "RoboTwin 2.0 双臂策略只在 clean 初始场景跑过 10 个 episode；论文想声称视觉鲁棒性和 sim-to-real 泛化，但没有 randomized 阶段和逐任务统计。",
            "已有大量 RoboTwin 轨迹，训练集和评估任务名相同但初始状态未核对。下一轮要比较单任务微调、多任务和跨 embodiment 能力。",
            "当前策略在固定光照和桌面高度上成功，团队准备直接把平均成功率写进报告；还没有按 clean/random、语言变化和任务维度拆分结果。",
        ],
    },
    {
        "id": "libero_transfer_leak",
        "projects": ["libero"],
        "questions": ["research_stage", "split_protocol", "benchmark", "eval_protocol", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "benchmark", "split_protocol": "libero_90_10", "benchmark": "libero", "eval_protocol": "fixed_seed", "evidence": "per_task", "next_action": "make_split", "blocked": True, "risk": "阻断"},
        "templates": [
            "研究目标是 LIBERO-100 的 lifelong knowledge transfer，但训练脚本把 LIBERO-90 和 LIBERO-10 一起加载了。当前结果看起来很高，却无法证明下游迁移。",
            "团队声称测 LIBERO 的 Object 泛化，只做了随机 episode split，没有固定 task suite、任务顺序和 seed，也没有保留独立下游任务。",
            "LIBERO 实验需要在有限 GPU 上把训练和评估拆开；目前评估脚本读不到 checkpoint 对应的 task order 和 seed，结果无法复现。",
        ],
    },
    {
        "id": "octo_target_adaptation",
        "projects": ["octo"],
        "questions": ["research_stage", "finetune_mode", "action_space", "benchmark", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "finetune", "finetune_mode": "head_mlp", "action_space": "match_schema", "benchmark": "custom_gym", "evidence": "learning_curve", "next_action": "lora_smoke", "blocked": False, "risk": "中"},
        "templates": [
            "Octo 预训练模型要适配一台新 Panda 机械臂，只有约 100 条目标域示教，新增腕部相机并把动作改成关节位置。团队想先做从头训练。",
            "目标域数据很少，但观测从单 RGB 变成 RGB 加 wrist camera，动作空间也从末端位姿改成 joint position；需要比较小适配层与 scratch 基线。",
            "Octo checkpoint 可以处理语言和多相机输入；当前目标只改 action head 和少量 observation，预算不足以支持全量长跑。",
        ],
    },
    {
        "id": "openpi_memory_budget",
        "projects": ["openpi"],
        "questions": ["research_stage", "finetune_mode", "gpu_plan", "normalization", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "finetune", "finetune_mode": "lora", "gpu_plan": "single_lora", "normalization": "compare", "evidence": "learning_curve", "next_action": "lora_smoke", "blocked": False, "risk": "中"},
        "templates": [
            "要在一张 24GB GPU 上适配 Pi0 到 ALOHA 任务。团队想全量微调，但没有 70GB 级显存；目标数据和源数据的动作统计也尚未对比。",
            "openpi 的 LoRA 资源预算可行，全量方案明显超出当前单卡；本轮只需要验证数据管线、loss、checkpoint 加载和一条短 rollout。",
            "实验室只有 24GB 显卡，计划直接跑 Pi0 full fine-tune；同时沿用源 checkpoint 的 normalization stats，没有安排 target stats 对照。",
        ],
    },
    {
        "id": "openpi_normalization_drift",
        "projects": ["openpi", "lerobot"],
        "questions": ["research_stage", "data_format", "normalization", "failure_mode", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "data_contract", "data_format": "lerobot", "normalization": "compare", "failure_mode": "action_scale", "evidence": "normalization", "next_action": "compute_stats", "blocked": True, "risk": "高"},
        "templates": [
            "目标数据来自不同夹爪和控制频率，动作范围比 Pi0 源数据大一倍；当前 rollout 出现持续过冲，团队只看到了训练 loss 下降。",
            "迁移到新机器人后，状态量纲、夹爪开合范围和 action chunk 频率变化，仍沿用源 normalization；失败轨迹没有记录反归一化前后的数值。",
            "LeRobot 数据能正常读取，模型也能加载，但真机动作明显偏大。实验尚未做 fresh target stats 与 reload source stats 的对照。",
        ],
    },
    {
        "id": "openvla_unnorm_key",
        "projects": ["openvla"],
        "questions": ["research_stage", "action_space", "failure_mode", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "real_robot", "action_space": "unnorm_key", "failure_mode": "action_scale", "evidence": "video", "next_action": "calibrate_control", "blocked": True, "risk": "阻断"},
        "templates": [
            "OpenVLA 在仿真预测合理，但接 WidowX 真机后末端位移和夹爪幅度都过大。服务调用没有传与 BridgeData V2 对应的 unnorm_key。",
            "团队更换了数据集和机器人，却仍用旧 action statistics 执行 OpenVLA 输出；视频显示第一步就出现明显 overshoot。",
            "模型 checkpoint、图像预处理和语言 prompt 都能运行，真机失败集中在动作尺度与控制周期，尚未检查 dataset key 和控制接口单位。",
        ],
    },
    {
        "id": "openx_metadata_gap",
        "projects": ["openx", "openvla"],
        "questions": ["research_stage", "data_format", "action_space", "split_protocol", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "cross_embodiment", "data_format": "rlds", "action_space": "match_schema", "split_protocol": "embodiment_disjoint", "evidence": "manifest", "next_action": "schema_audit", "blocked": True, "risk": "高"},
        "templates": [
            "混合多个 Open X-Embodiment 来源时，数据仍是 RLDS episode，但没有记录 robot、camera、action dimension 和 language coverage；训练后无法解释跨本体结果。",
            "OpenVLA mixture 包含多种机器人和传感器，有些来源没有 task string，有些没有 wrist camera。团队计划不做字段审计，直接按 episode 数加权。",
            "跨 embodiment 论文只保留了统一后的图像和 action 张量，原始来源 ID、机器人形态和缺失语言字段都丢失了。",
        ],
    },
    {
        "id": "real_rollout_small_n",
        "projects": ["lerobot", "octo"],
        "questions": ["research_stage", "benchmark", "eval_protocol", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "real_robot", "benchmark": "real_rollout", "eval_protocol": "real_10", "evidence": "per_task", "next_action": "run_benchmark", "blocked": True, "risk": "高"},
        "templates": [
            "真机策略只成功了 3/3 次，团队准备把 100% 写进论文；没有固定任务初始状态、失败视频、重复 rollout 或每任务统计。",
            "机器人 demo 看起来可行，但目前每个任务只跑了三次，没有报告 episode 数、硬件、控制频率和失败类型。",
            "已有一条真实控制链路，想比较两个 checkpoint；目前没有至少 10 次 rollout，也没有与仿真结果分开记录。",
        ],
    },
    {
        "id": "robotwin_language_shift",
        "projects": ["robotwin"],
        "questions": ["research_stage", "split_protocol", "benchmark", "failure_mode", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "benchmark", "split_protocol": "language_disjoint", "benchmark": "robotwin", "failure_mode": "language_shift", "evidence": "per_task", "next_action": "run_benchmark", "blocked": False, "risk": "高"},
        "templates": [
            "策略只见过每个 RoboTwin 任务的一种英文指令，在同一场景换成 paraphrase 后成功率大幅下降；视觉和动作回放没有明显异常。",
            "RoboTwin 数据中语言模板变化很少，评估却声称语言鲁棒性；需要把语言多样性作为独立测试轴，而不是混进总平均。",
            "同一个双臂操作任务换了指令表达后失败，固定相机和物体布局下仍然如此；当前没有按语言模板拆分结果。",
        ],
    },
    {
        "id": "deployment_watchdog",
        "projects": ["openvla", "lerobot"],
        "questions": ["research_stage", "action_space", "evidence", "failure_mode", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "deployment", "action_space": "action_token", "evidence": "latency", "failure_mode": "control_loop", "next_action": "serve_policy", "blocked": True, "risk": "阻断"},
        "templates": [
            "模型已经通过 REST 服务接入机器人，但没有 watchdog、超时动作、P99 推理延迟和动作 chunk 中断策略；控制周期要求 100ms。",
            "部署脚本能返回 action，现场控制线程却没有记录服务延迟、丢帧和异常恢复。模型在仿真成功不能替代安全控制门槛。",
            "策略要从服务器远程推理，网络抖动时可能超过控制周期；目前没有本地 fallback、限幅和急停策略。",
        ],
    },
    {
        "id": "failure_camera_shift",
        "projects": ["robotwin", "octo"],
        "questions": ["research_stage", "benchmark", "failure_mode", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "failure_analysis", "benchmark": "robotwin", "failure_mode": "camera_shift", "evidence": "video", "next_action": "collect_failures", "blocked": False, "risk": "高"},
        "templates": [
            "clean 场景成功、换光照和桌面高度后失败，动作日志没有明显爆炸；需要先判断是视觉域偏移还是控制器问题。",
            "策略在训练相机分辨率上稳定，换到 wrist camera 后抓取偏移；已有失败视频但未对齐传感器时间戳。",
            "RoboTwin randomized 阶段失败集中在遮挡和背景变化，模型 action magnitude 正常，团队准备直接增加学习率。",
        ],
    },
    {
        "id": "cross_embodiment_action",
        "projects": ["octo", "openx"],
        "questions": ["research_stage", "finetune_mode", "action_space", "split_protocol", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "cross_embodiment", "finetune_mode": "head_mlp", "action_space": "match_schema", "split_protocol": "embodiment_disjoint", "evidence": "per_task", "next_action": "lora_smoke", "blocked": False, "risk": "高"},
        "templates": [
            "预训练策略来自多个 Open X-Embodiment 机器人，目标是迁移到新形态；输入增加 proprioception，action 由末端位姿变成 joint position。",
            "跨本体试验同时改变相机数量、状态向量和动作定义，现有结果只报告 pooled success rate，没有新机器人单独的 baseline。",
            "目标机器人数据量有限，团队希望证明预训练带来正迁移；需要保留 scratch 与 pretrained adaptation 对照，并按 embodiment 留出测试。",
        ],
    },
    {
        "id": "benchmark_reporting",
        "projects": ["lerobot", "libero"],
        "questions": ["research_stage", "benchmark", "eval_protocol", "evidence", "next_action", "blocked", "risk"],
        "answers": {"research_stage": "benchmark", "benchmark": "libero", "eval_protocol": "suite_50", "evidence": "per_task", "next_action": "run_benchmark", "blocked": False, "risk": "中"},
        "templates": [
            "要发布一个 LIBERO policy checkpoint，当前只有平均成功率，没有 suite 名、episode 数、GPU、checkpoint 版本和可复现 eval 命令。",
            "模型已经能被 LeRobot eval 加载，但 benchmark 表格缺少每个 suite 的 n_episodes 和任务级结果，无法比较 Spatial/Object/Goal/10。",
            "团队准备只展示最高的一组 seed；评估脚本没有固定 seed，也没有保留逐任务日志。",
        ],
    },
]


def make_state(case, rng, idx):
    prefix = rng.choice(["实验记录：", "项目状态：", "研究周报：", "请审阅下面的 VLA 计划："])
    suffix = rng.choice([
        "请给出能在今天验证的决定，而不是泛泛介绍模型。",
        "当前目标是形成可复现的科研结论。",
        "如果前置条件不满足，应明确指出阻塞。",
        "需要同时考虑数据、训练、评估和部署之间的依赖。",
    ])
    extras = f"机器人形态：{rng.choice(COMMON['robot'])}；约束：{rng.choice(COMMON['constraint'])}；评估随机性：{rng.choice(COMMON['seed'])}。"
    return f"{prefix}{rng.choice(case['templates'])} {extras}{suffix}"


def build_record(case, idx, split, rng):
    questions = {}
    answers = case["answers"]
    for qid in case["questions"]:
        spec = QUESTION_BANK[qid]
        instruction = rng.choice(spec["instructions"])
        criteria = deepcopy(spec["criteria"])
        label_value = answers[qid]
        if spec["type"] == "choice":
            questions[qid] = choice(instruction, criteria, label_value)
        elif spec["type"] == "score":
            questions[qid] = score(instruction, criteria, label_value)
        else:
            questions[qid] = noul(instruction, label_value)
    refs = [PROJECTS[p] for p in case["projects"]]
    return {
        "state": make_state(case, rng, idx),
        "questions": questions,
        "_meta": {
            "source": "vla_github_experience_v1",
            "id": f"vla_github_experience_v1/{split}/{idx:05d}",
            "group_id": f"vla_github_experience_v1/{case['id']}/{idx:05d}",
            "domain": "vla_research",
            "case": case["id"],
            "projects": case["projects"],
            "source_urls": [p["repo"] for p in refs],
            "knowledge_version": KNOWLEDGE["version"],
            "question_count": len(questions),
        },
    }


def build(split, n, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        case = CASES[i % len(CASES)]
        rows.append(build_record(case, i, split, rng))
    rng.shuffle(rows)
    return rows


def write(name, rows):
    path = OUT / name
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    built = {"train": build("train", 1600, 101), "dev": build("dev", 240, 103), "test": build("test", 480, 107)}
    paths = [write(f"{split}.jsonl", rows) for split, rows in built.items()]
    manifest = {
        "version": "vla-experience-v1",
        "generator": "research/build_experience_dataset.py",
        "knowledge_version": KNOWLEDGE["version"],
        "counts": {split: {"records": len(rows), "questions": sum(len(r["questions"]) for r in rows)} for split, rows in built.items()},
        "question_counts": {split: {qid: sum(1 for r in rows if qid in r["questions"]) for qid in sorted(QUESTION_BANK)} for split, rows in built.items()},
        "projects": [{"id": p["id"], "name": p["name"], "repo": p["repo"]} for p in KNOWLEDGE["projects"]],
        "case_families": [c["id"] for c in CASES],
        "note": "Programmatic, source-grounded research decision data; not visual-action trajectories.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", *(f"{p}: {sum(1 for _ in p.open(encoding='utf-8'))}" for p in paths))
    print("cases", len(CASES), "questions/record", sorted({len(c['questions']) for c in CASES}))
