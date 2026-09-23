"""Build deterministic Chinese VLA research decisions for a Kev-style research run.

This is a decision dataset about VLA experiments (data, vision-language alignment,
action policies, simulation, robots, deployment and failure analysis); it is not a
replacement for image/action trajectories used to train a VLA policy.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"

TASKS = {
    "data_pipeline": ["VLA 数据管线建设", "机器人轨迹数据整理", "示教数据清洗"],
    "vision_language_alignment": ["视觉语言对齐", "图像与语言指令对齐", "视觉编码器适配"],
    "action_policy_training": ["动作策略训练", "VLA 策略微调", "动作预测模型训练"],
    "simulation_evaluation": ["仿真评估", "模拟器基准测试", "仿真任务成功率评测"],
    "real_robot_evaluation": ["真机评估", "真实机器人实验", "实机操作成功率测试"],
    "deployment_control": ["策略部署与控制", "机器人推理服务上线", "控制接口部署"],
    "failure_analysis": ["失败轨迹分析", "VLA 误差分析", "机器人失败案例复盘"],
}
MODULES = {
    "data": ["轨迹数据模块", "数据清洗模块", "示教数据模块"],
    "vision_encoder": ["视觉编码器模块", "视觉表征模块", "图像特征模块"],
    "policy": ["动作策略模块", "策略微调模块", "动作头模块"],
    "simulation": ["仿真评估模块", "模拟器模块", "基准任务模块"],
    "robot_eval": ["真机评估模块", "机器人实验模块", "成功率统计模块"],
    "deployment": ["部署控制模块", "推理服务模块", "机器人控制接口模块"],
    "analysis": ["失败分析模块", "误差分析模块", "轨迹可视化模块"],
}
PRIORITIES = ["低", "中", "高", "紧急"]
NEXT_ACTIONS = {
    "data_pipeline": "整理并划分轨迹数据",
    "vision_language_alignment": "检查视觉特征与语言指令对齐",
    "action_policy_training": "启动动作策略 LoRA 微调",
    "simulation_evaluation": "在仿真基准上运行评估",
    "real_robot_evaluation": "安排真实机器人测试",
    "deployment_control": "启动策略推理和控制服务",
    "failure_analysis": "抽取失败轨迹并分析误差",
}

TASK_MODULE = {
    "data_pipeline": "data", "vision_language_alignment": "vision_encoder",
    "action_policy_training": "policy", "simulation_evaluation": "simulation",
    "real_robot_evaluation": "robot_eval", "deployment_control": "deployment",
    "failure_analysis": "analysis",
}

VLA_DATASETS = [
    "RoboTwin 2.0", "LIBERO", "Open X-Embodiment", "CALVIN", "RLBench",
    "ALOHA", "BridgeData V2", "LeRobot 轨迹数据",
]
ROBOT_TASKS = [
    "抓取并放置方块", "将杯子移动到托盘", "打开抽屉并取出物体",
    "双臂协同整理物品", "按照语言指令完成桌面操作", "在域随机化场景中完成操作",
]
SUBJECTS = [
    "Qwen 视觉语言动作模型", "小型 VLA 策略", "机器人操作策略",
    "双臂操作模型", "语言条件动作预测器", "视觉编码器与动作头",
]
CONTEXTS = [
    "用于论文实验", "用于实验室机器人流水线", "用于比较不同 VLA 策略",
    "用于 RoboTwin 仿真任务", "用于 LIBERO 操作基准", "用于真实机器人验证",
]
EVIDENCE = [
    "目前只有少量示教轨迹，还没有完成去重和质量检查",
    "已经有相机图像、语言指令和动作序列，但还没有划分训练集和测试集",
    "仿真策略已经部署，但是没有完成跨任务成功率评估",
    "训练日志显示动作损失下降，但还没有做视觉语言对齐检查",
    "测试结果已经生成，需要定位抓取失败和时序误差",
    "服务器上已有 GPU 和机器人推理接口，可以开始实验",
]


def make_state(task, module, priority, rng, idx):
    task_text = rng.choice(TASKS[task])
    module_text = rng.choice(MODULES[module])
    subject = rng.choice(SUBJECTS)
    context = rng.choice(CONTEXTS)
    evidence = rng.choice(EVIDENCE)
    dataset = rng.choice(VLA_DATASETS)
    robot_task = rng.choice(ROBOT_TASKS)
    blocked = evidence != "服务器上已有 GPU 和机器人推理接口，可以开始实验"
    forms = [
        f"当前总任务是{task_text}，对象是{subject}，数据或基准采用{dataset}，{context}。{evidence}。目标任务是{robot_task}。",
        f"请判断现在最主要的 VLA 工作：{task_text}。项目是{subject}，使用{dataset}，目标是{context}；当前情况：{evidence}。",
        f"VLA 研究项目进展：{evidence}。接下来要围绕{subject}完成{task_text}，在{dataset}上验证{robot_task}，请给出主任务、优先级和模块。",
        f"任务单 {idx}：我们需要推进{task_text}，对应{module_text}。场景：{context}；数据集：{dataset}；机器人任务：{robot_task}；背景：{evidence}。",
    ]
    return rng.choice(forms), blocked


def record(state, task, module, priority, blocked, idx):
    return {
        "state": state,
        "questions": {
            "total_task": {"type": "choice", "instructions": "目前的 VLA 科研总任务是什么？", "criteria": {k: v[0] for k, v in TASKS.items()}, "label": task},
            "priority": {"type": "score", "instructions": "当前任务的优先程度如何？", "criteria": PRIORITIES, "label": PRIORITIES.index(priority)},
            "module": {"type": "choice", "instructions": "完成当前 VLA 任务应该使用什么模块？", "criteria": {k: v[0] for k, v in MODULES.items()}, "label": module},
            "next_action": {"type": "choice", "instructions": "下一步最应该执行什么动作？", "criteria": {k: v for k, v in NEXT_ACTIONS.items()}, "label": task},
            "blocked": {"type": "noul", "instructions": "当前是否存在会阻塞任务推进的问题？", "criteria": {"true": "存在数据、设备或评估阻塞", "false": "没有明显阻塞，可以继续执行"}, "label": blocked},
        },
        "_meta": {"source": "zh_vla_research", "id": f"zh_vla_research/{idx}", "group_id": f"zh_vla_research/{idx}", "domain": "vla"},
    }


def build(n, seed):
    rng = random.Random(seed)
    rows = []
    task_keys = list(TASKS)
    for i in range(n):
        task = task_keys[i % len(task_keys)]
        module = TASK_MODULE[task]
        if task in {"deployment_control", "action_policy_training", "real_robot_evaluation"}:
            priority = rng.choices(PRIORITIES, weights=[1, 3, 6, 2])[0]
        elif task in {"simulation_evaluation", "failure_analysis"}:
            priority = rng.choices(PRIORITIES, weights=[1, 4, 5, 1])[0]
        else:
            priority = rng.choices(PRIORITIES, weights=[3, 5, 3, 1])[0]
        state, blocked = make_state(task, module, priority, rng, i)
        rows.append(record(state, task, module, priority, blocked, i))
    rng.shuffle(rows)
    return rows


def write(name, rows):
    path = OUT / name
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write("train.jsonl", build(1200, 17))
    write("dev.jsonl", build(210, 23))
    write("test.jsonl", build(420, 29))
    print("wrote", *(f"{p.name}: {sum(1 for _ in p.open())}" for p in OUT.glob("*.jsonl")))
