export type VLAQuestion =
  | { type: "noul"; instructions: string; criteria: Record<string, string> }
  | { type: "choice"; instructions: string; criteria: Record<string, string> }
  | { type: "score"; instructions: string; criteria: string[] };

export type VLARequest = {
  state: string;
  model: string;
  questions: Record<string, VLAQuestion>;
};

export type VLAAnswer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> }
  | { type: "score"; score: number; confidence: number; legend: Record<string, string>; probabilities: Record<string, number> };

export type VLAResponse = {
  model: string;
  answers: Record<string, VLAAnswer>;
  usage: { input_tokens: number; output_tokens: number };
  latency_ms: number;
};

export const VLA_MODELS = {
  "vla-exp-qwen35-4b-v2": {
    label: "Qwen3.5-4B · 经验版",
    shortLabel: "4B 经验版",
    route: "4b",
  },
  "vla-exp-qwen05b": {
    label: "Qwen2.5-0.5B · 轻量版",
    shortLabel: "0.5B 轻量版",
    route: "05b",
  },
} as const;

export type VLAModel = keyof typeof VLA_MODELS;

export const VLA_LEGACY_QUESTIONS: Record<string, VLAQuestion> = {
  total_task: {
    type: "choice",
    instructions: "目前的 VLA 科研总任务是什么？",
    criteria: {
      data_pipeline: "VLA 数据管线建设",
      vision_language_alignment: "视觉语言对齐",
      action_policy_training: "动作策略训练",
      simulation_evaluation: "仿真评估",
      real_robot_evaluation: "真机评估",
      deployment_control: "策略部署与控制",
      failure_analysis: "失败轨迹分析",
    },
  },
  priority: {
    type: "score",
    instructions: "当前任务的优先程度如何？",
    criteria: ["低", "中", "高", "紧急"],
  },
  module: {
    type: "choice",
    instructions: "完成当前 VLA 任务应该使用什么模块？",
    criteria: {
      data: "轨迹数据模块",
      vision_encoder: "视觉编码器模块",
      policy: "动作策略模块",
      simulation: "仿真评估模块",
      robot_eval: "真机评估模块",
      deployment: "部署控制模块",
      analysis: "失败分析模块",
    },
  },
  next_action: {
    type: "choice",
    instructions: "下一步最应该执行什么动作？",
    criteria: {
      data_pipeline: "整理并划分轨迹数据",
      vision_language_alignment: "检查视觉特征与语言指令对齐",
      action_policy_training: "启动动作策略 LoRA 微调",
      simulation_evaluation: "在仿真基准上运行评估",
      real_robot_evaluation: "安排真实机器人测试",
      deployment_control: "启动策略推理和控制服务",
      failure_analysis: "抽取失败轨迹并分析误差",
    },
  },
  blocked: {
    type: "noul",
    instructions: "当前是否存在会阻塞任务推进的问题？",
    criteria: {
      true: "存在数据、设备或评估阻塞",
      false: "没有明显阻塞，可以继续执行",
    },
  },
};

export const VLA_EXPERIENCE_QUESTIONS: Record<string, VLAQuestion> = {
  research_stage: {
    type: "choice",
    instructions: "从项目状态看，主研究阶段是什么？",
    criteria: {
      data_contract: "数据契约与轨迹整理",
      finetune: "策略微调与适配",
      benchmark: "仿真基准评估",
      real_robot: "真机验证",
      deployment: "策略部署与控制",
      failure_analysis: "失败归因与复盘",
      cross_embodiment: "跨机器人泛化",
    },
  },
  data_format: {
    type: "choice",
    instructions: "当前数据应该采用什么数据契约？",
    criteria: {
      rlds: "统一为 RLDS episode 格式并保留来源元数据",
      lerobot: "采用 LeRobotDataset 的视频、状态、动作和 task 契约",
      native: "保留原生轨迹格式，只写适配读取器",
      manifest: "先建立 episode manifest 和字段覆盖审计",
    },
  },
  split_protocol: {
    type: "choice",
    instructions: "训练和测试应该怎样划分？",
    criteria: {
      episode_disjoint: "按 episode 去重后划分，避免同轨迹泄漏",
      task_disjoint: "按任务或语言目标留出，测组合泛化",
      language_disjoint: "按语言模板或指令表达留出，测语言鲁棒性",
      embodiment_disjoint: "按机器人和传感器形态留出，测跨本体泛化",
      libero_90_10: "按 LIBERO-90 预训练、LIBERO-10 下游测试",
      clean_random: "保留 clean 与 randomized 两个评估阶段",
    },
  },
  normalization: {
    type: "choice",
    instructions: "动作和状态归一化应该怎么处理？",
    criteria: {
      recompute: "在目标数据上重新计算并保存统计量",
      reuse: "直接复用源 checkpoint 的统计量",
      compare: "同时跑复用与重算两条对照实验",
      audit: "先审计量纲、范围和缺失值，再决定统计量",
    },
  },
  finetune_mode: {
    type: "choice",
    instructions: "在当前数据量和 GPU 约束下先选哪种微调方式？",
    criteria: {
      lora: "LoRA 微调，冻结大部分基座参数",
      head_only: "先只训练动作头或输出头",
      head_mlp: "训练输出头和小型 MLP 适配层",
      full: "全量微调视觉、语言和动作模块",
      scratch: "从头训练并作为容量基线",
    },
  },
  gpu_plan: {
    type: "choice",
    instructions: "当前硬件下哪种训练计划可行？",
    criteria: {
      single_lora: "单卡运行 LoRA，并把 batch/累积写入记录",
      multi_lora: "多卡 LoRA，固定全局 batch 和通信配置",
      a100_full: "使用 80GB 级 GPU 才考虑全量微调",
      cpu_debug: "仅用 CPU 做数据和接口 smoke test，不报告训练效果",
    },
  },
  action_space: {
    type: "choice",
    instructions: "动作输出在执行前最需要确认什么？",
    criteria: {
      match_schema: "先对齐目标机器人 action/state schema",
      unnorm_key: "推理时使用与数据集对应的 unnormalization key",
      ee_delta: "明确使用末端位姿增量和夹爪维度",
      joint_position: "明确使用关节位置控制并记录频率",
      action_token: "固定动作 tokenizer、chunk 长度和控制周期",
    },
  },
  benchmark: {
    type: "choice",
    instructions: "哪种评估基准最匹配当前主张？",
    criteria: {
      libero: "LIBERO suite，按知识迁移轴选择 Spatial/Object/Goal/10",
      robotwin: "RoboTwin clean/random 双阶段任务评估",
      bridge: "BridgeData V2 WidowX 真实环境评估",
      custom_gym: "封装成 Gym 接口后做可复现实验",
      real_rollout: "真机固定任务和固定 episode 数 rollout",
    },
  },
  eval_protocol: {
    type: "choice",
    instructions: "本轮结果应采用什么评估协议？",
    criteria: {
      task_100: "每个任务至少 100 个 episode 并按任务报告",
      suite_50: "每个 suite 至少 50 个 episode 并报告硬件",
      real_10: "真机至少 10 次 rollout 并保存失败视频",
      fixed_seed: "固定 seed、任务顺序和 checkpoint 后再比较",
      clean_random: "分别报告 clean 与 randomized 成功率",
    },
  },
  evidence: {
    type: "choice",
    instructions: "为了让结论可复现，下一步应该保存什么？",
    criteria: {
      manifest: "数据 manifest、字段覆盖和去重报告",
      per_task: "逐任务成功率、失败类型和置信区间",
      learning_curve: "按 epoch/step 保存 loss 与 success 曲线",
      normalization: "源统计量与目标统计量的对照结果",
      latency: "控制周期、P50/P95/P99 延迟和 watchdog 日志",
      video: "失败轨迹视频、传感器同步和动作回放",
    },
  },
  failure_mode: {
    type: "choice",
    instructions: "当前失败现象最应该先归因到哪里？",
    criteria: {
      data_quality: "轨迹质量、重复、缺失字段或标签问题",
      language_shift: "语言模板或任务指令分布发生变化",
      camera_shift: "视角、分辨率、光照或相机同步变化",
      action_scale: "动作量纲、归一化或控制频率错误",
      task_shift: "物体、布局、任务组合或 embodiment 分布变化",
      control_loop: "推理延迟、chunk 执行或安全控制问题",
    },
  },
  next_action: {
    type: "choice",
    instructions: "如果只做一件事，下一步应该是什么？",
    criteria: {
      schema_audit: "审计字段、单位、时间戳和 episode manifest",
      make_split: "生成去重且与论文主张一致的 train/dev/test split",
      compute_stats: "计算目标域 normalization 并做复用/重算对照",
      lora_smoke: "先做短步数 LoRA smoke test，确认 loss 和 checkpoint 可加载",
      run_benchmark: "运行固定任务、seed 和 episode 数的基准评估",
      collect_failures: "保存失败视频、状态和动作并建立失败标签",
      calibrate_control: "校准动作反归一化、控制周期、延迟和安全边界",
      serve_policy: "通过独立推理服务接入机器人控制回路并加 watchdog",
    },
  },
  blocked: {
    type: "noul",
    instructions: "当前是否存在必须先解决的阻塞？",
    criteria: { true: "存在缺失字段、数据泄漏、资源或安全阻塞", false: "前置条件齐全，可以开始该实验" },
  },
  risk: {
    type: "score",
    instructions: "当前结论被误读或不可复现的风险是多少？",
    criteria: ["低", "中", "高", "阻断"],
  },
};

export const DEFAULT_VLA_STATE =
  "项目使用 RoboTwin 2.0 和 LIBERO 操作基准，已有相机图像、语言指令和动作序列，但还没有划分训练集和测试集。当前需要整理轨迹并准备 VLA 策略训练。";

export async function vlaSystemOne(state: string, model: VLAModel): Promise<VLAResponse> {
  const response = await fetch(`/vla-api/${VLA_MODELS[model].route}/v1/systemone`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ state, model, questions: VLA_EXPERIENCE_QUESTIONS }),
  });
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json() as Promise<VLAResponse>;
}
