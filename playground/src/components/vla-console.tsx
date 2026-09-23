"use client";

import { useState } from "react";
import { DEFAULT_VLA_STATE, VLA_EXPERIENCE_QUESTIONS, VLA_MODELS, type VLAAnswer, type VLAModel, type VLAResponse, vlaSystemOne } from "@/lib/vla";

const EXAMPLES = [
  {
    name: "数据管线",
    state: DEFAULT_VLA_STATE,
  },
  {
    name: "仿真评估",
    state: "动作策略已经完成一轮 LoRA 微调，准备在 RLBench 和 LIBERO 上做基准测试；目前还没有统一成功率、碰撞率和语言完成度的评估脚本。",
  },
  {
    name: "失败分析",
    state: "RoboTwin 双臂整理任务在仿真中成功，但真机测试经常抓取偏移；已有失败视频和末端位姿日志，需要定位视觉、标定还是动作策略的问题。",
  },
];

const LABELS: Record<string, string> = {
  data_contract: "数据契约",
  finetune: "策略微调",
  benchmark: "仿真评估",
  real_robot: "真机验证",
  deployment: "部署控制",
  cross_embodiment: "跨机器人泛化",
  rlds: "RLDS episode",
  lerobot: "LeRobotDataset",
  native: "原生轨迹读取器",
  manifest: "episode manifest",
  episode_disjoint: "episode 去重划分",
  task_disjoint: "任务留出",
  language_disjoint: "语言模板留出",
  embodiment_disjoint: "机器人形态留出",
  libero_90_10: "LIBERO-90/10",
  clean_random: "clean/random 双阶段",
  recompute: "重算目标域统计量",
  reuse: "复用源统计量",
  compare: "复用/重算对照",
  audit: "先审计量纲和范围",
  lora: "LoRA 微调",
  head_only: "只训练动作头",
  head_mlp: "动作头 + MLP",
  full: "全量微调",
  scratch: "从头训练",
  single_lora: "单卡 LoRA",
  multi_lora: "多卡 LoRA",
  a100_full: "80GB GPU 全量微调",
  cpu_debug: "CPU 数据 smoke test",
  match_schema: "对齐 action/state schema",
  unnorm_key: "确认 unnormalization key",
  ee_delta: "末端位姿增量",
  joint_position: "关节位置控制",
  action_token: "动作 tokenizer/chunk",
  libero: "LIBERO suite",
  robotwin: "RoboTwin clean/random",
  bridge: "BridgeData V2 WidowX",
  custom_gym: "Gym 可复现实验",
  real_rollout: "真机 rollout",
  task_100: "每任务 100 episodes",
  suite_50: "每 suite 50 episodes",
  real_10: "真机 10 次 rollout",
  fixed_seed: "固定 seed/任务顺序",
  per_task: "逐任务成功率",
  learning_curve: "loss/success 曲线",
  normalization: "归一化统计量对照",
  latency: "P50/P95/P99 延迟",
  video: "失败视频与回放",
  data_quality: "轨迹质量/字段问题",
  language_shift: "语言分布变化",
  camera_shift: "相机/视觉域偏移",
  action_scale: "动作尺度/频率错误",
  task_shift: "任务或 embodiment 偏移",
  control_loop: "推理延迟/控制回路",
  schema_audit: "审计字段和 episode manifest",
  make_split: "生成无泄漏数据划分",
  compute_stats: "计算目标域 normalization",
  lora_smoke: "运行短步数 LoRA smoke test",
  run_benchmark: "运行固定协议基准评估",
  collect_failures: "采集并标注失败轨迹",
  calibrate_control: "校准动作、延迟和安全边界",
  serve_policy: "部署推理服务并加 watchdog",
  data_pipeline: "数据管线",
  vision_language_alignment: "视觉语言对齐",
  action_policy_training: "动作策略训练",
  simulation_evaluation: "仿真评估",
  real_robot_evaluation: "真机评估",
  deployment_control: "部署与控制",
  failure_analysis: "失败分析",
  data: "轨迹数据模块",
  vision_encoder: "视觉编码器模块",
  policy: "动作策略模块",
  simulation: "仿真评估模块",
  robot_eval: "真机评估模块",
  analysis: "失败分析模块",
};

function label(key: string) {
  return LABELS[key] ?? key;
}

function probabilityRows(answer: VLAAnswer) {
  if (answer.type === "noul") return [["存在", answer.noul], ["不存在", 1 - answer.noul]] as [string, number][];
  return Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]).map(([key, value]) => [
    answer.type === "score" ? answer.legend[key] ?? key : label(key), value,
  ] as [string, number]);
}

function headline(answer: VLAAnswer) {
  if (answer.type === "noul") return answer.noul >= 0.5 ? "存在阻塞" : "可以继续";
  if (answer.type === "choice") return label(answer.choice);
  const level = answer.legend[String(Math.round(answer.score))] ?? "";
  return `${level} · ${answer.score.toFixed(1)}`;
}

function ResultCard({ name, answer }: { name: string; answer: VLAAnswer }) {
  const rows = probabilityRows(answer);
  const confidence = answer.type === "noul" ? Math.max(answer.noul, 1 - answer.noul) : answer.confidence;
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_12px_40px_rgba(15,23,42,0.05)] dark:border-slate-800 dark:bg-slate-950">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-slate-400">{name}</p>
          <h3 className="mt-2 text-lg font-semibold tracking-tight text-slate-900 dark:text-slate-50">{headline(answer)}</h3>
        </div>
        <span className="rounded-full bg-cyan-50 px-2.5 py-1 text-xs font-medium text-cyan-700 dark:bg-cyan-950/60 dark:text-cyan-300">
          {(confidence * 100).toFixed(0)}%
        </span>
      </div>
      <div className="mt-5 space-y-2.5">
        {rows.map(([text, value]) => (
          <div key={text}>
            <div className="mb-1 flex justify-between gap-3 text-xs text-slate-500 dark:text-slate-400">
              <span className={value === Math.max(...rows.map(([, v]) => v)) ? "font-medium text-slate-800 dark:text-slate-200" : ""}>{text}</span>
              <span className="tabular-nums">{(value * 100).toFixed(0)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div className="h-full rounded-full bg-cyan-500 transition-all duration-500" style={{ width: `${Math.max(1, value * 100)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

export function VLAConsole() {
  const [state, setState] = useState(DEFAULT_VLA_STATE);
  const [model, setModel] = useState<VLAModel>("vla-exp-qwen35-4b-v2");
  const [result, setResult] = useState<VLAResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await vlaSystemOne(state, model));
    } catch (err) {
      setError(err instanceof Error ? err.message : "请求失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f7fafc] text-slate-900 dark:bg-[#070b14] dark:text-slate-100">
      <div className="mx-auto max-w-6xl px-5 py-8 sm:px-8 sm:py-12">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-xl bg-slate-900 text-sm font-bold text-white dark:bg-cyan-400 dark:text-slate-950">VLA</div>
            <div>
              <p className="text-sm font-semibold tracking-tight">科研任务控制台</p>
              <p className="text-xs text-slate-500 dark:text-slate-400">Kev decision model · 中文 VLA 研究版</p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300">
            <span className="size-1.5 rounded-full bg-emerald-500" /> {VLA_MODELS[model].label}已连接
          </div>
        </header>

        <section className="mt-12 max-w-3xl">
          <p className="text-sm font-medium text-cyan-600 dark:text-cyan-400">VLA RESEARCH ROUTER</p>
          <h1 className="mt-3 text-4xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-5xl dark:text-white">先判断研究状态，<br className="hidden sm:block" />再决定下一步。</h1>
          <p className="mt-5 max-w-2xl text-base leading-7 text-slate-600 dark:text-slate-400">输入当前实验、数据或机器人状态。模型会根据场景选择数据契约、划分策略、训练方式、评估证据、失败归因和下一步动作。</p>
        </section>

        <div className="mt-10 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_12px_40px_rgba(15,23,42,0.05)] sm:p-6 dark:border-slate-800 dark:bg-slate-950">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="font-semibold tracking-tight">当前研究状态</h2>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">支持 RoboTwin、LIBERO、RLBench、ALOHA 等场景</p>
              </div>
              <span className="rounded-md bg-slate-100 px-2 py-1 font-mono text-[11px] text-slate-500 dark:bg-slate-900">/v1/systemone</span>
            </div>
            <div className="mt-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-cyan-100 bg-cyan-50/70 px-3 py-2.5 dark:border-cyan-950 dark:bg-cyan-950/30">
              <div>
                <p className="text-xs font-semibold text-cyan-800 dark:text-cyan-200">推理模型</p>
                <p className="mt-0.5 text-[11px] text-cyan-700/70 dark:text-cyan-300/70">切换后下一次分析使用所选模型</p>
              </div>
              <select
                value={model}
                onChange={(event) => { setModel(event.target.value as VLAModel); setResult(null); setError(null); }}
                disabled={busy}
                aria-label="选择 VLA 推理模型"
                className="rounded-lg border border-cyan-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 outline-none transition focus:border-cyan-500 focus:ring-4 focus:ring-cyan-500/10 disabled:cursor-not-allowed disabled:opacity-60 dark:border-cyan-900 dark:bg-slate-950 dark:text-slate-200"
              >
                {Object.entries(VLA_MODELS).map(([id, info]) => <option key={id} value={id}>{info.label}</option>)}
              </select>
            </div>
            <textarea
              value={state}
              onChange={(event) => setState(event.target.value)}
              aria-label="当前研究状态"
              className="mt-5 min-h-56 w-full resize-y rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm leading-6 outline-none transition focus:border-cyan-500 focus:ring-4 focus:ring-cyan-500/10 dark:border-slate-800 dark:bg-slate-900/70"
            />
            <div className="mt-4 flex flex-wrap gap-2">
              {EXAMPLES.map((example) => (
                <button key={example.name} type="button" onClick={() => { setState(example.state); setResult(null); setError(null); }} className="rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-600 transition hover:border-cyan-400 hover:text-cyan-700 dark:border-slate-700 dark:text-slate-400 dark:hover:text-cyan-300">
                  {example.name}
                </button>
              ))}
            </div>
            <button type="button" onClick={run} disabled={busy || !state.trim()} className="mt-6 flex h-11 w-full items-center justify-center rounded-xl bg-slate-900 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-cyan-400 dark:text-slate-950 dark:hover:bg-cyan-300">
              {busy ? "分析中…" : "分析当前状态"}
            </button>
            {error && <p className="mt-3 rounded-lg bg-red-50 p-3 text-xs leading-5 text-red-700 dark:bg-red-950/40 dark:text-red-300">{error}</p>}
          </section>

          <section>
            {!result ? (
              <div className="flex min-h-full flex-col justify-center rounded-2xl border border-dashed border-slate-300 bg-white/60 p-8 text-center dark:border-slate-700 dark:bg-slate-950/40">
                <div className="mx-auto grid size-12 place-items-center rounded-2xl bg-cyan-50 text-xl text-cyan-600 dark:bg-cyan-950/50 dark:text-cyan-300">↗</div>
                <h2 className="mt-5 font-semibold tracking-tight">等待一次分析</h2>
                <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-slate-500 dark:text-slate-400">左侧输入实验状态后，模型会在一次前向推理中返回一组随场景变化的科研决策。</p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="mb-5 flex items-end justify-between gap-4">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-[0.18em] text-cyan-600 dark:text-cyan-400">分析结果</p>
                    <h2 className="mt-1 text-2xl font-semibold tracking-tight">下一步工作建议</h2>
                  </div>
                  <p className="text-right text-xs tabular-nums text-slate-500 dark:text-slate-400">{VLA_MODELS[model].shortLabel}<br />{result.latency_ms.toFixed(0)} ms · {result.usage.input_tokens} tokens</p>
                </div>
                {Object.entries(result.answers).map(([id, answer]) => <ResultCard key={id} name={VLA_EXPERIENCE_QUESTIONS[id].instructions} answer={answer} />)}
              </div>
            )}
          </section>
        </div>

        <footer className="mt-12 flex flex-wrap justify-between gap-3 border-t border-slate-200 pt-5 text-xs text-slate-400 dark:border-slate-800">
          <span>本页面支持切换本机 8014（4B）与 8013（0.5B）VLA 经验模型</span>
          <span>科研任务路由模型，不是完整的视觉动作策略</span>
        </footer>
      </div>
    </main>
  );
}
