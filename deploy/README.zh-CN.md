# 本机 Kev 部署

目录：`/data/ssd/xyt/data/jev`。上游：<https://github.com/jaredpalmer/kev>，提交 `90990a5`。

部署 Kev-4B（Qwen3.5-4B-Base + 官方 LoRA/指针头），使用 GPU 0、BF16。

2026-09-22 验证结果：100 项离线测试通过（排除 1 项需额外训练数据的研究测试），7 项 API 测试全部通过；真实浏览器预设切换、Run 按钮、网页代理到 GPU 的推理链路均通过。示例请求返回 billing 分类，模型耗时 106.7 ms；首次请求会编译 GPU 内核，耗时较长。完整记录见 `deploy/deployment.json`。

## 访问

- 网页：<http://127.0.0.1:3001>
- API 模型信息：<http://127.0.0.1:8009/v1/models>
- API 文档：<http://127.0.0.1:8009/docs>
- 网页同源 API：`http://127.0.0.1:3001/kev/v1/systemone`

在自己的电脑上建立 SSH 隧道（将服务器地址换成实际 SSH 地址）：

```bash
ssh -N -L 3001:127.0.0.1:3001 -L 8009:127.0.0.1:8009 yutongxiao@服务器地址
```

然后在电脑浏览器访问 `http://localhost:3001`。服务仅监听服务器回环地址，与上游默认一致。

## 管理

```bash
systemctl --user status kev-api kev-web
systemctl --user restart kev-api kev-web
systemctl --user stop kev-api kev-web
systemctl --user start kev-api kev-web
tail -f /data/ssd/xyt/data/jev/deploy/logs/api.log
tail -f /data/ssd/xyt/data/jev/deploy/logs/web.log
```

服务定义在 `~/.config/systemd/user/kev-{api,web}.service`，副本保存在此目录。
已启用用户 linger，使用户服务不依赖 SSH 会话。
修改 GPU 时调整 API 服务中的 `CUDA_VISIBLE_DEVICES`，随后运行 `systemctl --user daemon-reload` 并重启 API。

## 来源和环境

- 源码通过 GitHub 直连克隆；Python 依赖使用清华 PyPI，npm 使用 npmmirror。
- Python 3.12 虚拟环境：`.venv`，上游锁定的服务依赖导出至 `deploy/requirements.lock.txt`。
- CUDA 加速按上游 `modal_app.py` 使用 `flash-linear-attention==0.5.2`、`triton==3.7.1`。这会覆盖 Torch 2.8 元数据中的 Triton 3.4 固定依赖，`uv pip check` 因此会报告一项已知冲突；这是上游为新内核采取的配置，实际兼容性以 GPU 推理测试为准。完整安装清单见 `deploy/installed-requirements.txt`。
- 官方 Kev 权重通过 ghfast 下载 GitHub Release；SHA-256 同时与 GitHub API 的 digest 和发布校验文件比对。
- 基座来自 ModelScope 的 `Qwen/Qwen3.5-4B-Base`；固定的文件修订号及 SHA-256 保存在 `deploy/base-model-files.json`，下载脚本逐文件验证。
- `models/kev-4b` 保留官方权重。`deploy/serve_local.py` 仅在内存中把基座位置切换为本地 ModelScope 模型目录，不改写官方 checkpoint。
- ModelScope 修订号与 Hugging Face 修订号不同；具体来源以下载清单为准。
- 运行服务启用离线模式，并清除 HTTP/HTTPS/ALL_PROXY，无需代理或再次下载模型。
- 网页使用 `npm run build` 生成的生产构建。

## 调用示例

```bash
curl --noproxy '*' http://127.0.0.1:8009/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice. Please refund the duplicate charge.","model":"kev-latest","questions":{"billing":{"type":"noul","instructions":"Is this request about billing?"}}}'
```

项目返回判断、分类、评分及概率，不生成聊天回复。

## 验证

```bash
cd /data/ssd/xyt/data/jev
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  KEV_BASE_URL=http://127.0.0.1:8009 .venv/bin/python -m pytest tests/test_api.py -q
```

安装、构建与测试日志保存在 `deploy/logs/`。
单元测试所需的 Qwen2.5-0.5B 分词器也从 ModelScope 下载并校验，保存在 `models/test-tokenizer/`；`Qwen/Qwen2.5-0.5B` 是供上游测试离线查找的本地链接，不包含该模型权重。
研究测试 `test_training_plan_matches_registered_screen` 依赖 Hugging Face 上额外的训练数据，本机离线环境无法下载；因此离线测试命令通过 `-k 'not test_training_plan_matches_registered_screen'` 排除这一项。原始失败原因保留在 `deploy/logs/unit-tests.log`。
