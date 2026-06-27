# Mahjong Web Table

<p align="center">
  <img src="./logo.png" alt="Mahjong Web Table logo" width="180">
</p>

## 🀄 让 AI 进行麻将对决？

你可以使用此项目自行创建更多规则来让 AI 打麻将，内置支持北方推倒和规则可以直接使用，快让你自己的 AI 来对决吧！此项目同时支持自行编写适配器，内置 LLM 适配器，你可以自行编写适配器来运行自己的麻将模型。

一个独立的麻将规则引擎、浏览器调试牌桌和 LLM 控制器测试项目。

[English README](./README.md)

![调试牌桌截图](./test_1.png)

## ✨ 项目内容

- 🎲 确定性的四人麻将牌桌引擎。
- 🧩 北方推倒和规则插件。
- 🖥️ 浏览器调试牌桌，支持完整牌面可视化。
- 🤖 支持 Human、Random、Debug、本地/远程 LLM 控制玩家。
- 🔌 OpenAI-compatible LLM 适配器预设，支持 OpenAI、OpenRouter、DeepSeek、Gemini、Mistral、Groq、Together AI、xAI 和本地 OpenAI-compatible 服务。
- 🧠 可选的本地 `transformers` Qwen 服务。
- 📜 FullLog 导出，用于查看 prompt、模型决策、牌桌状态、事件和计分记录。

## ✅ 环境要求

- Python 3.11 或更新版本
- virtualenv、conda 或其他隔离 Python 环境
- 可选：用于本地 Qwen 推理的 GPU 或 Apple Silicon 加速

## 📦 安装

```bash
python -m pip install -e .
```

如果需要本地 Qwen 推理：

```bash
python -m pip install -e '.[llm]'
```

## 🚀 启动 Web 牌桌

```bash
uvicorn mahjong_ai.web.app:app --host 127.0.0.1 --port 8765
```

打开：

```text
http://127.0.0.1:8765/
```

UI 会显示所有玩家手牌、合法动作、响应窗口、完整状态 JSON、控制器 trace 和 FullLog 导出入口。

## 🤖 LLM 控制器预设

每个玩家都可以在 Debug Table 的玩家二级菜单里单独配置控制器。

内置预设：

- `debug`：本地确定性控制器，适合测试。
- `apple-fm`：通过 `fm` CLI 调用 Apple Foundation Models。
- `openai`：OpenAI Chat Completions API。
- `openrouter`：OpenRouter，默认模型为 `qwen/qwen3.5-flash-02-23`。
- `deepseek`：DeepSeek V4 Flash，tool call 时关闭 thinking。
- `deepseek-v4-pro`：DeepSeek V4 Pro，tool call 时关闭 thinking。
- `gemini`：Google Gemini OpenAI-compatible endpoint。
- `mistral`：Mistral function calling endpoint。
- `groq`：Groq OpenAI-compatible endpoint。
- `together`：Together AI OpenAI-compatible endpoint。
- `xai`：xAI OpenAI-compatible endpoint。
- `local-openai`：本地 OpenAI-compatible endpoint，默认 `http://127.0.0.1:8001/v1`。

适配器会要求模型调用 `choose_mahjong_action` tool，并从合法动作列表中选择一个 `action_id`。Prompt 会包含当前规则、当前玩家可见手牌、公开弃牌、通过副露可知的其他玩家牌、最近公开事件和合法动作。

不要把 API Key 提交到 Git。Token 只应在本地 UI 中填写。

## 🧠 启动本地 Qwen OpenAI-Compatible API

下载 Qwen 到 `models/`：

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3.5-2B", local_dir="models/Qwen3.5-2B")'
```

启动本地 OpenAI-compatible 服务：

```bash
python -u scripts/serve_qwen_transformers_openai.py \
  --model-path models/Qwen3.5-2B \
  --served-model-name qwen3.5-2b-transformers \
  --host 127.0.0.1 \
  --port 8001
```

服务接口：

```text
GET  http://127.0.0.1:8001/v1/models
POST http://127.0.0.1:8001/v1/chat/completions
```

Web UI 中可以这样配置 LLM 玩家：

```text
Provider: Local OpenAI-compatible
Base URL: http://127.0.0.1:8001/v1
Token: local
Model: qwen3.5-2b-transformers
```

## 🧪 运行 Smoke Hand

```bash
python scripts/run_table_smoke.py --seed 42
```

## ✅ 测试

```bash
python -m unittest discover -s tests
```

预期结果：

```text
OK
```

## 📁 项目结构

```text
configs/rules/       规则 YAML 配置
scripts/             工具脚本和本地模型服务
src/mahjong_ai/      引擎、规则、智能体、观测构造和 Web UI
tests/               引擎、规则、LLM 适配器和 Web API 测试
logo.png             项目 logo
test_1.png           示例 UI 截图
```

## 📄 开源协议

本项目使用 MIT License，详见 [LICENSE](./LICENSE)。
