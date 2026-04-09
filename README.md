---

<p align="center">
  <img src="assets/logo.png" width="300" alt="AutoResearcher Logo">
</p>

<h3 align="center">AutoResearcher — 深度自动化网络调研流水线</h3>
<p align="center"><b>一句话描述 → 一份媲美投行的深度研究报告</b></p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00?style=for-the-badge" alt="LangGraph"></a>
  <a href="https://textual.textualize.io/"><img src="https://img.shields.io/badge/Textual-TUI-009688?style=for-the-badge&logo=python&logoColor=white" alt="Textual"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Beta-yellow?style=flat-square" alt="Status" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" />
</p>

---

*Give the AI a research topic. It searches, reasons, filters, synthesizes, and writes — all by itself.*

<br/>

### 📑 最新自动生成的报告样例
> 以下报告完全由 AutoResearcher 自动采集、交叉验证并执笔编写：

| 课题 | 类型 | 核心模型 | 耗时 |
|:----:|:----:|:-------:|:-------:|
| 🤖 **[2026年3月发布的大模型及跑分调研](docs/examples/llm_releases_2026_03.md)** | 行业动态 | `gpt-5.4` | ~ 20 分钟 |
| 🏆 **[2025诺贝尔物理学奖实际应用前景](docs/examples/nobel_prize_2025.md)** | 科技前沿 | `gpt-4.1` | ~ 3.5 分钟 |

<br/>

## 📖 目录
- [💡 项目简介](#-项目简介)
- [🎬 工作原理](#-工作原理)
- [🏗️ 系统架构](#️-系统架构)
- [🚀 快速开始](#-快速开始)
- [⚙️ 配置说明](#️-配置说明)
- [📁 项目结构](#-项目结构)
- [🎯 功能特性](#-功能特性)
- [🛠️ 技术栈](#️-技术栈)
- [🤝 贡献指南](#-贡献指南)
- [📜 License](#-license)

---

## 💡 项目简介
**AutoResearcher** 是一个基于多 Agent 协作的深度自动化网络调研系统。

突破了常见的“大模型搜搜索搜就直接写”的粗糙路径，引入了 LATS（语言智能体树搜索）、私有 RAG 召回，以及极其独特的三层模型降本机制。你只需要给出一个调研主题，它就能不知疲倦地发散多路 Researcher 去全网收集、自我批判、交叉比对数据，最后合并输出极具深度的万字长文。

| | 特性 | 说明 |
|:---:|------|------|
| 🤖 | **全自动流水线** | 从细化提纲到最终截稿，全程无需人工干预 |
| 🌲 | **LATS 树搜索** | 基于 UCB1 算法发散搜索路径，自我反思并主动抛弃低价值信息 |
| 🪓 | **三层降本模型** | "好钢用在刀刃上"，让最高智商的模型做决策，让廉价模型洗数据 |
| 📺 | **高级 TUI 界面** | 绝赞的 Textual 双屏界面，左眼看 Log，右眼看实时草稿 |
| 📊 | **批量跑分框架** | 一键跑 N 个话题组合，自动输出评测对比 CSV （专为调优设计） |

---

## 🎬 工作原理

当你输入一个调研主题时，系统会按以下步骤全自动执行：

```text
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "调研一下近期关于可控核聚变的最新商业进展"                                │
│         │                                                       │
│         ▼                                                       │
│   ┌──────────┐                                                  │
│   │ 🗣️ 澄清  │  (可选) 交互式追问，替你把宽泛的概念缩小边界               │
│   └────┬─────┘                                                  │
│        ▼                                                        │
│   ┌──────────┐                                                  │
│   │ 📋 简报  │  自动分析并制定具体的研究路径和核心考核指标                │
│   └────┬─────┘                                                  │
│        ▼                                                        │
│   ┌──────────┐     ┌──────────┐   ┌──────────┐                  │
│   │ 🧠 调度  │────▶│ 🌐 发散  │──▶│ ✂️ 压缩  │ (并行派发 N 组调查员)   │
│   │Supervisor│◀────│ 搜索/RAG │◀──│ 提纲精炼 │                     │
│   └────┬─────┘     └──────────┘   └──────────┘                  │
│        ▼                                                        │
│   ┌──────────┐                                                  │
│   │ 📝 成文  │  (只有满足完备度时) 根据所有过滤出的纯净材料撰写研报       │
│   └────┬─────┘                                                  │
│        ▼                                                        │
│   ┌──────────┐                                                  │
│   │ ✅ 完成！ │  Markdown 报告 + 耗时/Token 数据统计 + 持久化存档      │
│   └──────────┘                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> 💡 整个研究流程通常在 **3-10 分钟** 内完成（取决于你想让它往下挖几层）。

---

## 🏗️ 系统架构

系统设计最大的亮点在于 **三层分级模型调度**（Tiered Models）与 **多图嵌套流转**（Nested Graphs）。

```text
                          ┌─────────────────────────┐
                          │     🎯 Supervisor        │
                          │   （全局调度 - Hard模型）   │
                          └────┬───────────┬─────────┘
                               │           │
                    ┌──────────▼──┐   ┌────▼──────────┐
                    │    LATS     │   │     RAG       │
                    │   树搜索子图   │   │  私域知识子图    │
                    └──────┬──────┘   └────┬──────────┘
                           │               │
                           ▼               ▼
                 (节点自动展开/反思/剪枝)    (本地文档混合召回)
                 (多用 Simple 模型省钱)     (保留 Hard 模型判断)
```

**三层分级防挥霍机制（Cost Optimization Tiering）**

| 层级 | 默认配置 | 职责范围 |
|------|---------|------|
| **🟢 Simple** | `gpt-4.1-mini` | 意图澄清、摘要精简、节点扩展拆分、机翻 (脏活累活) |
| **🟡 Medium** | `gpt-5.4` | 最终万字报告整理与排版书写 (结构化与文笔担当) |
| **🔴 Hard**   | `gpt-5.4` | Supervisor策略路由、树节点打分评估 (智商担当) |

---

## 🚀 快速开始

### 🛠️ 环境要求
| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 核心运行环境 |
| Conda | *推荐* | 虚拟环境隔离管理 |

### 1. 克隆并安装
```bash
conda activate base
pip install -e ".[dev]"
```

### 2. 配置环境变量
在项目根目录创建一个 `.env` 文件。
```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # 取决于你使用的代理服务 (如 OpenRouter)
TAVILY_API_KEY=tvly-...
```

### 3. 一键开车 (TUI 体验)
```bash
python src/tui.py "请调研诺贝尔物理学奖得主的具体实际应用前景"
```

---

## ⚙️ 配置说明

所有核心控制项通过环境变量全局穿透。

### 必填 API 密钥
| 变量 | 说明 | 获取方式 |
|------|------|---------|
| `OPENAI_API_KEY` | 主要大语言模型调用鉴权 | [OpenAI](https://platform.openai.com/) |
| `OPENAI_BASE_URL` | (可选) API 代理地址，针对第三方接入 (例如 OpenRouter) | [OpenRouter](https://openrouter.ai/) |
| `TAVILY_API_KEY` | 负责核心网络抓取和事实追踪 | [Tavily](https://tavily.com/) |

### 大模型分级覆写 (可选，动态替换)
| 变量 | 说明 |
|------|------|
| `SIMPLE_MODEL` | 用什么模型干累活（默认 `openai:gpt-4.1-mini`） |
| `MEDIUM_MODEL` | 用什么模型写报告（默认 `openai:gpt-5.4`） |
| `HARD_MODEL` | 用什么模型当总管（默认 `openai:gpt-5.4`） |

---

## 📁 项目结构

```text
AutoResearcher/
│
├── ⚙️ src/configuration.py       # 三层模型全局配置中心
├── 🚪 src/tui_advanced.py        # 全新 Textual 终端双屏 UI
├── 🤖 src/graph.py               # 主图拓扑 (Supervisor - Report)
├── 🌲 src/lats_subgraph.py       # LATS 树搜索核心逻辑
├── 📚 src/rag_subgraph.py        # 本地 RAG 查询分析与召回
├── 📐 src/state.py               # 图状态与 Pydantic 约束
├── 📋 src/prompts.py             # Prompt 神仙打架管理
├── 🖥️ src/runner.py              # 事件流分发与检查点(DB)基座
│
├── 📊 scripts/batch_runner.py    # 无人值守自动化测试编排引擎
├── 🧪 tests/                     # 模块端到端集成测试用例
└── 📦 logs/                      # 生成的报告产物及 LangGraph 回放快照
```

---

## 🎯 功能特性

*   ✅ **复杂目标自动瓦解** — 能够将大到发空的宏观概念，自动拆解为几十个并存的微型调查路径。
*   ✅ **LATS 自我反省** — 搜索结果如果不尽人意，立刻自我批判并回溯节点重搜，告别“睁眼说瞎话”。
*   ✅ **高倍微操透明度** — `events.jsonl` 不放过任意一次 LLM 的 API 调用细节，绝不搞闭门造车。
*   ✅ **端到端批处理流** — 睡前丢一个包含几百个课题的 YAML，醒来后直接看跑分表格对比。

---

## 🛠️ 技术栈
<table>
<tr>
<td align="center"><b>类别</b></td>
<td align="center"><b>技术</b></td>
<td align="center"><b>用途</b></td>
</tr>
<tr>
<td>🧠 Agent核心编排</td>
<td>LangGraph</td>
<td>多 Agent 循环路由、状态树持久化、打断与恢复</td>
</tr>
<tr>
<td>🌐 信息获取源</td>
<td>Tavily Search API</td>
<td>高可用性、针对 LLM 优化的网络内容提取</td>
</tr>
<tr>
<td>📱 Terminal UI</td>
<td>Textual</td>
<td>极客质感的终端双屏流式渲染框架</td>
</tr>
<tr>
<td>🧪 测试管理</td>
<td>PyTest + SQLite</td>
<td>状态与错误边界快照留存追踪</td>
</tr>
</table>

---

## 🤝 贡献指南
欢迎向 AutoResearcher 提交你的 Pull Request！

```bash
# 执行完整主图黑盒测试，确保你的 PR 不会破坏三层模型底层拓扑
python tests/test_full_pipeline.py
```

---

[MIT License](LICENSE) © 2024-2026

---

<div align="center">

**Built with ❤️ and 🤖**

*If you find this project useful, please consider giving it a ⭐!*

</div>
