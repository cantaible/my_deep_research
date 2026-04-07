# 工作区协作规则

## 项目目标

本项目的目标是**完全复现 langchain-ai/open_deep_research**，一个基于 LangGraph 的自动化深度研究 Agent。

复现分 4 个阶段，每阶段独立可测试：
1. Phase 1: 基础设施（state/config/prompts/utils）
2. Phase 2: Researcher 子图（ReAct 循环 + 压缩）
3. Phase 3: Supervisor 子图 + 报告生成
4. Phase 4: 完整主图 + 端到端验证

## 开始工作前必读

1. **进度追踪**: `docs/PROGRESS.md` — 查看当前阶段和待办项
2. **复现指南**: `docs/REPRODUCTION_GUIDE.md` — 每个阶段的详细规格和陷阱
3. **原版参考**: `reference/` 目录 — 原版 5 个核心源文件，用于 diff 对比

## 环境管理

- 使用 Conda 管理 Python 环境，默认使用 `base` 环境。
- 不要创建 `.venv`、`venv` 或其他新的虚拟环境。
- 安装依赖到当前激活的 Conda `base` 环境中。

## 密钥与配置

- 优先使用工作区根目录下的 `.env`
- 非经用户明确要求，不要额外创建项目级 `.env`。

## 协作方式

- 每完成一个子任务，更新 `docs/PROGRESS.md` 中对应的 checkbox。
- 写代码时，始终对照 `reference/` 目录中的原版文件，确保关键逻辑一致。
- Prompt 必须原文复制，不做任何修改。
- Configuration 默认值必须与原版一致。
