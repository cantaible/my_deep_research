# Open Deep Research 完全复现指南

## 总体方案

4阶段自底向上构建，在已有 my_deep_research 项目基础上扩充：

```
Phase 1(扩充基础) → Phase 2(Researcher子图) → Phase 3(Supervisor+报告) → Phase 4(完整主图)
```

## 项目结构（在已有基础上扩充）

```
my_deep_research/
├── pyproject.toml          ← 补充依赖
├── langgraph.json          ← 更新图注册
├── src/
│   ├── state.py            ← Phase 1 扩充
│   ├── configuration.py    ← Phase 1 扩充
│   ├── prompts.py          ← Phase 1 替换
│   ├── utils.py            ← Phase 1 新增
│   ├── graph.py            ← Phase 2-4 重写
│   ├── model_factory.py    ← 保留（不在 deep research 流程中用）
│   ├── debug_trace.py      ← 保留
│   └── cli.py              ← 保留，后期改造
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   └── test_phase4.py
└── docs/
    ├── PROGRESS.md
    └── REPRODUCTION_GUIDE.md
```

## 关键差异：my_deep_research vs 原版

| 项目 | 原版 | 本项目 |
|------|------|--------|
| import 路径 | `from open_deep_research.xxx` | `from xxx` (flat) |
| 模型创建 | `configurable_model` 全局单例 | graph.py 中用原版模式 |
| model_factory.py | 无 | 保留但不在流程中用 |
| debug_trace.py | 无 | 保留，不影响一致性 |

## 一致性原则

必须一致：Prompt原文、State定义、Configuration默认值、Graph拓扑、退出条件顺序
可以偏离：包名/import路径、测试入口、MCP先stub、注释风格

## 原版文件位置

```
/Users/dongzhiming/Documents/Codes/deep-research-agent-reference/open_deep_research/src/open_deep_research/
├── state.py (96行) ├── configuration.py (252行) ├── prompts.py (368行)
├── utils.py (926行) └── deep_researcher.py (719行)
```

---

## Phase 1: 基础设施扩充

### pyproject.toml — 补充依赖
在现有基础上添加（版本对齐原版）：
- langchain-anthropic>=0.3.15
- langchain-mcp-adapters>=0.1.6
- langchain-tavily
- tavily-python>=0.5.0
- mcp>=1.9.4
- aiohttp (MCP token交换)

### state.py — 扩充（当前只有2个空State类）
需要添加：
1. 5个Pydantic Schema: ConductResearch, ResearchComplete, Summary, ClarifyWithUser, ResearchQuestion
2. override_reducer函数
3. 扩充AgentState字段: supervisor_messages, research_brief, raw_notes, notes, final_report
4. 新增: SupervisorState, ResearcherState, ResearcherOutputState

### configuration.py — 扩充（当前只有4个配置项）
需要添加：
- SearchAPI枚举, MCPConfig
- 多模型配置: research_model, summarization_model, compression_model, final_report_model
- 流程控制: max_concurrent_research_units(5), max_researcher_iterations(6), max_react_tool_calls(10) 等

### prompts.py — 完全替换
替换为原版的8个prompt模板（原文复制）

### utils.py — 新增
从原版复制，调整import路径。MCP部分先stub。

### 验证
- 所有import正常
- pytest test_phase1.py 通过
- 与原版各文件diff无实质差异

---

## Phase 2: Researcher 子图

在 graph.py 中实现（替换现有单节点图）：
1. configurable_model = init_chat_model(configurable_fields=("model","max_tokens","api_key"))
2. execute_tool_safely() — try/except包装
3. researcher() — bind_tools, 调用LLM, Command(goto="researcher_tools")
4. researcher_tools() — 先检查无tool_calls→compress, 执行工具, 后检查迭代限制
5. compress_research() — compression_model蒸馏, 重试3次
6. 组装: StateGraph(ResearcherState, output=ResearcherOutputState)

关键陷阱：
- output=ResearcherOutputState 不可遗漏（状态隔离）
- researcher_tools退出条件顺序：先无tool_calls→compress，执行后再检查迭代限制

独立测试：researcher_subgraph.ainvoke({...}) 直接调用

---

## Phase 3: Supervisor + 报告生成

在 graph.py 中新增：
1. supervisor() — bind_tools([ConductResearch, ResearchComplete, think_tool])
2. supervisor_tools() — 退出条件检查, 处理think_tool, 并行启动researcher_subgraph
3. supervisor_subgraph组装
4. write_research_brief() — structured_output(ResearchQuestion), override重置messages
5. final_report_generation() — 合并notes, token超限重试(先4x截断,后每次减10%)

关键陷阱：
- supervisor_tools异常处理有 `or True`（任何异常都退出）
- write_research_brief用override语义重置supervisor_messages
- researcher_subgraph.ainvoke传入格式必须正确

独立测试：supervisor_subgraph.ainvoke({...}) 直接调用

---

## Phase 4: 完整主图

在 graph.py 中新增：
1. clarify_with_user() — structured_output(ClarifyWithUser), Command路由
2. 主图组装: clarify→(Command)→write_brief→supervisor→report→END
3. 更新langgraph.json指向新图

关键陷阱：
- clarify→write_research_brief之间没有add_edge，靠Command路由

---

## 常见复现失败原因

1. LangGraph版本不对 → 锁定>=0.5.4
2. override_reducer写错 → 消息累积失控
3. Prompt改了措辞 → LLM行为不一致
4. 遗漏output=ResearcherOutputState → 状态泄漏
5. 退出条件顺序错 → 死循环或跳步
6. Command goto遗漏 → 路由失败
7. or True被删除 → 异常崩溃
8. InjectedToolArg遗漏 → LLM看到不该看的参数
9. 异步用法不对 → 并行变串行
10. API key逻辑错误 → 模型调用失败
