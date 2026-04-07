# 潜在优化待办事项 (TODO)

由于当前项目核心架构（Open Deep Research）的复现工作仍处于初期阶段，部分架构层面的优化想法暂不便立即推翻并实施。特在此建立 TODO 列表，待整个 4 个复现阶段端到端跑通后再统一着手迭代。

## 架构与逻辑优化

- [ ] **优化大模型上下文边界管理**
  - **现状**：目前在处理 Token 超限（遇到大量网页文本撑死模型）时，系统采用了一种极其“粗暴的截断”做法——要么利用 `remove_up_to_last_ai_message` 直接抛弃最新一轮对话，要么在写最终报告时对资料按硬性比例截短。这很容易导致刚查出的关键资料丢失。
  - **优化思路**：后续应当研究更加平滑和智能的记忆机制。例如：引入 RAG 切块检索在压缩节点前先行清洗资料，或者通过更聪明的滑动窗口配合专门的小模型做实时摘要。


- [ ] **代码结构重构：剥离 Researcher 子图**
  - **现状**：目前原版设计将主图 (`AgentState`)、Supervisor 子图以及 Researcher 子图全塞在一个臃肿的 `deep_researcher.py` 文件中，后期维护和阅读非常痛苦。
  - **优化思路**：在基础流程复现跑通后，将打工节点模块（`ResearcherState`、`researcher_tools`、压缩逻辑及路由）单独解耦抽离到一个独立的 python 文件（如 `researcher_subgraph.py`）中。


- [ ] **增强搜索质量：强制 Tavily 使用英文 Query**
  - **现状**：目前用户即使输入中文进行研究，Researcher 大模型在生成 `tavily_search` 的 `queries` 参数时也极大概率会直接使用中文搜索。而在外网搜索质量上，英文 query 的检索召回率和资料纯度通常远高于中文。
  - **优化思路**：在工具的参数描述（`tool definition`）或者 Researcher 节点的 `system_prompt` 中，通过强行约束（如：“Always translate the search intent to English before passing it to the queries argument”），来硬性保证对底层搜索引擎的数据投喂语言。
