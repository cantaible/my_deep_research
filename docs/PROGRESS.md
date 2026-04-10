# 项目进度

## 当前状态
- **阶段**: 优化阶段 — Step 2（Supervisor 接入）
- **下一步**: graph.py supervisor bind_tools 增加 ConductRAGResearch

## 复现阶段 ✅

### Phase 1: 基础设施 ✅
- [x] pyproject.toml 补充依赖
- [x] state.py 扩充
- [x] configuration.py 扩充
- [x] prompts.py 替换
- [x] 新增 utils.py
- [x] 验证通过

### Phase 2: Researcher 子图 ✅
- [x] graph.py (researcher + researcher_tools + compress_research)
- [x] 搜索质量优化（英文查询 + advanced 深度）
- [x] AsyncSqliteSaver 持久化 checkpointer
- [x] 验证通过

### Phase 3: Supervisor + 报告 ✅
- [x] graph.py (supervisor + write_research_brief + final_report_generation)
- [x] 验证通过

### Phase 4: 完整主图 + TUI ✅
- [x] clarify_with_user + 主图组装
- [x] runner.py + tui.py
- [x] 端到端验证通过

## 优化阶段

### Step 1: RAG 子图（Plan-and-Execute） ✅
- [x] rag_subgraph.py（plan + execute + compress）
- [x] RAGResearcherState 定义
- [x] 精确日期范围过滤（start_date / end_date 替代 days）
- [x] category 枚举约束（AI / GAMES）
- [x] rag_search + bm25_search 支持日期范围
- [x] 独立测试通过（test_rag_subgraph.py）

### Step 2: Supervisor 接入
- [x] state.py 新增 ConductRAGResearch
- [ ] graph.py supervisor bind_tools 增加 RAG
- [ ] prompts.py Supervisor 提示词增加 RAG 说明
- [ ] TUI 验证通过

### Step 3: 结构化去重
- [ ] dedup.py（ResearchFinding 抽取 + 精确去重）
- [ ] 主图拓扑加入 deduplicate_findings 节点
- [ ] 端到端验证通过

### 辅助工具
- [x] scripts/export_rag_run.py（导出单次运行的 RAG 查询与命中结果）
- [x] scripts/export_researcher_run.py（导出单次运行的 researcher 子图轨迹）
- [x] scripts/export_rag_subgraph_analysis.py（导出 RAG 子图 query/反馈/迭代详细报告）
- [x] OpenSearch 词法检索后端（替换手写 BM25）
- [x] docker-compose.opensearch.yml + build_opensearch_index.py
- [x] docs/ANTIGRAVITY_PROXY_FIX.md（Clash Verge / Antigravity 代理稳定性排障记录）
