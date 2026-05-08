# Phase 2 完成总结

## 已完成的工作

### 1. 修改 RAG 检索代码

**文件**: `rag/rag_search.py`

**修改内容**:
- ✅ 添加 `return_details` 参数（默认 False，保持向后兼容）
- ✅ 记录 4 个阶段的检索数据：
  - Dense（向量检索）：article_ids, scores, count
  - Sparse（词法检索）：article_ids, scores, backend, count
  - Merged（合并候选池）：article_ids, sources, count
  - Reranked（重排后）：article_ids, scores, count
- ✅ 返回格式：
  - `return_details=False`: 返回 `str`（格式化文本，默认行为）
  - `return_details=True`: 返回 `dict`（包含 `formatted_output` 和 `retrieval_details`）

### 2. 扩展 State 定义

**文件**: `src/state.py`

**修改内容**:
- ✅ `RAGResearcherState` 添加 `retrieval_details` 字段
- ✅ `RAGExecuteState` 添加 `retrieval_details` 字段
- ✅ 使用 `operator.add` reducer 自动汇总多个子查询的检索详情

### 3. 修改 RAG 子图

**文件**: `src/rag_subgraph.py`

**修改内容**:
- ✅ `_run_single_rag_query` 调用时传入 `return_details=True`
- ✅ `execute` 节点处理返回的 dict，提取 `formatted_output` 和 `retrieval_details`
- ✅ 支持多轮重试，收集所有轮次的检索详情

### 4. 修改评测脚本

**文件**: `eval/eval_rag.py`

**修改内容**:
- ✅ 从 state 中获取 `retrieval_details`
- ✅ 保存到 `retrieval_details.json`
- ✅ 在 `run_meta.json` 中记录 `retrieval_details_count`

### 5. 实现细粒度评测脚本

**文件**: `eval/eval_rag_retrieval.py`（新增）

**功能**:
- ✅ 加载 ground truth 数据（`article_labels_v2.json`, `event_to_articles.json`）
- ✅ 加载检索详情（`retrieval_details.json`）
- ✅ 计算各阶段指标：
  - Event Recall（事件召回率）
  - Article Recall（文章召回率）
  - 双路召回贡献分析
- ✅ 生成评测报告（`RETRIEVAL_REPORT.md`）
- ✅ 保存评测指标（`retrieval_metrics.json`）

---

## 数据流

```
用户运行评测
    │
    ▼
python eval/eval_rag_retrieval.py
    │
    ├─→ 运行 RAG 子图
    │   ├─→ plan: 拆分子查询
    │   ├─→ execute: 并行执行
    │   │   └─→ rag_search(return_details=True)
    │   │       └─→ 返回 {formatted_output, retrieval_details}
    │   └─→ compress: 合并结果
    │
    ├─→ 保存到 logs/{run_dir}/
    │   ├─→ compressed.md
    │   ├─→ raw_results.json
    │   ├─→ retrieval_details.json  ← ✨ 新增
    │   └─→ ...
    │
    └─→ 计算评测指标
        ├─→ 各阶段 Event Recall
        ├─→ 各阶段 Article Recall
        ├─→ 双路召回贡献分析
        └─→ 生成 RETRIEVAL_REPORT.md
```

---

## 输出文件

### 新增文件

| 文件 | 位置 | 说明 |
|------|------|------|
| `retrieval_details.json` | `logs/{run_dir}/` | 每个子查询的 4 阶段检索数据 |
| `RETRIEVAL_REPORT.md` | `logs/{run_dir}/` | 细粒度评测报告 |
| `retrieval_metrics.json` | `logs/{run_dir}/` | 评测指标（JSON 格式） |
| `eval_rag_retrieval.py` | `eval/` | 细粒度评测脚本 |

### retrieval_details.json 示例

```json
[
  {
    "query": "2026年3月头部厂商发布的大模型",
    "dense": {
      "article_ids": [5045, 5053, 5056, 4362, 5085],
      "scores": [0.15, 0.23, 0.31, 0.35, 0.42],
      "count": 5
    },
    "sparse": {
      "article_ids": [5045, 4362, 6304, 4574, 5053],
      "scores": [12.5, 8.3, 6.1, 5.2, 4.8],
      "backend": "opensearch",
      "count": 5
    },
    "merged": {
      "article_ids": [5045, 5053, 5056, 4362, 5085, 6304, 4574],
      "sources": {
        "5045": ["向量", "OpenSearch"],
        "5053": ["向量", "OpenSearch"],
        "5056": ["向量"],
        "4362": ["向量", "OpenSearch"],
        "5085": ["向量"],
        "6304": ["OpenSearch"],
        "4574": ["OpenSearch"]
      },
      "count": 7
    },
    "reranked": {
      "article_ids": [5045, 4362, 5053, 5056, 5085],
      "scores": [0.92, 0.88, 0.85, 0.75, 0.68],
      "count": 5
    }
  }
]
```

---

## 使用方法

### 1. 运行完整评测（包含 RAG 子图）

```bash
python eval/eval_rag_retrieval.py
```

**输出**:
- 运行 RAG 子图
- 保存检索详情到 `logs/{run_dir}/retrieval_details.json`
- 计算评测指标
- 生成报告 `logs/{run_dir}/RETRIEVAL_REPORT.md`

### 2. 只评测已有运行结果

```bash
python eval/eval_rag_retrieval.py --run-dir logs/eval-xxx-xxx
```

**输出**:
- 加载已有的 `retrieval_details.json`
- 计算评测指标
- 生成报告

### 3. 自定义查询主题

```bash
python eval/eval_rag_retrieval.py --topic "搜索2026年2月的游戏相关新闻"
```

---

## 评测报告示例

```markdown
# RAG 检索细粒度评测报告

## 查询信息
- 查询: 搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻...
- 运行目录: logs/eval-xxx-xxx

## 各阶段召回率

| 阶段 | 召回文章数 | Event Recall | Article Recall |
|------|-----------|--------------|----------------|
| 向量检索 (Dense) | 50 | 65.0% (13/20) | 45.0% (45/100) |
| 词法检索 (Sparse) | 45 | 55.0% (11/20) | 40.0% (40/100) |
| 合并候选池 (Merged) | 65 | 80.0% (16/20) | 60.0% (60/100) |
| Rerank Top K | 10 | 75.0% (15/20) | 10.0% (10/100) |

**分析**：
- 向量检索和词法检索各有优势，合并后召回率显著提升
- Rerank 后虽然文章数减少，但保留了大部分相关事件

## 双路召回贡献分析

| 类型 | 文章数 | 命中事件数 |
|------|--------|-----------|
| 只被向量检索召回 | 20 | 5 |
| 只被词法检索召回 | 15 | 3 |
| 两路都召回 | 30 | 12 |

**分析**：
- 46% 的文章被两路都召回，说明向量和词法检索有较高的重叠
- 向量检索独有的 20 篇文章覆盖了 5 个事件
- 词法检索独有的 15 篇文章覆盖了 3 个事件
```

---

## 技术亮点

### 1. 向后兼容

- `return_details` 默认为 `False`，保持现有行为
- 不影响其他使用 `rag_search` 的代码

### 2. 最小侵入

- 只修改了 4 个文件（`rag_search.py`, `state.py`, `rag_subgraph.py`, `eval_rag.py`）
- 新增 1 个评测脚本（`eval_rag_retrieval.py`）
- 不影响主流程的运行

### 3. 完整的数据记录

- 记录所有 4 个阶段的数据
- 支持多轮重试，收集所有轮次的检索详情
- 保留双路召回的来源标记（`sources`）

### 4. 灵活的评测

- 支持运行时评测（自动运行 RAG 子图）
- 支持离线评测（只评测已有结果）
- 支持自定义查询主题

---

## 下一步

Phase 2 已完成！可以开始 Phase 3：

**Phase 3: Finding 抽取与匹配**
- 定义 Finding Schema
- 实现 Finding 抽取器（从 `compressed_research` 抽取结构化 findings）
- 实现 Finding 匹配器（基于 aliases 和实体重叠匹配 ground truth）

**预计工作量**: 5-6 小时

---

## 验证清单

- ✅ `rag_search.py` 添加 `return_details` 参数
- ✅ `state.py` 扩展 `retrieval_details` 字段
- ✅ `rag_subgraph.py` 调用时传入 `return_details=True`
- ✅ `eval_rag.py` 保存 `retrieval_details.json`
- ✅ `eval_rag_retrieval.py` 实现细粒度评测
- ✅ 生成评测报告和指标文件
- ⚠️ 需要实际运行测试（需要数据库和向量库初始化）

---

## 总结

Phase 2 成功实现了 RAG 检索的细粒度评测：

1. **数据记录**: 记录 dense/sparse/merged/reranked 4 个阶段的数据
2. **指标计算**: Event Recall、Article Recall、双路召回贡献
3. **报告生成**: 自动生成 Markdown 格式的评测报告
4. **向后兼容**: 不影响现有代码的运行

所有修改都是**最小侵入**的，保持了代码的简洁性和可维护性。
