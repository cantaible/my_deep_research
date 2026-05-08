# Phase 2: Researcher 检索评测增强

## 目标

在现有 `eval_rag.py` 基础上，增加**细粒度的检索评测**，分析 RAG 混合检索的各个阶段表现。

---

## 背景

当前 RAG 检索流程：

```
用户查询
    │
    ├─→ 向量检索 (ChromaDB)  ──┐
    │                          │
    └─→ 词法检索 (BM25/OpenSearch) ──┤
                                  │
                                  ▼
                            合并候选池
                                  │
                                  ▼
                            Rerank (本地模型)
                                  │
                                  ▼
                            最终结果 (Top K)
```

**问题**：
- 现有 `eval_rag.py` 只评测**最终结果**（rerank 后）
- 不知道各阶段的召回贡献：
  - 向量检索召回了多少相关文章？
  - 词法检索召回了多少相关文章？
  - 合并后候选池的质量如何？
  - Rerank 提升了多少排序质量？

---

## Phase 2 要做的事情

### 任务 1: 修改 RAG 检索代码，记录中间结果

**文件**：`rag/rag_search.py`

**修改点**：在 `rag_search()` 函数中，记录各阶段的中间结果

**需要记录的数据**：

```python
# 1. 向量检索阶段
dense_candidates = {
    "stage": "dense",
    "article_ids": [123, 456, 789, ...],  # 向量检索召回的文章 ID
    "scores": [0.95, 0.87, 0.76, ...],    # 向量相似度分数
}

# 2. 词法检索阶段
sparse_candidates = {
    "stage": "sparse",
    "article_ids": [456, 999, 234, ...],  # 词法检索召回的文章 ID
    "scores": [12.5, 8.3, 6.1, ...],      # BM25 分数
    "backend": "opensearch",              # 使用的后端 (bm25/opensearch)
}

# 3. 合并候选池阶段
merged_candidates = {
    "stage": "merged",
    "article_ids": [123, 456, 789, 999, 234, ...],  # 去重后的候选池
    "sources": {
        123: ["dense"],                   # 只来自向量检索
        456: ["dense", "sparse"],         # 两路都召回
        789: ["dense"],
        999: ["sparse"],                  # 只来自词法检索
        234: ["sparse"],
    }
}

# 4. Rerank 阶段
reranked_results = {
    "stage": "reranked",
    "article_ids": [456, 123, 999, ...],  # Rerank 后的排序
    "scores": [0.92, 0.88, 0.75, ...],    # Rerank 分数
}
```

**实现方式**：

**选项 A（推荐）**：返回结构化数据，不改变现有接口

```python
# 在 rag_search() 函数末尾，添加一个可选的 return_details 参数
@tool
def rag_search(query: str, ..., return_details: bool = False) -> str | dict:
    # ... 现有逻辑 ...
    
    if return_details:
        # 返回详细数据（用于评测）
        return {
            "formatted_output": "\n".join(output),  # 原有的格式化输出
            "retrieval_details": {
                "dense": {...},
                "sparse": {...},
                "merged": {...},
                "reranked": {...},
            }
        }
    else:
        # 返回格式化字符串（现有行为）
        return "\n".join(output)
```

**选项 B**：写入日志文件

```python
# 在 rag_search() 函数中，将中间结果写入 JSON 文件
import json
from pathlib import Path

def rag_search(query: str, ...) -> str:
    # ... 现有逻辑 ...
    
    # 记录中间结果（可选，通过环境变量控制）
    if os.getenv("RAG_LOG_DETAILS"):
        log_file = Path("logs/rag_retrieval_details.jsonl")
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, "a") as f:
            json.dump({
                "query": query,
                "timestamp": time.time(),
                "dense": {...},
                "sparse": {...},
                "merged": {...},
                "reranked": {...},
            }, f)
            f.write("\n")
    
    return "\n".join(output)
```

**推荐**：选项 A，因为更灵活，不依赖文件 I/O。

---

### 任务 2: 实现细粒度检索评测脚本

**文件**：`eval/eval_rag_retrieval.py`（新增）

**功能**：

1. **运行 RAG 子图**（复用 `eval_rag.py` 的逻辑）
2. **提取各阶段的检索结果**（从 `return_details` 获取）
3. **计算各阶段的指标**：

#### 指标 1: 各阶段的 Event Recall

```python
# 对于每个阶段，计算命中了多少事件
def compute_stage_recall(stage_article_ids: list[int], 
                         event_index: dict,
                         total_events: int) -> float:
    """计算某阶段的事件召回率"""
    hit_events = set()
    for article_id in stage_article_ids:
        if str(article_id) in event_index:
            hit_events.add(event_index[str(article_id)])
    
    return len(hit_events) / total_events

# 示例输出
{
    "dense_recall": 0.65,      # 向量检索召回了 65% 的事件
    "sparse_recall": 0.55,     # 词法检索召回了 55% 的事件
    "merged_recall": 0.80,     # 合并后召回了 80% 的事件
    "reranked_recall": 0.75,   # Rerank 后（Top K）召回了 75% 的事件
}
```

#### 指标 2: 双路召回的贡献分析

```python
# 分析向量和词法检索的贡献
def analyze_dual_recall(merged_candidates: dict, event_index: dict) -> dict:
    """分析双路召回的贡献"""
    dense_only = []   # 只被向量检索召回
    sparse_only = []  # 只被词法检索召回
    both = []         # 两路都召回
    
    for article_id, sources in merged_candidates["sources"].items():
        if len(sources) == 2:
            both.append(article_id)
        elif "dense" in sources:
            dense_only.append(article_id)
        else:
            sparse_only.append(article_id)
    
    # 统计各类文章命中的事件数
    dense_only_events = count_events(dense_only, event_index)
    sparse_only_events = count_events(sparse_only, event_index)
    both_events = count_events(both, event_index)
    
    return {
        "dense_only_count": len(dense_only),
        "sparse_only_count": len(sparse_only),
        "both_count": len(both),
        "dense_only_events": dense_only_events,
        "sparse_only_events": sparse_only_events,
        "both_events": both_events,
    }

# 示例输出
{
    "dense_only_count": 20,        # 20 篇文章只被向量检索召回
    "sparse_only_count": 15,       # 15 篇文章只被词法检索召回
    "both_count": 30,              # 30 篇文章两路都召回
    "dense_only_events": 5,        # 这 20 篇覆盖了 5 个事件
    "sparse_only_events": 3,       # 这 15 篇覆盖了 3 个事件
    "both_events": 12,             # 这 30 篇覆盖了 12 个事件
}
```

#### 指标 3: Rerank 的排序提升

```python
# 计算 Rerank 前后的 NDCG 变化
def compute_rerank_gain(merged_candidates: list[dict],
                        reranked_results: list[dict],
                        labels: dict) -> dict:
    """计算 Rerank 的排序提升"""
    # Rerank 前：按原始分数排序（向量相似度或 BM25 分数）
    # 这里简化处理：假设合并后的候选池是无序的
    ndcg_before = compute_ndcg(merged_candidates, labels, k=10)
    
    # Rerank 后：按 Rerank 分数排序
    ndcg_after = compute_ndcg(reranked_results, labels, k=10)
    
    return {
        "ndcg_before_rerank": ndcg_before,
        "ndcg_after_rerank": ndcg_after,
        "ndcg_gain": ndcg_after - ndcg_before,
    }

# 示例输出
{
    "ndcg_before_rerank": 0.65,
    "ndcg_after_rerank": 0.78,
    "ndcg_gain": 0.13,  # Rerank 提升了 13% 的排序质量
}
```

---

### 任务 3: 生成细粒度评测报告

**输出**：`eval/RETRIEVAL_REPORT.md`

**报告内容**：

```markdown
# RAG 检索细粒度评测报告

## 查询信息
- 查询: "搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻"
- 时间范围: 2026-03-01 至 2026-03-31
- 分类: AI

## 各阶段召回率

| 阶段 | 召回文章数 | Event Recall | Article Recall |
|------|-----------|--------------|----------------|
| 向量检索 (Dense) | 50 | 65% (13/20) | 45% (45/100) |
| 词法检索 (Sparse) | 45 | 55% (11/20) | 40% (40/100) |
| 合并候选池 (Merged) | 65 | 80% (16/20) | 60% (60/100) |
| Rerank Top 10 | 10 | 75% (15/20) | 10% (10/100) |

**分析**：
- 向量检索和词法检索各有优势，合并后召回率显著提升
- Rerank 后虽然文章数减少，但保留了大部分相关事件

## 双路召回贡献分析

| 类型 | 文章数 | 命中事件数 | 占比 |
|------|--------|-----------|------|
| 只被向量检索召回 | 20 | 5 | 31% |
| 只被词法检索召回 | 15 | 3 | 23% |
| 两路都召回 | 30 | 12 | 46% |

**分析**：
- 46% 的文章被两路都召回，说明向量和词法检索有较高的重叠
- 向量检索独有的 20 篇文章覆盖了 5 个事件，说明向量检索能捕获语义相关的内容
- 词法检索独有的 15 篇文章覆盖了 3 个事件，说明词法检索能捕获关键词匹配

## Rerank 排序提升

| 指标 | Rerank 前 | Rerank 后 | 提升 |
|------|----------|----------|------|
| NDCG@10 | 0.65 | 0.78 | +0.13 |
| NDCG@20 | 0.70 | 0.82 | +0.12 |

**分析**：
- Rerank 显著提升了排序质量（+13%）
- 相关文章被排到了更靠前的位置

## 未命中的事件

以下事件未被检索到：
- ❌ Midjourney V8
- ❌ MAI-Image-2
- ❌ Vidu Q3
- ❌ LongCat-Flash-Prover

**可能原因**：
- 查询关键词不匹配（如 "大模型" 不包含图像生成模型）
- 时间范围过滤（这些事件可能在 3 月之外）
- 文章数量少（每个事件只有 1-2 篇报道）
```

---

## 实施步骤

### Step 1: 修改 `rag/rag_search.py`（1-2 小时）

```python
# 在 rag_search() 函数中添加 return_details 参数
@tool
def rag_search(query: str, ..., return_details: bool = False) -> str | dict:
    # ... 现有逻辑 ...
    
    # 记录各阶段数据
    retrieval_details = {
        "dense": {
            "article_ids": [int(meta["article_id"]) for meta in vec["metadatas"][0]],
            "scores": vec["distances"][0] if "distances" in vec else [],
        },
        "sparse": {
            "article_ids": [hit["metadata"]["article_id"] for hit in lexical_hits],
            "scores": [hit.get("score", 0) for hit in lexical_hits],
            "backend": LEXICAL_BACKEND,
        },
        "merged": {
            "article_ids": [int(c["metadata"]["article_id"]) for c in candidates],
            "sources": {int(c["metadata"]["article_id"]): c["sources"] for c in candidates},
        },
        "reranked": {
            "article_ids": [int(item["metadata"]["article_id"]) for item in reranked],
            "scores": [item["rerank_score"] for item in reranked],
        },
    }
    
    if return_details:
        return {
            "formatted_output": "\n".join(output),
            "retrieval_details": retrieval_details,
        }
    else:
        return "\n".join(output)
```

### Step 2: 实现 `eval/eval_rag_retrieval.py`（2-3 小时）

```python
# 主流程
async def main():
    # 1. 运行 RAG 子图（复用 eval_rag.py 的逻辑）
    run_dir, result = await run_rag_subgraph(topic)
    
    # 2. 提取检索详情（需要修改 RAG 子图，传递 return_details=True）
    retrieval_details = extract_retrieval_details(result)
    
    # 3. 加载 ground truth
    labels_v2 = json.load(open("eval/article_labels_v2.json"))
    event_index = json.load(open("eval/event_to_articles.json"))
    
    # 4. 计算各阶段指标
    metrics = {
        "stage_recall": compute_stage_recall(retrieval_details, event_index),
        "dual_recall": analyze_dual_recall(retrieval_details, event_index),
        "rerank_gain": compute_rerank_gain(retrieval_details, labels_v2),
    }
    
    # 5. 生成报告
    generate_report(metrics, run_dir)
```

### Step 3: 测试和验证（1 小时）

```bash
# 运行细粒度评测
python eval/eval_rag_retrieval.py

# 检查输出
cat logs/xxx/RETRIEVAL_REPORT.md
```

---

## 预期产出

1. **修改后的代码**：
   - `rag/rag_search.py`（添加 `return_details` 参数）

2. **新增脚本**：
   - `eval/eval_rag_retrieval.py`（细粒度检索评测）

3. **评测报告**：
   - `eval/RETRIEVAL_REPORT.md`（报告模板）
   - `logs/xxx/RETRIEVAL_REPORT.md`（实际运行结果）

4. **评测指标**：
   - 各阶段 Event Recall
   - 双路召回贡献分析
   - Rerank 排序提升

---

## 可选任务：Pairwise Preference 评测

如果没有完整的排序真值，可以用 LLM 做 pairwise 比较：

```python
def pairwise_preference(query: str, result_a: dict, result_b: dict) -> str:
    """用 LLM 判断两个结果哪个更相关"""
    prompt = f"""
    查询: {query}
    
    结果 A: {result_a["title"]}
    {result_a["preview"][:200]}
    
    结果 B: {result_b["title"]}
    {result_b["preview"][:200]}
    
    问题: 哪个结果与查询更相关？
    回答: A / B / 相同
    """
    response = llm.invoke(prompt)
    return response.strip()

# 计算 pairwise accuracy
correct = 0
total = 0
for i in range(len(results)):
    for j in range(i+1, len(results)):
        preference = pairwise_preference(query, results[i], results[j])
        # 根据 ground truth 判断是否正确
        if is_correct(preference, results[i], results[j], labels):
            correct += 1
        total += 1

pairwise_accuracy = correct / total
```

**成本估算**：
- 10 个结果 → 45 次比较
- 每次比较 ~500 tokens
- 总成本：45 × 500 × $0.003/1K ≈ $0.07（可接受）

---

## 总结

Phase 2 的核心是**拆解 RAG 检索流程**，分析各阶段的召回贡献：

1. **向量检索**：召回语义相关的文章
2. **词法检索**：召回关键词匹配的文章
3. **合并候选池**：去重，提高召回率
4. **Rerank**：提升排序质量

通过细粒度评测，可以：
- 发现瓶颈（哪个阶段召回率低？）
- 优化策略（是否需要调整候选池大小？Rerank 模型是否有效？）
- 对比实验（不同 embedding 模型、不同 rerank 模型的效果）

**预计工作量**：4-5 小时
