# Phase 2: 数据流和保存位置说明

## 问题：修改 RAG 检索代码后，结果会保存在哪里？

---

## 当前数据流（Phase 1）

```
RAG 子图运行
    │
    ├─→ events.jsonl          # 事件流日志
    ├─→ compressed.md         # 压缩后的研究摘要
    ├─→ raw_notes.json        # 原始笔记
    ├─→ raw_results.json      # 原始搜索结果（文本格式）
    ├─→ sub_queries.json      # 子查询列表
    ├─→ run_meta.json         # 运行元数据
    └─→ checkpoints.db        # LangGraph 检查点
```

**保存位置**：`logs/eval-{topic[:20]}-{timestamp}/`

**问题**：
- `raw_results.json` 只包含**格式化的文本输出**
- **没有保存中间结果**（dense/sparse/merged/reranked 的 article_ids 和 scores）

---

## Phase 2 修改后的数据流

### 方案 A：扩展 State，保存到 JSON（推荐）

#### 1. 修改 `rag_search.py`

```python
@tool
def rag_search(query: str, ..., return_details: bool = False) -> str | dict:
    # ... 现有逻辑 ...
    
    # 记录各阶段数据
    retrieval_details = {
        "dense": {
            "article_ids": [123, 456, 789, ...],
            "scores": [0.95, 0.87, 0.76, ...],
        },
        "sparse": {
            "article_ids": [456, 999, 234, ...],
            "scores": [12.5, 8.3, 6.1, ...],
            "backend": "opensearch",
        },
        "merged": {
            "article_ids": [123, 456, 789, 999, 234, ...],
            "sources": {
                123: ["dense"],
                456: ["dense", "sparse"],
                789: ["dense"],
                999: ["sparse"],
                234: ["sparse"],
            }
        },
        "reranked": {
            "article_ids": [456, 123, 999, ...],
            "scores": [0.92, 0.88, 0.75, ...],
        },
    }
    
    if return_details:
        return {
            "formatted_output": "\n".join(output),  # 原有的文本输出
            "retrieval_details": retrieval_details,  # 新增的结构化数据
        }
    else:
        return "\n".join(output)  # 保持向后兼容
```

#### 2. 修改 `rag_subgraph.py` 的 `execute` 节点

```python
async def execute(state: RAGExecuteState, config) -> dict:
    # ... 现有逻辑 ...
    
    # 调用 rag_search 时传入 return_details=True
    result = await _run_single_rag_query_with_details({
        **sub_query,
        "query": current_query,
        "return_details": True,  # 新增参数
    })
    
    # result 现在是一个 dict，包含 formatted_output 和 retrieval_details
    formatted_output = result["formatted_output"]
    retrieval_details = result["retrieval_details"]
    
    # 保存到 state
    return {
        "raw_results": [f"--- 查询: {sub_query['query']} ---\n{formatted_output}"],
        "raw_notes": [f"[RAG] {sub_query['query']}"],
        "retrieval_details": [retrieval_details],  # 新增字段
    }
```

#### 3. 扩展 `RAGResearcherState`

```python
# 在 src/state.py 中
class RAGResearcherState(TypedDict):
    research_topic: str
    sub_queries: list[dict]
    raw_results: Annotated[list[str], operator.add]
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer]
    
    # 新增字段：保存每个子查询的检索详情
    retrieval_details: Annotated[list[dict], operator.add]
```

#### 4. 修改 `eval_rag.py`，保存检索详情

```python
async def run_rag_subgraph(topic: str) -> tuple[Path, dict]:
    # ... 现有逻辑 ...
    
    state = await subgraph.aget_state(config)
    result = state.values
    
    # 保存产出
    compressed = result.get("compressed_research", "")
    raw_notes = result.get("raw_notes", [])
    raw_results = result.get("raw_results", [])
    retrieval_details = result.get("retrieval_details", [])  # 新增
    
    # ... 保存现有文件 ...
    
    # 保存检索详情（新增）
    with open(run_dir / "retrieval_details.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_details, f, ensure_ascii=False, indent=2)
    
    return run_dir, result
```

#### 5. 最终保存位置

```
logs/eval-{topic[:20]}-{timestamp}/
├── events.jsonl                # 事件流日志
├── compressed.md               # 压缩后的研究摘要
├── raw_notes.json              # 原始笔记
├── raw_results.json            # 原始搜索结果（文本格式）
├── sub_queries.json            # 子查询列表
├── run_meta.json               # 运行元数据
├── checkpoints.db              # LangGraph 检查点
└── retrieval_details.json      # ✨ 新增：检索详情（结构化数据）
```

**`retrieval_details.json` 内容示例**：

```json
[
  {
    "query": "2026年3月头部厂商发布的大模型",
    "dense": {
      "article_ids": [5045, 5053, 5056, 4362, 5085],
      "scores": [0.95, 0.87, 0.76, 0.72, 0.68]
    },
    "sparse": {
      "article_ids": [5045, 4362, 6304, 4574, 5053],
      "scores": [12.5, 8.3, 6.1, 5.2, 4.8],
      "backend": "opensearch"
    },
    "merged": {
      "article_ids": [5045, 5053, 5056, 4362, 5085, 6304, 4574],
      "sources": {
        "5045": ["dense", "sparse"],
        "5053": ["dense", "sparse"],
        "5056": ["dense"],
        "4362": ["dense", "sparse"],
        "5085": ["dense"],
        "6304": ["sparse"],
        "4574": ["sparse"]
      }
    },
    "reranked": {
      "article_ids": [5045, 4362, 5053, 5056, 5085],
      "scores": [0.92, 0.88, 0.85, 0.75, 0.68]
    }
  },
  {
    "query": "2026年3月国产大模型发布情况",
    "dense": {...},
    "sparse": {...},
    "merged": {...},
    "reranked": {...}
  }
]
```

---

### 方案 B：写入独立日志文件（备选）

如果不想修改 State，可以在 `rag_search.py` 中直接写日志：

```python
import os
import json
from pathlib import Path

@tool
def rag_search(query: str, ...) -> str:
    # ... 现有逻辑 ...
    
    # 如果设置了环境变量，记录详情
    if os.getenv("RAG_LOG_DETAILS"):
        log_file = Path("logs/rag_retrieval_details.jsonl")
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.time(),
                "query": query,
                "dense": {...},
                "sparse": {...},
                "merged": {...},
                "reranked": {...},
            }, f, ensure_ascii=False)
            f.write("\n")
    
    return "\n".join(output)
```

**保存位置**：`logs/rag_retrieval_details.jsonl`（全局日志）

**优点**：
- 不需要修改 State
- 不需要修改 `rag_subgraph.py`

**缺点**：
- 所有运行的日志混在一起，不好区分
- 需要通过 timestamp 关联到具体的运行

---

## 推荐方案对比

| 方案 | 保存位置 | 优点 | 缺点 |
|------|---------|------|------|
| **方案 A**<br>扩展 State | `logs/{run_dir}/retrieval_details.json` | ✅ 数据与运行绑定<br>✅ 结构清晰<br>✅ 易于评测 | ⚠️ 需要修改 State<br>⚠️ 需要修改多个文件 |
| **方案 B**<br>独立日志 | `logs/rag_retrieval_details.jsonl` | ✅ 实现简单<br>✅ 不修改 State | ❌ 日志混在一起<br>❌ 需要手动关联 |

**推荐**：**方案 A**，因为：
1. 数据与运行绑定，便于追溯
2. 结构化存储，便于评测脚本读取
3. 符合现有的数据组织方式

---

## Phase 2 评测脚本如何读取数据

### 新增 `eval/eval_rag_retrieval.py`

```python
import json
from pathlib import Path

def load_retrieval_details(run_dir: Path) -> list[dict]:
    """从运行目录加载检索详情"""
    details_file = run_dir / "retrieval_details.json"
    if not details_file.exists():
        raise FileNotFoundError(f"找不到 {details_file}")
    
    return json.loads(details_file.read_text(encoding="utf-8"))

async def main():
    # 1. 运行 RAG 子图
    run_dir, result = await run_rag_subgraph(topic)
    
    # 2. 加载检索详情
    retrieval_details = load_retrieval_details(run_dir)
    
    # 3. 加载 ground truth
    labels_v2 = json.load(open("eval/article_labels_v2.json"))
    event_index = json.load(open("eval/event_to_articles.json"))
    
    # 4. 计算各阶段指标
    for query_details in retrieval_details:
        dense_recall = compute_recall(
            query_details["dense"]["article_ids"],
            event_index
        )
        sparse_recall = compute_recall(
            query_details["sparse"]["article_ids"],
            event_index
        )
        # ... 更多指标
    
    # 5. 生成报告
    generate_report(metrics, run_dir)
```

---

## 数据流总结

```
用户运行评测
    │
    ▼
python eval/eval_rag.py
    │
    ├─→ 运行 RAG 子图
    │   ├─→ plan: 拆分子查询
    │   ├─→ execute: 并行执行（调用 rag_search with return_details=True）
    │   │   └─→ 每个子查询返回 {formatted_output, retrieval_details}
    │   └─→ compress: 合并结果
    │
    ├─→ 保存到 logs/{run_dir}/
    │   ├─→ compressed.md
    │   ├─→ raw_results.json
    │   ├─→ retrieval_details.json  ← ✨ 新增
    │   └─→ ...
    │
    └─→ 计算评测指标
        ├─→ 从 retrieval_details.json 读取各阶段数据
        ├─→ 对照 ground truth 计算 Recall/Precision/NDCG
        └─→ 生成 RETRIEVAL_REPORT.md
```

---

## 总结

**修改后的结果保存在**：

```
logs/eval-{topic[:20]}-{timestamp}/retrieval_details.json
```

**包含内容**：
- 每个子查询的 4 个阶段数据（dense/sparse/merged/reranked）
- 每个阶段的 article_ids 和 scores
- 双路召回的来源标记（sources）

**评测脚本读取方式**：
```python
retrieval_details = json.load(open(run_dir / "retrieval_details.json"))
```

**优点**：
- 数据与运行绑定，便于追溯
- 结构化存储，便于评测
- 不影响现有功能（向后兼容）
