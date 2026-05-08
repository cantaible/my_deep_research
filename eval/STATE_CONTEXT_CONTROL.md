# LangGraph State 上下文控制机制详解

## 核心原则

**State 不会自动进入 LLM 上下文，只有显式引用的字段才会进入。**

---

## 控制方式：在节点函数中显式引用

### 方式 1：在 Prompt 中引用（进入上下文）

```python
async def plan(state: RAGResearcherState, config) -> dict:
    model = get_model()
    
    # ✅ 这些字段会进入 LLM 上下文
    result = await model.ainvoke([
        SystemMessage(content=RAG_PLAN_PROMPT),
        HumanMessage(content=f"研究主题：{state['research_topic']}")  # ← 显式引用
    ])
    
    # ❌ 这些字段不会进入 LLM 上下文（只是存储）
    # state['sub_queries']  ← 没有在 prompt 中引用
    # state['retrieval_details']  ← 没有在 prompt 中引用
    
    return {"sub_queries": result.sub_queries}
```

**进入上下文的内容**：
```
SystemMessage: "你是一个查询规划助手..."
HumanMessage: "研究主题：搜索本地新闻数据库，查找2026年3月..."
```

**不进入上下文的内容**：
- `state['sub_queries']`
- `state['retrieval_details']`
- `state['raw_results']`
- 其他所有未在 prompt 中引用的字段

---

### 方式 2：不引用（不进入上下文）

```python
async def compress(state: RAGResearcherState, config) -> dict:
    model = get_model()
    
    # 只引用 research_topic 和 raw_results
    all_results = "\n\n".join(state.get("raw_results", []))
    
    result = await model.ainvoke([
        SystemMessage(content=RAG_COMPRESS_PROMPT),
        HumanMessage(content=f"研究主题：{state['research_topic']}\n\n搜索结果：\n\n{all_results}")
    ])
    
    # ❌ retrieval_details 不会进入上下文
    # 因为没有在 prompt 中引用
    
    return {"compressed_research": result.content}
```

**进入上下文的内容**：
```
SystemMessage: "你是一个研究结果整合助手..."
HumanMessage: "研究主题：搜索本地新闻数据库...\n\n搜索结果：\n\n[raw_results 的内容]"
```

**不进入上下文的内容**：
- `state['retrieval_details']`  ← 虽然存在于 state，但没有引用
- `state['sub_queries']`
- `state['raw_notes']`

---

## 实际案例分析

### 案例 1：RAG 子图的 3 个节点

#### plan 节点
```python
async def plan(state: RAGResearcherState, config) -> dict:
    result = await model.ainvoke([
        SystemMessage(content=RAG_PLAN_PROMPT),
        HumanMessage(content=f"请将以下研究主题拆分为子查询：\n\n{state['research_topic']}")
        #                                                              ↑
        #                                                    只引用了 research_topic
    ])
    return {"sub_queries": result.sub_queries}
```

**进入上下文**：
- ✅ `state['research_topic']`

**不进入上下文**：
- ❌ `state['sub_queries']`（还没生成）
- ❌ `state['retrieval_details']`（还没生成）
- ❌ `state['raw_results']`（还没生成）

---

#### execute 节点
```python
async def execute(state: RAGExecuteState, config) -> dict:
    # ... 执行搜索 ...
    
    # 评估搜索结果
    evaluation = await evaluator.ainvoke([
        SystemMessage(content=RAG_EVALUATE_PROMPT),
        HumanMessage(content=(
            f"研究主题：{research_topic}\n"      # ← 引用
            f"子查询：{current_query}\n"         # ← 引用
            f"搜索结果：\n{formatted_output}"    # ← 引用
        )),
    ])
    
    return {
        "raw_results": [...],
        "retrieval_details": all_retrieval_details,  # ← 只是返回，不进入上下文
    }
```

**进入上下文**：
- ✅ `research_topic`（从 state 读取）
- ✅ `current_query`（局部变量）
- ✅ `formatted_output`（从 rag_search 返回）

**不进入上下文**：
- ❌ `retrieval_details`（只是返回值，没有在 prompt 中引用）

---

#### compress 节点
```python
async def compress(state: RAGResearcherState, config) -> dict:
    all_results = "\n\n".join(state.get("raw_results", []))
    
    response = await model.ainvoke([
        SystemMessage(content=RAG_COMPRESS_PROMPT),
        HumanMessage(content=f"研究主题：{state['research_topic']}\n\n搜索结果：\n\n{all_results}")
        #                                ↑                                          ↑
        #                        引用 research_topic                        引用 raw_results
    ])
    
    return {"compressed_research": response.content}
```

**进入上下文**：
- ✅ `state['research_topic']`
- ✅ `state['raw_results']`（通过 all_results 变量）

**不进入上下文**：
- ❌ `state['retrieval_details']`（没有引用）
- ❌ `state['sub_queries']`（没有引用）
- ❌ `state['raw_notes']`（没有引用）

---

## 如何验证哪些字段进入了上下文？

### 方法 1：查看代码中的 prompt 构造

```bash
# 搜索所有传给 LLM 的内容
grep -A 5 "ainvoke\|invoke" src/rag_subgraph.py | grep "HumanMessage\|SystemMessage"
```

### 方法 2：查看 LangSmith 追踪（如果启用）

LangSmith 会显示每次 LLM 调用的完整输入：
```
Input:
  - SystemMessage: "..."
  - HumanMessage: "研究主题：xxx\n\n搜索结果：xxx"
```

### 方法 3：添加日志

```python
async def compress(state: RAGResearcherState, config) -> dict:
    all_results = "\n\n".join(state.get("raw_results", []))
    
    prompt_content = f"研究主题：{state['research_topic']}\n\n搜索结果：\n\n{all_results}"
    
    # 打印实际传给 LLM 的内容
    print(f"[DEBUG] 传给 LLM 的内容长度: {len(prompt_content)} 字符")
    print(f"[DEBUG] 包含的 state 字段: research_topic, raw_results")
    print(f"[DEBUG] 不包含的 state 字段: retrieval_details, sub_queries, raw_notes")
    
    response = await model.ainvoke([
        SystemMessage(content=RAG_COMPRESS_PROMPT),
        HumanMessage(content=prompt_content)
    ])
    
    return {"compressed_research": response.content}
```

---

## 特殊情况：MessagesState

### MessagesState 的特殊行为

如果你的 State 继承自 `MessagesState`：

```python
from langgraph.graph import MessagesState

class MyState(MessagesState):
    research_topic: str
    retrieval_details: list[dict]
```

**特殊点**：
- `messages` 字段会**自动**作为对话历史传给 LLM
- 其他字段（如 `retrieval_details`）仍然需要**显式引用**

**示例**：
```python
async def my_node(state: MyState, config) -> dict:
    model = get_model()
    
    # ✅ messages 会自动传给 LLM（作为对话历史）
    # ❌ retrieval_details 不会自动传给 LLM
    
    result = await model.ainvoke(state["messages"])  # 只传 messages
    
    return {"messages": [result]}
```

**但是**：RAG 子图**不使用** MessagesState，所以没有这个特殊行为。

---

## 总结

### 控制 State 字段是否进入上下文的方式

| 方式 | 是否进入上下文 | 示例 |
|------|--------------|------|
| **在 prompt 中显式引用** | ✅ 进入 | `HumanMessage(content=f"主题：{state['research_topic']}")` |
| **只存储在 state，不引用** | ❌ 不进入 | `state['retrieval_details']`（没有在 prompt 中使用） |
| **作为返回值** | ❌ 不进入 | `return {"retrieval_details": [...]}` |
| **MessagesState 的 messages** | ✅ 自动进入 | `await model.ainvoke(state["messages"])` |

### 当前项目中的情况

**进入 LLM 上下文的字段**：
- ✅ `research_topic`（在 plan 和 compress 节点中引用）
- ✅ `raw_results`（在 compress 节点中引用）
- ✅ `current_query`（在 execute 节点的评估阶段引用）
- ✅ `formatted_output`（在 execute 节点的评估阶段引用）

**不进入 LLM 上下文的字段**：
- ❌ `retrieval_details`（从未在 prompt 中引用）
- ❌ `sub_queries`（只用于路由，不传给 LLM）
- ❌ `raw_notes`（只用于记录，不传给 LLM）

### 为什么这样设计？

1. **节省 tokens**：只传必要的信息给 LLM
2. **提高性能**：减少 LLM 输入长度，加快响应速度
3. **数据持久化**：State 可以存储评测数据，但不影响 LLM 推理

---

## 如果想让 retrieval_details 进入上下文？

**不推荐**，但如果真的需要：

```python
async def compress(state: RAGResearcherState, config) -> dict:
    all_results = "\n\n".join(state.get("raw_results", []))
    retrieval_details = state.get("retrieval_details", [])
    
    # 格式化检索详情
    details_summary = "\n".join([
        f"查询 {i+1}: dense召回{d['dense']['count']}篇, sparse召回{d['sparse']['count']}篇"
        for i, d in enumerate(retrieval_details)
    ])
    
    response = await model.ainvoke([
        SystemMessage(content=RAG_COMPRESS_PROMPT),
        HumanMessage(content=(
            f"研究主题：{state['research_topic']}\n\n"
            f"检索详情：\n{details_summary}\n\n"  # ← 显式引用
            f"搜索结果：\n\n{all_results}"
        ))
    ])
    
    return {"compressed_research": response.content}
```

**但这样做的缺点**：
- ❌ 增加 tokens 消耗
- ❌ 检索详情对 LLM 压缩摘要没有帮助
- ❌ 可能干扰 LLM 的理解

**所以当前设计是正确的**：retrieval_details 只用于评测，不传给 LLM。
