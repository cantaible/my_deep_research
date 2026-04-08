# LATS Researcher 子图设计方案

## 核心思路

标准 LATS 是为了找**一条**最优路径（如解题）。而新闻研究的目标是**最大化召回率**——找到所有相关信息。因此我们做一个面向信息检索的变体：**LATS-R (LATS for Retrieval)**。

核心差异：不是找到一个好答案就停，而是**遍历所有有价值的分支，剪掉无价值的**。

## 为什么 LATS 适合找新闻

找新闻天然是树形发散的：

```
"2026年3月有哪些重大AI新闻"
├── 按地区 → 国内 / 国外
│   ├── 国内 → 阿里、百度、腾讯、字节、小米...
│   │   └── 阿里 → Qwen3.5、通义千问...  ✅ 有结果
│   └── 国外 → OpenAI、Google、Meta、NVIDIA...
│       ├── OpenAI → GPT-5.4  ✅ 有结果
│       └── Meta → Llama4 推迟  ✅ 有结果
├── 按类型 → 大语言模型 / 图像 / 视频 / Agent
│   ├── 大语言模型 → ...
│   └── 图像 → ... ❌ 召回少，剪枝
└── 按事件类型 → 发布 / 融资 / 收购 / 政策
    └── ...
```

每个分支独立搜索 + 评估，低价值分支被剪掉，高价值分支继续深入展开。

## 算法流程

```
                    ┌─────────┐
                    │ 初始化   │  创建根节点
                    └────┬────┘
                         ▼
               ┌─────────────────┐
          ┌───►│    SELECT 选择   │  UCB 选最有潜力的节点
          │    └────────┬────────┘
          │             ▼
          │    ┌─────────────────┐
          │    │   EXPAND 展开    │  LLM 生成 N 个子查询
          │    └────────┬────────┘
          │             ▼
          │    ┌─────────────────┐
          │    │ EVALUATE 评估    │  执行搜索 + LLM 打分
          │    └────────┬────────┘
          │             ▼
          │    ┌─────────────────┐
          │    │ BACKPROP 回传   │  更新祖先节点分数
          │    └────────┬────────┘
          │             ▼
          │    ┌─────────────────┐
          │    │ SHOULD_CONTINUE │  预算耗尽？收敛？
          │    └───┬────────┬───┘
          │        │ 继续   │ 结束
          └────────┘        ▼
                   ┌─────────────────┐
                   │  AGGREGATE 聚合  │  收集全部叶子节点结果
                   └─────────────────┘
```

## 状态设计

```python
class TreeNode(BaseModel):
    """搜索树中的一个节点"""
    id: str                          # 唯一标识
    query: str                       # 该节点的搜索查询
    dimension: str                   # 拆分维度: "region", "company", "type", "time"
    parent_id: str | None            # 父节点（根为 None）
    children_ids: list[str]          # 子节点列表
    depth: int                       # 树深度（根=0）
    status: str                      # "pending" | "expanded" | "leaf" | "pruned"

    # 搜索结果
    search_results: str              # RAG/Web 返回的原始文本
    result_count: int                # 命中条数

    # 评估分数（LLM 打分）
    relevance_score: float           # 相关性 0-1
    completeness_score: float        # 完整性 0-1（是否还有遗漏子方向）

    # MCTS 统计
    visits: int                      # 被访问次数
    value: float                     # 累积价值（用于 UCB 计算）


class LATSResearcherState(TypedDict):
    """LATS 研究子图的完整状态"""
    research_topic: str              # 研究主题
    tree: dict[str, TreeNode]        # 节点 ID → 节点
    root_id: str                     # 根节点 ID
    current_node_id: str             # 当前正在处理的节点
    iteration: int                   # 当前迭代轮次
    max_iterations: int              # 最大迭代次数（预算）
    max_depth: int                   # 树最大深度
    collected_findings: list[str]    # 所有叶子节点的结果汇总
```

## 各节点详细设计

### 1. INITIALIZE（初始化）
- 创建根节点，`query = research_topic`
- 对根节点执行一次宽泛搜索，获取初始结果
- LLM 分析初始结果，识别可以展开的维度（地区/类型/公司/时间段）

### 2. SELECT（选择）
- 遍历所有 `status == "expanded"` 且有 `pending` 子节点的节点
- 对叶子节点使用 **UCB1** 公式选择最有探索价值的：

```
UCB = value/visits + C * sqrt(ln(parent_visits) / visits)
```

- C 值偏大（如 sqrt(2)），鼓励探索未访问分支
- 如果没有可展开的节点，跳转 AGGREGATE

### 3. EXPAND（展开）
- LLM 基于当前节点的 query 和 search_results，生成 2-5 个子查询
- Prompt 要点：

```
你是一个研究规划助手。当前研究节点的查询是："{query}"
搜索结果摘要：{search_results_summary}

请判断是否需要进一步细分探索。如果需要，生成子查询列表。
考虑以下维度拆分：
- 按公司/厂商拆分
- 按技术类型拆分
- 按时间窗口拆分
- 按应用场景拆分

如果当前结果已经足够详细完整，返回空列表（表示该节点为叶子节点）。
```

- 生成的子查询创建为子节点，`status = "pending"`

### 4. EVALUATE（评估）
- 对当前 pending 节点执行搜索（调用 `rag_search`）
- LLM 对搜索结果打两个分：
  - **relevance_score**：这些结果和研究主题的相关性（0-1）
  - **completeness_score**：该方向还有没有值得深入的子方向（0=不需要深入, 1=强烈需要深入）
- 剪枝规则：
  - `relevance_score < 0.3` → `status = "pruned"`（该方向无关，砍掉）
  - `completeness_score < 0.3` → `status = "leaf"`（已足够完整，不再展开）
  - 否则 → `status = "expanded"`（值得继续展开）

### 5. BACKPROPAGATE（回传）
- 从当前节点沿父链向上更新 `value` 和 `visits`
- `value += relevance_score * result_count`（高相关+高命中=高价值）

### 6. SHOULD_CONTINUE（是否继续）
- 终止条件（满足任一）：
  - `iteration >= max_iterations`
  - 所有叶子节点 `status` 都不是 `"expanded"`（即无节点可展开）
  - 最近 3 次迭代没有新发现（收敛）

### 7. AGGREGATE（聚合）
- 收集所有 `status == "leaf"` 的节点的 `search_results`
- 输出 `collected_findings` + `compressed_research`
- 输出接口与 RAGResearcherState 兼容

## 与现有架构的关系

```
Supervisor
├── ConductResearch       → Researcher 子图（网络搜索，ReAct 循环）
├── ConductRAGResearch    → RAG 子图（Plan-Execute-Compress，本地新闻库）
└── ConductLATSResearch   → LATS 子图（树搜索，适合发散探索）  ← 新增
```

三个工具互补：

| 工具 | 适合场景 | 搜索方式 | 特点 |
|------|---------|----------|------|
| ConductResearch | 通用研究 | 网络搜索 | ReAct 循环，灵活但浅 |
| ConductRAGResearch | 近期新闻 | 本地 RAG | 并行时间窗口，快速高召回 |
| ConductLATSResearch | 复杂调研 | RAG + Web | 树搜索，深度探索+剪枝 |

## 关于搜索后端

LATS 子图的每个节点搜索时，可以灵活选择后端：
- **本地 RAG**：快速，适合近期新闻，是默认后端
- **Web 搜索**：可选，适合 RAG 数据库没覆盖的内容
- 每个节点可以同时调两个后端，合并结果

## 示例：执行流程

研究主题："2026年3月有哪些重大AI新闻"

```
迭代 0: 初始化
  根节点 → "2026年3月重大AI新闻" → 搜索 → 找到 GPT-5.4, Grok 4.20 等
  LLM 评估: relevance=0.9, completeness=0.8 → 需要展开

迭代 1: 展开根节点
  子节点: ["国内AI厂商3月发布", "国外AI厂商3月发布", "AI政策与监管"]

迭代 2: SELECT="国内AI厂商3月发布" → 搜索 → 找到 Qwen3.5, Yuan3.0
  LLM 评估: relevance=0.95, completeness=0.7 → 展开

迭代 3: 展开"国内AI厂商3月发布"
  子节点: ["阿里 AI 3月", "百度 AI 3月", "腾讯 AI 3月", "小米 AI 3月"]

迭代 4: SELECT="阿里 AI 3月" → 搜索 → 找到 Qwen3.5 小模型, Fun-CineForge
  LLM 评估: relevance=0.9, completeness=0.2 → 标记为 leaf

迭代 5: SELECT="国外AI厂商3月发布"（UCB 最高，还没访问）
  搜索 → 找到 GPT-5.4, Grok 4.20, Nemotron 3 ...

迭代 6: SELECT="AI政策与监管" → 搜索 → 结果很少
  LLM 评估: relevance=0.2 → 剪枝

... 直到预算耗尽或收敛
```

## 参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| max_iterations | 10-15 | 每次迭代 1 次搜索 + 1 次 LLM 评估 |
| max_depth | 3 | 超过 3 层粒度太细，噪声增大 |
| max_children | 5 | 每次展开不超过 5 个子方向 |
| prune_threshold | 0.3 | relevance < 0.3 剪掉 |
| leaf_threshold | 0.3 | completeness < 0.3 不再深入 |
| UCB_C | 1.414 | sqrt(2)，平衡探索与利用 |

## 开放问题

> [!IMPORTANT]
> 需要你决策的 4 个问题：

1. **搜索后端选择**：LATS 默认只用 RAG，还是 RAG + Web 都用？Web 搜索慢但覆盖广。
2. **与 RAG 子图的关系**：LATS 是替代 RAG 子图，还是并列共存？如果共存，Supervisor 什么时候用 LATS 什么时候用 RAG？
3. **并行度**：展开子节点后，是串行逐个 evaluate，还是并行 evaluate 所有 pending 子节点？
4. **预算分配**：max_iterations=15 大约需要 15 次 RAG search + 15 次 LLM 调用，耗时预估 3-5 分钟，可以接受吗？
