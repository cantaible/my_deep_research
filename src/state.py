"""图状态定义。"""

import operator
from typing import Annotated, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出定义
###################
class ConductResearch(BaseModel):
    """调用此工具对指定主题进行研究。"""
    research_topic: str = Field(
        description="要研究的主题。应为单一主题，并以高度详细的方式描述（至少一个段落）。",
    )

class ResearchComplete(BaseModel):
    """调用此工具表示研究已完成。"""

class Summary(BaseModel):
    """研究摘要，包含关键发现。"""
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """用户意图澄清模型。"""
    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题。",
    )
    question: str = Field(
        description="向用户提出的澄清问题，用于明确报告范围。",
    )
    verification: str = Field(
        description="确认消息，告知用户我们将在获得必要信息后开始研究。",
    )

class ResearchQuestion(BaseModel):
    """研究问题及简报，用于指导研究方向。"""
    research_brief: str = Field(
        description="用于指导研究的研究问题。",
    )

class ConductRAGResearch(BaseModel):
    """委派本地新闻库研究任务。Supervisor 调用此工具搜索本地新闻数据库。"""
    research_topic: str = Field(
        description="要在本地新闻库中搜索的研究主题。",
    )

class RAGSubQuery(BaseModel):
    """Plan 阶段产出的单个子查询。"""
    search_intent: str = Field(description="一句话说明这个子查询的搜索目的，想找什么信息")
    query: str = Field(description="描述性自然语言搜索短句（中文为主，可混合英文术语）")
    start_date: str = Field(
        description="搜索时间范围的起始日期，格式 YYYY-MM-DD。例如 '2026-03-01'",
    )
    end_date: str = Field(
        description="搜索时间范围的结束日期，格式 YYYY-MM-DD。例如 '2026-03-07'",
    )
    category: Literal["AI", "GAMES", ""] = Field(
        default="AI",
        description="新闻分类：'AI'=人工智能相关, 'GAMES'=游戏相关, ''=不限",
    )

class RAGQueryPlan(BaseModel):
    """Plan 阶段的完整输出：将研究主题拆分为多个子查询。"""
    sub_queries: list[RAGSubQuery] = Field(
        description="拆分后的子查询列表，按时间窗口或子主题拆分以提高召回率",
    )

class SearchEvaluation(BaseModel):
    """LLM 对单次 RAG 搜索结果的结构化评估。"""
    quality: Literal["good", "insufficient", "off_topic"] = Field(
        description="结果质量：good=满足需求, insufficient=信息不足需补搜, off_topic=偏题需改写查询",
    )
    reason: str = Field(description="评估理由")
    refined_query: Optional[str] = Field(
        default=None,
        description="修正后的查询（quality != good 时必填）",
    )


# ── LATS 树搜索子图 ──

class TreeNode(BaseModel):
    """搜索树中的一个节点。"""
    id: str
    query: str
    dimension: str = ""                 # 拆分维度: region/company/type/time
    parent_id: Optional[str] = None
    children_ids: list[str] = Field(default_factory=list)
    depth: int = 0
    status: str = "pending"             # pending | expanded | leaf | pruned
    search_results: str = ""
    result_count: int = 0
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    visits: int = 0
    value: float = 0.0

class NodeEvaluation(BaseModel):
    """LLM 对搜索结果的评估打分。"""
    relevance_score: float = Field(description="结果与研究主题的相关性，0-1")
    completeness_score: float = Field(description="该方向是否还需深入，0=已完整 1=需要深入")
    reasoning: str = Field(description="打分理由")

class LATSExpandResult(BaseModel):
    """LLM 展开节点时生成的子查询列表。"""
    sub_queries: list[str] = Field(description="子查询列表，空列表表示不需要展开")
    dimensions: list[str] = Field(description="每个子查询对应的拆分维度")

class ConductLATSResearch(BaseModel):
    """委派 LATS 树搜索研究任务。适合需要发散探索的复杂调研场景。"""
    research_topic: str = Field(description="要研究的主题")

class LATSResearcherState(TypedDict):
    """LATS 树搜索子图的完整状态。"""
    research_topic: str
    tree: dict                          # node_id → TreeNode.model_dump()
    root_id: str
    current_node_id: str
    iteration: int
    collected_findings: list[str]
    compressed_research: str


class AgentInputState(MessagesState):
    """图输入状态，只暴露 messages 给外部调用者。"""


###################
# 状态定义
###################
#
# 整体架构是三层嵌套图：
#
#   主图 (AgentState)
#    ├─ clarify_with_user    → 用户意图澄清
#    ├─ write_research_brief → 生成研究简报
#    ├─ research_supervisor  → Supervisor 子图 (SupervisorState)
#    │   ├─ supervisor       → 决定研究策略
#    │   └─ supervisor_tools → 执行工具，启动 N 个 Researcher
#    │       └─ researcher_subgraph → Researcher 子图 (ResearcherState)
#    │           ├─ researcher       → ReAct 循环：搜索 + 思考
#    │           ├─ researcher_tools → 执行搜索工具
#    │           └─ compress_research → 压缩研究结果
#    └─ final_report_generation → 生成最终报告
#
# 每层子图有独立的 State，通过 output State 控制向上暴露哪些字段。

def override_reducer(current_value, new_value):
    """状态归约器：支持覆盖语义。
    
    当 new_value 是 {"type": "override", "value": ...} 时，
    直接用 value 替换当前值（而非追加）。
    否则使用 operator.add 追加。
    
    用途：write_research_brief 需要重置 supervisor_messages，
    而不是在原有基础上追加。
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

class AgentState(MessagesState):
    """主图运行状态，贯穿整个研究流程。
    
    继承 MessagesState，自带 messages 字段（用户对话历史）。
    """
    # Supervisor 子图的消息列表，用 override_reducer 支持重置
    # write_research_brief 会用 {"type": "override", "value": [...]} 来初始化
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    # 由 write_research_brief 生成的研究简报文本
    research_brief: Optional[str]
    # 所有 Researcher 的原始笔记（每个 Researcher 的工具调用结果）
    raw_notes: Annotated[list[str], override_reducer] = []
    # Supervisor 汇总后的笔记，传给 final_report_generation
    notes: Annotated[list[str], override_reducer] = []
    # 最终生成的研究报告
    final_report: str

class SupervisorState(TypedDict):
    """Supervisor 子图状态，管理研究任务的分配和协调。
    
    用 TypedDict 而非 MessagesState，因为 Supervisor 有自己独立的消息流，
    不与主图的 messages 混合。
    """
    # Supervisor 自己的对话历史（系统提示 + 研究简报 + 工具调用结果）
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    # 从主图传入的研究简报
    research_brief: str
    # 汇总后的研究笔记，最终传回主图
    notes: Annotated[list[str], override_reducer] = []
    # 当前迭代次数，超过 max_researcher_iterations 就强制停止
    research_iterations: int = 0
    # 原始笔记，从 Researcher 子图汇总上来
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """Researcher 子图状态，执行具体的研究任务。
    
    每个 Researcher 独立运行，互不干扰。
    用 operator.add 追加消息（不需要 override 语义）。
    """
    # Researcher 的对话历史（系统提示 + 搜索结果 + 思考过程）
    # 注意：用 operator.add 而非 override_reducer，因为研究过程只需追加
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    # 工具调用次数，超过 max_react_tool_calls 就停止搜索
    tool_call_iterations: int = 0
    # 由 Supervisor 分配的具体研究主题
    research_topic: str
    # compress_research 节点生成的压缩摘要
    compressed_research: str
    # 原始笔记（工具调用的原始返回内容）
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Researcher 子图的输出状态（状态隔离的关键）。
    
    这是 StateGraph(ResearcherState, output=ResearcherOutputState) 的 output 参数。
    当 Researcher 子图完成时，只有这两个字段会传回给 Supervisor，
    researcher_messages、tool_call_iterations 等内部状态不会泄漏。
    
    为什么重要？如果不隔离，Researcher 的几十条搜索消息会混入
    Supervisor 的状态，导致 token 爆炸和逻辑混乱。
    """
    # 压缩后的研究成果，作为 ToolMessage 返回给 Supervisor
    compressed_research: str
    # 原始笔记，用于备份和调试
    raw_notes: Annotated[list[str], override_reducer] = []

class RAGResearcherState(TypedDict):
    """RAG 子图状态（Plan → 并行 Execute → Compress）。

    plan 节点拆分子查询后，通过 Send API 并行分发到 execute 节点。
    execute 的结果通过 operator.add reducer 自动汇入 raw_results。
    输出仍复用 ResearcherOutputState，确保 Supervisor 端代码零改动。
    """
    # 由 Supervisor 分配的研究主题
    research_topic: str
    # plan 节点产出的子查询列表
    sub_queries: list[dict]
    # execute 节点汇入的原始结果
    raw_results: Annotated[list[str], operator.add]
    # compress 节点产出的压缩摘要
    compressed_research: str
    # 原始笔记（与 ResearcherOutputState 的 reducer 一致）
    raw_notes: Annotated[list[str], override_reducer]
    # 检索详情（用于评测，记录 dense/sparse/merged/reranked 各阶段数据）
    retrieval_details: Annotated[list[dict], operator.add]

class RAGExecuteState(TypedDict):
    """单个 RAG execute 节点的状态，由 Send 分发。"""
    # 来自 plan 的单个子查询（dict 形式的 RAGSubQuery）
    sub_query: dict
    # 原始研究主题（评估时需要上下文）
    research_topic: str
    # execute 输出，通过 reducer 汇入 RAGResearcherState
    raw_results: Annotated[list[str], operator.add]
    raw_notes: Annotated[list[str], override_reducer]
    # 检索详情（用于评测）
    retrieval_details: Annotated[list[dict], operator.add]

