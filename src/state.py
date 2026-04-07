"""图状态定义。"""

import operator
from typing import Annotated, Optional

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

