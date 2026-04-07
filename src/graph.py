"""Deep Research Agent 的 LangGraph 图定义。

按阶段逐步实现：
- Phase 2: Researcher 子图（ReAct 循环 + 压缩）
- Phase 3: Supervisor 子图 + 主图
"""

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from configuration import Configuration, PROJECT_ROOT
from prompts import research_system_prompt
from state import ResearcherOutputState, ResearcherState
from utils import (
    get_all_tools, get_api_key_for_model, get_today_str,
)

# 初始化可配置模型（通过 configurable_fields 动态切换模型）
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


##########################
# Researcher 子图节点
##########################

async def researcher(state: ResearcherState, config: RunnableConfig):
    """Researcher 节点：ReAct 循环的主体。

    读取当前消息历史 → 绑定工具 → 调用 LLM → 返回带工具调用的响应。
    每次调用后 tool_call_iterations +1，用于控制循环次数上限。
    """
    configurable = Configuration.from_runnable_config(config)
    # 获取可用工具（搜索 + think_tool）
    tools = await get_all_tools(config)
    if not tools:
        raise ValueError("没有可用的研究工具，请配置搜索 API 或 MCP 工具。")
    # 配置研究模型
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config({
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.research_model, config),
            "tags": ["langsmith:nostream"],
        })
    )
    # 构建消息：系统提示 + 历史消息
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", date=get_today_str()
    )
    messages = [SystemMessage(content=researcher_prompt)] + state.get("researcher_messages", [])
    response = await research_model.ainvoke(messages)
    # → 进入 researcher_tools 执行工具调用
    from langgraph.types import Command
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        }
    )


async def _execute_tool_safely(tool, args, config):
    """安全执行单个工具，捕获异常返回错误信息。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"工具执行错误: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig):
    """执行 Researcher 的工具调用（搜索、think_tool 等）。

    流程：
    1. 检查是否有工具调用 → 没有则直接进入压缩
    2. 并行执行所有工具调用
    3. 判断是否继续循环：超过次数上限或无更多工具调用 → 压缩
    """
    import asyncio
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command
    from utils import openai_websearch_called, anthropic_websearch_called

    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent = researcher_messages[-1]

    # 无工具调用且无原生搜索 → 直接进入压缩
    has_tool_calls = bool(most_recent.tool_calls)
    has_native = openai_websearch_called(most_recent) or anthropic_websearch_called(most_recent)
    if not has_tool_calls and not has_native:
        return Command(goto="compress_research")

    # 获取所有工具并建立名称→工具的映射
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # 并行执行所有工具调用
    tool_calls = most_recent.tool_calls
    observations = await asyncio.gather(*[
        _execute_tool_safely(tools_by_name[tc["name"]], tc["args"], config)
        for tc in tool_calls
    ])

    # 将执行结果包装为 ToolMessage
    tool_outputs = [
        ToolMessage(content=obs, name=tc["name"], tool_call_id=tc["id"])
        for obs, tc in zip(observations, tool_calls)
    ]

    # 判断是否结束循环
    exceeded = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    if exceeded:
        # 超过工具调用次数上限 → 进入压缩
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # 未达上限 → 继续 ReAct 循环
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """压缩节点：将搜索记录精炼为结构化摘要。

    使用 compression_model 将 Researcher 的完整对话历史
    压缩为一段精炼摘要。带 token 超限重试逻辑。
    """
    from langchain_core.messages import filter_messages
    from prompts import compress_research_system_prompt, compress_research_simple_human_message
    from utils import is_token_limit_exceeded, remove_up_to_last_ai_message

    configurable = Configuration.from_runnable_config(config)
    synth_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"],
    })

    researcher_messages = state.get("researcher_messages", [])
    # 添加压缩指令作为 HumanMessage
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    compress_prompt = compress_research_system_prompt.format(date=get_today_str())

    # 最多重试 3 次（处理 token 超限）
    for attempt in range(3):
        try:
            messages = [SystemMessage(content=compress_prompt)] + researcher_messages
            response = await synth_model.ainvoke(messages)
            raw = "\n".join(str(m.content) for m in filter_messages(
                researcher_messages, include_types=["tool", "ai"]))
            return {"compressed_research": str(response.content), "raw_notes": [raw]}
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.compression_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
            continue

    return {"compressed_research": "研究压缩失败：超过最大重试次数", "raw_notes": []}


##########################
# Researcher 子图组装
##########################
researcher_builder = StateGraph(
    ResearcherState, output=ResearcherOutputState, config_schema=Configuration
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)

# 默认编译（无 checkpointer，用于快速测试）
# 需要持久化时，使用 researcher_builder 自行编译：
#   async with AsyncSqliteSaver.from_conn_string("logs/checkpoints.db") as saver:
#       graph = researcher_builder.compile(checkpointer=saver)
researcher_subgraph = researcher_builder.compile()
