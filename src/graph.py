"""Deep Research Agent 的 LangGraph 图定义。

按阶段逐步实现：
- Phase 2: Researcher 子图（ReAct 循环 + 压缩）
- Phase 3: Supervisor 子图 + 主图
"""

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from configuration import Configuration, PROJECT_ROOT
from prompts import (
    clarify_with_user_instructions,
    research_system_prompt, lead_researcher_prompt,
    transform_messages_into_research_topic_prompt, final_report_generation_prompt,
)
from state import (
    AgentInputState, ClarifyWithUser,
    ConductResearch, ResearchComplete, ResearchQuestion,
    ResearcherOutputState, ResearcherState, SupervisorState, AgentState,
)
from utils import (
    get_all_tools, get_api_key_for_model, get_model_token_limit,
    get_today_str, is_token_limit_exceeded, think_tool,
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
    research_complete_called = any(
        tc["name"] == "ResearchComplete" for tc in tool_calls
    )
    if exceeded or research_complete_called:
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


##########################
# Supervisor 子图节点
##########################

async def supervisor(state: SupervisorState, config: RunnableConfig):
    """Supervisor 节点：研究策略规划和任务分配。

    读取 supervisor_messages → 绑定 3 个工具 → 调用 LLM → 路由到 supervisor_tools。
    与 researcher() 的区别：Supervisor 不直接搜索，而是通过
    ConductResearch 将任务委派给 Researcher 子图执行。
    """
    configurable = Configuration.from_runnable_config(config)
    # Supervisor 可用的 3 个工具：
    #   ConductResearch — 委派研究任务给子 Researcher
    #   ResearchComplete — 表示研究已完成
    #   think_tool      — 策略反思
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config({
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.research_model, config),
            "tags": ["langsmith:nostream"],
        })
    )
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    # → 进入 supervisor_tools 执行工具调用
    from langgraph.types import Command
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig):
    """执行 Supervisor 的工具调用。

    处理 3 种工具调用：
    1. think_tool       → 记录反思，继续循环
    2. ConductResearch  → 并行启动 Researcher 子图（Step 3 补充）
    3. ResearchComplete → 结束研究阶段

    退出条件（任一命中即退出）：
    - 超过 max_researcher_iterations
    - LLM 没有生成任何工具调用
    - LLM 调用了 ResearchComplete
    """
    from langchain_core.messages import ToolMessage, HumanMessage
    from langgraph.types import Command
    from utils import get_notes_from_tool_calls, is_token_limit_exceeded
    import asyncio

    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent = supervisor_messages[-1]

    # ── 退出条件检查 ──
    exceeded = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent.tool_calls
    research_complete = any(
        tc["name"] == "ResearchComplete" for tc in most_recent.tool_calls
    )
    if exceeded or no_tool_calls or research_complete:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            }
        )

    # ── 处理工具调用 ──
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # 1) think_tool：记录反思
    for tc in most_recent.tool_calls:
        if tc["name"] == "think_tool":
            all_tool_messages.append(ToolMessage(
                content=f"反思已记录: {tc['args']['reflection']}",
                name="think_tool",
                tool_call_id=tc["id"],
            ))

    # 2) ConductResearch：并行执行研究任务
    conduct_research_calls = [tc for tc in most_recent.tool_calls if tc["name"] == "ConductResearch"]
    if conduct_research_calls:
        try:
            # 限制并发数
            allowed_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_calls = conduct_research_calls[configurable.max_concurrent_research_units:]

            # 并行启动 Researcher 子图
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])],
                    "research_topic": tc["args"]["research_topic"]
                }, config)
                for tc in allowed_calls
            ]
            tool_results = await asyncio.gather(*research_tasks)

            # 处理研究结果
            for obs, tc in zip(tool_results, allowed_calls):
                all_tool_messages.append(ToolMessage(
                    content=obs.get("compressed_research", "研究报告合成错误：超过最大重试次数"),
                    name=tc["name"],
                    tool_call_id=tc["id"]
                ))
            
            # 处理溢出的任务
            for tc in overflow_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"错误：未执行此研究，因为已超过最大并行研究单元数。请保证在 {configurable.max_concurrent_research_units} 个以内。",
                    name="ConductResearch",
                    tool_call_id=tc["id"]
                ))
            
            # 汇总所有新增的 raw_notes
            raw_notes_concat = "\n".join([
                "\n".join(obs.get("raw_notes", [])) for obs in tool_results
            ])
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception as e:
            # token 超限或其他异常，直接结束研究阶段
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", ""),
                    }
                )

    # 返回结果，继续 supervisor 循环
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


##########################
# Supervisor 子图组装
##########################
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


##########################
# 主图节点：用户澄清
##########################

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """分析用户消息，判断研究范围是否清晰，必要时提出澄清问题。

    如果配置中禁用了澄清（allow_clarification=False），直接跳到研究简报生成。
    否则使用 ClarifyWithUser schema 做结构化分析，决定是追问还是继续。
    """
    # Step 1: 检查是否启用了澄清功能
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")

    # Step 2: 配置模型，使用 ClarifyWithUser 结构化输出
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # Step 3: 调用模型分析是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # Step 4: 根据分析结果路由
    if response.need_clarification:
        # 需要澄清 → 结束对话，向用户追问
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # 不需要澄清 → 带确认消息进入研究简报生成
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


##########################
# 主图节点：研究简报生成
##########################

async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转化为结构化的研究简报，并初始化 Supervisor 可见的上下文。

    使用 override 语义重置 supervisor_messages，使其只包含
    system_prompt 和简报内容，以防止历史消息过长。
    """
    from langgraph.types import Command
    
    configurable = Configuration.from_runnable_config(config)
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config({
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.research_model, config),
            "tags": ["langsmith:nostream"],
        })
    )
    
    # 根据用户对话生成研究简报
    prompt = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt)])
    
    # 组装 Supervisor 的系统提示词
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    # 进入 research_supervisor 子图，并使用 override_reducer 完全替换 supervisor_messages
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


##########################
# 主图节点：最终报告生成
##########################

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """将所有研究笔记合成为最终报告。

    带 token 超限重试逻辑：
    - 首次重试：用 model_token_limit * 4 作为字符截断基准
    - 后续重试：每次减 10%
    """
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    configurable = Configuration.from_runnable_config(config)
    writer_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }

    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            final_report = await configurable_model.with_config(writer_config).ainvoke(
                [HumanMessage(content=prompt)]
            )
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                if current_retry == 1:
                    limit = get_model_token_limit(configurable.final_report_model)
                    if not limit:
                        return {
                            "final_report": f"报告生成失败：token 超限，且无法确定模型上限。{e}",
                            "messages": [AIMessage(content="报告生成因 token 超限失败")],
                            **cleared_state
                        }
                    findings_token_limit = limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                findings = findings[:findings_token_limit]
                continue
            else:
                return {
                    "final_report": f"报告生成失败：{e}",
                    "messages": [AIMessage(content="报告生成遇到错误")],
                    **cleared_state
                }

    return {
        "final_report": "报告生成失败：超过最大重试次数",
        "messages": [AIMessage(content="报告生成超过最大重试次数")],
        **cleared_state
    }


##########################
# 完整主图组装
##########################

# 创建主图：用户输入 → 澄清 → 研究简报 → Supervisor 研究 → 最终报告
deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration
)

# 添加 4 个节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# 定义边
# 注意：clarify_with_user → write_research_brief 没有 edge，靠 Command 路由
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# 编译完整主图
deep_researcher = deep_researcher_builder.compile()
