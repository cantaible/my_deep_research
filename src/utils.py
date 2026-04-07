"""工具函数和辅助模块。

包含搜索工具、反思工具、模型提供商检测、token 限制处理等。
MCP 相关功能暂以 stub 形式保留。
"""

import os
from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from configuration import Configuration, SearchAPI


##########################
# 反思工具
##########################

@tool(description="研究过程中的策略反思工具")
def think_tool(reflection: str) -> str:
    """用于研究进度的策略反思和决策工具。

    在每次搜索后使用此工具分析结果并系统地规划下一步。
    """
    return f"反思已记录: {reflection}"


##########################
# 通用工具函数
##########################

def get_today_str() -> str:
    """获取当前日期的格式化字符串，用于提示词。"""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """从配置中提取值，处理枚举和 None。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value


##########################
# API Key 获取函数
##########################

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """根据模型名称从环境变量获取对应的 API Key。

    根据模型前缀（openai:/anthropic:/google:）自动选择正确的环境变量。
    """
    model_name = model_name.lower()
    if model_name.startswith("openai:"):
        return os.getenv("OPENAI_API_KEY")
    elif model_name.startswith("anthropic:"):
        return os.getenv("ANTHROPIC_API_KEY")
    elif model_name.startswith("google"):
        return os.getenv("GOOGLE_API_KEY")
    return None

def get_tavily_api_key(config: RunnableConfig):
    """获取 Tavily 搜索 API Key。"""
    return os.getenv("TAVILY_API_KEY")


##########################
# 消息处理工具
##########################

def get_notes_from_tool_calls(messages):
    """从工具调用消息中提取笔记内容。

    遍历所有 ToolMessage，提取 content 作为研究笔记。
    用于 supervisor_tools 退出时汇总所有研究结果。
    """
    from langchain_core.messages import filter_messages
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

def remove_up_to_last_ai_message(messages):
    """截断消息历史到最后一条 AI 消息之前。

    用于处理 token 超限时，移除最近的上下文以减少输入长度。
    从后往前扫描，找到最后一条 AIMessage，返回它之前的所有消息。
    """
    from langchain_core.messages import AIMessage
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]
    return messages


##########################
# Tavily 搜索工具
##########################

async def tavily_search_async(
    search_queries, max_results=5, topic="general",
    include_raw_content=True, config=None
):
    """异步并行执行多个 Tavily 搜索查询。

    将多个查询同时发送给 Tavily API，利用 asyncio.gather 并行化。
    """
    import asyncio
    from tavily import AsyncTavilyClient

    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    search_tasks = [
        tavily_client.search(
            query, max_results=max_results,
            include_raw_content=include_raw_content, topic=topic,
            search_depth="advanced"
        )
        for query in search_queries
    ]
    return await asyncio.gather(*search_tasks)

async def summarize_webpage(model, webpage_content: str) -> str:
    """使用 AI 模型摘要网页内容，带 60 秒超时保护。

    将原始网页内容压缩为 summary + key_excerpts 的结构化格式。
    如果摘要失败或超时，返回原始内容作为降级方案。
    """
    import asyncio
    import logging
    from langchain_core.messages import HumanMessage
    from prompts import summarize_webpage_prompt

    try:
        prompt = summarize_webpage_prompt.format(
            webpage_content=webpage_content, date=get_today_str()
        )
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt)]),
            timeout=60.0
        )
        return (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
    except asyncio.TimeoutError:
        logging.warning("网页摘要超时（60秒），返回原始内容")
        return webpage_content
    except Exception as e:
        logging.warning(f"网页摘要失败: {e}，返回原始内容")
        return webpage_content


@tool(description="针对全面、准确结果优化的搜索引擎，用于回答时事问题。")
async def tavily_search(
    queries: list[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """执行 Tavily 搜索，去重并摘要网页内容。

    流程：并行搜索 → URL 去重 → AI 摘要每个网页 → 格式化输出。
    InjectedToolArg 标记的参数对 LLM 不可见，由系统自动注入。
    """
    import asyncio
    from langchain.chat_models import init_chat_model
    from state import Summary
    # 1. 搜索 + 按 URL 去重
    results = await tavily_search_async(
        queries, max_results=max_results, topic=topic,
        include_raw_content=True, config=config
    )
    unique = {}
    for resp in results:
        for r in resp.get('results', []):
            if r['url'] not in unique:
                unique[r['url']] = {**r, "query": resp.get('query', '')}
    if not unique:
        return "未找到有效的搜索结果。请尝试不同的搜索查询。"
    # 2. 配置摘要模型（用 Summary schema 做结构化输出）
    cfg = Configuration.from_runnable_config(config)
    sum_model = init_chat_model(
        model=cfg.summarization_model, max_tokens=cfg.summarization_model_max_tokens,
        api_key=get_api_key_for_model(cfg.summarization_model, config),
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=cfg.max_structured_output_retries
    )
    # 3. 并行摘要所有网页
    async def noop(): return None
    tasks = [
        noop() if not r.get("raw_content")
        else summarize_webpage(sum_model, r['raw_content'][:cfg.max_content_length])
        for r in unique.values()
    ]
    summaries = await asyncio.gather(*tasks)
    # 4. 格式化输出
    output = "搜索结果:\n\n"
    for i, (url, result, summary) in enumerate(
        zip(unique.keys(), unique.values(), summaries)
    ):
        content = result['content'] if summary is None else summary
        output += f"\n--- 来源 {i+1}: {result['title']} ---\n"
        output += f"URL: {url}\n\n摘要:\n{content}\n\n" + "-" * 80 + "\n"
    return output


##########################
# 工具集管理
##########################

def get_search_tool(config: RunnableConfig):
    """根据配置返回对应的搜索工具。

    SearchAPI.TAVILY → 返回 tavily_search 工具
    SearchAPI.OPENAI / ANTHROPIC → 返回 None（原生搜索在模型层面处理）
    SearchAPI.NONE → 返回 None（不使用搜索）
    """
    cfg = Configuration.from_runnable_config(config)
    search_api = get_config_value(cfg.search_api)

    if search_api == "tavily":
        return tavily_search
    # OpenAI / Anthropic 原生搜索由模型自带，不需要额外工具
    # NONE 表示不使用任何搜索
    return None

async def get_all_tools(config: RunnableConfig):
    """获取 Researcher 可用的所有工具列表。

    这是 Researcher 节点获取工具的唯一入口。
    组合：搜索工具 + think_tool + MCP 工具（暂为空）。
    """
    tools = []

    # 1. 添加搜索工具（如果有）
    search_tool = get_search_tool(config)
    if search_tool:
        tools.append(search_tool)

    # 2. 添加反思工具（始终可用）
    tools.append(think_tool)

    # 3. MCP 工具（暂时 stub，后续阶段实现）
    # TODO: 从 MCP 服务器加载外部工具


    return tools


##########################
# Token 限制处理
##########################

# 常见模型的上下文窗口大小（token 数）
# 用于 final_report_generation 的 token 超限重试逻辑
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4o": 128000,
    "openai:gpt-4o-mini": 128000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
}

def get_model_token_limit(model_string):
    """查找模型的 token 上限。

    通过子串匹配在 MODEL_TOKEN_LIMITS 中查找。
    返回 token 数（int），未找到返回 None。
    """
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    return None

def is_token_limit_exceeded(error, model_name: str) -> bool:
    """检测异常是否为 token 超限错误。

    不同模型提供商的超限错误消息格式不同，
    通过统一检测关键词来判断。
    """
    error_str = str(error).lower()
    token_limit_keywords = [
        "context_length_exceeded",
        "context window",
        "token limit",
        "max_tokens",
        "too many tokens",
        "maximum context length",
        "request too large",
    ]
    return any(kw in error_str for kw in token_limit_keywords)


##########################
# 原生搜索检测
##########################

def openai_websearch_called(message) -> bool:
    """检测 OpenAI 模型是否使用了原生网络搜索。

    OpenAI 原生搜索的标志是 message.content 中包含
    type='url_citation' 的 annotation 块。
    """
    if not hasattr(message, "content") or not isinstance(message.content, list):
        return False
    for block in message.content:
        annotations = block.get("annotations", []) if isinstance(block, dict) else []
        if any(a.get("type") == "url_citation" for a in annotations):
            return True
    return False

def anthropic_websearch_called(message) -> bool:
    """检测 Anthropic 模型是否使用了原生网络搜索。

    Anthropic 原生搜索的标志是 message.content 中包含
    type='web_search_tool_result' 的内容块。
    """
    if not hasattr(message, "content") or not isinstance(message.content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "web_search_tool_result"
        for block in message.content
    )
