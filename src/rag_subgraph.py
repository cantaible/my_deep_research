"""RAG 子图：Plan-and-Execute 模式。

将研究主题拆分为多个子查询，并行搜索本地新闻数据库，
合并后压缩为摘要。输出复用 ResearcherOutputState。
"""

import asyncio
import sys
from pathlib import Path

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from configuration import Configuration
from state import RAGQueryPlan, RAGResearcherState, ResearcherOutputState
from graph import configurable_model
from utils import get_api_key_for_model

# ── Plan 阶段的系统提示 ──
RAG_PLAN_PROMPT = """你是一个查询规划助手。将用户的研究主题拆分为多个子查询，
以提高在本地新闻数据库中的召回率。今天是 {today}。

拆分策略：
1. **按时间窗口**：如果涉及一段时间（如"3月"），按周拆分
2. **按子主题**：如果涉及多个方面（如"语言模型、图像模型"），按方面拆分
3. **组合拆分**：时间 × 子主题

字段说明：
- query: 具体的英文搜索关键词（本地新闻库为英文内容）
- start_date: 时间范围起始日期，格式 YYYY-MM-DD
- end_date: 时间范围结束日期，格式 YYYY-MM-DD
- category: 新闻分类，必须为以下之一：
  - "AI"：人工智能、大模型、机器学习相关
  - "GAMES"：游戏相关
  - ""：不限分类

确保覆盖完整，宁多勿漏。"""


async def plan(state: RAGResearcherState, config) -> dict:
    """Plan 节点：LLM 将研究主题拆分为子查询列表。"""
    configurable = Configuration.from_runnable_config(config)
    model = configurable_model.with_config({
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    })
    structured_model = model.with_structured_output(RAGQueryPlan)

    prompt = RAG_PLAN_PROMPT.format(today=date.today().isoformat())
    result = await structured_model.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"请将以下研究主题拆分为子查询：\n\n{state['research_topic']}"),
    ])

    sub_queries = [q.model_dump() for q in result.sub_queries]
    print(f"  📝 RAG Plan: 拆分为 {len(sub_queries)} 个子查询")
    return {"sub_queries": sub_queries}


async def _run_single_rag_query(sub_query: dict, semaphore: asyncio.Semaphore) -> str:
    """在线程池中执行单个 RAG 查询（限制并发避免 ChromaDB 冲突）。"""
    from rag_search import rag_search
    async with semaphore:
        result = await asyncio.to_thread(
            rag_search.invoke,
            {
                "query": sub_query["query"],
                "start_date": sub_query.get("start_date", ""),
                "end_date": sub_query.get("end_date", ""),
                "category": sub_query.get("category", ""),
                "top_k": 10,
            },
        )
    return result


async def execute(state: RAGResearcherState, config) -> dict:
    """Execute 节点：并行执行所有子查询（限制并发数）。"""
    sub_queries = state["sub_queries"]
    print(f"  🔍 RAG Execute: 并行查询 {len(sub_queries)} 个子查询...")

    # 预初始化所有模型（避免多线程同时初始化冲突）
    from rag_search import get_collection, get_embedding_model
    from reranker import get_reranker
    get_collection()
    get_embedding_model()
    get_reranker()

    # 限制并发数为 5，避免 ChromaDB SQLite 锁冲突
    semaphore = asyncio.Semaphore(5)
    results = await asyncio.gather(*[
        _run_single_rag_query(q, semaphore) for q in sub_queries
    ])

    # 组装带标签的结果，方便 compress 和调试
    raw_results = []
    for q, r in zip(sub_queries, results):
        raw_results.append(f"--- 查询: {q['query']} ---\n{r}")

    raw_notes = [f"[RAG] {q['query']}" for q in sub_queries]
    return {"raw_results": raw_results, "raw_notes": raw_notes}


# ── Compress 阶段的系统提示 ──
RAG_COMPRESS_PROMPT = """你是一个研究结果整合助手。将多个查询的搜索结果合并为一份结构化的研究摘要。

要求：
1. 保留所有发现的信息，不要遗漏（去重将在后续步骤统一处理）
2. 保留关键细节：名称、日期、厂商、性能指标
3. 按时间或主题组织，结构清晰
4. 输出为纯文本摘要"""


async def compress(state: RAGResearcherState, config) -> dict:
    """Compress 节点：合并去重所有子查询结果，压缩为摘要。"""
    configurable = Configuration.from_runnable_config(config)
    model = configurable_model.with_config({
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    })

    all_results = "\n\n".join(state.get("raw_results", []))

    response = await model.ainvoke([
        SystemMessage(content=RAG_COMPRESS_PROMPT),
        HumanMessage(
            content=f"研究主题：{state['research_topic']}\n\n搜索结果：\n\n{all_results}"
        ),
    ])

    result_text = response.content
    print(f"  📦 RAG Compress: {len(result_text)} 字符")
    return {"compressed_research": result_text}


# ── 图组装：plan → execute → compress ──
from langgraph.graph import END, START, StateGraph  # noqa: E402

rag_researcher_builder = StateGraph(
    RAGResearcherState, output=ResearcherOutputState
)
rag_researcher_builder.add_node("plan", plan)
rag_researcher_builder.add_node("execute", execute)
rag_researcher_builder.add_node("compress", compress)

rag_researcher_builder.add_edge(START, "plan")
rag_researcher_builder.add_edge("plan", "execute")
rag_researcher_builder.add_edge("execute", "compress")
rag_researcher_builder.add_edge("compress", END)
