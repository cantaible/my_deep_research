"""RAG 子图：Plan → 并行 Execute-with-Retry → Compress。"""

import asyncio
import sys
from pathlib import Path

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from configuration import Configuration
from state import (
    RAGExecuteState, RAGQueryPlan, RAGResearcherState,
    ResearcherOutputState, SearchEvaluation,
)

def _get_model():
    """延迟导入 configurable_model 以避免循环依赖。"""
    from graph import configurable_model
    return configurable_model
from utils import get_api_key_for_model

# ── Plan 阶段的系统提示 ──
RAG_PLAN_PROMPT = """你是一个查询规划助手。将用户的研究主题拆分为多个子查询，
以提高在本地新闻数据库中的召回率。今天是 {today}。

## 搜索系统能力说明
你的每个子查询会被发送到一个混合检索系统，该系统会：
1. 用 query 文本做**向量语义匹配**（适合自然语言描述性短句）
2. 同时做**关键词全文检索**（适合精确术语和名称）
3. 用 start_date/end_date 参数在搜索层做**时间过滤**——因此 query 文本中不需要包含日期信息
4. 用 category 参数按分类过滤

## 拆分原则
1. **按角度/维度拆分**：每个子查询聚焦一个具体的子主题、厂商群或技术方向，而非重复同一个泛化 query
2. **不要按时间窗口拆分**：时间过滤已交给搜索引擎参数处理，所有子查询共享用户指定的完整时间范围即可
3. **query 应为描述性自然语言短句**：向量检索对完整语义的句子效果远好于松散的关键词堆叠
4. **适度补充英文名/别名**：对于知名厂商和模型，在 query 中混入英文名有助于提升召回率
6. **子查询之间应尽量正交**：不同子查询的预期返回结果重叠度应尽可能低

## 字段说明
- search_intent: 一句话说明这个子查询想找什么信息
- query: 描述性自然语言搜索短句（中文为主，可混合英文术语）
- start_date: 时间范围起始日期，格式 YYYY-MM-DD
- end_date: 时间范围结束日期，格式 YYYY-MM-DD
- category: 新闻分类，"AI" / "GAMES" / ""（不限）"""


async def plan(state: RAGResearcherState, config) -> dict:
    """Plan 节点：LLM 将研究主题拆分为子查询列表。"""
    configurable = Configuration.from_runnable_config(config)
    # Plan 是 RAG 流程中最关键的环节，使用 hard_model 确保查询质量
    model = _get_model().with_config({
        "model": configurable.hard_model,
        "max_tokens": configurable.hard_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.hard_model, config),
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


def route_plan(state: RAGResearcherState) -> list[Send]:
    """路由函数：将 plan 产出的子查询通过 Send 并行分发到 execute。"""
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        print("  ⚠️ RAG Plan: 未生成子查询，直接进入 compress")
        return [Send("compress", {"research_topic": state["research_topic"]})]
    return [
        Send("execute", {
            "sub_query": sq,
            "research_topic": state["research_topic"],
        })
        for sq in sub_queries
    ]


async def _run_single_rag_query(sub_query: dict) -> str:
    """在线程池中执行单个 RAG 查询。"""
    from rag_search import rag_search
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


# ── Execute 阶段的评估提示 ──
RAG_EVALUATE_PROMPT = """你是一个搜索结果质量评估助手。评估以下搜索结果是否充分回答了子查询。

评估标准：
1. **good**：结果直接相关且信息充足，无需补搜
2. **insufficient**：结果部分相关但信息不足，需要补充搜索
3. **off_topic**：结果偏离主题，需要改写查询

如果判断为 insufficient 或 off_topic，请提供修正后的查询（refined_query）：
- 保留原查询的时间范围和主题边界
- 可调整关键词、补充中文/英文别名、缩窄或扩展范围
- 修正后的查询应具体、可搜索"""


async def execute(state: RAGExecuteState, config) -> dict:
    """执行单个子查询，带结构化重试。

    由 Send 分发，每个实例独立处理一个子查询。
    内部用 Python 循环实现重试，不需要图的边来编排。
    """
    sub_query = state["sub_query"]
    research_topic = state["research_topic"]
    configurable = Configuration.from_runnable_config(config)
    max_retries = configurable.max_rag_retries

    model = _get_model().with_config({
        "model": configurable.simple_model,
        "max_tokens": configurable.simple_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.simple_model, config),
    })
    evaluator = model.with_structured_output(SearchEvaluation)

    all_results = []
    current_query = sub_query["query"]

    for attempt in range(max_retries + 1):
        # 1. 执行搜索
        result = await _run_single_rag_query({**sub_query, "query": current_query})
        all_results.append(f"[第{attempt+1}轮] 查询: {current_query}\n{result}")
        print(f"  🔍 RAG execute [{attempt+1}/{max_retries+1}]: {current_query}")

        # 最后一轮不再评估
        if attempt == max_retries:
            break

        # 2. 结构化评估
        evaluation = await evaluator.ainvoke([
            SystemMessage(content=RAG_EVALUATE_PROMPT),
            HumanMessage(content=(
                f"研究主题：{research_topic}\n"
                f"子查询：{current_query}\n"
                f"搜索结果：\n{result}"
            )),
        ])

        if evaluation.quality == "good":
            print(f"  ✅ RAG evaluate: {evaluation.reason}")
            break

        # 3. 不满意 → 用修正后的 query 重试
        current_query = evaluation.refined_query or current_query
        print(f"  🔄 RAG retry: {evaluation.reason} → {current_query}")

    combined = "\n\n".join(all_results)
    return {
        "raw_results": [f"--- 查询: {sub_query['query']} ---\n{combined}"],
        "raw_notes": [f"[RAG] {sub_query['query']}"],
    }


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
    model = _get_model().with_config({
        "model": configurable.simple_model,
        "max_tokens": configurable.simple_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.simple_model, config),
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


# ── 图组装：plan → [Send ×N] execute → compress ──
from langgraph.graph import END, START, StateGraph  # noqa: E402

rag_researcher_builder = StateGraph(
    RAGResearcherState, output=ResearcherOutputState
)
rag_researcher_builder.add_node("plan", plan)
rag_researcher_builder.add_node("execute", execute)
rag_researcher_builder.add_node("compress", compress)

rag_researcher_builder.add_edge(START, "plan")
rag_researcher_builder.add_conditional_edges("plan", route_plan, ["execute", "compress"])
rag_researcher_builder.add_edge("execute", "compress")
rag_researcher_builder.add_edge("compress", END)

