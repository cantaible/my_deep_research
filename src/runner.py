"""核心运行器：事件采集 + 持久化。

将 LangGraph astream_events 标准化，实时写入 events.jsonl，
并通过回调通知 TUI 渲染层。
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_run_dir(topic: str) -> Path:
    """从研究主题生成 run 目录：logs/{slug}-{timestamp}/"""
    slug = re.sub(r'[^\w\s]', '', topic).strip()
    slug = re.sub(r'\s+', '-', slug)[:30]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).resolve().parent.parent / "logs" / f"{slug}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_event(path: Path, event: dict):
    """追加一条事件到 JSONL 文件（实时持久化，崩溃也不丢）。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def save_run_meta(run_dir: Path, meta: dict):
    """保存运行元数据到 run_meta.json。"""
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_report(run_dir: Path, content: str):
    """保存最终报告到 report.md。"""
    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(content)


def save_retrieval_details(run_dir: Path, details: list):
    """保存检索详情到 retrieval_details.json。"""
    with open(run_dir / "retrieval_details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)


def normalize_event(raw: dict) -> dict | None:
    """将 astream_events v2 事件标准化为 TUI 可消费的格式。

    返回 None 表示此事件不需要展示/持久化（噪音过滤）。
    只保留 6 种关键事件：node_start, llm_start, llm_stream, llm_end,
    tool_start, tool_end。
    """
    etype = raw.get("event", "")
    name = raw.get("name", "")
    data = raw.get("data", {})
    node = raw.get("metadata", {}).get("langgraph_node", "")
    ts = datetime.now().isoformat()

    # LLM 逐 token 流式输出
    if etype == "on_chat_model_stream":
        chunk = data.get("chunk")
        if chunk and hasattr(chunk, "content") and chunk.content:
            return {"type": "llm_stream", "node": node, "token": chunk.content, "ts": ts}

    # LLM 调用完成（含 token 用量）
    if etype == "on_chat_model_end":
        output = data.get("output")
        usage = {}
        if output and hasattr(output, "usage_metadata") and output.usage_metadata:
            usage = dict(output.usage_metadata)
        content = str(output.content) if output and hasattr(output, "content") else ""
        return {"type": "llm_end", "node": node, "content": content[:500],
                "usage": usage, "ts": ts}

    # LLM 调用开始
    if etype == "on_chat_model_start":
        return {"type": "llm_start", "node": node, "model": name, "ts": ts}

    # 工具调用开始
    if etype == "on_tool_start":
        return {"type": "tool_start", "node": node, "tool": name,
                "args": str(data.get("input", ""))[:200], "ts": ts}

    # 工具调用结束
    if etype == "on_tool_end":
        return {"type": "tool_end", "node": node, "tool": name,
                "result": str(data.get("output", ""))[:300], "ts": ts}

    # 节点开始（只捕获图节点级别，过滤内部 chain）
    if etype == "on_chain_start" and raw.get("metadata", {}).get("langgraph_triggers"):
        return {"type": "node_start", "node": node, "ts": ts}

    return None


async def run_research(
    topic: str, *,
    on_event: Callable[[dict], Awaitable[None]],
    on_clarify: Callable[[str], Awaitable[str]] | None = None,
) -> Path:
    """运行完整研究流水线，实时产出事件并持久化。

    Args:
        topic: 研究主题
        on_event: 每产出一个事件就调用（TUI 渲染用）
        on_clarify: 需要澄清时调用（获取用户输入），None 则不支持交互

    Returns:
        run_dir: 本次运行的日志目录
    """
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from graph import deep_researcher_builder

    run_dir = make_run_dir(topic)
    events_path = run_dir / "events.jsonl"
    db_path = str(run_dir / "checkpoints.db")
    thread_id = run_dir.name
    config = {"configurable": {"thread_id": thread_id}}
    start_time = datetime.now()
    final_report = ""

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = deep_researcher_builder.compile(checkpointer=checkpointer)
        messages = [HumanMessage(content=topic)]

        while True:
            # 流式执行主图，逐事件处理
            # astream_events() 返回异步迭代器：图在后台执行，每发生一件事就吐出一个事件
            try:
                event_stream = graph.astream_events(
                    {"messages": messages}, config, version="v2"
                )
                while True:
                    try:
                        raw = await event_stream.__anext__()  # 等待下一个事件（可能等几秒）
                    except StopAsyncIteration:
                        break  # 没有更多事件了（图执行完毕）
                    evt = normalize_event(raw)
                    if evt:
                        append_event(events_path, evt)  # 实时写入 events.jsonl
                        await on_event(evt)             # 通知 TUI 渲染
            except Exception as e:
                # 图执行过程中出错：记录并通知 UI，但不中断 runner
                import traceback
                error_msg = f"研究执行出错: {e}\n{traceback.format_exc()}"
                print(error_msg, file=sys.stderr)
                append_event(events_path, {
                    "type": "error", "message": str(e),
                    "ts": datetime.now().isoformat()
                })

            # 检查运行结果：完成 or 需要澄清
            state = await graph.aget_state(config)
            result = state.values
            final_report = result.get("final_report", "")
            retrieval_details = result.get("retrieval_details", [])

            if final_report:
                save_report(run_dir, final_report)
                if retrieval_details:
                    save_retrieval_details(run_dir, retrieval_details)
                await on_event({"type": "report", "content": final_report,
                                "ts": datetime.now().isoformat()})
                break
            elif on_clarify and result.get("messages"):
                q = result["messages"][-1].content
                await on_event({"type": "clarify", "question": q,
                                "ts": datetime.now().isoformat()})
                answer = await on_clarify(q)
                messages = [HumanMessage(content=answer)]
                # 循环继续 → re-invoke 图（checkpointer 保留上下文）
            else:
                break

    # 保存运行元数据
    elapsed = (datetime.now() - start_time).total_seconds()
    save_run_meta(run_dir, {
        "topic": topic, "thread_id": thread_id,
        "elapsed_seconds": round(elapsed, 1),
        "completed": bool(final_report),
        "report_length": len(final_report),
    })
    return run_dir
