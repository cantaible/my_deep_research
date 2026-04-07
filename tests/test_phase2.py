"""Phase 2 测试：实际运行 Researcher 子图 + 导出状态历史。

需要环境变量：OPENAI_API_KEY, TAVILY_API_KEY
运行后会在 logs/ 目录生成 JSON 格式的完整状态历史。
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage


async def test_researcher_subgraph():
    """实际调用 Researcher 子图，测试完整 ReAct 循环。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from graph import researcher_builder

    topic = "给我调研一下2026年2月有哪些大语言模型发布了"
    thread_id = f"test-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"📋 研究主题: {topic}")
    print(f"🔑 Thread ID: {thread_id}")
    print("⏳ 启动 Researcher 子图...\n")

    # 用 AsyncSqliteSaver 持久化 state 到 logs/checkpoints.db
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    db_path = str(logs_dir / "checkpoints.db")

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        subgraph = researcher_builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = await subgraph.ainvoke({
            "researcher_messages": [HumanMessage(content=topic)],
            "research_topic": topic,
        }, config=config)

        # 打印摘要
        compressed = result.get("compressed_research", "")
        print("=" * 60)
        print("📝 压缩研究摘要（前 500 字）:")
        print("-" * 60)
        print(compressed[:500] if compressed else "❌ 无摘要输出")
        print(f"\n📏 摘要总长度: {len(compressed)} 字符")

        # 导出完整状态历史到 JSON（含 token 用量）
        from langchain_core.messages import AIMessage
        history = []
        total_tokens = {"input": 0, "output": 0, "total": 0}
        async for snapshot in subgraph.aget_state_history(config):
            # 从 AIMessage 中提取 token 用量
            step_tokens = {"input": 0, "output": 0, "total": 0}
            for msg in snapshot.values.get("researcher_messages", []):
                usage = getattr(msg, "usage_metadata", None)
                if usage and isinstance(msg, AIMessage):
                    step_tokens["input"] += usage.get("input_tokens", 0)
                    step_tokens["output"] += usage.get("output_tokens", 0)
                    step_tokens["total"] += usage.get("total_tokens", 0)
            step = {
                "step": snapshot.metadata.get("step", -1),
                "node": snapshot.metadata.get("source", "unknown"),
                "writes_keys": list((snapshot.metadata.get("writes") or {}).keys()),
                "msg_count": len(snapshot.values.get("researcher_messages", [])),
                "iterations": snapshot.values.get("tool_call_iterations", 0),
                "cumulative_tokens": step_tokens,
            }
            history.append(step)
            total_tokens = {k: max(total_tokens[k], step_tokens[k]) for k in total_tokens}

    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"{thread_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"topic": topic, "thread_id": thread_id,
                   "summary_len": len(compressed),
                   "total_tokens": total_tokens, "steps": history},
                  f, ensure_ascii=False, indent=2)
    print(f"\n📁 状态历史已保存: {log_path}（{len(history)} 步）")
    print(f"💰 Token 用量: 输入={total_tokens['input']}, 输出={total_tokens['output']}, 总计={total_tokens['total']}")

    assert compressed, "❌ compressed_research 不应为空"
    print("\n🎉 Phase 2 Researcher 子图测试通过！")


if __name__ == "__main__":
    asyncio.run(test_researcher_subgraph())
