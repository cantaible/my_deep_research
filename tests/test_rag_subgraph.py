"""RAG 子图测试：实际运行 Plan-and-Execute + 导出状态历史。

需要环境变量：OPENAI_API_KEY
需要本地 RAG 数据库已构建（rag/vectordb/）
运行后在 logs/ 目录生成独立 run 目录。
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))


async def test_rag_subgraph():
    """实际调用 RAG 子图，测试完整 Plan-Execute-Compress 流程。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from rag_subgraph import rag_researcher_builder
    from runner import append_event, normalize_event

    # topic = "搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻、产品发布、榜单变化，重点覆盖基础大语言模型、图像生成模型、视频生成模型、Agent、代码生成模型，以及Lmarena leaderboard在2026年3月的相关报道或引用。优先提取发布日期、厂商、模型类别、功能亮点、技术规格和榜单排名信息。"
    topic = "搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻，看看有哪些新的大模型发布了，尤其要关注头部厂商。"

    # 创建 run 目录（复用 runner 的命名逻辑）
    from runner import make_run_dir
    run_dir = make_run_dir(f"rag-test-{topic[:20]}")
    thread_id = run_dir.name
    db_path = str(run_dir / "checkpoints.db")
    events_path = run_dir / "events.jsonl"

    print(f"📋 研究主题: {topic}")
    print(f"📁 Run 目录: {run_dir}")
    print(f"🔑 Thread ID: {thread_id}")
    print("⏳ 启动 RAG 子图...\n")

    start_time = datetime.now()

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        subgraph = rag_researcher_builder.compile(checkpointer=checkpointer)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 500,
        }

        event_stream = subgraph.astream_events(
            {"research_topic": topic},
            config=config,
            version="v2",
        )
        async for raw in event_stream:
            evt = normalize_event(raw)
            if evt:
                append_event(events_path, evt)

        state = await subgraph.aget_state(config)
        result = state.values

        # 打印结果
        compressed = result.get("compressed_research", "")
        raw_notes = result.get("raw_notes", [])
        print("=" * 60)
        print("📝 压缩研究摘要（前 500 字）:")
        print("-" * 60)
        print(compressed[:500] if compressed else "❌ 无摘要输出")
        print(f"\n📏 摘要总长度: {len(compressed)} 字符")
        print(f"📌 原始笔记条数: {len(raw_notes)}")

        # 导出状态历史
        history = []
        total_tokens = {"input": 0, "output": 0, "total": 0}
        async for snapshot in subgraph.aget_state_history(config):
            step_tokens = {"input": 0, "output": 0, "total": 0}
            # RAGResearcherState 没有 messages，从 snapshot.values 直接提取信息
            step = {
                "step": snapshot.metadata.get("step", -1),
                "node": snapshot.metadata.get("source", "unknown"),
                "writes_keys": list((snapshot.metadata.get("writes") or {}).keys()),
                "sub_query_count": len(snapshot.values.get("sub_queries", [])),
                "raw_result_count": len(snapshot.values.get("raw_results", [])),
            }
            history.append(step)

    elapsed = (datetime.now() - start_time).total_seconds()

    # 保存运行元数据
    meta = {
        "test": "rag_subgraph",
        "topic": topic,
        "thread_id": thread_id,
        "elapsed_seconds": round(elapsed, 1),
        "compressed_length": len(compressed),
        "raw_notes_count": len(raw_notes),
        "steps": history,
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 保存压缩摘要
    if compressed:
        with open(run_dir / "compressed.md", "w", encoding="utf-8") as f:
            f.write(compressed)

    # 保存原始笔记
    with open(run_dir / "raw_notes.json", "w", encoding="utf-8") as f:
        json.dump(raw_notes, f, ensure_ascii=False, indent=2)

    # 保存子查询计划（从最终 state 提取）
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        subgraph = rag_researcher_builder.compile(checkpointer=checkpointer)
        state = await subgraph.aget_state(config)
        sub_queries = state.values.get("sub_queries", [])
        with open(run_dir / "sub_queries.json", "w", encoding="utf-8") as f:
            json.dump(sub_queries, f, ensure_ascii=False, indent=2)

    print(f"\n⏱️  耗时: {elapsed:.1f}s")
    print(f"📁 结果已保存: {run_dir}")
    print(f"   ├── run_meta.json      (运行元数据)")
    print(f"   ├── compressed.md      (压缩摘要)")
    print(f"   ├── raw_notes.json     (原始笔记)")
    print(f"   ├── sub_queries.json   (拆分计划)")
    print(f"   └── checkpoints.db     (完整 checkpoint)")

    assert compressed, "❌ compressed_research 不应为空"
    assert len(raw_notes) > 0, "❌ raw_notes 不应为空"
    print("\n🎉 RAG 子图测试通过！")


if __name__ == "__main__":
    asyncio.run(test_rag_subgraph())
