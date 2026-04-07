"""将 checkpoints.db 的完整运行数据导出为可读的 Markdown 文件。

用法: python scripts/export_run.py [thread_id]
如不指定 thread_id，默认导出最近一次运行。
"""
import asyncio, sys
sys.path.insert(0, "src")

async def export(thread_id=None):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from graph import researcher_builder
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
    from pathlib import Path

    db_path = str(Path(__file__).resolve().parent.parent / "logs" / "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        subgraph = researcher_builder.compile(checkpointer=checkpointer)

        # 如果没指定 thread_id，找最近一次
        if not thread_id:
            import sqlite3
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1").fetchone()
            conn.close()
            thread_id = row[0] if row else None
            if not thread_id:
                print("❌ 数据库中没有运行记录"); return

        config = {"configurable": {"thread_id": thread_id}}

        # 取最终状态
        snapshots = []
        async for s in subgraph.aget_state_history(config):
            snapshots.append(s)
        snapshots.reverse()
        final_msgs = snapshots[-1].values.get("researcher_messages", [])

        # 生成 Markdown
        lines = [f"# 运行记录: {thread_id}\n"]
        lines.append(f"- 总消息数: {len(final_msgs)}")
        lines.append(f"- 总 checkpoint 数: {len(snapshots)}\n")
        lines.append("---\n")

        for i, msg in enumerate(final_msgs):
            if isinstance(msg, HumanMessage):
                lines.append(f"## 📨 [{i}] HumanMessage\n")
                lines.append(f"```\n{msg.content}\n```\n")
            elif isinstance(msg, AIMessage):
                lines.append(f"## 🤖 [{i}] AIMessage\n")
                # token 用量
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    lines.append(f"> Token: input={usage.get('input_tokens',0)}, output={usage.get('output_tokens',0)}\n")
                # 文本内容
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                if text.strip():
                    lines.append(f"**文本回复:**\n```\n{text[:2000]}\n```\n")
                # 工具调用
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        lines.append(f"**工具调用: `{tc['name']}`**\n")
                        if tc["name"] == "tavily_search":
                            queries = tc["args"].get("queries", [])
                            lines.append("搜索 Queries:\n")
                            for q in queries:
                                lines.append(f"- `{q}`\n")
                        elif tc["name"] == "think_tool":
                            thought = tc["args"].get("thought", "")
                            lines.append(f"```\n{thought[:1000]}\n```\n")
                        lines.append("")
            elif isinstance(msg, ToolMessage):
                lines.append(f"## 🔧 [{i}] ToolMessage (name={msg.name})\n")
                content = msg.content
                lines.append(f"<details>\n<summary>展开查看（{len(content)} 字符）</summary>\n\n```\n{content}\n```\n</details>\n")

        # 压缩摘要
        compressed = snapshots[-1].values.get("compressed_research", "")
        if compressed:
            lines.append("---\n")
            lines.append("## 📝 最终压缩摘要\n")
            lines.append(f"```\n{compressed}\n```\n")

        # 写文件
        out_path = Path(__file__).resolve().parent.parent / "logs" / f"{thread_id}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅ 已导出: {out_path}")

if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(export(tid))
