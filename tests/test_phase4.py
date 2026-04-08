"""Phase 4 验证：完整主图端到端测试。

真实调用 LLM + 搜索，验证从用户输入到最终报告的全流程。
每次运行创建独立目录保存 checkpoint、元数据和报告。

需要环境变量：OPENAI_API_KEY (或 ANTHROPIC_API_KEY), TAVILY_API_KEY
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")

from langchain_core.messages import AIMessage, HumanMessage


def _make_run_dir(topic: str) -> Path:
    """从研究主题生成 run 目录：logs/{slug}-{timestamp}/"""
    # 保留中文、字母、数字，空格转连字符，截断30字符
    slug = re.sub(r'[^\w\s]', '', topic).strip()
    slug = re.sub(r'\s+', '-', slug)[:30]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).resolve().parent.parent / "logs" / f"{slug}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def test_full_pipeline():
    """完整主图端到端测试：用户输入 → 澄清 → 研究简报 → Supervisor → 报告。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from graph import deep_researcher_builder

    topic = "2025年诺贝尔物理学奖的研究内容是什么，有什么实际应用前景"
    run_dir = _make_run_dir(topic)
    thread_id = run_dir.name
    db_path = str(run_dir / "checkpoints.db")

    print(f"📋 研究主题: {topic}")
    print(f"📁 Run 目录: {run_dir}")
    print(f"🔑 Thread ID: {thread_id}")
    print("⏳ 启动完整主图...\n")

    start_time = datetime.now()

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = deep_researcher_builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=topic)]},
            config=config,
        )

        # ── 提取结果 ──
        final_report = result.get("final_report", "")
        messages = result.get("messages", [])

        if final_report:
            print("=" * 60)
            print("📝 最终报告（前 500 字）:")
            print("-" * 60)
            print(final_report[:500])
            print(f"\n📏 报告总长度: {len(final_report)} 字符")
        else:
            last_msg = messages[-1].content if messages else "无消息"
            print(f"⚠️  图在澄清阶段结束，AI 追问：\n{last_msg}")

        # ── 导出步骤历史 + token 用量 ──
        history = []
        total_tokens = {"input": 0, "output": 0, "total": 0}
        async for snapshot in graph.aget_state_history(config):
            step_tokens = {"input": 0, "output": 0, "total": 0}
            for msg in snapshot.values.get("messages", []):
                usage = getattr(msg, "usage_metadata", None)
                if usage and isinstance(msg, AIMessage):
                    step_tokens["input"] += usage.get("input_tokens", 0)
                    step_tokens["output"] += usage.get("output_tokens", 0)
                    step_tokens["total"] += usage.get("total_tokens", 0)
            history.append({
                "step": snapshot.metadata.get("step", -1),
                "node": snapshot.metadata.get("source", "unknown"),
                "writes_keys": list((snapshot.metadata.get("writes") or {}).keys()),
                "msg_count": len(snapshot.values.get("messages", [])),
                "tokens": step_tokens,
            })
            total_tokens = {k: max(total_tokens[k], step_tokens[k]) for k in total_tokens}

    # ── 保存结果到 run 目录 ──
    elapsed = (datetime.now() - start_time).total_seconds()

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "topic": topic, "thread_id": thread_id,
            "elapsed_seconds": round(elapsed, 1),
            "completed": bool(final_report),
            "report_length": len(final_report),
            "total_tokens": total_tokens,
            "steps": history,
        }, f, ensure_ascii=False, indent=2)

    if final_report:
        with open(run_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(final_report)

    print(f"\n⏱️  耗时: {elapsed:.1f}s")
    print(f"💰 Token: 输入={total_tokens['input']}, 输出={total_tokens['output']}, 总计={total_tokens['total']}")
    print(f"📁 结果已保存: {run_dir}")

    if final_report:
        print("\n🎉 Phase 4 端到端测试通过！")
    else:
        print("\n⚠️  未生成报告（可能在澄清阶段结束），请检查日志")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
