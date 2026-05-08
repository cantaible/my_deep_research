"""抽取单次运行里的 researcher 输出为 JSON。

用途：
    python scripts/extract_researcher_results.py <log_dir>
    python scripts/extract_researcher_results.py --latest

默认输出：
    <log_dir>/researcher_results.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_ROOT = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抽取 researcher 运行结果")
    parser.add_argument("log_dir", nargs="?", help="logs 下某次运行目录")
    parser.add_argument("--latest", action="store_true", help="使用最近一次包含 checkpoints.db 的运行")
    parser.add_argument("--output", help="输出 JSON 路径，默认 <log_dir>/researcher_results.json")
    return parser.parse_args()


def find_latest_log_dir() -> Path:
    candidates = [p for p in LOGS_ROOT.iterdir() if p.is_dir() and (p / "checkpoints.db").exists()]
    if not candidates:
        raise FileNotFoundError(f"在 {LOGS_ROOT} 下未找到包含 checkpoints.db 的运行目录")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def loads(serializer: JsonPlusSerializer, row: tuple[str, bytes] | None) -> Any:
    if row is None:
        return None
    return serializer.loads_typed(row)


def load_latest_channel(
    cur: sqlite3.Cursor,
    serializer: JsonPlusSerializer,
    namespace: str,
    channel: str,
) -> Any:
    row = cur.execute(
        """
        SELECT type, value
        FROM writes
        WHERE checkpoint_ns = ? AND channel = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (namespace, channel),
    ).fetchone()
    return loads(serializer, row)


def load_channel_rows(
    cur: sqlite3.Cursor,
    serializer: JsonPlusSerializer,
    namespace: str,
    channel: str,
) -> list[Any]:
    rows = cur.execute(
        """
        SELECT type, value
        FROM writes
        WHERE checkpoint_ns = ? AND channel = ?
        ORDER BY rowid
        """,
        (namespace, channel),
    ).fetchall()
    values = [serializer.loads_typed(row) for row in rows]
    # LangGraph 有时把追加消息包成单元素 list，这里摊平一层。
    flattened: list[Any] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


def text_preview(text: str, limit: int = 500) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"type": "human", "content": message.content}
    if isinstance(message, AIMessage):
        return {
            "type": "ai",
            "content": message.content,
            "tool_calls": message.tool_calls,
            "usage": message.usage_metadata or {},
        }
    if isinstance(message, ToolMessage):
        return {
            "type": "tool",
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "content": message.content,
            "content_preview": text_preview(message.content),
        }
    return {"type": type(message).__name__, "content": str(message)}


def build_tool_steps(messages: list[Any]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, AIMessage):
            usage = message.usage_metadata or {}
            for tool_call in message.tool_calls or []:
                pending.append({
                    "tool_name": tool_call.get("name", ""),
                    "tool_args": tool_call.get("args", {}) or {},
                    "tool_call_id": tool_call.get("id", ""),
                    "usage": usage,
                })
            continue

        if isinstance(message, ToolMessage):
            call = pending.pop(0) if pending else {}
            content = message.content or ""
            steps.append({
                "step_no": len(steps) + 1,
                "tool_name": call.get("tool_name") or message.name,
                "tool_args": call.get("tool_args", {}),
                "tool_call_id": call.get("tool_call_id") or message.tool_call_id,
                "usage": call.get("usage", {}),
                "tool_result": content,
                "tool_result_preview": text_preview(content),
            })

    return steps


def extract_researcher_sections(cur: sqlite3.Cursor, serializer: JsonPlusSerializer) -> list[dict[str, Any]]:
    namespaces = [
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT checkpoint_ns FROM writes WHERE channel = 'researcher_messages' ORDER BY checkpoint_ns"
        )
    ]

    sections: list[dict[str, Any]] = []
    for namespace in namespaces:
        messages = load_channel_rows(cur, serializer, namespace, "researcher_messages")
        human_prompts = [m.content for m in messages if isinstance(m, HumanMessage)]
        sections.append({
            "namespace": namespace,
            "research_topic": load_latest_channel(cur, serializer, namespace, "research_topic") or "",
            "human_prompts": human_prompts,
            "compressed_research": load_latest_channel(cur, serializer, namespace, "compressed_research") or "",
            "raw_notes": load_latest_channel(cur, serializer, namespace, "raw_notes") or [],
            "tool_steps": build_tool_steps(messages),
            "messages": [message_to_dict(m) for m in messages],
        })
    return sections


def extract_rag_sections(cur: sqlite3.Cursor, serializer: JsonPlusSerializer) -> list[dict[str, Any]]:
    namespaces = [
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT checkpoint_ns FROM writes WHERE channel = 'sub_queries' ORDER BY checkpoint_ns"
        )
    ]

    sections: list[dict[str, Any]] = []
    for namespace in namespaces:
        sections.append({
            "namespace": namespace,
            "research_topic": load_latest_channel(cur, serializer, namespace, "research_topic") or "",
            "sub_queries": load_latest_channel(cur, serializer, namespace, "sub_queries") or [],
            "raw_results": load_latest_channel(cur, serializer, namespace, "raw_results") or [],
            "raw_notes": load_latest_channel(cur, serializer, namespace, "raw_notes") or [],
            "compressed_research": load_latest_channel(cur, serializer, namespace, "compressed_research") or "",
        })
    return sections


def extract_run(log_dir: Path) -> dict[str, Any]:
    checkpoints_path = log_dir / "checkpoints.db"
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"找不到 {checkpoints_path}")

    serializer = JsonPlusSerializer()
    conn = sqlite3.connect(checkpoints_path)
    try:
        cur = conn.cursor()
        result = {
            "log_dir": str(log_dir),
            "report_path": str(log_dir / "report.md") if (log_dir / "report.md").exists() else None,
            "final_report": (log_dir / "report.md").read_text(encoding="utf-8")
            if (log_dir / "report.md").exists() else "",
            "ordinary_researchers": extract_researcher_sections(cur, serializer),
            "rag_researchers": extract_rag_sections(cur, serializer),
        }
        result["counts"] = {
            "ordinary_researchers": len(result["ordinary_researchers"]),
            "rag_researchers": len(result["rag_researchers"]),
            "ordinary_tool_steps": sum(len(s["tool_steps"]) for s in result["ordinary_researchers"]),
            "rag_raw_results": sum(len(s["raw_results"]) for s in result["rag_researchers"]),
        }
        return result
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    if args.latest or not args.log_dir:
        log_dir = find_latest_log_dir()
    else:
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = PROJECT_ROOT / log_dir

    result = extract_run(log_dir)
    output_path = Path(args.output) if args.output else log_dir / "researcher_results.json"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = result["counts"]
    print(f"输出: {output_path}")
    print(
        "抽取完成: "
        f"ordinary={counts['ordinary_researchers']}, "
        f"rag={counts['rag_researchers']}, "
        f"tool_steps={counts['ordinary_tool_steps']}, "
        f"rag_raw_results={counts['rag_raw_results']}"
    )


if __name__ == "__main__":
    main()
