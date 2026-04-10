"""导出单次运行中的 researcher 子图记录为可读 Markdown。

优先从 logs/<run_dir>/checkpoints.db 读取结构化状态：
- research_topic
- researcher_messages
- compressed_research

输出按 researcher 子图分组的总览和逐轮记录，适合复盘：
- HumanMessage 研究题目
- 每一轮 AI 的工具调用
- 对应 ToolMessage 返回内容
- tavily_search 的查询批次与命中来源
- think_tool 的反思
- ResearchComplete 收尾

用法：
    python scripts/export_researcher_run.py <log_dir>
    python scripts/export_researcher_run.py <log_dir> --output <output_path>
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


@dataclass
class SearchSource:
    rank: int
    title: str
    url: str
    summary: str


@dataclass
class StepRecord:
    step_no: int
    tool_name: str
    tool_args: dict[str, Any]
    tool_result: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class ResearcherSection:
    namespace: str
    research_topic: str
    human_prompt: str
    steps: list[StepRecord]
    compressed_research: str


SOURCE_RE = re.compile(
    r"--- 来源 (?P<rank>\d+): (?P<title>.*?) ---\n"
    r"URL: (?P<url>.*?)\n"
    r"(?:\n摘要:\n<summary>\n(?P<summary>.*?)(?:\n</summary>|</summary>))?",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 researcher 子图运行记录")
    parser.add_argument(
        "log_dir",
        nargs="?",
        help="运行日志目录，例如 logs/某次运行目录",
    )
    parser.add_argument(
        "--output",
        help="输出 Markdown 路径，默认写入 <log_dir>/RESEARCHER_RUN_README.md",
    )
    return parser.parse_args()


def find_latest_log_dir(logs_root: Path) -> Path:
    candidates = [p for p in logs_root.iterdir() if p.is_dir() and (p / "checkpoints.db").exists()]
    if not candidates:
        raise FileNotFoundError(f"在 {logs_root} 下未找到包含 checkpoints.db 的运行目录")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def deserialize(serializer: JsonPlusSerializer, typ: str, value: bytes) -> Any:
    obj = serializer.loads_typed((typ, value))
    if isinstance(obj, list) and len(obj) == 1:
        return obj[0]
    return obj


def load_researcher_sections(checkpoints_path: Path) -> list[ResearcherSection]:
    serializer = JsonPlusSerializer()
    conn = sqlite3.connect(checkpoints_path)
    cur = conn.cursor()

    namespaces = [
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT checkpoint_ns FROM writes WHERE channel = 'researcher_messages' ORDER BY checkpoint_ns"
        )
    ]
    if not namespaces:
        raise ValueError(f"{checkpoints_path} 中没有找到 researcher 子图记录")

    sections: list[ResearcherSection] = []
    for namespace in namespaces:
        topic = load_single_value(cur, serializer, namespace, "research_topic") or ""
        compressed = load_single_value(cur, serializer, namespace, "compressed_research") or ""

        message_rows = cur.execute(
            """
            SELECT rowid, type, value
            FROM writes
            WHERE checkpoint_ns = ? AND channel = 'researcher_messages'
            ORDER BY rowid
            """,
            (namespace,),
        ).fetchall()
        messages = [deserialize(serializer, typ, value) for _, typ, value in message_rows]

        human_prompt, steps = build_steps(messages)
        sections.append(
            ResearcherSection(
                namespace=namespace,
                research_topic=topic,
                human_prompt=human_prompt,
                steps=steps,
                compressed_research=compressed,
            )
        )

    conn.close()
    return sections


def load_single_value(
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
        ORDER BY rowid
        LIMIT 1
        """,
        (namespace, channel),
    ).fetchone()
    if row is None:
        return None
    return serializer.loads_typed(row)


def build_steps(messages: list[Any]) -> tuple[str, list[StepRecord]]:
    human_prompt = ""
    ai_queue: list[tuple[AIMessage, str, dict[str, Any], dict[str, int]]] = []
    steps: list[StepRecord] = []
    step_no = 0

    for message in messages:
        if isinstance(message, HumanMessage):
            human_prompt = message.content
            continue

        if isinstance(message, AIMessage):
            tool_call = message.tool_calls[0] if message.tool_calls else None
            if tool_call is None:
                continue
            usage = message.usage_metadata or {}
            ai_queue.append(
                (
                    message,
                    tool_call["name"],
                    tool_call.get("args", {}) or {},
                    {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                )
            )
            continue

        if isinstance(message, ToolMessage):
            if not ai_queue:
                continue
            _, tool_name, tool_args, usage = ai_queue.pop(0)
            step_no += 1
            steps.append(
                StepRecord(
                    step_no=step_no,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=message.content or "",
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    total_tokens=usage["total_tokens"],
                )
            )

    return human_prompt, steps


def parse_search_sources(text: str) -> list[SearchSource]:
    sources: list[SearchSource] = []
    for match in SOURCE_RE.finditer(text):
        summary = normalize_text(match.group("summary") or "")
        sources.append(
            SearchSource(
                rank=int(match.group("rank")),
                title=normalize_text(match.group("title")),
                url=normalize_text(match.group("url")),
                summary=summary,
            )
        )
    return sources


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def truncate(text: str, limit: int = 240) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def escape_pipes(text: str) -> str:
    return text.replace("|", "\\|")


def classify_section(section: ResearcherSection) -> str:
    topic = section.research_topic
    lower = topic.lower()
    if "补充" in topic or "补查" in topic:
        return "补充核验"
    if "头部" in topic or "人工智能厂商" in topic:
        return "厂商发布研究"
    if "lmarena" in lower or "leaderboard" in lower or "arena" in lower or "榜单" in topic:
        return "榜单研究"
    return "Researcher 子图"


def render_markdown(log_dir: Path, sections: list[ResearcherSection]) -> str:
    lines: list[str] = [
        "# Researcher 运行记录解析",
        "",
        "## 总览",
        "",
        f"- 日志目录: `{log_dir}`",
        f"- Researcher 子图数量: {len(sections)}",
    ]

    total_steps = sum(len(section.steps) for section in sections)
    tool_counter: Counter[str] = Counter()
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for section in sections:
        for step in section.steps:
            tool_counter[step.tool_name] += 1
            input_tokens += step.input_tokens
            output_tokens += step.output_tokens
            total_tokens += step.total_tokens

    lines.extend(
        [
            f"- 总步骤数: {total_steps}",
            f"- 总输入 tokens: {input_tokens}",
            f"- 总输出 tokens: {output_tokens}",
            f"- 总 tokens: {total_tokens}",
            "",
            "## 工具调用统计",
            "",
        ]
    )
    for tool_name, count in tool_counter.items():
        lines.append(f"- {tool_name}: {count} 次")

    lines.extend(
        [
            "",
            "## 子图概览",
            "",
            "| 子图 | 类型 | 步骤数 | tavily_search | think_tool | ResearchComplete | tokens |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for section in sections:
        section_counter = Counter(step.tool_name for step in section.steps)
        section_tokens = sum(step.total_tokens for step in section.steps)
        lines.append(
            f"| `{short_namespace(section.namespace)}` | {classify_section(section)} | {len(section.steps)} | "
            f"{section_counter.get('tavily_search', 0)} | {section_counter.get('think_tool', 0)} | "
            f"{section_counter.get('ResearchComplete', 0)} | {section_tokens} |"
        )

    for section in sections:
        lines.extend(render_section(section))

    return "\n".join(lines).strip() + "\n"


def short_namespace(namespace: str) -> str:
    parts = namespace.split("|")
    if parts and parts[-1].isdigit() and len(parts) >= 2:
        return "|".join(parts[-2:])
    return parts[-1]


def render_section(section: ResearcherSection) -> list[str]:
    section_counter = Counter(step.tool_name for step in section.steps)
    section_input = sum(step.input_tokens for step in section.steps)
    section_output = sum(step.output_tokens for step in section.steps)
    section_total = sum(step.total_tokens for step in section.steps)

    lines = [
        "",
        f"## 子图：`{section.namespace}`",
        "",
        f"- 类型: {classify_section(section)}",
        f"- 研究主题: {section.research_topic}",
        f"- 步骤数: {len(section.steps)}",
        f"- 工具统计: `tavily_search={section_counter.get('tavily_search', 0)}`, "
        f"`think_tool={section_counter.get('think_tool', 0)}`, "
        f"`ResearchComplete={section_counter.get('ResearchComplete', 0)}`",
        f"- Tokens: input={section_input}, output={section_output}, total={section_total}",
        "",
        "### 初始任务",
        "",
        "```text",
        section.human_prompt.strip(),
        "```",
        "",
        "### 步骤概览",
        "",
        "| 步骤 | 工具 | 关键内容 |",
        "|---:|---|---|",
    ]

    for step in section.steps:
        lines.append(
            f"| {step.step_no} | `{step.tool_name}` | {escape_pipes(step_headline(step))} |"
        )

    lines.extend(["", "### 逐步记录", ""])
    for step in section.steps:
        lines.extend(render_step(step))

    lines.extend(
        [
            "",
            "### 子图压缩摘要",
            "",
            "<details>",
            "<summary>展开查看 compress_research 输出</summary>",
            "",
            section.compressed_research.strip(),
            "",
            "</details>",
        ]
    )
    return lines


def step_headline(step: StepRecord) -> str:
    if step.tool_name == "tavily_search":
        queries = step.tool_args.get("queries", [])
        first = queries[0] if queries else "无查询"
        return f"{len(queries)} 个查询；首条：{truncate(first, 90)}"
    if step.tool_name == "think_tool":
        reflection = step.tool_args.get("reflection", "")
        return truncate(reflection, 110)
    if step.tool_name == "ResearchComplete":
        return "结束 researcher 子图"
    return truncate(step.tool_result, 110)


def render_step(step: StepRecord) -> list[str]:
    lines = [
        f"#### Step {step.step_no}. `{step.tool_name}`",
        "",
        f"- Tokens: input={step.input_tokens}, output={step.output_tokens}, total={step.total_tokens}",
    ]

    if step.tool_name == "tavily_search":
        queries = step.tool_args.get("queries", [])
        sources = parse_search_sources(step.tool_result)
        lines.extend(
            [
                f"- 查询数: {len(queries)}",
                f"- 命中来源数: {len(sources)}",
                "",
                "**查询批次**",
                "",
            ]
        )
        for query in queries:
            lines.append(f"- `{query}`")

        lines.extend(["", "<details>", f"<summary>展开查看 {len(sources)} 个命中来源</summary>", ""])
        if sources:
            lines.extend(
                [
                    "| 来源 | 标题 | URL | 摘要预览 |",
                    "|---:|---|---|---|",
                ]
            )
            for source in sources:
                lines.append(
                    f"| {source.rank} | {escape_pipes(source.title)} | {escape_pipes(source.url)} | "
                    f"{escape_pipes(truncate(source.summary, 160))} |"
                )
        else:
            lines.append("未解析到来源。")
        lines.extend(["", "</details>", ""])
        return lines

    if step.tool_name == "think_tool":
        reflection = step.tool_args.get("reflection", "")
        lines.extend(
            [
                "",
                "**反思内容**",
                "",
                "```text",
                normalize_text(reflection),
                "```",
                "",
                "**工具返回**",
                "",
                "```text",
                normalize_text(step.tool_result),
                "```",
                "",
            ]
        )
        return lines

    if step.tool_name == "ResearchComplete":
        lines.extend(
            [
                "",
                "```text",
                step.tool_result.strip() or "(空返回，表示 researcher 正常结束)",
                "```",
                "",
            ]
        )
        return lines

    lines.extend(
        [
            "",
            "```text",
            normalize_text(step.tool_result),
            "```",
            "",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    logs_root = repo_root / "logs"

    log_dir = Path(args.log_dir).expanduser() if args.log_dir else find_latest_log_dir(logs_root)
    if not log_dir.is_absolute():
        log_dir = (repo_root / log_dir).resolve()
    if not log_dir.exists():
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")

    checkpoints_path = log_dir / "checkpoints.db"
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"缺少 checkpoints.db: {checkpoints_path}")

    sections = load_researcher_sections(checkpoints_path)
    markdown = render_markdown(log_dir, sections)

    output_path = Path(args.output).expanduser() if args.output else log_dir / "RESEARCHER_RUN_README.md"
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()
    output_path.write_text(markdown, encoding="utf-8")

    print(f"✅ 已导出 Researcher README: {output_path}")
    print(f"   - 子图数: {len(sections)}")
    print(f"   - 总步骤数: {sum(len(section.steps) for section in sections)}")


if __name__ == "__main__":
    main()
