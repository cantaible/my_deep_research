"""导出单次运行中的 RAG 子图记录为可读 Markdown。

优先从 logs/<run_dir>/checkpoints.db 读取结构化状态：
- sub_queries
- raw_results
- compressed_research
- research_topic

同时从 events.jsonl 补充：
- rag_search 调用次数
- 执行开始/结束时间

用法：
    python scripts/export_rag_run.py <log_dir>
    python scripts/export_rag_run.py <log_dir> --output <output_path>

如果不传 <log_dir>，默认选择 logs/ 下最近修改的一个运行目录。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


@dataclass
class ResultHit:
    rank: int
    retrieval_mode: str
    title: str
    metadata: str
    source: str
    rerank_score: str
    preview: str


@dataclass
class QueryRecord:
    index: int
    query: str
    start_date: str
    end_date: str
    category: str
    topic: str
    hits: list[ResultHit]


@dataclass
class RagRunSection:
    namespace: str
    research_topic: str
    sub_queries: list[dict]
    raw_results: list[str]
    compressed_research: str


RESULT_BLOCK_RE = re.compile(
    r"^--- 结果 (?P<rank>\d+) \[(?P<mode>.*?)\] ---\n"
    r"标题: (?P<title>.*?)\n"
    r"元数据: (?P<meta>.*?)\n"
    r"Rerank分数: (?P<score>[^\n]+)\n"
    r"预览: (?P<preview>.*?)(?=^--- 结果 \d+ \[|\Z)",
    re.MULTILINE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 RAG 子图运行记录")
    parser.add_argument(
        "log_dir",
        nargs="?",
        help="运行日志目录，例如 logs/某次运行目录",
    )
    parser.add_argument(
        "--output",
        help="输出 Markdown 路径，默认写入 <log_dir>/RAG_RUN_README.md",
    )
    return parser.parse_args()


def find_latest_log_dir(logs_root: Path) -> Path:
    candidates = [p for p in logs_root.iterdir() if p.is_dir() and (p / "checkpoints.db").exists()]
    if not candidates:
        raise FileNotFoundError(f"在 {logs_root} 下未找到包含 checkpoints.db 的运行目录")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_sections(checkpoints_path: Path) -> list[RagRunSection]:
    serializer = JsonPlusSerializer()
    conn = sqlite3.connect(checkpoints_path)
    cur = conn.cursor()

    namespaces = [
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT checkpoint_ns FROM writes WHERE channel = 'sub_queries' ORDER BY checkpoint_ns"
        )
    ]
    if not namespaces:
        raise ValueError(f"{checkpoints_path} 中没有找到 RAG 子图的 sub_queries")

    sections: list[RagRunSection] = []
    for namespace in namespaces:
        channel_values = {}
        for channel in ("research_topic", "sub_queries", "raw_results", "compressed_research"):
            row = cur.execute(
                """
                SELECT type, value
                FROM writes
                WHERE checkpoint_ns = ? AND channel = ?
                ORDER BY idx
                LIMIT 1
                """,
                (namespace, channel),
            ).fetchone()
            if row is None:
                channel_values[channel] = None
                continue
            channel_values[channel] = serializer.loads_typed(row)

        sections.append(
            RagRunSection(
                namespace=namespace,
                research_topic=channel_values["research_topic"] or "",
                sub_queries=channel_values["sub_queries"] or [],
                raw_results=channel_values["raw_results"] or [],
                compressed_research=channel_values["compressed_research"] or "",
            )
        )

    conn.close()
    return sections


def parse_metadata_source(metadata: str) -> str:
    matches = re.findall(r"\[(.*?)\]", metadata)
    if len(matches) >= 2:
        return matches[1]
    if matches:
        return matches[0]
    return metadata.strip()


def normalize_preview(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def parse_hits(raw_result: str) -> list[ResultHit]:
    hits: list[ResultHit] = []
    for match in RESULT_BLOCK_RE.finditer(raw_result):
        metadata = match.group("meta").strip()
        hits.append(
            ResultHit(
                rank=int(match.group("rank")),
                retrieval_mode=match.group("mode").strip(),
                title=match.group("title").strip(),
                metadata=metadata,
                source=parse_metadata_source(metadata),
                rerank_score=match.group("score").strip(),
                preview=normalize_preview(match.group("preview")),
            )
        )
    return hits


def classify_topic(query: str) -> str:
    lower = query.lower()
    if "foundation large language model" in lower:
        return "基础大语言模型"
    if "image generation model" in lower:
        return "图像生成模型"
    if "video generation model" in lower:
        return "视频生成模型"
    if "agent model" in lower:
        return "Agent 模型"
    if "code generation model" in lower:
        return "代码生成模型"
    if "lmarena" in lower:
        return "LMArena 榜单"
    return "其他"


def build_query_records(section: RagRunSection) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for idx, sub_query in enumerate(section.sub_queries, start=1):
        raw_result = section.raw_results[idx - 1] if idx - 1 < len(section.raw_results) else ""
        records.append(
            QueryRecord(
                index=idx,
                query=sub_query.get("query", ""),
                start_date=sub_query.get("start_date", ""),
                end_date=sub_query.get("end_date", ""),
                category=sub_query.get("category", ""),
                topic=classify_topic(sub_query.get("query", "")),
                hits=parse_hits(raw_result),
            )
        )
    return records


def load_event_stats(events_path: Path) -> dict:
    stats = {
        "rag_tool_start": 0,
        "rag_tool_end": 0,
        "first_rag_ts": None,
        "last_rag_ts": None,
    }
    if not events_path.exists():
        return stats

    with events_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            if event.get("node") != "execute" or event.get("tool") != "rag_search":
                continue

            if event.get("type") == "tool_start":
                stats["rag_tool_start"] += 1
                stats["first_rag_ts"] = stats["first_rag_ts"] or event.get("ts")
            elif event.get("type") == "tool_end":
                stats["rag_tool_end"] += 1
                stats["last_rag_ts"] = event.get("ts")
    return stats


def format_duration(start_ts: str | None, end_ts: str | None) -> str:
    if not start_ts or not end_ts:
        return "未知"
    start = datetime.fromisoformat(start_ts)
    end = datetime.fromisoformat(end_ts)
    duration = end - start
    return str(duration)


def build_summary_lines(sections: Iterable[RagRunSection], event_stats: dict, log_dir: Path) -> list[str]:
    section_list = list(sections)
    total_queries = sum(len(section.sub_queries) for section in section_list)
    total_hits = 0
    topic_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()

    for section in section_list:
        for record in build_query_records(section):
            topic_counter[record.topic] += 1
            total_hits += len(record.hits)
            for hit in record.hits:
                source_counter[hit.source] += 1

    lines = [
        "# RAG 运行记录解析",
        "",
        "## 总览",
        "",
        f"- 日志目录: `{log_dir}`",
        f"- RAG 子图数量: {len(section_list)}",
        f"- 子查询总数: {total_queries}",
        f"- 结果命中总数: {total_hits}",
        f"- `rag_search` 调用次数: {event_stats['rag_tool_end']} 次",
        f"- 执行窗口: {event_stats['first_rag_ts'] or '未知'} -> {event_stats['last_rag_ts'] or '未知'}",
        f"- 执行耗时: {format_duration(event_stats['first_rag_ts'], event_stats['last_rag_ts'])}",
        "",
        "## 查询主题分布",
        "",
    ]

    for topic, count in topic_counter.items():
        lines.append(f"- {topic}: {count} 个子查询")

    lines.extend(["", "## 命中来源 Top 10", ""])
    for source, count in source_counter.most_common(10):
        lines.append(f"- {source}: {count} 条命中")

    return lines


def build_section_markdown(section: RagRunSection) -> list[str]:
    records = build_query_records(section)
    lines = [
        "",
        f"## RAG 子图：`{section.namespace}`",
        "",
        f"- 研究主题: {section.research_topic}",
        f"- 子查询数: {len(records)}",
        f"- `raw_results` 数量: {len(section.raw_results)}",
        "",
        "### 查询概览",
        "",
        "| # | 主题 | 时间范围 | Query | Top1 标题 | Top1 来源 | 命中数 |",
        "|---:|---|---|---|---|---|---:|",
    ]

    for record in records:
        top1 = record.hits[0] if record.hits else None
        top1_title = top1.title if top1 else "无结果"
        top1_source = top1.source if top1 else "-"
        lines.append(
            f"| {record.index} | {record.topic} | {record.start_date} ~ {record.end_date} | "
            f"`{record.query}` | {escape_pipes(top1_title)} | {escape_pipes(top1_source)} | {len(record.hits)} |"
        )

    lines.extend(["", "### 逐条查询结果", ""])
    for record in records:
        lines.extend(build_record_markdown(record))

    lines.extend(
        [
            "",
            "### 子图压缩摘要",
            "",
            "<details>",
            "<summary>展开查看 compress 阶段输出</summary>",
            "",
            section.compressed_research.strip(),
            "",
            "</details>",
        ]
    )
    return lines


def escape_pipes(text: str) -> str:
    return text.replace("|", "\\|")


def build_record_markdown(record: QueryRecord) -> list[str]:
    lines = [
        f"#### {record.index:02d}. {record.topic} | {record.start_date} ~ {record.end_date}",
        "",
        f"- Query: `{record.query}`",
        f"- Category: `{record.category or '未指定'}`",
        f"- 命中数: {len(record.hits)}",
    ]

    if record.hits:
        top1 = record.hits[0]
        lines.append(
            f"- Top1: {top1.title}（来源：{top1.source}，Rerank: {top1.rerank_score}，检索方式：{top1.retrieval_mode}）"
        )
    else:
        lines.append("- Top1: 无")

    lines.extend(
        [
            "",
            "<details>",
            f"<summary>展开查看该查询的 {len(record.hits)} 条命中</summary>",
            "",
        ]
    )

    if not record.hits:
        lines.extend(["无结果。", "", "</details>", ""])
        return lines

    lines.extend(
        [
            "| 排名 | 检索方式 | 来源 | Rerank | 标题 |",
            "|---:|---|---|---:|---|",
        ]
    )
    for hit in record.hits:
        lines.append(
            f"| {hit.rank} | {escape_pipes(hit.retrieval_mode)} | {escape_pipes(hit.source)} | "
            f"{escape_pipes(hit.rerank_score)} | {escape_pipes(hit.title)} |"
        )

    lines.append("")
    for hit in record.hits:
        lines.extend(
            [
                f"**命中 {hit.rank}. {hit.title}**",
                "",
                f"- 检索方式: `{hit.retrieval_mode}`",
                f"- 来源: {hit.metadata}",
                f"- Rerank分数: `{hit.rerank_score}`",
                "- 预览:",
                "",
                "```text",
                hit.preview,
                "```",
                "",
            ]
        )

    lines.extend(["</details>", ""])
    return lines


def render_markdown(log_dir: Path, sections: list[RagRunSection], event_stats: dict) -> str:
    lines = build_summary_lines(sections, event_stats, log_dir)
    for section in sections:
        lines.extend(build_section_markdown(section))
    return "\n".join(lines).strip() + "\n"


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
    events_path = log_dir / "events.jsonl"
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"缺少 checkpoints.db: {checkpoints_path}")

    sections = load_sections(checkpoints_path)
    event_stats = load_event_stats(events_path)
    markdown = render_markdown(log_dir, sections, event_stats)

    output_path = Path(args.output).expanduser() if args.output else log_dir / "RAG_RUN_README.md"
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()
    output_path.write_text(markdown, encoding="utf-8")

    total_queries = sum(len(section.sub_queries) for section in sections)
    print(f"✅ 已导出 RAG README: {output_path}")
    print(f"   - RAG 子图: {len(sections)}")
    print(f"   - 子查询: {total_queries}")
    print(f"   - rag_search 调用: {event_stats['rag_tool_end']}")


if __name__ == "__main__":
    main()
