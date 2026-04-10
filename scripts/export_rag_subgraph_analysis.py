"""导出 RAG 子图成功运行的详细分析报告。

重点解析：
- sub_queries.json：计划出的子查询
- events.jsonl：每个子查询的 rag_search / think_tool / 反馈 / execute 边界
- run_meta.json：总耗时与摘要长度
- compressed.md：最终压缩摘要

输出：
- 默认写入 <log_dir>/RAG_SUBGRAPH_ANALYSIS.md

用法：
    python scripts/export_rag_subgraph_analysis.py <log_dir>
    python scripts/export_rag_subgraph_analysis.py <log_dir> --output <output_path>
    python scripts/export_rag_subgraph_analysis.py
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "src")))
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "rag")))


RESULT_BLOCK_RE = re.compile(
    r"^--- 结果 (?P<rank>\d+) \[(?P<mode>.*?)\] ---\n"
    r"标题: (?P<title>.*?)\n"
    r"元数据: (?P<meta>.*?)\n"
    r"Rerank分数: (?P<score>[^\n]+)\n"
    r"预览: (?P<preview>.*?)(?=^--- 结果 \d+ \[|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class ResultHit:
    rank: int
    mode: str
    title: str
    metadata: str
    score: str
    preview: str


@dataclass
class SearchIteration:
    args: dict
    result_text: str = ""
    hits: list[ResultHit] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    index: int
    sub_query: dict
    searches: list[SearchIteration] = field(default_factory=list)
    thinks: list[str] = field(default_factory=list)
    feedbacks: list[str] = field(default_factory=list)
    final_result_text: str = ""
    final_hits: list[ResultHit] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 RAG 子图详细分析报告")
    parser.add_argument("log_dir", nargs="?", help="运行日志目录，例如 logs/某次运行目录")
    parser.add_argument("--output", help="输出 Markdown 路径，默认写入 <log_dir>/RAG_SUBGRAPH_ANALYSIS.md")
    return parser.parse_args()


def find_latest_log_dir(logs_root: Path) -> Path:
    candidates = [
        p for p in logs_root.iterdir()
        if p.is_dir() and (p / "events.jsonl").exists() and (p / "sub_queries.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"在 {logs_root} 下未找到包含 events.jsonl + sub_queries.json 的运行目录")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_tool_args(raw: str) -> dict:
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return {"raw": raw}


def parse_hits(result_text: str) -> list[ResultHit]:
    hits: list[ResultHit] = []
    for match in RESULT_BLOCK_RE.finditer(result_text):
        hits.append(
            ResultHit(
                rank=int(match.group("rank")),
                mode=match.group("mode").strip(),
                title=match.group("title").strip(),
                metadata=match.group("meta").strip(),
                score=match.group("score").strip(),
                preview=match.group("preview").strip(),
            )
        )
    return hits


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_full_raw_results(log_dir: Path) -> list[str]:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from rag_subgraph import rag_researcher_builder

    thread_id = log_dir.name
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}

    async def _load() -> list[str]:
        async with AsyncSqliteSaver.from_conn_string(str(log_dir / "checkpoints.db")) as cp:
            graph = rag_researcher_builder.compile(checkpointer=cp)
            state = await graph.aget_state(config)
            return state.values.get("raw_results", [])

    return asyncio.run(_load())


def build_query_analyses(log_dir: Path) -> list[QueryAnalysis]:
    sub_queries = load_json(log_dir / "sub_queries.json", [])
    analyses = [QueryAnalysis(index=i, sub_query=q) for i, q in enumerate(sub_queries, start=1)]
    raw_results = load_full_raw_results(log_dir)
    for idx, result_text in enumerate(raw_results):
        if idx >= len(analyses):
            break
        analyses[idx].final_result_text = result_text
        analyses[idx].final_hits = parse_hits(result_text)

    events_path = log_dir / "events.jsonl"
    current: QueryAnalysis | None = None
    search_index = 0

    with events_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            evt = json.loads(line)
            evt_type = evt.get("type")
            node = evt.get("node")
            tool = evt.get("tool")

            if evt_type == "tool_start" and tool == "rag_search":
                if current is None:
                    current = analyses[len([a for a in analyses if a.searches or a.thinks or a.feedbacks])]
                current.searches.append(SearchIteration(args=parse_tool_args(evt.get("args", ""))))
                search_index += 1
                continue

            if evt_type == "tool_end" and tool == "rag_search":
                if current and current.searches:
                    current.searches[-1].result_text = evt.get("result", "")
                    current.searches[-1].hits = parse_hits(evt.get("result", ""))
                continue

            if evt_type == "tool_end" and tool == "think_tool":
                if current is None:
                    current = analyses[len([a for a in analyses if a.searches or a.thinks or a.feedbacks])]
                current.thinks.append(str(evt.get("result", "")).strip())
                continue

            if evt_type == "llm_end" and node == "rag_researcher":
                content = str(evt.get("content", "")).strip()
                if content:
                    if current is None:
                        current = analyses[len([a for a in analyses if a.searches or a.thinks or a.feedbacks])]
                    current.feedbacks.append(content)
                continue

            if evt_type == "node_start" and node == "execute":
                current = None

    return analyses


def summarize_patterns(analyses: list[QueryAnalysis]) -> Counter[tuple[int, int, int]]:
    return Counter((len(a.searches), len(a.thinks), len(a.feedbacks)) for a in analyses)


def collect_findings(analyses: list[QueryAnalysis]) -> list[str]:
    findings: list[str] = []

    extra_search = [a.index for a in analyses if len(a.searches) > 1]
    if extra_search:
        findings.append(f"发生补充检索的子查询: Q{', Q'.join(f'{i:02d}' for i in extra_search)}")

    user_style_feedback = [a.index for a in analyses if a.feedbacks]
    if user_style_feedback:
        findings.append(f"出现额外自然语言反馈的子查询: Q{', Q'.join(f'{i:02d}' for i in user_style_feedback)}")

    leaderboard_leak = []
    for a in analyses:
        if a.index == 21:
            continue
        if any("lmarena" in json.dumps(s.args, ensure_ascii=False).lower() for s in a.searches[1:]):
            leaderboard_leak.append(a.index)
    if leaderboard_leak:
        findings.append(
            "非榜单子查询中出现 Lmarena 补搜: "
            + ", ".join(f"Q{i:02d}" for i in leaderboard_leak)
        )

    return findings


def format_search_args(args: dict) -> str:
    ordered = []
    for key in ("query", "start_date", "end_date", "category", "top_k", "days"):
        if key in args and args[key] not in ("", None):
            ordered.append(f"{key}={args[key]}")
    remaining = [k for k in args.keys() if k not in {"query", "start_date", "end_date", "category", "top_k", "days"}]
    for key in remaining:
        ordered.append(f"{key}={args[key]}")
    return ", ".join(ordered)


def truncate(text: str, limit: int = 220) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def render_markdown(log_dir: Path, analyses: list[QueryAnalysis]) -> str:
    meta = load_json(log_dir / "run_meta.json", {})
    compressed = (log_dir / "compressed.md").read_text(encoding="utf-8") if (log_dir / "compressed.md").exists() else ""

    event_counts = Counter()
    for line in (log_dir / "events.jsonl").open("r", encoding="utf-8"):
        evt = json.loads(line)
        event_counts[evt.get("type", "?")] += 1

    lines = [
        "# RAG 子图运行分析报告",
        "",
        "## 总览",
        "",
        f"- 日志目录: `{log_dir}`",
        f"- 研究主题: {meta.get('topic', '')}",
        f"- 总耗时: {meta.get('elapsed_seconds', '未知')}s",
        f"- 子查询数量: {len(analyses)}",
        f"- 原始笔记: {meta.get('raw_notes_count', 0)}",
        f"- 压缩摘要长度: {meta.get('compressed_length', 0)} 字符",
        f"- 事件统计: node_start={event_counts.get('node_start', 0)}, llm_start={event_counts.get('llm_start', 0)}, llm_end={event_counts.get('llm_end', 0)}, tool_start={event_counts.get('tool_start', 0)}, tool_end={event_counts.get('tool_end', 0)}",
        "",
        "## 计划出的 Query",
        "",
    ]

    for a in analyses:
        q = a.sub_query
        lines.append(
            f"- Q{a.index:02d}: `{q.get('start_date', '')} ~ {q.get('end_date', '')}` | `{q.get('query', '')}` | `{q.get('category', '')}`"
        )

    lines.extend(["", "## 迭代模式统计", ""])
    pattern_counter = summarize_patterns(analyses)
    for (searches, thinks, feedbacks), count in sorted(pattern_counter.items()):
        lines.append(
            f"- `{searches} 次 rag_search + {thinks} 次 think_tool + {feedbacks} 段反馈`: {count} 个子查询"
        )

    findings = collect_findings(analyses)
    if findings:
        lines.extend(["", "## 关键观察", ""])
        for finding in findings:
            lines.append(f"- {finding}")

    lines.extend(["", "## 逐 Query 复盘", ""])
    for a in analyses:
        q = a.sub_query
        lines.extend(
            [
                f"### Q{a.index:02d}",
                "",
                f"- 时间范围: `{q.get('start_date', '')} ~ {q.get('end_date', '')}`",
                f"- 计划查询: `{q.get('query', '')}`",
                f"- 分类: `{q.get('category', '')}`",
                f"- 检索次数: {len(a.searches)}",
                f"- 反思次数: {len(a.thinks)}",
                f"- 额外反馈: {len(a.feedbacks)}",
                "",
            ]
        )

        for idx, search in enumerate(a.searches, start=1):
            lines.append(f"#### Q{a.index:02d} Search {idx}")
            lines.append("")
            lines.append(f"- 参数: `{format_search_args(search.args)}`")
            hits = search.hits
            if idx == len(a.searches) and a.final_hits:
                hits = a.final_hits
            if hits:
                lines.append("- Top3 命中:")
                for hit in hits[:3]:
                    lines.append(
                        f"  - #{hit.rank} [{hit.mode}] {hit.title} | {hit.metadata} | score={hit.score}"
                    )
            else:
                lines.append("- Top3 命中: 无法从结果中解析结构化命中块")
            lines.append("")

        for idx, think in enumerate(a.thinks, start=1):
            lines.append(f"- Think {idx}: {truncate(think, 500)}")
        for idx, feedback in enumerate(a.feedbacks, start=1):
            lines.append(f"- Feedback {idx}: {truncate(feedback, 500)}")
        lines.append("")

    lines.extend(
        [
            "## 最终压缩摘要节选",
            "",
            "```text",
            compressed[:3000].strip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logs_root = Path("logs")
    log_dir = Path(args.log_dir) if args.log_dir else find_latest_log_dir(logs_root)
    output_path = Path(args.output) if args.output else (log_dir / "RAG_SUBGRAPH_ANALYSIS.md")

    analyses = build_query_analyses(log_dir)
    markdown = render_markdown(log_dir, analyses)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
