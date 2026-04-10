"""导出 RAG 子图运行记录（v2，适配 Send 并行架构）。

通过 LangGraph aget_state 读取完整累积状态，解析每个子查询的
多轮搜索结果和重试信息，生成 Markdown 报告。

用法：
    python scripts/export_rag_run_v2.py <log_dir>
    python scripts/export_rag_run_v2.py              # 自动选最近的
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

# ── 结果解析 ──

RESULT_BLOCK_RE = re.compile(
    r"^--- 结果 (?P<rank>\d+) \[(?P<mode>.*?)\] ---\n"
    r"标题: (?P<title>.*?)\n"
    r"元数据: (?P<meta>.*?)\n"
    r"Rerank分数: (?P<score>[^\n]+)\n"
    r"预览: (?P<preview>.*?)(?=^--- 结果 \d+ \[|\Z)",
    re.MULTILINE | re.DOTALL,
)

ROUND_RE = re.compile(r"^\[第(\d+)轮\] 查询: (.+?)$", re.MULTILINE)


@dataclass
class Hit:
    rank: int
    mode: str
    title: str
    metadata: str
    score: str
    preview: str


@dataclass
class SearchRound:
    attempt: int
    query: str
    hits: list[Hit] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class SubQueryResult:
    original_query: str
    rounds: list[SearchRound] = field(default_factory=list)


def parse_hits(text: str) -> list[Hit]:
    return [
        Hit(
            rank=int(m.group("rank")),
            mode=m.group("mode").strip(),
            title=m.group("title").strip(),
            metadata=m.group("meta").strip(),
            score=m.group("score").strip(),
            preview=m.group("preview").strip()[:300],
        )
        for m in RESULT_BLOCK_RE.finditer(text)
    ]


def parse_raw_result(raw: str) -> SubQueryResult:
    """解析单个 raw_result 条目，格式：
    --- 查询: <原始query> ---
    [第1轮] 查询: <query>
    <搜索结果...>

    [第2轮] 查询: <refined_query>
    <搜索结果...>
    """
    # 提取原始查询
    header_match = re.match(r"^--- 查询: (.+?) ---\n", raw)
    original_query = header_match.group(1) if header_match else "未知"
    body = raw[header_match.end():] if header_match else raw

    # 按 [第N轮] 分割
    round_matches = list(ROUND_RE.finditer(body))
    result = SubQueryResult(original_query=original_query)

    if not round_matches:
        # 没有轮次标记，整体作为一轮
        result.rounds.append(SearchRound(
            attempt=1, query=original_query,
            hits=parse_hits(body), raw_text=body,
        ))
        return result

    for i, rm in enumerate(round_matches):
        start = rm.end()
        end = round_matches[i + 1].start() if i + 1 < len(round_matches) else len(body)
        chunk = body[start:end].strip()
        result.rounds.append(SearchRound(
            attempt=int(rm.group(1)),
            query=rm.group(2),
            hits=parse_hits(chunk),
            raw_text=chunk,
        ))

    return result


# ── 状态加载 ──

def load_state_via_graph(log_dir: Path) -> dict:
    """通过 LangGraph aget_state 读取完整累积状态。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from rag_subgraph import rag_researcher_builder

    thread_id = log_dir.name
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}

    async def _load():
        async with AsyncSqliteSaver.from_conn_string(
            str(log_dir / "checkpoints.db")
        ) as cp:
            graph = rag_researcher_builder.compile(checkpointer=cp)
            state = await graph.aget_state(config)
            return dict(state.values)

    return asyncio.run(_load())


# ── Markdown 渲染 ──

def render_markdown(
    log_dir: Path, state: dict, sub_results: list[SubQueryResult],
    meta: dict, compressed: str,
) -> str:
    lines = [
        "# RAG 子图运行分析报告",
        "",
        "## 总览",
        "",
        f"- 日志目录: `{log_dir.name}`",
        f"- 研究主题: {state.get('research_topic', '')}",
        f"- 总耗时: {meta.get('elapsed', '未知')}s",
        f"- 子查询数量: {len(sub_results)}",
        f"- 搜索总轮次: {sum(len(r.rounds) for r in sub_results)}",
        f"- 压缩摘要长度: {len(compressed)} 字符",
        "",
    ]

    # 查询概览表
    lines.extend([
        "## 查询概览",
        "",
        "| # | 原始 Query | 轮次 | 首轮命中 | 是否重试 | 最终 Query |",
        "|--:|-----------|:---:|:------:|:------:|-----------|",
    ])
    for i, sr in enumerate(sub_results, 1):
        n_rounds = len(sr.rounds)
        first_hits = len(sr.rounds[0].hits) if sr.rounds else 0
        retried = "🔄 是" if n_rounds > 1 else "✅ 否"
        final_q = sr.rounds[-1].query if sr.rounds else sr.original_query
        final_display = final_q if final_q == sr.original_query else f"~~{sr.original_query}~~ → {final_q}"
        lines.append(
            f"| {i} | `{sr.original_query}` | {n_rounds} | {first_hits} | {retried} | {final_display} |"
        )

    # 逐 Query 详情
    lines.extend(["", "## 逐 Query 详情", ""])
    for i, sr in enumerate(sub_results, 1):
        lines.extend([
            f"### Q{i:02d}: `{sr.original_query}`",
            "",
        ])
        for rd in sr.rounds:
            query_note = f" (改写)" if rd.query != sr.original_query else ""
            lines.extend([
                f"#### 第 {rd.attempt} 轮{query_note}",
                "",
                f"- 查询: `{rd.query}`",
                f"- 命中数: {len(rd.hits)}",
            ])
            if rd.hits:
                lines.append("- Top 3:")
                for h in rd.hits[:3]:
                    lines.append(
                        f"  - #{h.rank} [{h.mode}] **{h.title}** (score={h.score})"
                    )
            else:
                lines.append("- 命中: 无结构化结果")
            lines.append("")

        # 折叠显示原始文本
        if sr.rounds:
            lines.extend([
                "<details>",
                f"<summary>展开 Q{i:02d} 原始搜索结果</summary>",
                "",
                "```text",
                "\n".join(rd.raw_text[:1000] for rd in sr.rounds),
                "```",
                "",
                "</details>",
                "",
            ])

    # 压缩摘要
    lines.extend([
        "## 压缩摘要",
        "",
        compressed.strip(),
        "",
    ])

    return "\n".join(lines)


# ── 主流程 ──

def find_latest_log_dir(logs_root: Path) -> Path:
    candidates = [
        p for p in logs_root.iterdir()
        if p.is_dir() and (p / "checkpoints.db").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"在 {logs_root} 下未找到运行目录")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="导出 RAG 子图运行报告 (v2)")
    parser.add_argument("log_dir", nargs="?", help="运行日志目录")
    parser.add_argument("--output", help="输出路径，默认 <log_dir>/RAG_REPORT.md")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    logs_root = repo_root / "logs"

    log_dir = Path(args.log_dir) if args.log_dir else find_latest_log_dir(logs_root)
    if not log_dir.is_absolute():
        log_dir = (repo_root / log_dir).resolve()

    # 加载状态
    state = load_state_via_graph(log_dir)
    raw_results = state.get("raw_results", [])

    # 加载元数据
    meta_path = log_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    compressed_path = log_dir / "compressed.md"
    compressed = compressed_path.read_text() if compressed_path.exists() else state.get("compressed_research", "")

    # 解析
    sub_results = [parse_raw_result(r) for r in raw_results]

    # 渲染
    markdown = render_markdown(log_dir, state, sub_results, meta, compressed)

    output_path = Path(args.output) if args.output else (log_dir / "RAG_REPORT.md")
    output_path.write_text(markdown, encoding="utf-8")

    print(f"✅ 报告已生成: {output_path}")
    print(f"   - 子查询: {len(sub_results)}")
    print(f"   - 搜索轮次: {sum(len(r.rounds) for r in sub_results)}")
    print(f"   - 重试子查询: {sum(1 for r in sub_results if len(r.rounds) > 1)}")


if __name__ == "__main__":
    main()
