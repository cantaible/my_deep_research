"""解析 RAG 子图运行记录，逐一还原每次 execute 的查询与结果。

由于并发搜索中 tool_start / tool_end 返回顺序不一致，
本脚本以 (tool_end, evaluate llm_end) 时序配对为基准，
再通过 tool_start.args.query 匹配回原始查询。

用法:
    python scripts/analyze_rag_run.py <logs_dir> [--output report.md]

示例:
    python scripts/analyze_rag_run.py "logs/ragtest搜索本地新闻数据库查找2026年3月1-20260410-124455"
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path


# ── 工具函数 ──────────────────────────────────────────────

def pts(ts_str: str) -> datetime:
    """解析 ISO 时间戳"""
    return datetime.fromisoformat(ts_str)


def duration_str(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def truncate(s: str, maxlen: int = 120) -> str:
    return s[:maxlen] + "..." if len(s) > maxlen else s


def parse_result_blocks(raw: str) -> list[dict]:
    """从 tool_end 的 result 字符串中解析结构化搜索结果。"""
    results = []
    chunks = raw.split("--- 结果 ")
    for chunk in chunks[1:]:
        lines = chunk.strip().split("\n")
        item = {}
        # 第一行: "1 [向量+OpenSearch] ---"
        header = lines[0] if lines else ""
        if "[" in header and "]" in header:
            item["retrieval_method"] = header.split("[", 1)[1].split("]", 1)[0]
        idx_part = header.split("[")[0].strip().split(" ")[0] if header else ""
        item["index"] = idx_part

        for line in lines[1:]:
            line = line.strip()
            if line.startswith("标题:"):
                item["title"] = line[len("标题:"):].strip()
            elif line.startswith("元数据:"):
                item["metadata"] = line[len("元数据:"):].strip()
            elif line.startswith("Rerank分数:"):
                try:
                    item["rerank_score"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    item["rerank_score"] = line.split(":", 1)[1].strip()
            elif line.startswith("预览:"):
                item["preview"] = line[len("预览:"):].strip()

        if item.get("title"):
            results.append(item)
    return results


# ── 主解析逻辑 ──────────────────────────────────────────────

def parse_run(log_dir: Path) -> dict:
    """解析一次 RAG 运行的全部事件，返回结构化结果。"""

    with open(log_dir / "events.jsonl") as f:
        events = [json.loads(line) for line in f]
    with open(log_dir / "run_meta.json") as f:
        meta = json.load(f)
    with open(log_dir / "sub_queries.json") as f:
        sub_queries = json.load(f)
    with open(log_dir / "compressed.md") as f:
        compressed = f.read()

    # ── 1. 收集各类事件（按时间排序） ──
    tool_starts = sorted(
        [e for e in events if e["type"] == "tool_start" and e["node"] == "execute"],
        key=lambda e: e["ts"],
    )
    tool_ends = sorted(
        [e for e in events if e["type"] == "tool_end" and e["node"] == "execute"],
        key=lambda e: e["ts"],
    )
    eval_llm_ends = sorted(
        [e for e in events if e["type"] == "llm_end" and e["node"] == "execute"],
        key=lambda e: e["ts"],
    )
    compress_usage = None
    for e in events:
        if e["type"] == "llm_end" and e["node"] == "compress":
            compress_usage = e.get("usage", {})

    # ── 2. 提取所有 tool_start 的 query（建立 query → tool_start 映射）──
    def parse_args(args_str):
        try:
            return eval(args_str) if isinstance(args_str, str) else args_str
        except Exception:
            return {}

    # 用于消费的 tool_start 列表（同一 query 可能有多个 retry）
    available_starts = []
    for ts_evt in tool_starts:
        args = parse_args(ts_evt.get("args", "{}"))
        available_starts.append({
            "ts": ts_evt["ts"],
            "query": args.get("query", "?"),
            "start_date": args.get("start_date", "?"),
            "end_date": args.get("end_date", "?"),
            "category": args.get("category", "?"),
            "top_k": args.get("top_k", "?"),
            "consumed": False,
        })

    # ── 3. 核心配对：tool_end 和 evaluate 按时序 1:1 配对 ──
    # tool_end[i] 对应 eval_llm_ends[i]（都按时间排序，每个搜索完成后立即评估）
    search_calls = []
    for i, te_evt in enumerate(tool_ends):
        result_raw = te_evt.get("result", "")
        results_parsed = parse_result_blocks(result_raw)

        # 取对应的 evaluate
        evaluate = None
        if i < len(eval_llm_ends):
            content = eval_llm_ends[i].get("content", "")
            try:
                evaluate = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                evaluate = {"raw": content}

        # 匹配回 tool_start：找 result_raw 或 evaluate 能匹配的 query
        # 策略：找时间戳早于 tool_end 且尚未消费的 tool_start
        # 由于并发，优先用 evaluate.reason 来匹配
        matched_start = None

        # 候选 = 所有时间 < tool_end 且未消费的 starts
        te_ts = pts(te_evt["ts"])
        candidates = [
            (idx, s) for idx, s in enumerate(available_starts)
            if not s["consumed"] and pts(s["ts"]) < te_ts
        ]

        if len(candidates) == 1:
            # 唯一匹配
            idx, s = candidates[0]
            matched_start = s
            available_starts[idx]["consumed"] = True
        elif len(candidates) > 1:
            # 多个候选：用 evaluate.reason 中的关键词来匹配
            eval_reason = (evaluate or {}).get("reason", "")
            best_idx = None
            best_score = -1
            for idx, s in candidates:
                # 简单评分：reason 中包含 query 关键词的数量
                query_words = set(s["query"].replace("，", " ").replace("、", " ").split())
                score = sum(1 for w in query_words if len(w) > 1 and w in eval_reason)
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    matched_start = s
            if best_idx is not None:
                available_starts[best_idx]["consumed"] = True

        call = {
            "ts_end": te_evt["ts"],
            "ts_eval": eval_llm_ends[i]["ts"] if i < len(eval_llm_ends) else None,
            "results_raw": result_raw,
            "results_parsed": results_parsed,
            "evaluate": evaluate,
        }

        if matched_start:
            call.update({
                "ts_start": matched_start["ts"],
                "query": matched_start["query"],
                "start_date": matched_start["start_date"],
                "end_date": matched_start["end_date"],
                "category": matched_start["category"],
                "top_k": matched_start["top_k"],
            })
        else:
            call.update({
                "ts_start": None,
                "query": "（未匹配到原始查询）",
                "start_date": "?",
                "end_date": "?",
                "category": "?",
                "top_k": "?",
            })

        search_calls.append(call)

    # ── 4. 按轮次分组（基于 tool_end 时间间隔 > 10s） ──
    rounds = []
    current_round = []
    for call in search_calls:
        if current_round:
            prev_ts = pts(current_round[-1]["ts_end"])
            curr_ts = pts(call["ts_end"])
            if (curr_ts - prev_ts).total_seconds() > 10:
                rounds.append(current_round)
                current_round = []
        current_round.append(call)
    if current_round:
        rounds.append(current_round)

    return {
        "meta": meta,
        "sub_queries": sub_queries,
        "compressed": compressed,
        "compress_usage": compress_usage,
        "search_calls": search_calls,
        "rounds": rounds,
        "total_events": len(events),
    }


# ── 报告生成 ──────────────────────────────────────────────

def generate_report(data: dict) -> str:
    meta = data["meta"]
    sub_queries = data["sub_queries"]
    compressed = data["compressed"]
    compress_usage = data["compress_usage"] or {}
    search_calls = data["search_calls"]
    rounds = data["rounds"]

    lines: list[str] = []

    def w(s=""):
        lines.append(s)

    w("# RAG 子图运行分析报告")
    w()
    w(f"> **研究主题**: {meta['topic']}")
    w(f"> **运行 ID**: `{meta['thread_id']}`")
    w(f"> **总耗时**: {meta['elapsed_seconds']}s | **子查询数**: {len(sub_queries)} "
      f"| **搜索调用数**: {len(search_calls)} | **摘要长度**: {meta['compressed_length']} 字符")
    w()

    # ── 子查询计划 ──
    w("---")
    w()
    w("## 1. 子查询计划 (Plan)")
    w()
    w("| # | 搜索意图 | 查询关键词 | 时间范围 |")
    w("|---|---|---|---|")
    for i, sq in enumerate(sub_queries):
        intent = sq.get("search_intent", "")
        query = sq.get("query", "")
        date_range = f"{sq.get('start_date', '?')} ~ {sq.get('end_date', '?')}"
        w(f"| Q{i + 1} | {intent} | {query} | {date_range} |")
    w()

    # ── 逐轮解析 ──
    w("---")
    w()
    w("## 2. Execute 阶段逐轮详情")
    w()

    global_call_idx = 0
    for round_idx, round_calls in enumerate(rounds):
        w(f"### Round {round_idx + 1}（{len(round_calls)} 条搜索）")
        w()

        for call in round_calls:
            global_call_idx += 1

            # 耗时
            if call.get("ts_start") and call.get("ts_end"):
                dur = (pts(call["ts_end"]) - pts(call["ts_start"])).total_seconds()
                dur_s = duration_str(dur)
            else:
                dur_s = "?"

            evaluate = call.get("evaluate") or {}
            quality = evaluate.get("quality", "?")
            reason = evaluate.get("reason", "")
            refined = evaluate.get("refined_query")

            if quality == "good":
                emoji = "✅"
            elif quality == "insufficient":
                emoji = "🔄"
            else:
                emoji = "❓"

            w(f"#### 搜索 #{global_call_idx}  {emoji} `{quality}`")
            w()
            w(f"- **查询**: {call['query']}")
            w(f"- **时间范围**: {call['start_date']} ~ {call['end_date']}  |  "
              f"**类目**: {call['category']}  |  **top_k**: {call['top_k']}")
            w(f"- **搜索耗时**: {dur_s}")
            w()

            # 搜索结果
            results = call.get("results_parsed", [])
            if results:
                w(f"**返回 {len(results)} 条结果:**")
                w()
                w("| # | Rerank 分数 | 检索方式 | 标题 |")
                w("|---|---|---|---|")
                for r in results:
                    score = r.get("rerank_score", "?")
                    score_s = f"{score:.4f}" if isinstance(score, float) else str(score)
                    method = r.get("retrieval_method", "?")
                    title = truncate(r.get("title", "?"), 70)
                    w(f"| {r.get('index', '?')} | {score_s} | {method} | {title} |")
                w()

                # 显示 top-1 预览
                if results[0].get("preview"):
                    w(f"> **Top-1 预览**: {truncate(results[0]['preview'], 150)}")
                    w()
            else:
                w("> ⚠️ 未解析到结构化搜索结果")
                w()

            # 评估
            if reason:
                w(f"**评估理由**: {reason}")
                w()
            if refined:
                w(f"**→ Retry 查询**: `{refined}`")
                w()

            w("---")
            w()

    # ── Compress ──
    w("## 3. Compress 阶段")
    w()
    if compress_usage:
        w("| 指标 | 数值 |")
        w("|---|---|")
        inp = compress_usage.get("input_tokens", "?")
        out = compress_usage.get("output_tokens", "?")
        tot = compress_usage.get("total_tokens", "?")
        w(f"| Input Tokens | {inp:,} |" if isinstance(inp, int) else f"| Input Tokens | {inp} |")
        w(f"| Output Tokens | {out:,} |" if isinstance(out, int) else f"| Output Tokens | {out} |")
        w(f"| Total Tokens | {tot:,} |" if isinstance(tot, int) else f"| Total Tokens | {tot} |")
        w()

    w(f"### 压缩摘要（全文 {len(compressed)} 字符）")
    w()
    w("```")
    w(compressed)
    w("```")
    w()

    # ── 统计 ──
    w("---")
    w()
    w("## 4. 统计汇总")
    w()

    good = sum(1 for c in search_calls if (c.get("evaluate") or {}).get("quality") == "good")
    insuf = sum(1 for c in search_calls if (c.get("evaluate") or {}).get("quality") == "insufficient")
    unknown = len(search_calls) - good - insuf

    w("| 指标 | 数值 |")
    w("|---|---|")
    w(f"| 总搜索次数 | {len(search_calls)} |")
    w(f"| ✅ good | {good} |")
    w(f"| 🔄 insufficient | {insuf} |")
    w(f"| ❓ 未知/无评估 | {unknown} |")
    w(f"| 搜索轮次 | {len(rounds)} |")
    if rounds:
        r1 = len(rounds[0])
        r1_good = sum(
            1 for c in rounds[0]
            if (c.get("evaluate") or {}).get("quality") == "good"
        )
        w(f"| 首轮通过率 | {r1_good}/{r1} ({r1_good / r1 * 100:.0f}%) |")
    w()

    # Rerank 统计
    all_scores = []
    top1_scores = []
    for c in search_calls:
        for j, r in enumerate(c.get("results_parsed", [])):
            s = r.get("rerank_score")
            if isinstance(s, (int, float)):
                all_scores.append(s)
                if j == 0:
                    top1_scores.append(s)

    if all_scores:
        w("### Rerank 分数")
        w()
        w("| 指标 | 全部结果 | Top-1 结果 |")
        w("|---|---|---|")
        w(f"| 数量 | {len(all_scores)} | {len(top1_scores)} |")
        w(f"| 最高 | {max(all_scores):.4f} | {max(top1_scores):.4f} |")
        w(f"| 最低 | {min(all_scores):.4f} | {min(top1_scores):.4f} |")
        w(f"| 平均 | {sum(all_scores) / len(all_scores):.4f} | "
          f"{sum(top1_scores) / len(top1_scores):.4f} |")
        w()

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="解析 RAG 子图运行记录")
    parser.add_argument("log_dir", help="日志目录路径")
    parser.add_argument("--output", "-o", help="输出报告路径（默认写到日志目录下 RAG_REPORT.md）")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"❌ 目录不存在: {log_dir}")
        sys.exit(1)

    print(f"📂 解析日志: {log_dir}")
    data = parse_run(log_dir)

    print(f"  子查询: {len(data['sub_queries'])} 条")
    print(f"  搜索调用: {len(data['search_calls'])} 次")
    print(f"  搜索轮次: {len(data['rounds'])} 轮")

    report = generate_report(data)

    output_path = Path(args.output) if args.output else log_dir / "RAG_REPORT.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"✅ 报告已保存: {output_path}")
    print(f"   大小: {len(report)} 字符")


if __name__ == "__main__":
    main()
