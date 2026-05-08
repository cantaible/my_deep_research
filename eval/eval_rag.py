"""RAG 子图评估脚本：运行完整 Plan-Execute-Compress 流程，然后对照 ground truth 计算指标。

运行主逻辑和 tests/test_rag_subgraph.py 完全一致。
评估部分从搜索结果中提取 article_id，与 eval/article_labels.json 对齐计算 Recall/Precision。

用法：
    python eval/eval_rag.py                                    # 使用默认 topic
    python eval/eval_rag.py --topic "搜索本地新闻数据库..."     # 自定义 topic
    python eval/eval_rag.py --eval-only <run_dir>              # 只对已有运行结果做评估
"""

import argparse
import asyncio
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

# ── 配置 ──

LABELS_FILE = Path(__file__).parent / "article_labels.json"

DEFAULT_TOPIC = (
    "搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻，"
    "看看有哪些新的大模型发布了，尤其要关注头部厂商。"
)

# 只评估 model_release 类文章，二值相关度（相关=1，不相关=0）
RELEVANT_EVENT_TYPES = {"model_release"}


# ── 第一部分：运行 RAG 子图（和 test_rag_subgraph.py 完全一致）──

async def run_rag_subgraph(topic: str) -> tuple[Path, dict]:
    """完整运行 RAG 子图，返回 (run_dir, final_state)。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from rag_subgraph import rag_researcher_builder
    from runner import append_event, make_run_dir, normalize_event

    run_dir = make_run_dir(f"eval-{topic[:20]}")
    thread_id = run_dir.name
    db_path = str(run_dir / "checkpoints.db")
    events_path = run_dir / "events.jsonl"

    print(f"📋 研究主题: {topic}")
    print(f"📁 Run 目录: {run_dir}")
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

        # 保存产出
        compressed = result.get("compressed_research", "")
        raw_notes = result.get("raw_notes", [])
        raw_results = result.get("raw_results", [])
        retrieval_details = result.get("retrieval_details", [])  # 新增：获取检索详情

        with open(run_dir / "compressed.md", "w", encoding="utf-8") as f:
            f.write(compressed or "")
        with open(run_dir / "raw_notes.json", "w", encoding="utf-8") as f:
            json.dump(raw_notes, f, ensure_ascii=False, indent=2)
        with open(run_dir / "raw_results.json", "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)

        # 新增：保存检索详情
        with open(run_dir / "retrieval_details.json", "w", encoding="utf-8") as f:
            json.dump(retrieval_details, f, ensure_ascii=False, indent=2)

        sub_queries = state.values.get("sub_queries", [])
        with open(run_dir / "sub_queries.json", "w", encoding="utf-8") as f:
            json.dump(sub_queries, f, ensure_ascii=False, indent=2)

    elapsed = (datetime.now() - start_time).total_seconds()

    meta = {
        "topic": topic,
        "thread_id": thread_id,
        "elapsed_seconds": round(elapsed, 1),
        "compressed_length": len(compressed),
        "raw_notes_count": len(raw_notes),
        "raw_results_count": len(raw_results),
        "sub_query_count": len(sub_queries),
        "retrieval_details_count": len(retrieval_details),  # 新增：记录检索详情数量
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n⏱️  耗时: {elapsed:.1f}s")
    print(f"📁 结果已保存: {run_dir}")
    return run_dir, result


# ── 第二部分：提取检索到的 article_id ──

def extract_article_ids(raw_results: list[str]) -> list[dict]:
    """从 raw_results 文本中提取所有检索到的 article_id 和 rerank 分数。"""
    pattern = re.compile(
        r"ArticleID:\s*(\d+).*?Rerank分数:\s*([\d.]+)",
        re.DOTALL,
    )
    hits = []
    seen = set()
    for result_text in raw_results:
        for match in pattern.finditer(result_text):
            art_id = int(match.group(1))
            score = float(match.group(2))
            if art_id not in seen:
                hits.append({"article_id": art_id, "rerank_score": score})
                seen.add(art_id)
    return hits


# ── 第三部分：计算评估指标（基于事件级别）──


def _build_event_groups(labels: dict[int, dict], relevant_types: set[str]) -> dict[str, list[int]]:
    """将 model_release 文章按 release_event 分组。

    返回 {event_name: [article_id, ...]} 映射。
    """
    groups: dict[str, list[int]] = {}
    for art_id, info in labels.items():
        if info["event_type"] not in relevant_types:
            continue
        event_name = info.get("release_event", "")
        if not event_name:
            # 没有 release_event 的 fallback：用 entities 拼接
            event_name = "_".join(info.get("entities", ["unknown"]))
        groups.setdefault(event_name, []).append(art_id)
    return groups


def _compute_ndcg(ranked_hits: list[dict], labels: dict,
                  relevant_types: set[str], k: int = 0) -> float:
    """计算 NDCG@K（二值相关度：model_release=1, 其他=0）。"""
    if not ranked_hits:
        return 0.0
    if k <= 0:
        k = len(ranked_hits)
    k = min(k, len(ranked_hits))

    def _rel(art_id: int) -> int:
        if art_id not in labels:
            return 0
        return 1 if labels[art_id]["event_type"] in relevant_types else 0

    # 实际 DCG
    dcg = 0.0
    for i in range(k):
        rel = _rel(ranked_hits[i]["article_id"])
        dcg += rel / math.log2(i + 2)

    # Ideal DCG：把全库所有相关文章排最前
    total_relevant = sum(1 for info in labels.values() if info["event_type"] in relevant_types)
    idcg = 0.0
    for i in range(min(k, total_relevant)):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    retrieved: list[dict],
    labels: dict[int, dict],
    relevant_types: set[str],
) -> dict:
    """基于事件级别的评估指标。

    核心逻辑：
    - 将 model_release 文章按 release_event 分组（72 个独立事件）
    - 检索到某事件的任意一篇报道 → 该事件被命中
    - Event Recall = 命中事件数 / 总事件数
    """

    # 事件分组
    event_groups = _build_event_groups(labels, relevant_types)
    total_events = len(event_groups)

    # 检索到的文章
    retrieved_ids = {h["article_id"] for h in retrieved}

    # 文章级别统计
    gt_relevant_articles = {
        art_id for art_id, info in labels.items()
        if info["event_type"] in relevant_types
    }
    retrieved_relevant = retrieved_ids & gt_relevant_articles
    retrieved_irrelevant = retrieved_ids - gt_relevant_articles

    # ── 事件级别指标 ──
    hit_events = []     # 命中的事件
    missed_events = []  # 未命中的事件
    for event_name, article_ids in event_groups.items():
        if retrieved_ids & set(article_ids):
            hit_events.append(event_name)
        else:
            missed_events.append(event_name)

    event_recall = len(hit_events) / total_events if total_events else 0

    # Article Precision（检索结果中 model_release 的占比）
    article_precision = len(retrieved_relevant) / len(retrieved_ids) if retrieved_ids else 0

    # F1（基于 event_recall 和 article_precision 不太合理，用文章级的算）
    article_recall = len(retrieved_relevant) / len(gt_relevant_articles) if gt_relevant_articles else 0
    f1 = (
        2 * article_recall * article_precision / (article_recall + article_precision)
        if (article_recall + article_precision) > 0 else 0
    )

    # NDCG（二值相关度）
    ranked = sorted(retrieved, key=lambda h: h["rerank_score"], reverse=True)
    ndcg_all = _compute_ndcg(ranked, labels, relevant_types)
    ndcg_10 = _compute_ndcg(ranked, labels, relevant_types, k=10)
    ndcg_20 = _compute_ndcg(ranked, labels, relevant_types, k=20)

    # Entity Coverage（基于事件维度）
    gt_event_entities = set(event_groups.keys())
    hit_event_entities = set(hit_events)
    entity_coverage = len(hit_event_entities) / len(gt_event_entities) if gt_event_entities else 0

    # Rerank 分数分析
    scores = [h["rerank_score"] for h in retrieved]
    relevant_scores = [
        h["rerank_score"] for h in retrieved if h["article_id"] in gt_relevant_articles
    ]
    irrelevant_scores = [
        h["rerank_score"] for h in retrieved if h["article_id"] not in gt_relevant_articles
    ]

    # 检索到的文章的事件类型分布
    type_dist = Counter()
    for art_id in retrieved_ids:
        if art_id in labels:
            type_dist[labels[art_id]["event_type"]] += 1

    return {
        # 事件级别
        "total_events": total_events,
        "hit_events": len(hit_events),
        "missed_events": len(missed_events),
        "event_recall": round(event_recall, 4),
        "missed_event_list": sorted(missed_events),
        # 文章级别
        "gt_relevant_articles": len(gt_relevant_articles),
        "total_retrieved": len(retrieved_ids),
        "retrieved_relevant": len(retrieved_relevant),
        "retrieved_irrelevant": len(retrieved_irrelevant),
        "article_precision": round(article_precision, 4),
        "article_recall": round(article_recall, 4),
        "f1": round(f1, 4),
        # 排序
        "ndcg": round(ndcg_all, 4),
        "ndcg_10": round(ndcg_10, 4),
        "ndcg_20": round(ndcg_20, 4),
        # 分数
        "avg_rerank_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "avg_relevant_score": round(sum(relevant_scores) / len(relevant_scores), 4) if relevant_scores else 0,
        "avg_irrelevant_score": round(sum(irrelevant_scores) / len(irrelevant_scores), 4) if irrelevant_scores else 0,
        "type_distribution": dict(type_dist.most_common()),
    }


# ── 第四部分：输出报告 ──

def print_report(metrics: dict, run_dir: Path):
    """打印评估报告并保存。"""
    lines = []
    lines.append("=" * 60)
    lines.append("📊 RAG 检索评估报告（事件级别）")
    lines.append("=" * 60)
    lines.append("")
    lines.append("## 🎯 核心指标：事件召回")
    lines.append(f"  全库独立发布事件数: {metrics['total_events']}")
    lines.append(f"  检索命中事件数:     {metrics['hit_events']}")
    lines.append(f"  未命中事件数:       {metrics['missed_events']}")
    lines.append(f"")
    lines.append(f"  📈 Event Recall: {metrics['event_recall']:.2%}  ({metrics['hit_events']}/{metrics['total_events']})")
    lines.append(f"  （检索到某事件的任意一篇报道即算命中）")
    lines.append("")
    lines.append("## 文章级指标")
    lines.append(f"  实际检索到的文章数 (去重):  {metrics['total_retrieved']}")
    lines.append(f"  其中 model_release:        {metrics['retrieved_relevant']}")
    lines.append(f"  其中非 model_release:      {metrics['retrieved_irrelevant']}")
    lines.append(f"")
    lines.append(f"  📈 Article Precision: {metrics['article_precision']:.2%}  ({metrics['retrieved_relevant']}/{metrics['total_retrieved']})")
    lines.append(f"  📈 Article Recall:    {metrics['article_recall']:.2%}  ({metrics['retrieved_relevant']}/{metrics['gt_relevant_articles']})")
    lines.append(f"  📈 F1:                {metrics['f1']:.2%}")
    lines.append("")
    lines.append("## 排序质量 (NDCG, 二值相关度)")
    lines.append(f"  📈 NDCG@10:   {metrics['ndcg_10']:.4f}")
    lines.append(f"  📈 NDCG@20:   {metrics['ndcg_20']:.4f}")
    lines.append(f"  📈 NDCG@All:  {metrics['ndcg']:.4f}")
    lines.append("")
    lines.append("## Rerank 分数")
    lines.append(f"  全体平均:   {metrics['avg_rerank_score']:.4f}")
    lines.append(f"  相关文章:   {metrics['avg_relevant_score']:.4f}")
    lines.append(f"  无关文章:   {metrics['avg_irrelevant_score']:.4f}")
    lines.append("")
    lines.append("## 检索结果事件类型分布")
    for t, c in metrics["type_distribution"].items():
        lines.append(f"  {t:20s} {c:5d}")
    lines.append("")
    lines.append("## 未命中的发布事件")
    for event in metrics["missed_event_list"]:
        lines.append(f"  ❌ {event}")

    report = "\n".join(lines)
    print(report)

    # 保存到 run_dir
    report_path = run_dir / "EVAL_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    with open(run_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n📁 评估报告: {report_path}")
    print(f"📁 评估指标: {run_dir / 'eval_metrics.json'}")


# ── 主流程 ──

async def main():
    parser = argparse.ArgumentParser(description="RAG 子图评估")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="研究主题")
    parser.add_argument("--eval-only", type=str, default="", help="只评估已有运行结果（传入 run_dir 路径）")
    args = parser.parse_args()

    # 加载 ground truth
    if not LABELS_FILE.exists():
        print(f"❌ 标签文件不存在: {LABELS_FILE}")
        print(f"   请先运行 eval/label_articles.py 生成标签")
        sys.exit(1)

    labels_list = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    labels = {item["article_id"]: item for item in labels_list}
    print(f"📂 加载标签: {len(labels)} 篇")

    if args.eval_only:
        # 只评估模式
        run_dir = Path(args.eval_only)
        raw_results_path = run_dir / "raw_results.json"
        if not raw_results_path.exists():
            print(f"❌ 找不到 {raw_results_path}")
            sys.exit(1)
        raw_results = json.loads(raw_results_path.read_text(encoding="utf-8"))
    else:
        # 完整运行模式
        run_dir, result = await run_rag_subgraph(args.topic)
        raw_results = result.get("raw_results", [])

    # 提取检索到的 article_id
    retrieved = extract_article_ids(raw_results)
    print(f"\n🔍 检索到 {len(retrieved)} 篇去重文章")

    # 计算指标
    metrics = compute_metrics(retrieved, labels, RELEVANT_EVENT_TYPES)

    # 输出报告
    print_report(metrics, run_dir)


if __name__ == "__main__":
    asyncio.run(main())
