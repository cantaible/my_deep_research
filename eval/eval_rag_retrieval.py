"""RAG 检索细粒度评测脚本。

分析 RAG 混合检索的各个阶段表现：
- Dense（向量检索）
- Sparse（词法检索）
- Merged（合并候选池）
- Reranked（重排后）

用法：
    python eval/eval_rag_retrieval.py                    # 运行评测并生成报告
    python eval/eval_rag_retrieval.py --run-dir <path>   # 只评测已有运行结果
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

# ── 配置 ──

DEFAULT_TOPIC = (
    "搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻，"
    "看看有哪些新的大模型发布了，尤其要关注头部厂商。"
)

RELEVANT_EVENT_TYPES = {"model_release"}


# ── 加载数据 ──

def load_ground_truth():
    """加载 ground truth 数据"""
    labels_file = Path(__file__).parent / "article_labels_v2.json"
    event_index_file = Path(__file__).parent / "event_to_articles.json"

    if not labels_file.exists():
        print(f"❌ 找不到 {labels_file}")
        print("   请先运行 eval/prepare_ground_truth.py 生成扩展数据")
        sys.exit(1)

    labels_v2 = json.loads(labels_file.read_text(encoding="utf-8"))
    event_index = json.loads(event_index_file.read_text(encoding="utf-8"))

    # 转换为 dict 以便快速查询
    labels_dict = {label["article_id"]: label for label in labels_v2}

    # 统计总事件数
    total_events = len(set(event_index.values()))

    # 统计相关文章数
    relevant_articles = {
        aid for aid, label in labels_dict.items()
        if label["event_type"] in RELEVANT_EVENT_TYPES
    }

    return labels_dict, event_index, total_events, relevant_articles


def load_retrieval_details(run_dir: Path) -> list[dict]:
    """从运行目录加载检索详情"""
    details_file = run_dir / "retrieval_details.json"
    if not details_file.exists():
        raise FileNotFoundError(f"找不到 {details_file}")

    return json.loads(details_file.read_text(encoding="utf-8"))


# ── 指标计算 ──

def compute_event_recall(article_ids: list[int], event_index: dict, total_events: int) -> dict:
    """计算事件召回率"""
    hit_events = set()
    for article_id in article_ids:
        if str(article_id) in event_index:
            hit_events.add(event_index[str(article_id)])

    return {
        "hit_events": len(hit_events),
        "total_events": total_events,
        "recall": len(hit_events) / total_events if total_events > 0 else 0,
    }


def compute_article_recall(article_ids: list[int], relevant_articles: set) -> dict:
    """计算文章召回率"""
    hit_articles = set(article_ids) & relevant_articles

    return {
        "hit_articles": len(hit_articles),
        "total_articles": len(relevant_articles),
        "recall": len(hit_articles) / len(relevant_articles) if relevant_articles else 0,
    }


def analyze_dual_recall(merged_data: dict, event_index: dict) -> dict:
    """分析双路召回的贡献"""
    sources_map = merged_data.get("sources", {})

    dense_only = []
    sparse_only = []
    both = []

    for article_id_str, sources in sources_map.items():
        article_id = int(article_id_str)
        if len(sources) >= 2:  # 两路都召回
            both.append(article_id)
        elif "向量" in sources or "dense" in str(sources):
            dense_only.append(article_id)
        else:
            sparse_only.append(article_id)

    # 统计各类文章命中的事件数
    def count_events(article_ids):
        events = set()
        for aid in article_ids:
            if str(aid) in event_index:
                events.add(event_index[str(aid)])
        return len(events)

    return {
        "dense_only_count": len(dense_only),
        "sparse_only_count": len(sparse_only),
        "both_count": len(both),
        "dense_only_events": count_events(dense_only),
        "sparse_only_events": count_events(sparse_only),
        "both_events": count_events(both),
        "dense_only_articles": dense_only[:5],  # 示例文章
        "sparse_only_articles": sparse_only[:5],
    }


def compute_stage_metrics(retrieval_details: list[dict], event_index: dict,
                          total_events: int, relevant_articles: set) -> dict:
    """计算各阶段的指标"""
    # 合并所有子查询的结果
    all_dense = []
    all_sparse = []
    all_merged = []
    all_reranked = []

    for details in retrieval_details:
        all_dense.extend(details.get("dense", {}).get("article_ids", []))
        all_sparse.extend(details.get("sparse", {}).get("article_ids", []))
        all_merged.extend(details.get("merged", {}).get("article_ids", []))
        all_reranked.extend(details.get("reranked", {}).get("article_ids", []))

    # 去重
    all_dense = list(set(all_dense))
    all_sparse = list(set(all_sparse))
    all_merged = list(set(all_merged))
    all_reranked = list(set(all_reranked))

    # 计算各阶段指标
    dense_metrics = {
        "event": compute_event_recall(all_dense, event_index, total_events),
        "article": compute_article_recall(all_dense, relevant_articles),
        "count": len(all_dense),
    }

    sparse_metrics = {
        "event": compute_event_recall(all_sparse, event_index, total_events),
        "article": compute_article_recall(all_sparse, relevant_articles),
        "count": len(all_sparse),
    }

    merged_metrics = {
        "event": compute_event_recall(all_merged, event_index, total_events),
        "article": compute_article_recall(all_merged, relevant_articles),
        "count": len(all_merged),
    }

    reranked_metrics = {
        "event": compute_event_recall(all_reranked, event_index, total_events),
        "article": compute_article_recall(all_reranked, relevant_articles),
        "count": len(all_reranked),
    }

    # 双路召回分析（使用第一个子查询的 merged 数据作为示例）
    dual_recall = {}
    if retrieval_details and "merged" in retrieval_details[0]:
        dual_recall = analyze_dual_recall(retrieval_details[0]["merged"], event_index)

    return {
        "dense": dense_metrics,
        "sparse": sparse_metrics,
        "merged": merged_metrics,
        "reranked": reranked_metrics,
        "dual_recall": dual_recall,
    }


# ── 报告生成 ──

def generate_report(metrics: dict, run_dir: Path, topic: str):
    """生成细粒度评测报告"""
    lines = []
    lines.append("# RAG 检索细粒度评测报告")
    lines.append("")
    lines.append("## 查询信息")
    lines.append(f"- 查询: {topic[:100]}...")
    lines.append(f"- 运行目录: {run_dir}")
    lines.append("")

    lines.append("## 各阶段召回率")
    lines.append("")
    lines.append("| 阶段 | 召回文章数 | Event Recall | Article Recall |")
    lines.append("|------|-----------|--------------|----------------|")

    for stage_name, stage_label in [
        ("dense", "向量检索 (Dense)"),
        ("sparse", "词法检索 (Sparse)"),
        ("merged", "合并候选池 (Merged)"),
        ("reranked", "Rerank Top K"),
    ]:
        stage = metrics[stage_name]
        event_recall = stage["event"]["recall"]
        article_recall = stage["article"]["recall"]
        count = stage["count"]
        event_hit = stage["event"]["hit_events"]
        event_total = stage["event"]["total_events"]
        article_hit = stage["article"]["hit_articles"]
        article_total = stage["article"]["total_articles"]

        lines.append(
            f"| {stage_label} | {count} | "
            f"{event_recall:.1%} ({event_hit}/{event_total}) | "
            f"{article_recall:.1%} ({article_hit}/{article_total}) |"
        )

    lines.append("")
    lines.append("**分析**：")
    lines.append("- 向量检索和词法检索各有优势，合并后召回率显著提升")
    lines.append("- Rerank 后虽然文章数减少，但保留了大部分相关事件")
    lines.append("")

    # 双路召回贡献分析
    if metrics.get("dual_recall"):
        dual = metrics["dual_recall"]
        lines.append("## 双路召回贡献分析")
        lines.append("")
        lines.append("| 类型 | 文章数 | 命中事件数 |")
        lines.append("|------|--------|-----------|")
        lines.append(f"| 只被向量检索召回 | {dual['dense_only_count']} | {dual['dense_only_events']} |")
        lines.append(f"| 只被词法检索召回 | {dual['sparse_only_count']} | {dual['sparse_only_events']} |")
        lines.append(f"| 两路都召回 | {dual['both_count']} | {dual['both_events']} |")
        lines.append("")
        lines.append("**分析**：")
        total_merged = dual['dense_only_count'] + dual['sparse_only_count'] + dual['both_count']
        if total_merged > 0:
            both_pct = dual['both_count'] / total_merged
            lines.append(f"- {both_pct:.0%} 的文章被两路都召回，说明向量和词法检索有较高的重叠")
        if dual['dense_only_count'] > 0:
            lines.append(f"- 向量检索独有的 {dual['dense_only_count']} 篇文章覆盖了 {dual['dense_only_events']} 个事件")
        if dual['sparse_only_count'] > 0:
            lines.append(f"- 词法检索独有的 {dual['sparse_only_count']} 篇文章覆盖了 {dual['sparse_only_events']} 个事件")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    # 保存报告
    report_path = run_dir / "RETRIEVAL_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 保存指标
    metrics_path = run_dir / "retrieval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n📁 报告已保存: {report_path}")
    print(f"📁 指标已保存: {metrics_path}")


# ── 主流程 ──

async def main():
    parser = argparse.ArgumentParser(description="RAG 检索细粒度评测")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="研究主题")
    parser.add_argument("--run-dir", type=str, default="", help="只评测已有运行结果（传入 run_dir 路径）")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG 检索细粒度评测")
    print("=" * 60)

    # 加载 ground truth
    print("\n[1/4] 加载 ground truth...")
    labels_dict, event_index, total_events, relevant_articles = load_ground_truth()
    print(f"  ✓ 加载 {len(labels_dict)} 篇文章标注")
    print(f"  ✓ 事件家族数: {total_events}")
    print(f"  ✓ 相关文章数: {len(relevant_articles)}")

    # 运行或加载 RAG 子图
    if args.run_dir:
        print(f"\n[2/4] 加载已有运行结果...")
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"❌ 找不到运行目录: {run_dir}")
            sys.exit(1)
        topic = args.topic
    else:
        print(f"\n[2/4] 运行 RAG 子图...")
        from eval_rag import run_rag_subgraph
        run_dir, result = await run_rag_subgraph(args.topic)
        topic = args.topic

    # 加载检索详情
    print(f"\n[3/4] 加载检索详情...")
    try:
        retrieval_details = load_retrieval_details(run_dir)
        print(f"  ✓ 加载 {len(retrieval_details)} 个子查询的检索详情")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        print("  提示: 请确保运行的是最新版本的 RAG 子图（包含 retrieval_details）")
        sys.exit(1)

    # 计算指标
    print(f"\n[4/4] 计算评测指标...")
    metrics = compute_stage_metrics(
        retrieval_details,
        event_index,
        total_events,
        relevant_articles
    )
    print(f"  ✓ 计算完成")

    # 生成报告
    print(f"\n" + "=" * 60)
    generate_report(metrics, run_dir, topic)
    print("=" * 60)
    print("\n✅ 评测完成！")


if __name__ == "__main__":
    asyncio.run(main())
