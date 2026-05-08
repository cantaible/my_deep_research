"""Finding 级别评测脚本。

从 RAG 压缩报告中抽取 findings，与 ground truth 匹配，计算评测指标。

用法：
    python eval/eval_findings.py                    # 运行评测并生成报告
    python eval/eval_findings.py --run-dir <path>   # 只评测已有运行结果
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from finding_extractor import extract_findings
from finding_matcher import compute_evidence_support, match_findings
from finding_schema import FindingMatch


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

    return labels_dict, event_index, total_events


def load_research_report(run_dir: Path) -> str:
    """从运行目录加载研究报告（优先 compressed.md，其次 report.md）"""
    compressed_file = run_dir / "compressed.md"
    report_file = run_dir / "report.md"

    if compressed_file.exists():
        print(f"  ✓ 使用 RAG 压缩报告: compressed.md")
        return compressed_file.read_text(encoding="utf-8")
    elif report_file.exists():
        print(f"  ✓ 使用完整研究报告: report.md")
        return report_file.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"找不到 compressed.md 或 report.md")


# ── 指标计算 ──

def compute_finding_metrics(
    matches: list[FindingMatch],
    total_events: int,
) -> dict:
    """计算 Finding 级别的评测指标。

    Args:
        matches: 匹配结果列表
        total_events: ground truth 中的总事件数

    Returns:
        评测指标字典
    """
    # 统计匹配成功的 findings（置信度 >= 0.6）
    matched_findings = [m for m in matches if m.matched_event is not None]

    # 统计命中的 ground truth 事件（去重）
    hit_events = set(m.matched_event for m in matched_findings)

    # 计算 Recall, Precision, F1
    recall = len(hit_events) / total_events if total_events > 0 else 0
    precision = len(matched_findings) / len(matches) if matches else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    # 计算 Evidence Support Rate
    total_evidence = sum(len(m.evidence_article_ids) for m in matched_findings)
    total_evidence_in_gold = sum(m.evidence_in_gold for m in matched_findings)
    evidence_support_rate = (
        total_evidence_in_gold / total_evidence if total_evidence > 0 else 0
    )

    return {
        "total_findings": len(matches),
        "matched_findings": len(matched_findings),
        "hit_events": len(hit_events),
        "total_events": total_events,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "evidence_support_rate": evidence_support_rate,
    }


# ── 报告生成 ──

def generate_report(
    matches: list[FindingMatch],
    metrics: dict,
    run_dir: Path,
    topic: str,
    event_index: dict,
):
    """生成 Finding 评测报告"""
    lines = []
    lines.append("# Finding 抽取与匹配评测报告")
    lines.append("")
    lines.append("## 查询信息")
    lines.append(f"- 查询: {topic[:100]}...")
    lines.append(f"- 运行目录: {run_dir}")
    lines.append("")

    lines.append("## 整体指标")
    lines.append("")
    lines.append(f"- **Finding Recall**: {metrics['recall']:.1%} ({metrics['hit_events']}/{metrics['total_events']})")
    lines.append(f"- **Finding Precision**: {metrics['precision']:.1%} ({metrics['matched_findings']}/{metrics['total_findings']})")
    lines.append(f"- **Finding F1**: {metrics['f1']:.1%}")
    lines.append(f"- **Evidence Support Rate**: {metrics['evidence_support_rate']:.1%}")
    lines.append("")

    # 正确识别的事件
    matched_findings = [m for m in matches if m.matched_event is not None]
    if matched_findings:
        lines.append(f"## 正确识别的事件 ({len(matched_findings)})")
        lines.append("")
        for i, match in enumerate(matched_findings, 1):
            finding = match.finding
            lines.append(f"### {i}. {finding.vendor} - {finding.model_name} ✓")
            lines.append(f"- **匹配事件**: {match.matched_event}")
            lines.append(f"- **匹配置信度**: {match.confidence:.2f}")
            lines.append(f"- **匹配原因**: {match.match_reason}")
            if finding.release_date:
                lines.append(f"- **发布日期**: {finding.release_date}")
            if finding.key_features:
                lines.append(f"- **关键特性**: {', '.join(finding.key_features[:3])}")
            lines.append(f"- **证据支撑**: {match.evidence_in_gold}/{len(match.evidence_article_ids)} 篇在 gold_evidence 中")
            lines.append("")

    # 误报的 findings
    false_positives = [m for m in matches if m.matched_event is None]
    if false_positives:
        lines.append(f"## 误报的 findings ({len(false_positives)})")
        lines.append("")
        for i, match in enumerate(false_positives, 1):
            finding = match.finding
            lines.append(f"### {i}. {finding.vendor} - {finding.model_name} ✗")
            lines.append(f"- **原因**: {match.match_reason}")
            lines.append(f"- **证据**: {finding.evidence_text[:100]}...")
            lines.append("")

    # 漏报的事件
    hit_events = set(m.matched_event for m in matched_findings)
    all_events = set(event_index.values())
    missed_events = all_events - hit_events
    if missed_events:
        lines.append(f"## 漏报的事件 ({len(missed_events)})")
        lines.append("")
        for i, event_name in enumerate(sorted(missed_events), 1):
            lines.append(f"{i}. {event_name} ✗")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    # 保存报告
    report_path = run_dir / "FINDING_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 保存指标
    metrics_path = run_dir / "finding_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存匹配详情
    matches_data = [
        {
            "finding": {
                "event_type": m.finding.event_type,
                "model_name": m.finding.model_name,
                "vendor": m.finding.vendor,
                "release_date": m.finding.release_date,
                "key_features": m.finding.key_features,
                "evidence_text": m.finding.evidence_text,
            },
            "matched_event": m.matched_event,
            "confidence": m.confidence,
            "match_reason": m.match_reason,
            "evidence_article_ids": m.evidence_article_ids,
            "evidence_in_gold": m.evidence_in_gold,
        }
        for m in matches
    ]
    matches_path = run_dir / "finding_matches.json"
    with open(matches_path, "w", encoding="utf-8") as f:
        json.dump(matches_data, f, ensure_ascii=False, indent=2)

    print(f"\n📁 报告已保存: {report_path}")
    print(f"📁 指标已保存: {metrics_path}")
    print(f"📁 匹配详情已保存: {matches_path}")


# ── 主流程 ──

async def main():
    parser = argparse.ArgumentParser(description="Finding 级别评测")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="研究主题")
    parser.add_argument("--run-dir", type=str, default="", help="只评测已有运行结果（传入 run_dir 路径）")
    args = parser.parse_args()

    print("=" * 60)
    print("Finding 抽取与匹配评测")
    print("=" * 60)

    # 加载 ground truth
    print("\n[1/5] 加载 ground truth...")
    labels_dict, event_index, total_events = load_ground_truth()
    print(f"  ✓ 加载 {len(labels_dict)} 篇文章标注")
    print(f"  ✓ 事件家族数: {total_events}")

    # 运行或加载 RAG 子图
    if args.run_dir:
        print(f"\n[2/5] 加载已有运行结果...")
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"❌ 找不到运行目录: {run_dir}")
            sys.exit(1)
        topic = args.topic
    else:
        print(f"\n[2/5] 运行 RAG 子图...")
        from eval_rag import run_rag_subgraph
        run_dir, result = await run_rag_subgraph(args.topic)
        topic = args.topic

    # 加载研究报告
    print(f"\n[3/5] 加载研究报告...")
    try:
        research_report = load_research_report(run_dir)
        print(f"  ✓ 加载报告，长度: {len(research_report)} 字符")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    # 抽取 findings
    print(f"\n[4/5] 抽取 findings...")
    findings = await extract_findings(research_report)
    print(f"  ✓ 抽取到 {len(findings)} 个 findings")

    # 匹配 findings
    print(f"\n[5/5] 匹配 findings 与 ground truth...")
    matches = match_findings(findings, labels_dict, event_index)
    matches = compute_evidence_support(matches, labels_dict, event_index)
    print(f"  ✓ 匹配完成")

    # 计算指标
    metrics = compute_finding_metrics(matches, total_events)

    # 生成报告
    print(f"\n" + "=" * 60)
    generate_report(matches, metrics, run_dir, topic, event_index)
    print("=" * 60)
    print("\n✅ 评测完成！")


if __name__ == "__main__":
    asyncio.run(main())
