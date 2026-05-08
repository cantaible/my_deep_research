"""生成扩展后的 ground truth 数据。

功能：
1. 从 event_families.json 映射 canonical_name
2. 基于规则生成 aliases（事件名称的常见变体）
3. 从数据库提取 gold_evidence（标题 + 正文前 300 字）
4. 生成反向索引 event_to_articles.json

输入：
- eval/article_labels.json（原始标注）
- eval/event_families.json（事件家族）
- MariaDB 数据库（文章全文，可选）

输出：
- eval/article_labels_v2.json（扩展后的标注）
- eval/event_to_articles.json（反向索引）
- eval/ground_truth_stats.txt（统计报告）

用法：
    python eval/prepare_ground_truth.py              # 尝试连接数据库
    python eval/prepare_ground_truth.py --no-db      # 不连接数据库，只用标题作为 evidence
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# 添加 rag 模块到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

# ============================================================
# 配置
# ============================================================

LABELS_FILE = Path(__file__).parent / "article_labels.json"
EVENT_FAMILIES_FILE = Path(__file__).parent / "event_families.json"
OUTPUT_LABELS_FILE = Path(__file__).parent / "article_labels_v2.json"
OUTPUT_INDEX_FILE = Path(__file__).parent / "event_to_articles.json"
STATS_FILE = Path(__file__).parent / "ground_truth_stats.txt"

# gold_evidence 的长度（字符数）
EVIDENCE_LENGTH = 300


# ============================================================
# 1. 生成 aliases（基于规则）
# ============================================================

def generate_aliases(canonical_name: str) -> list[str]:
    """基于规则生成事件名称的常见变体。

    规则：
    1. 原始名称
    2. 去除连字符（GPT-5.4 → GPT5.4）
    3. 连字符改空格（GPT-5.4 → GPT 5.4）
    4. 小写（GPT-5.4 → gpt-5.4）
    5. 添加常见厂商前缀
    """
    aliases = set()

    # 规则 1: 原始名称
    aliases.add(canonical_name)

    # 规则 2: 去除连字符
    aliases.add(canonical_name.replace("-", ""))

    # 规则 3: 连字符改空格
    aliases.add(canonical_name.replace("-", " "))

    # 规则 4: 小写
    aliases.add(canonical_name.lower())
    aliases.add(canonical_name.replace("-", "").lower())
    aliases.add(canonical_name.replace("-", " ").lower())

    # 规则 5: 添加常见厂商前缀
    if "GPT" in canonical_name:
        aliases.add(f"OpenAI {canonical_name}")
        aliases.add(f"OpenAI {canonical_name.replace('-', '')}")
    elif "Qwen" in canonical_name:
        aliases.add(f"阿里 {canonical_name}")
        aliases.add(f"通义 {canonical_name}")
        aliases.add(f"Alibaba {canonical_name}")
    elif "DeepSeek" in canonical_name:
        aliases.add(f"深度求索 {canonical_name}")
    elif "GLM" in canonical_name:
        aliases.add(f"智谱 {canonical_name}")
        aliases.add(f"ChatGLM {canonical_name}")
    elif "Gemini" in canonical_name:
        aliases.add(f"Google {canonical_name}")
    elif "Grok" in canonical_name:
        aliases.add(f"xAI {canonical_name}")
    elif "Nemotron" in canonical_name:
        aliases.add(f"NVIDIA {canonical_name}")
    elif "Mistral" in canonical_name:
        aliases.add(f"Mistral AI {canonical_name}")
    elif "MiniMax" in canonical_name:
        aliases.add(f"MiniMax {canonical_name}")
    elif "Composer" in canonical_name:
        aliases.add(f"Composer {canonical_name}")

    # 去重并排序
    return sorted(list(aliases))


# ============================================================
# 2. 提取 gold_evidence（从数据库或标题）
# ============================================================

def extract_gold_evidence_from_db(article_id: int, conn) -> str:
    """从数据库提取文章的关键证据片段。

    提取策略：标题 + 正文前 300 字
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT title, raw_content FROM news_article WHERE id = %s",
            (article_id,)
        )
        row = cur.fetchone()

        if not row:
            return ""

        title, raw_content = row

        # 标题 + 正文前 300 字
        if raw_content:
            evidence = f"{title}\n{raw_content[:EVIDENCE_LENGTH]}"
        else:
            evidence = title

        return evidence.strip()


def extract_gold_evidence_from_label(label: dict) -> str:
    """从标注数据提取证据（不连接数据库时的备选方案）。

    只使用标题作为证据。
    """
    return label.get("title", "")


# ============================================================
# 3. 主流程
# ============================================================

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成扩展后的 Ground Truth 数据")
    parser.add_argument("--no-db", action="store_true", help="不连接数据库，只使用标题作为 evidence")
    args = parser.parse_args()

    print("=" * 60)
    print("生成扩展后的 Ground Truth 数据")
    print("=" * 60)

    # ── 加载输入数据 ──
    print("\n[1/5] 加载输入数据...")

    if not LABELS_FILE.exists():
        print(f"❌ 找不到 {LABELS_FILE}")
        sys.exit(1)

    if not EVENT_FAMILIES_FILE.exists():
        print(f"❌ 找不到 {EVENT_FAMILIES_FILE}")
        sys.exit(1)

    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    event_families = json.loads(EVENT_FAMILIES_FILE.read_text(encoding="utf-8"))

    print(f"  ✓ 加载 {len(labels)} 篇文章标注")
    print(f"  ✓ 加载 {len(event_families)} 个事件家族")

    # ── 生成反向索引 ──
    print("\n[2/5] 生成反向索引（article_id → canonical_name）...")

    article_to_event = {}
    for family_name, info in event_families.items():
        for article_id in info["article_ids"]:
            article_to_event[article_id] = family_name

    print(f"  ✓ 生成 {len(article_to_event)} 条映射")

    # ── 连接数据库（可选）──
    print("\n[3/5] 准备提取 gold_evidence...")

    conn = None
    use_db = not args.no_db

    if use_db:
        try:
            from scripts.db import get_connection
            conn = get_connection()
            print("  ✓ 数据库连接成功，将从数据库提取完整证据")
        except Exception as e:
            print(f"  ⚠️  数据库连接失败: {e}")
            print("  ℹ️  将使用标题作为 evidence（可用 --no-db 跳过连接尝试）")
            use_db = False
    else:
        print("  ℹ️  跳过数据库连接，将使用标题作为 evidence")

    # ── 扩展标注数据 ──
    print("\n[4/5] 扩展标注数据...")

    labels_v2 = []
    stats = {
        "total": len(labels),
        "with_canonical_name": 0,
        "without_canonical_name": 0,
        "event_types": Counter(),
        "evidence_extracted": 0,
        "evidence_failed": 0,
        "evidence_source": "database" if use_db else "title_only",
    }

    for i, label in enumerate(labels):
        if (i + 1) % 100 == 0:
            print(f"  处理中... {i + 1}/{len(labels)}")

        article_id = label["article_id"]
        label_v2 = label.copy()

        # 添加 canonical_name
        canonical_name = article_to_event.get(article_id, None)
        label_v2["canonical_name"] = canonical_name

        if canonical_name:
            stats["with_canonical_name"] += 1
            # 添加 aliases
            label_v2["aliases"] = generate_aliases(canonical_name)
        else:
            stats["without_canonical_name"] += 1
            label_v2["aliases"] = []

        # 添加 gold_evidence
        try:
            if use_db:
                evidence = extract_gold_evidence_from_db(article_id, conn)
            else:
                evidence = extract_gold_evidence_from_label(label)

            label_v2["gold_evidence"] = evidence
            if evidence:
                stats["evidence_extracted"] += 1
            else:
                stats["evidence_failed"] += 1
        except Exception as e:
            print(f"  ⚠️  提取证据失败 (article_id={article_id}): {e}")
            label_v2["gold_evidence"] = ""
            stats["evidence_failed"] += 1

        # 统计事件类型
        stats["event_types"][label["event_type"]] += 1

        labels_v2.append(label_v2)

    if conn:
        conn.close()

    print(f"  ✓ 扩展完成")

    # ── 保存输出 ──
    print("\n[5/5] 保存输出文件...")

    # 保存扩展后的标注
    with open(OUTPUT_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels_v2, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 保存 {OUTPUT_LABELS_FILE}")

    # 保存反向索引
    with open(OUTPUT_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(article_to_event, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 保存 {OUTPUT_INDEX_FILE}")

    # ── 生成统计报告 ──
    print("\n" + "=" * 60)
    print("统计报告")
    print("=" * 60)

    report_lines = []
    report_lines.append("Ground Truth 数据统计")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"总文章数: {stats['total']}")
    report_lines.append(f"  有 canonical_name: {stats['with_canonical_name']} ({stats['with_canonical_name']/stats['total']:.1%})")
    report_lines.append(f"  无 canonical_name: {stats['without_canonical_name']} ({stats['without_canonical_name']/stats['total']:.1%})")
    report_lines.append("")
    report_lines.append(f"Gold Evidence 提取:")
    report_lines.append(f"  来源: {stats['evidence_source']}")
    report_lines.append(f"  成功: {stats['evidence_extracted']}")
    report_lines.append(f"  失败: {stats['evidence_failed']}")
    report_lines.append("")
    report_lines.append("事件类型分布:")
    for event_type, count in stats["event_types"].most_common():
        report_lines.append(f"  {event_type:20s} {count:5d} ({count/stats['total']:.1%})")
    report_lines.append("")
    report_lines.append("事件家族统计:")
    report_lines.append(f"  事件家族数: {len(event_families)}")
    report_lines.append(f"  覆盖文章数: {len(article_to_event)}")
    report_lines.append("")
    report_lines.append("事件家族列表:")
    for family_name, info in sorted(event_families.items()):
        article_count = len(info["article_ids"])
        report_lines.append(f"  {family_name:30s} {article_count:3d} 篇")

    report = "\n".join(report_lines)
    print(report)

    # 保存统计报告
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✓ 统计报告已保存: {STATS_FILE}")

    # ── 示例输出 ──
    print("\n" + "=" * 60)
    print("示例输出（前 3 条）")
    print("=" * 60)

    for i, label in enumerate(labels_v2[:3]):
        print(f"\n[{i+1}] Article ID: {label['article_id']}")
        print(f"    Title: {label['title'][:60]}...")
        print(f"    Event Type: {label['event_type']}")
        print(f"    Canonical Name: {label['canonical_name']}")
        print(f"    Aliases: {label['aliases'][:3]}..." if len(label['aliases']) > 3 else f"    Aliases: {label['aliases']}")
        print(f"    Gold Evidence: {label['gold_evidence'][:100]}...")

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
