"""示例：如何使用扩展后的 ground truth 数据。

演示：
1. 加载数据
2. 查询文章的 canonical_name
3. 使用 aliases 进行模糊匹配
4. 使用 gold_evidence 验证 finding
"""

import json
from pathlib import Path

# ============================================================
# 加载数据
# ============================================================

LABELS_V2_FILE = Path(__file__).parent / "article_labels_v2.json"
EVENT_INDEX_FILE = Path(__file__).parent / "event_to_articles.json"

print("加载数据...")
labels_v2 = json.loads(LABELS_V2_FILE.read_text(encoding="utf-8"))
event_index = json.loads(EVENT_INDEX_FILE.read_text(encoding="utf-8"))

# 转换为 dict 以便快速查询
labels_dict = {label["article_id"]: label for label in labels_v2}

print(f"✓ 加载 {len(labels_v2)} 篇文章")
print(f"✓ 加载 {len(event_index)} 条反向索引")

# ============================================================
# 示例 1: 查询文章的 canonical_name
# ============================================================

print("\n" + "=" * 60)
print("示例 1: 查询文章的 canonical_name")
print("=" * 60)

article_id = 4362
if str(article_id) in event_index:
    canonical_name = event_index[str(article_id)]
    print(f"Article {article_id} 属于事件: {canonical_name}")
else:
    print(f"Article {article_id} 不属于任何事件家族")

# ============================================================
# 示例 2: 使用 aliases 进行模糊匹配
# ============================================================

print("\n" + "=" * 60)
print("示例 2: 使用 aliases 进行模糊匹配")
print("=" * 60)

# 模拟 Agent 输出的文本
agent_text = "OpenAI 在 3 月发布了 GPT5.4 和 GPT 5.4 mini"

# 收集所有事件的 aliases
all_aliases = {}
for label in labels_v2:
    if label["canonical_name"]:
        for alias in label["aliases"]:
            all_aliases[alias] = label["canonical_name"]

# 模糊匹配
matched_events = set()
for alias, canonical_name in all_aliases.items():
    if alias in agent_text:
        matched_events.add(canonical_name)

print(f"Agent 文本: {agent_text}")
print(f"匹配到的事件: {matched_events}")

# ============================================================
# 示例 3: 使用 gold_evidence 验证 finding
# ============================================================

print("\n" + "=" * 60)
print("示例 3: 使用 gold_evidence 验证 finding")
print("=" * 60)

# 模拟 Agent 输出的 finding
finding = {
    "text": "DeepSeek 即将推出 V4 版本，这是一个全新的多模态大模型",
    "evidence_article_ids": [4362]
}

print(f"Finding: {finding['text']}")
print(f"引用文章: {finding['evidence_article_ids']}")
print()

for article_id in finding["evidence_article_ids"]:
    label = labels_dict[article_id]
    print(f"Article {article_id}:")
    print(f"  Title: {label['title']}")
    print(f"  Gold Evidence: {label['gold_evidence'][:150]}...")
    print()
    print("  ✓ 可以用 LLM 判断 finding 是否与 gold_evidence 一致")

# ============================================================
# 示例 4: 统计事件覆盖率
# ============================================================

print("\n" + "=" * 60)
print("示例 4: 统计事件覆盖率")
print("=" * 60)

# 模拟检索结果
retrieved_article_ids = [4362, 5085, 6304, 4574, 5045, 5053]

# 统计命中的事件
hit_events = set()
for article_id in retrieved_article_ids:
    if str(article_id) in event_index:
        hit_events.add(event_index[str(article_id)])

# 统计总事件数
total_events = len(set(event_index.values()))

print(f"检索到的文章: {retrieved_article_ids}")
print(f"命中的事件: {hit_events}")
print(f"Event Recall: {len(hit_events)}/{total_events} = {len(hit_events)/total_events:.1%}")

# ============================================================
# 示例 5: 查看事件家族的所有文章
# ============================================================

print("\n" + "=" * 60)
print("示例 5: 查看事件家族的所有文章")
print("=" * 60)

target_event = "GPT-5.4"
articles_in_event = [
    article_id for article_id, event_name in event_index.items()
    if event_name == target_event
]

print(f"事件 '{target_event}' 包含 {len(articles_in_event)} 篇文章:")
for article_id in articles_in_event[:5]:  # 只显示前 5 篇
    label = labels_dict[int(article_id)]
    print(f"  - Article {article_id}: {label['title'][:60]}...")

if len(articles_in_event) > 5:
    print(f"  ... 还有 {len(articles_in_event) - 5} 篇")
