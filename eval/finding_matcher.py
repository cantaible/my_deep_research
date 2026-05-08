"""Finding 匹配器：将抽取的 findings 与 ground truth 进行匹配。

匹配策略：
1. 基于 aliases 的模糊匹配（主要）
2. 基于 vendor 和 model_name 的实体重叠
3. 基于时间窗口的过滤
"""

import json
import re
from pathlib import Path

from finding_schema import Finding, FindingMatch


def normalize_name(name: str) -> str:
    """标准化名称：去除空格、标点、统一大小写。"""
    # 转小写
    name = name.lower()
    # 去除常见分隔符
    name = re.sub(r"[\s\-_\.]", "", name)
    # 去除版本号前缀（如 "v"）
    name = re.sub(r"^v", "", name)
    return name


def compute_name_similarity(name1: str, name2: str) -> float:
    """计算两个名称的相似度。

    使用简单的字符串包含关系：
    - 完全匹配: 1.0
    - 一个包含另一个: 0.8
    - 否则: 0.0
    """
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)

    if norm1 == norm2:
        return 1.0
    if norm1 in norm2 or norm2 in norm1:
        return 0.8
    return 0.0


def match_finding_to_event(
    finding: Finding,
    event_canonical_name: str,
    event_aliases: list[str],
    event_vendor: str | None = None,
) -> tuple[float, str]:
    """将 finding 与单个 ground truth 事件进行匹配。

    Args:
        finding: 待匹配的 finding
        event_canonical_name: 事件的 canonical_name
        event_aliases: 事件的 aliases 列表
        event_vendor: 事件的 vendor（如果有）

    Returns:
        (confidence, reason): 匹配置信度和原因说明
    """
    # 1. 检查 model_name 是否匹配 canonical_name 或 aliases
    all_names = [event_canonical_name] + event_aliases
    max_name_sim = max(
        compute_name_similarity(finding.model_name, name)
        for name in all_names
    )

    if max_name_sim == 0.0:
        return 0.0, "模型名称不匹配"

    # 2. 检查 vendor 是否匹配（如果 ground truth 有 vendor 信息）
    vendor_match = True
    if event_vendor:
        vendor_sim = compute_name_similarity(finding.vendor, event_vendor)
        if vendor_sim < 0.8:
            vendor_match = False

    # 3. 计算最终置信度
    if max_name_sim >= 1.0 and vendor_match:
        confidence = 1.0
        reason = f"模型名称完全匹配 ({event_canonical_name})"
    elif max_name_sim >= 0.8 and vendor_match:
        confidence = 0.9
        reason = f"模型名称部分匹配 ({event_canonical_name})"
    elif max_name_sim >= 1.0 and not vendor_match:
        confidence = 0.7
        reason = f"模型名称匹配但厂商不匹配 ({event_canonical_name})"
    elif max_name_sim >= 0.8 and not vendor_match:
        confidence = 0.6
        reason = f"模型名称部分匹配但厂商不匹配 ({event_canonical_name})"
    else:
        confidence = 0.0
        reason = "匹配度过低"

    return confidence, reason


def match_findings(
    findings: list[Finding],
    labels_dict: dict,
    event_index: dict,
) -> list[FindingMatch]:
    """将所有 findings 与 ground truth 进行匹配。

    Args:
        findings: 抽取的 findings 列表
        labels_dict: article_id -> label 的映射（来自 article_labels_v2.json）
        event_index: article_id -> canonical_name 的映射（来自 event_to_articles.json）

    Returns:
        匹配结果列表
    """
    # 构建 canonical_name -> event_info 的映射
    event_info_map = {}
    for article_id_str, canonical_name in event_index.items():
        if canonical_name not in event_info_map:
            # 从 labels_dict 中获取该事件的信息
            article_id = int(article_id_str)
            if article_id in labels_dict:
                label = labels_dict[article_id]

                # 提取 vendor（从 entities 列表中获取第一个，如果存在）
                entities = label.get("entities", [])
                vendor = entities[0] if entities else None

                event_info_map[canonical_name] = {
                    "canonical_name": canonical_name,
                    "aliases": label.get("aliases", []),
                    "vendor": vendor,
                    "gold_evidence": label.get("gold_evidence", []),
                }

    # 对每个 finding 进行匹配
    matches = []
    for finding in findings:
        best_match = None
        best_confidence = 0.0
        best_reason = ""

        # 尝试匹配所有 ground truth 事件
        for canonical_name, event_info in event_info_map.items():
            confidence, reason = match_finding_to_event(
                finding,
                event_info["canonical_name"],
                event_info["aliases"],
                event_info["vendor"],
            )

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = canonical_name
                best_reason = reason

        # 创建匹配结果
        match = FindingMatch(
            finding=finding,
            matched_event=best_match if best_confidence >= 0.6 else None,
            confidence=best_confidence,
            match_reason=best_reason,
            evidence_article_ids=[],  # 后续通过文本匹配填充
            evidence_in_gold=0,
        )

        matches.append(match)

    return matches


def compute_evidence_support(
    matches: list[FindingMatch],
    labels_dict: dict,
    event_index: dict,
) -> list[FindingMatch]:
    """计算每个 finding 的证据支撑情况。

    通过文本匹配推断哪些文章支撑了该 finding，
    然后检查这些文章是否在 gold_evidence 中。

    Args:
        matches: 匹配结果列表
        labels_dict: article_id -> label 的映射
        event_index: article_id -> canonical_name 的映射

    Returns:
        更新后的匹配结果列表
    """
    for match in matches:
        if not match.matched_event:
            continue

        # 获取该事件的所有文章
        event_articles = [
            int(aid) for aid, cname in event_index.items()
            if cname == match.matched_event
        ]

        # 获取 gold_evidence
        gold_evidence = set()
        for article_id in event_articles:
            if article_id in labels_dict:
                gold_evidence.update(labels_dict[article_id].get("gold_evidence", []))

        # 简单策略：假设所有事件文章都是证据
        # （更精确的方法需要文本匹配，但这里简化处理）
        match.evidence_article_ids = event_articles
        match.evidence_in_gold = len(set(event_articles) & gold_evidence)

    return matches


# ── 测试 ──

if __name__ == "__main__":
    # 测试名称标准化和相似度计算
    print("测试名称匹配：")
    print(f"GPT-5.4 vs GPT 5.4: {compute_name_similarity('GPT-5.4', 'GPT 5.4')}")
    print(f"GPT-5.4 vs GPT5.4: {compute_name_similarity('GPT-5.4', 'GPT5.4')}")
    print(f"GPT-5.4 vs gpt-5.4: {compute_name_similarity('GPT-5.4', 'gpt-5.4')}")
    print(f"GPT-5.4 vs GPT-5: {compute_name_similarity('GPT-5.4', 'GPT-5')}")
    print(f"GPT-5.4 vs Claude 4.5: {compute_name_similarity('GPT-5.4', 'Claude 4.5')}")
