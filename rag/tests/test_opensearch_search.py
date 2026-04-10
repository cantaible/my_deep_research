"""OpenSearch 查询构造测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opensearch_search import build_search_body


def test_build_search_body_uses_weighted_fields_and_filters():
    body = build_search_body(
        query="2026年3月 大模型 发布 上线",
        top_k=5,
        category="AI",
        published_ts_gte=1772323200,
        published_ts_lte=1775001599,
    )

    assert body["size"] == 5
    assert {"term": {"category": "AI"}} in body["query"]["bool"]["filter"]
    assert {"range": {"published_ts": {"gte": 1772323200, "lte": 1775001599}}} in body["query"]["bool"]["filter"]

    zh_fields = body["query"]["bool"]["should"][0]["multi_match"]["fields"]
    en_fields = body["query"]["bool"]["should"][1]["multi_match"]["fields"]

    assert "title^8" in zh_fields
    assert "summary^5" in zh_fields
    assert "search_text^2" in zh_fields
    assert "title.en^5" in en_fields
    assert "search_text.en^1.5" in en_fields
