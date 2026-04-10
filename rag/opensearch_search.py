"""OpenSearch 词法检索模块。"""
from __future__ import annotations

from opensearchpy import TransportError

try:
    from .opensearch_client import get_opensearch_client, get_opensearch_index_name
except ImportError:
    from opensearch_client import get_opensearch_client, get_opensearch_index_name


class OpenSearchUnavailableError(RuntimeError):
    """OpenSearch 不可用或索引不存在。"""


def _build_filters(category: str, published_ts_gte: int | None, published_ts_lte: int | None) -> list[dict]:
    filters: list[dict] = []
    if category:
        filters.append({"term": {"category": category}})

    range_params = {}
    if published_ts_gte is not None:
        range_params["gte"] = published_ts_gte
    if published_ts_lte is not None:
        range_params["lte"] = published_ts_lte
    if range_params:
        filters.append({"range": {"published_ts": range_params}})

    return filters


def build_search_body(
    query: str,
    top_k: int,
    category: str,
    published_ts_gte: int | None,
    published_ts_lte: int | None,
) -> dict:
    filters = _build_filters(category, published_ts_gte, published_ts_lte)
    return {
        "size": top_k,
        "_source": [
            "article_id",
            "category",
            "source_name",
            "published_ts",
            "title",
            "summary",
            "preview",
            "raw_content",
        ],
        "query": {
            "bool": {
                "filter": filters,
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "type": "best_fields",
                            "fields": [
                                "title^8",
                                "summary^5",
                                "preview^3",
                                "source_name_text^1.5",
                                "search_text^2",
                            ],
                            "operator": "or",
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "type": "best_fields",
                            "fields": [
                                "title.en^5",
                                "summary.en^3",
                                "preview.en^2",
                                "source_name_text.en^1",
                                "search_text.en^1.5",
                            ],
                            "operator": "or",
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        "highlight": {
            "pre_tags": [""],
            "post_tags": [""],
            "fields": {
                "title": {},
                "summary": {"fragment_size": 180, "number_of_fragments": 1},
                "raw_content": {"fragment_size": 200, "number_of_fragments": 1},
            },
        },
    }


def opensearch_search(
    query: str,
    top_k: int = 10,
    category: str = "",
    published_ts_gte: int | None = None,
    published_ts_lte: int | None = None,
) -> list[dict]:
    client = get_opensearch_client()
    index_name = get_opensearch_index_name()

    try:
        if not client.ping():
            raise OpenSearchUnavailableError("OpenSearch 服务不可达，请先启动本地 OpenSearch。")
        if not client.indices.exists(index=index_name):
            raise OpenSearchUnavailableError(
                f"OpenSearch 索引 '{index_name}' 不存在，请先构建词法索引。"
            )
        response = client.search(
            index=index_name,
            body=build_search_body(query, top_k, category, published_ts_gte, published_ts_lte),
        )
    except TransportError as exc:
        raise OpenSearchUnavailableError(f"OpenSearch 查询失败: {exc}") from exc

    results = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        highlight = hit.get("highlight", {})
        title = source.get("title", "")
        summary = source.get("summary", "")
        preview = (
            next(iter(highlight.get("summary", [])), "")
            or next(iter(highlight.get("raw_content", [])), "")
            or source.get("preview", "")
            or summary
            or source.get("raw_content", "")[:300]
        )

        results.append({
            "id": f"article_{source.get('article_id')}",
            "score": float(hit.get("_score") or 0.0),
            "backend": "OpenSearch",
            "metadata": {
                "article_id": source.get("article_id"),
                "category": source.get("category"),
                "source_name": source.get("source_name"),
                "published_ts": source.get("published_ts"),
                "preview": preview,
            },
            "doc": f"{title}\n{summary}".strip(),
        })

    return results
