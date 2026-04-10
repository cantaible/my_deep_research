"""OpenSearch 客户端与索引定义。"""
from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from opensearchpy import OpenSearch

try:
    from .config import (
        OPENSEARCH_INDEX_NAME,
        OPENSEARCH_PASSWORD,
        OPENSEARCH_TIMEOUT,
        OPENSEARCH_URL,
        OPENSEARCH_USERNAME,
    )
except ImportError:
    from config import (
        OPENSEARCH_INDEX_NAME,
        OPENSEARCH_PASSWORD,
        OPENSEARCH_TIMEOUT,
        OPENSEARCH_URL,
        OPENSEARCH_USERNAME,
    )


def _build_hosts() -> list[dict]:
    parsed = urlparse(OPENSEARCH_URL)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 9200
    return [{"host": host, "port": port, "scheme": scheme}]


@lru_cache(maxsize=1)
def get_opensearch_client() -> OpenSearch:
    kwargs = {
        "hosts": _build_hosts(),
        "timeout": OPENSEARCH_TIMEOUT,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
    }
    if OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD:
        kwargs["http_auth"] = (OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD)
    return OpenSearch(**kwargs)


def get_opensearch_index_name() -> str:
    return OPENSEARCH_INDEX_NAME


def build_news_index_body() -> dict:
    """新闻索引映射。

    目标：
    1. 中文优先，使用内置 cjk analyzer 处理中文；
    2. 英文补充，使用 english analyzer 处理英文与词干；
    3. 字段加权时能区分 title / summary / raw_content。
    """
    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "zh_text": {
                        "type": "cjk",
                    },
                    "en_text": {
                        "type": "english",
                    },
                },
            },
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
        },
        "mappings": {
            "properties": {
                "article_id": {"type": "integer"},
                "category": {"type": "keyword"},
                "source_name": {"type": "keyword"},
                "published_ts": {"type": "long"},
                "title": {
                    "type": "text",
                    "analyzer": "zh_text",
                    "copy_to": ["search_text"],
                    "fields": {
                        "en": {"type": "text", "analyzer": "en_text"},
                        "keyword": {"type": "keyword", "ignore_above": 512},
                    },
                },
                "summary": {
                    "type": "text",
                    "analyzer": "zh_text",
                    "copy_to": ["search_text"],
                    "fields": {
                        "en": {"type": "text", "analyzer": "en_text"},
                    },
                },
                "preview": {
                    "type": "text",
                    "analyzer": "zh_text",
                    "copy_to": ["search_text"],
                    "fields": {
                        "en": {"type": "text", "analyzer": "en_text"},
                    },
                },
                "raw_content": {
                    "type": "text",
                    "analyzer": "zh_text",
                    "copy_to": ["search_text"],
                    "fields": {
                        "en": {"type": "text", "analyzer": "en_text"},
                    },
                },
                "source_name_text": {
                    "type": "text",
                    "analyzer": "zh_text",
                    "copy_to": ["search_text"],
                    "fields": {
                        "en": {"type": "text", "analyzer": "en_text"},
                    },
                },
                "search_text": {
                    "type": "text",
                    "analyzer": "zh_text",
                    "fields": {
                        "en": {"type": "text", "analyzer": "en_text"},
                    },
                },
            },
        },
    }
