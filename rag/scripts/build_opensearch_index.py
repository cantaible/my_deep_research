"""构建 OpenSearch 词法索引：读取文章 → 清洗 → 写入 OpenSearch。"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from opensearchpy.helpers import bulk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opensearch_client import (  # noqa: E402
    build_news_index_body,
    get_opensearch_client,
    get_opensearch_index_name,
)
from scripts.clean_html import clean_html  # noqa: E402
from scripts.db import fetch_all_articles  # noqa: E402


def parse_timestamp(published_at: str) -> int:
    if not published_at:
        return 0
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return 0


def iter_actions(articles: list[dict], index_name: str):
    for art in articles:
        title = (art.get("title") or "").strip()
        summary = clean_html(art.get("summary") or "")
        raw_content = clean_html(art.get("raw_content") or "")
        preview = raw_content[:500] if raw_content else summary[:500]
        source_name = (art.get("source_name") or "").strip()

        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": f"article_{art['id']}",
            "_source": {
                "article_id": int(art["id"]),
                "category": art.get("category") or "UNCATEGORIZED",
                "source_name": source_name,
                "source_name_text": source_name,
                "published_ts": parse_timestamp(art.get("published_at") or ""),
                "title": title,
                "summary": summary,
                "preview": preview,
                "raw_content": raw_content,
            },
        }


def build(recreate: bool = False) -> None:
    print("1/3 从 MariaDB 读取文章...")
    articles = fetch_all_articles()
    print(f"    共 {len(articles)} 篇")

    print("2/3 连接 OpenSearch...")
    client = get_opensearch_client()
    if not client.ping():
        raise RuntimeError("OpenSearch 服务不可达，请先启动 docker compose。")

    index_name = get_opensearch_index_name()
    if recreate and client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=build_news_index_body())

    print("3/3 写入 OpenSearch 词法索引...")
    success, errors = bulk(client, iter_actions(articles, index_name), chunk_size=200, raise_on_error=False)
    client.indices.refresh(index=index_name)
    print(f"    已写入 {success} 条文档")
    if errors:
        print(f"    警告: 有 {len(errors)} 条写入失败")
    print(f"✅ OpenSearch 索引构建完成: {index_name}")


if __name__ == "__main__":
    recreate = "--force" in sys.argv
    if recreate:
        print("⚠️  --force 已指定，将重建 OpenSearch 索引...")
    build(recreate=recreate)
