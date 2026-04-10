"""RAG 混合检索工具：向量检索 + 词法召回 + 本地 reranker 重排。"""
import os
import sys
import time
from pathlib import Path

import chromadb
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bm25_search import bm25_search
from config import (
    COLLECTION_NAME,
    DEFAULT_MAX_RESULTS,
    EMBEDDING_MODEL,
    LEXICAL_BACKEND,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    VECTORDB_DIR,
)
from opensearch_search import OpenSearchUnavailableError, opensearch_search
from reranker import rerank_candidates

_collection = None
_embedding_model = None
_init_lock = __import__("threading").Lock()


def _force_hf_offline() -> None:
    """强制 Hugging Face 只使用本地缓存，避免运行时意外联网。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def get_collection():
    global _collection
    if _collection is None:
        with _init_lock:
            if _collection is None:
                client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
                _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _init_lock:
            if _embedding_model is None:
                # 查询只有单条，优先稳定性，统一在 CPU 上做 embedding。
                _force_hf_offline()
                _embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    device="cpu",
                    local_files_only=True,
                    tokenizer_kwargs={"local_files_only": True},
                    config_kwargs={"local_files_only": True},
                )
    return _embedding_model


def embed_query(query: str) -> list[float]:
    embedding = get_embedding_model().encode(
        query,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def _collect_candidates(vec_results: dict, lexical_hits: list[dict]) -> list[dict]:
    """合并向量和词法候选，并保留召回来源，供 reranker 统一重排。"""
    merged_by_id: dict[str, dict] = {}

    for i in range(len(vec_results["ids"][0])):
        item = {
            "id": vec_results["ids"][0][i],
            "doc": vec_results["documents"][0][i],
            "metadata": vec_results["metadatas"][0][i],
            "sources": ["向量"],
        }
        merged_by_id[item["id"]] = item

    for hit in lexical_hits:
        existing = merged_by_id.get(hit["id"])
        if existing is None:
            lexical_label = hit.get("backend", "BM25")
            merged_by_id[hit["id"]] = {
                "id": hit["id"],
                "doc": hit.get("doc", ""),
                "metadata": hit["metadata"],
                "sources": [lexical_label],
            }
            continue

        lexical_label = hit.get("backend", "BM25")
        if lexical_label not in existing["sources"]:
            existing["sources"].append(lexical_label)
        if not existing.get("doc") and hit.get("doc"):
            existing["doc"] = hit["doc"]

    return list(merged_by_id.values())


def _lexical_search(
    query: str,
    top_k: int,
    category: str,
    published_ts_gte: int | None,
    published_ts_lte: int | None,
) -> list[dict]:
    backend = LEXICAL_BACKEND
    if backend == "bm25":
        return bm25_search(
            query,
            top_k=top_k,
            category=category,
            published_ts_gte=published_ts_gte,
            published_ts_lte=published_ts_lte,
        )

    try:
        return opensearch_search(
            query,
            top_k=top_k,
            category=category,
            published_ts_gte=published_ts_gte,
            published_ts_lte=published_ts_lte,
        )
    except OpenSearchUnavailableError:
        if backend != "auto":
            raise
        return bm25_search(
            query,
            top_k=top_k,
            category=category,
            published_ts_gte=published_ts_gte,
            published_ts_lte=published_ts_lte,
        )


@tool
def rag_search(query: str, days: int = 0, category: str = "",
               top_k: int = 10, start_date: str = "", end_date: str = "") -> str:
    """搜索本地新闻数据库，先做混合召回，再用本地 reranker 统一重排。

    Args:
        query: 搜索关键词
        days: 搜索最近 N 天（向后兼容，优先使用 start_date/end_date）
        category: 分类筛选（AI / GAMES / ""）
        top_k: 最多返回条数
        start_date: 起始日期 YYYY-MM-DD（含）
        end_date: 结束日期 YYYY-MM-DD（含）
    """
    from datetime import datetime

    if top_k <= 0:
        top_k = DEFAULT_MAX_RESULTS

    where_clauses = []
    ts_gte = None
    ts_lte = None

    # 优先使用精确日期范围
    if start_date:
        ts_gte = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        where_clauses.append({"published_ts": {"$gte": ts_gte}})
    if end_date:
        # end_date 当天的 23:59:59
        ts_lte = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) + 86399
        where_clauses.append({"published_ts": {"$lte": ts_lte}})

    # 向后兼容：如果没指定日期但指定了 days
    if not start_date and not end_date and days > 0:
        ts_gte = int(time.time() - days * 24 * 3600)
        where_clauses.append({"published_ts": {"$gte": ts_gte}})

    if category:
        where_clauses.append({"category": {"$eq": category}})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    candidate_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)

    vec = get_collection().query(
        query_embeddings=[embed_query(query)],
        where=where,
        n_results=candidate_k,
    )
    lexical_hits = _lexical_search(
        query,
        top_k=candidate_k,
        category=category,
        published_ts_gte=ts_gte,
        published_ts_lte=ts_lte,
    )

    candidates = _collect_candidates(vec, lexical_hits)
    reranked = rerank_candidates(query, candidates)[:top_k]
    if not reranked:
        return f"查询 '{query}' 没有找到匹配的新闻。"

    output = []
    for i, item in enumerate(reranked, start=1):
        meta = item["metadata"]
        doc = item.get("doc", "")
        title = doc.split("\n")[0] if doc else f"article_{meta.get('article_id', '?')}"
        source_label = "+".join(item.get("sources", [])) or "unknown"
        output.append(f"--- 结果 {i} [{source_label}] ---")
        output.append(f"标题: {title}")
        output.append(f"元数据: [{meta.get('category')}] | [{meta.get('source_name')}]")
        output.append(f"Rerank分数: {item.get('rerank_score', 0.0):.4f}")
        output.append(f"预览: {meta.get('preview', '')[:300]}\n")
    return "\n".join(output)
