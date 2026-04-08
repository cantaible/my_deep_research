"""BM25 关键词检索模块，和向量检索互补。"""
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COLLECTION_NAME, VECTORDB_DIR
from text_analyzer import analyze_text

_bm25 = None        # BM25 索引缓存
_doc_ids = None     # 文档 ID 列表，和 BM25 索引一一对应
_doc_metas = None   # 文档 metadata 列表
_doc_texts = None   # 原始文档文本，用于标题展示


def _tokenize(text: str) -> list[str]:
    """对外保留旧接口，内部统一走独立 analyzer。"""
    return analyze_text(text)


def _matches_filters(metadata: dict, category: str,
                     published_ts_gte: int | None, published_ts_lte: int | None = None) -> bool:
    if category and metadata.get("category") != category:
        return False
    ts = int(metadata.get("published_ts", 0))
    if published_ts_gte is not None and ts < published_ts_gte:
        return False
    if published_ts_lte is not None and ts > published_ts_lte:
        return False
    return True


def _load_bm25():
    """从 ChromaDB 读取所有文档文本，构建 BM25 索引。首次调用约 2-3 秒。"""
    global _bm25, _doc_ids, _doc_metas, _doc_texts
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    col = client.get_collection(name=COLLECTION_NAME)
    all_data = col.get(include=["documents", "metadatas"])

    _doc_ids = all_data["ids"]
    _doc_metas = all_data["metadatas"]
    _doc_texts = all_data["documents"]
    corpus = [_tokenize(doc) for doc in _doc_texts]
    _bm25 = BM25Okapi(corpus)


def bm25_search(
    query: str,
    top_k: int = 10,
    category: str = "",
    published_ts_gte: int | None = None,
    published_ts_lte: int | None = None,
) -> list[dict]:
    """BM25 关键词检索，返回 [{id, score, metadata, doc}, ...]。"""
    if _bm25 is None:
        _load_bm25()
    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue

        metadata = _doc_metas[idx]
        if not _matches_filters(
            metadata,
            category=category,
            published_ts_gte=published_ts_gte,
            published_ts_lte=published_ts_lte,
        ):
            continue

        results.append({
            "id": _doc_ids[idx],
            "score": float(scores[idx]),
            "metadata": metadata,
            "doc": _doc_texts[idx],
        })
        if len(results) >= top_k:
            break
    return results
