"""本地 reranker：对候选文档做统一重排。"""
import os

import torch
from sentence_transformers import CrossEncoder

from config import RERANKER_MAX_LENGTH, RERANKER_MODEL


def get_reranker_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_reranker = None
_reranker_lock = __import__("threading").Lock()


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                _reranker = CrossEncoder(
                    RERANKER_MODEL,
                    max_length=RERANKER_MAX_LENGTH,
                    device=get_reranker_device(),
                    local_files_only=True,
                )
    return _reranker


def rerank_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """对候选文档统一打分并按分数降序返回。"""
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, candidate.get("doc", "")) for candidate in candidates]
    scores = model.predict(pairs)

    reranked = []
    for candidate, score in zip(candidates, scores):
        updated = dict(candidate)
        updated["rerank_score"] = float(score)
        reranked.append(updated)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked
