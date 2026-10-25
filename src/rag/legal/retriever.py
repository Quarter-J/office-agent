"""法律法规独立向量索引检索。"""

from __future__ import annotations

import threading
from pathlib import Path

from src.config.settings import settings
from src.rag.validation import validate_top_k
from src.rag.vector_store import ScoredChunk, get_embeddings

_LEGAL_STORE = None
_LEGAL_LOCK = threading.Lock()


def _index_path() -> Path:
    return Path(settings.legal_vector_store_path)


def clear_cache() -> None:
    global _LEGAL_STORE
    with _LEGAL_LOCK:
        _LEGAL_STORE = None


def load_index():
    global _LEGAL_STORE
    if _LEGAL_STORE is None:
        with _LEGAL_LOCK:
            if _LEGAL_STORE is None:
                from langchain_community.vectorstores import FAISS

                path = _index_path()
                if not path.exists():
                    return None
                _LEGAL_STORE = FAISS.load_local(
                    str(path), get_embeddings(), allow_dangerous_deserialization=True
                )
    return _LEGAL_STORE


def _similarity(score: float) -> float:
    return 1.0 / (1.0 + float(score))


def search(query: str, top_k: int = 5, threshold: float = None, *, raise_errors: bool = False):
    """检索法律条文，返回 ScoredChunk 列表。"""
    validate_top_k(top_k)
    threshold = settings.similarity_threshold if threshold is None else threshold
    try:
        store = load_index()
        if store is None:
            raise FileNotFoundError(
                "法律向量索引不存在，请先运行 build_legal_index.py --embed"
            )
        results = []
        for doc, score in store.similarity_search_with_score(query, k=top_k):
            similarity = _similarity(score)
            if similarity < threshold:
                continue
            results.append(ScoredChunk(text=doc.page_content, meta=doc.metadata, score=similarity))
        return results
    except Exception:
        if raise_errors:
            raise
        return []


def source_label(meta: dict) -> str:
    title = str(meta.get("title") or "").strip()
    parts = [f"《{title}》" if title else str(meta.get("source") or "未知法规")]
    start = str(meta.get("article_start") or "")
    end = str(meta.get("article_end") or "")
    if start:
        parts.append(start if start == end or not end else f"{start}-{end}")
    heading = str(meta.get("heading_path") or "").strip()
    if heading and heading not in parts:
        parts.append(heading)
    return " · ".join(parts)


def citation(meta: dict) -> dict:
    return {
        "title": str(meta.get("title") or ""),
        "type": str(meta.get("type") or ""),
        "status": str(meta.get("status") or ""),
        "article_start": str(meta.get("article_start") or ""),
        "article_end": str(meta.get("article_end") or ""),
        "url": str(meta.get("url") or ""),
        "source": str(meta.get("source") or ""),
    }
