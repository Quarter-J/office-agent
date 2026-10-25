"""法律知识库混合检索：向量召回 + BM25 召回 + RRF 融合。"""

from __future__ import annotations

import threading

from src.config.settings import settings
from src.rag.bm25 import BM25Index
from src.rag.corpus import load_chunks
from src.rag.hybrid import rrf_fuse
from src.rag.legal.retriever import search as vector_search

_BM25 = None
_BM25_LOCK = threading.Lock()


def _get_bm25() -> BM25Index:
    global _BM25
    if _BM25 is None:
        with _BM25_LOCK:
            if _BM25 is None:
                chunks = load_chunks(settings.legal_corpus_path)
                if not chunks:
                    raise FileNotFoundError(
                        "法律语料不存在，请先运行 build_legal_index.py"
                    )
                index = BM25Index()
                index.index(chunks)
                _BM25 = index
    return _BM25


def clear_cache() -> None:
    global _BM25
    with _BM25_LOCK:
        _BM25 = None


def hybrid_search(query: str, top_k: int = 5, candidate_k: int = 20):
    """向量 + BM25 混合召回，再按 RRF 融合。"""
    bm25 = _get_bm25()
    vector_hits = vector_search(
        query, top_k=candidate_k, threshold=0.0, raise_errors=True
    )
    keyword_hits = bm25.search(query, k=candidate_k)
    # 法条语义匹配优先；BM25 用于补足法规名、条文号和专有名词的精确命中。
    return rrf_fuse([vector_hits, keyword_hits], weights=[2.0, 1.0])[:top_k]
