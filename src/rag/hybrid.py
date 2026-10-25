"""
混合检索：向量 + BM25 的倒数排名融合（RRF）。
"""

from __future__ import annotations

from src.rag import vector_store
from src.rag.bm25 import BM25Index


def _chunk_key(hit) -> str:
    meta = hit.meta or {}
    return f"{meta.get('source', '')}::{meta.get('heading_path', '')}::{hit.text}"


def rrf_fuse(ranked_lists, k=60, weights=None):
    """按 RRF 融合多路排序结果，返回融合后的 hit 列表（按融合分降序）。

    weights 可给不同召回路不同权重；默认等权。
    """
    fused = {}
    items = {}
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, hit in enumerate(ranked, start=1):
            key = _chunk_key(hit)
            items.setdefault(key, hit)
            fused[key] = fused.get(key, 0.0) + float(weight) / (k + rank)
    order = sorted(fused, key=lambda key: fused[key], reverse=True)
    return [items[key] for key in order]


class HybridRetriever:
    def __init__(self, bm25: BM25Index, top_n=20):
        self.bm25 = bm25
        self.top_n = top_n

    def search(self, query, k=3, threshold=0.0):
        vector_hits = vector_store.search(query, top_k=self.top_n, threshold=threshold)
        bm25_hits = self.bm25.search(query, k=self.top_n)
        return rrf_fuse([vector_hits, bm25_hits])[:k]
