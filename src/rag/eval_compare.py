"""
多路检索对比评测：同一测试集上跑多种检索方案，输出 hit@1/@3/@5 与 MRR。
"""

from __future__ import annotations

from src.rag.eval import first_relevant_rank


def hit_at_k(ranks, k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def summary(ranks, ks=(1, 3, 5)) -> dict:
    total = len(ranks)
    if total == 0:
        return {"queries": 0, "mrr": 0.0, **{f"hit@{k}": 0.0 for k in ks}}
    hits = [r for r in ranks if r is not None]
    mrr = sum(1.0 / r for r in hits) / total
    return {
        "queries": total,
        "mrr": round(mrr, 4),
        **{f"hit@{k}": round(hit_at_k(ranks, k), 4) for k in ks},
    }


def run_comparison(test_set, retrievers, top_k=5, ks=(1, 3, 5)):
    """retrievers：{名称: fn(query, k) -> list[hit]}。

    返回 {名称: {"summary": {...}, "per_query": [{query, rank, hit}]}}。
    """
    out = {}
    for name, fn in retrievers.items():
        ranks = []
        per_query = []
        for case in test_set:
            hits = fn(case["query"], top_k)
            rank = first_relevant_rank(case["relevant"], [h.meta for h in hits], top_k)
            ranks.append(rank)
            per_query.append({"query": case["query"], "rank": rank, "hit": rank is not None})
        out[name] = {"summary": summary(ranks, ks), "per_query": per_query}
    return out
