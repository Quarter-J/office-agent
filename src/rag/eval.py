"""
检索质量评测：Top-K 命中率（hit@K / recall@K）与 MRR。

测试集为 JSON 数组，每项：
  {"query": "问题", "relevant": [...]}
relevant 元素支持两种粒度：
  - 字符串：精确匹配命中块的 meta["source"]（文档级）
  - 对象  ：{"source": "...", "section": "..."}，source 精确匹配 +
            section 匹配 meta["section"] 或 meta["heading_path"]（段落级）

命中口径：Top-K 结果中命中任意一个 relevant 项即算该问命中。
"""

from __future__ import annotations

import json
from typing import Optional


def match_relevant(relevant, meta: dict) -> bool:
    """判断单个 relevant 规格是否命中某个 chunk 的元数据。"""
    if isinstance(relevant, str):
        return meta.get("source") == relevant
    if isinstance(relevant, dict):
        if meta.get("source") != relevant.get("source"):
            return False
        section = (relevant.get("section") or "").strip()
        if not section:
            return True
        return section == (meta.get("section") or meta.get("heading_path") or "")
    return False


def first_relevant_rank(relevant, metas, k: int) -> Optional[int]:
    """返回 Top-K 元数据列表中首个 relevant 项的排名（1 起）；未命中返回 None。"""
    for i, meta in enumerate(metas[:k], start=1):
        if any(match_relevant(r, meta) for r in relevant):
            return i
    return None


def compute_metrics(per_query_ranks, k: int) -> dict:
    """由每问首个相关命中排名聚合 hit@K 与 MRR。None 表示未命中/出错。"""
    total = len(per_query_ranks)
    if total == 0:
        return {"k": k, "queries": 0, "hits": 0, "hit_at_k": 0.0, "mrr": 0.0}
    hits = [r for r in per_query_ranks if r is not None]
    reciprocal_sum = sum(1.0 / r for r in hits)
    return {
        "k": k,
        "queries": total,
        "hits": len(hits),
        "hit_at_k": round(len(hits) / total, 4),
        "mrr": round(reciprocal_sum / total, 4),
    }


def load_test_set(path) -> list:
    """读取并校验测试集 JSON，返回 [{"query","relevant"}, ...]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("测试集必须是 JSON 数组")
    cases = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("query") or not item.get("relevant"):
            raise ValueError(f"第 {i} 项缺少 query 或 relevant")
        cases.append({"query": item["query"], "relevant": item["relevant"]})
    return cases


def run_eval(test_set, top_k: int = 3, threshold: float = 0.0) -> dict:
    """对测试集逐条检索并计算命中率。

    threshold=0.0 表示不做相似度过滤、按纯 Top-K 排名评测（评测排序而非阈值）。
    单条检索失败不计入命中，错误原因写入该条的 error 字段。
    """
    from src.rag import vector_store

    per_query = []
    for case in test_set:
        query = case["query"]
        relevant = case["relevant"]
        rank = None
        error = None
        try:
            hits = vector_store.search(query, top_k=top_k, threshold=threshold)
            rank = first_relevant_rank(relevant, [h.meta for h in hits], top_k)
        except Exception as e:  # 单条失败不影响整体
            error = str(e)
        per_query.append({
            "query": query,
            "rank": rank,
            "hit": rank is not None,
            "error": error,
        })

    return {
        "metrics": compute_metrics([q["rank"] for q in per_query], top_k),
        "per_query": per_query,
    }
