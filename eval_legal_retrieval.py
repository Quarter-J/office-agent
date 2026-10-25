"""法律法规检索评测：向量召回 vs 向量+BM25 混合召回。"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import settings
from src.rag.corpus import load_chunks
from src.rag.legal.hybrid import hybrid_search
from src.rag.legal.retriever import search as vector_search

QUERIES = [
    ("劳动合同试用期最长可以约定多久", "中华人民共和国劳动合同法", "第十九条"),
    ("劳动者提前多少天通知可以解除劳动合同", "中华人民共和国劳动合同法", "第三十七条"),
    ("用人单位应当自用工之日起多久内订立书面劳动合同", "中华人民共和国劳动合同法", "第十条"),
    ("劳动合同解除或终止的经济补偿怎么计算", "中华人民共和国劳动合同法", "第四十七条"),
    ("协议离婚的冷静期是多少天", "中华人民共和国民法典", "第一千零七十七条"),
    ("高空抛物造成他人损害由谁承担责任", "中华人民共和国民法典", "第一千二百五十四条"),
    ("普通民事诉讼时效期间是几年", "中华人民共和国民法典", "第一百八十八条"),
    ("法定继承的第一顺序继承人包括哪些人", "中华人民共和国民法典", "第一千一百二十七条"),
    ("夫妻共同债务如何认定", "中华人民共和国民法典", "第一千零六十四条"),
    ("正当防卫造成不法侵害人损害是否负刑事责任", "中华人民共和国刑法", "第二十条"),
    ("处理敏感个人信息需要取得什么同意", "中华人民共和国个人信息保护法", "第二十八条"),
    ("网络运营者收集使用个人信息应遵循什么原则", "中华人民共和国网络安全法", "第四十一条"),
    ("网购商品七天无理由退货的条件", "中华人民共和国消费者权益保护法", "第二十五条"),
    ("产品存在缺陷造成人身损害由生产者承担什么责任", "中华人民共和国产品质量法", "第四十一条"),
    ("机动车第三者责任强制保险制度", "中华人民共和国道路交通安全法", "第十七条"),
    ("有限责任公司股东认缴出资额应在几年内缴足", "中华人民共和国公司法", "第四十七条"),
    ("行政处罚听证的适用范围", "中华人民共和国行政处罚法", "第六十三条"),
    ("申请行政复议的期限是多久", "中华人民共和国行政复议法", "第二十条"),
    ("哪些情形应当认定为工伤", "工伤保险条例", "第十四条"),
]


def _chunk_key(item) -> str:
    meta = item.get("meta", {}) if isinstance(item, dict) else item.meta
    text = item.get("text", "") if isinstance(item, dict) else item.text
    meta = meta or {}
    return f"{meta.get('source', '')}::{meta.get('heading_path', '')}::{text}"


def _is_relevant(meta: dict, text: str, title: str, article: str) -> bool:
    """只匹配条文标题，排除其他条文中对目标条号的引用。"""
    if meta.get("title") != title:
        return False
    return re.search(rf"(?m)^\s*{re.escape(article)}(?=$|[\s　])", text) is not None


def _first_rank(hits, gold_keys: set[str]) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if _chunk_key(hit) in gold_keys:
            return index
    return None


def _metrics(ranked_sets, gold_sets, ks=(1, 3, 5, 10, 20)) -> dict:
    total = len(ranked_sets)
    output = {"queries": total}
    if total == 0:
        for k in ks:
            output[f"hit@{k}"] = 0.0
            output[f"recall@{k}"] = 0.0
        output["mrr"] = 0.0
        return output

    for k in ks:
        hits = 0
        recall_sum = 0.0
        for ranked, gold in zip(ranked_sets, gold_sets):
            retrieved = {_chunk_key(hit) for hit in ranked[:k]}
            found = len(retrieved & gold)
            hits += bool(found)
            recall_sum += found / len(gold)
        output[f"hit@{k}"] = round(hits / total, 4)
        output[f"recall@{k}"] = round(recall_sum / total, 4)

    ranks = [_first_rank(ranked, gold) for ranked, gold in zip(ranked_sets, gold_sets)]
    output["mrr"] = round(sum(1 / rank for rank in ranks if rank) / total, 4)
    return output


def _p95(values) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return round(ordered[index] * 1000, 2)


def main() -> int:
    chunks = load_chunks(settings.legal_corpus_path)
    if not chunks:
        print("❌ 法律语料不存在，请先运行 build_legal_index.py")
        return 1

    by_title: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_title.setdefault(chunk["meta"].get("title", ""), []).append(chunk)
    valid_queries = []
    for query, title, article in QUERIES:
        gold = {
            _chunk_key(chunk)
            for chunk in by_title.get(title, [])
            if _is_relevant(chunk["meta"], chunk["text"], title, article)
        }
        if gold:
            valid_queries.append((query, title, article, gold))
        else:
            print(f"⚠️ 跳过评测项（语料中未定位到）: {title} {article}")

    vector_rankings = []
    hybrid_rankings = []
    gold_sets = []
    vector_latencies = []
    hybrid_latencies = []

    # 预热索引和 BM25，避免把首次加载时间算进查询延迟。
    vector_search(valid_queries[0][0], top_k=20, threshold=0.0, raise_errors=True)
    hybrid_search(valid_queries[0][0], top_k=20)

    for query, title, article, gold in valid_queries:
        for _ in range(3):
            start = time.perf_counter()
            vector_search(query, top_k=5, threshold=0.0, raise_errors=True)
            vector_latencies.append(time.perf_counter() - start)
        vector_rankings.append(vector_search(query, top_k=20, threshold=0.0, raise_errors=True))

        for _ in range(3):
            start = time.perf_counter()
            hybrid_search(query, top_k=5)
            hybrid_latencies.append(time.perf_counter() - start)
        hybrid_rankings.append(hybrid_search(query, top_k=20))
        gold_sets.append(gold)

    result = {
        "dataset": "legal_curated_19",
        "relevance": "title + exact article heading",
        "recall_definition": "retrieved gold chunks / all gold chunks",
        "corpus_chunks": len(chunks),
        "queries": len(valid_queries),
        "vector": {
            **_metrics(vector_rankings, gold_sets),
            "p95_latency_ms": _p95(vector_latencies),
            "median_latency_ms": round(statistics.median(vector_latencies) * 1000, 2),
        },
        "hybrid": {
            **_metrics(hybrid_rankings, gold_sets),
            "p95_latency_ms": _p95(hybrid_latencies),
            "median_latency_ms": round(statistics.median(hybrid_latencies) * 1000, 2),
        },
        "details": [
            {
                "query": query,
                "title": title,
                "article": article,
                "gold_chunks": len(gold),
                "vector_rank": _first_rank(vector_hits, gold),
                "hybrid_rank": _first_rank(hybrid_hits, gold),
            }
            for (query, title, article, gold), vector_hits, hybrid_hits in zip(
                valid_queries, vector_rankings, hybrid_rankings
            )
        ],
    }
    output = Path("data/eval/legal_retrieval_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
