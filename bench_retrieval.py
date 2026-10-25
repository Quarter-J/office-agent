"""
检索延迟压测：对一组查询反复调用语义检索，输出 P50/P90/P95/P99 延迟（毫秒）。

注意：vector_store.search 每次调用都会重新 load_index()（从磁盘加载 FAISS），
因此测得的是「磁盘加载 + embedding + 检索」的端到端单次延迟，即真实线上路径。

用法：
  python bench_retrieval.py                          # 内置查询集，默认 20 轮
  python bench_retrieval.py --queries q.json --runs 50 --top-k 5
  python bench_retrieval.py --threshold 0.0          # 关闭相似度过滤（纯排名）
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import settings
from src.rag import vector_store
from src.rag.metrics import latency_summary

DEFAULT_QUERIES = [
    "Spring Boot 自动装配怎么配置",
    "应用配置文件 application.yml",
    "公司通讯录 技术部",
]


def load_queries(path=None):
    if path is None:
        return DEFAULT_QUERIES
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("查询文件必须是 JSON 数组")
    return [
        item["query"] if isinstance(item, dict) and item.get("query") else str(item)
        for item in data
    ]


def ensure_index():
    try:
        if vector_store.load_index() is None:
            print("未找到向量索引，请先运行 python build_index.py")
            sys.exit(1)
    except Exception as e:
        print(f"无法加载向量索引（检查 embedding 是否已配置）: {e}")
        sys.exit(1)


def time_search(query, top_k, threshold):
    start = time.perf_counter()
    vector_store.search(query, top_k=top_k, threshold=threshold)
    return (time.perf_counter() - start) * 1000.0


def main():
    parser = argparse.ArgumentParser(description="检索延迟压测")
    parser.add_argument("--queries", help="查询 JSON 文件路径（字符串数组或含 query 字段的对象数组）")
    parser.add_argument("--runs", type=int, default=20, help="每查询重复次数")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None, help="相似度阈值，默认用配置值")
    parser.add_argument("--index-dir", default=None, help="向量索引目录（默认用配置的向量库路径）")
    args = parser.parse_args()

    if args.index_dir:
        settings.vector_store_path = args.index_dir
    ensure_index()
    queries = load_queries(args.queries)

    # 预热一次：初始化 embedding 单例，避免冷启动污染首个样本
    time_search(queries[0], args.top_k, args.threshold)

    all_samples = []
    per_query = {q: [] for q in queries}
    for q in queries:
        for _ in range(args.runs):
            ms = time_search(q, args.top_k, args.threshold)
            all_samples.append(ms)
            per_query[q].append(ms)

    summary = latency_summary(all_samples)
    print()
    print("=" * 62)
    print("  检索延迟压测结果（毫秒）")
    print("=" * 62)
    print(f"  查询数 {len(queries)} × 每查询 {args.runs} 轮 = {summary['count']} 次")
    print(f"  汇总  mean={summary['mean_ms']}  p50={summary['p50_ms']}"
          f"  p90={summary['p90_ms']}  p95={summary['p95_ms']}"
          f"  p99={summary['p99_ms']}  max={summary['max_ms']}")
    print()
    for q, samples in per_query.items():
        s = latency_summary(samples)
        print(f"  · {q}")
        print(f"      mean={s['mean_ms']}  p50={s['p50_ms']}  p95={s['p95_ms']}  max={s['max_ms']}")
    print()


if __name__ == "__main__":
    main()
