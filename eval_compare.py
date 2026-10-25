"""
多路检索对比评测：向量 / BM25 / 混合 / 重排序，输出 hit@1/@3/@5 与 MRR。

用法：
  python eval_compare.py --test-set data/eval/medical_queries.json \
      --index-dir data/db/vector_store_medical --chunks data/db/medical_chunks.json
  python eval_compare.py ... --rerank      # 追加 LLM 重排序（调用 LLM）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import settings
from src.rag import vector_store
from src.rag.bm25 import BM25Index
from src.rag.corpus import load_chunks
from src.rag.eval import load_test_set
from src.rag.eval_compare import run_comparison
from src.rag.hybrid import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="多路检索对比评测")
    parser.add_argument("--test-set", default="data/eval/medical_queries.json")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--chunks", default="data/db/medical_chunks.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="追加 LLM 重排序（调用 LLM）")
    args = parser.parse_args()

    if args.index_dir:
        settings.vector_store_path = args.index_dir
    if vector_store.load_index() is None:
        print("未找到向量索引，先运行 build_medical_index.py --embed")
        sys.exit(1)

    test_set = load_test_set(args.test_set)
    chunks = load_chunks(args.chunks)
    if not chunks:
        print(f"未找到语料文件 {args.chunks}，先运行 build_medical_index.py")
        sys.exit(1)

    bm25 = BM25Index()
    bm25.index(chunks)
    hybrid = HybridRetriever(bm25)

    retrievers = {
        "vector": lambda q, k: vector_store.search(q, top_k=k, threshold=0.0),
        "bm25": lambda q, k: bm25.search(q, k=k),
        "hybrid": lambda q, k: hybrid.search(q, k=k, threshold=0.0),
    }
    if args.rerank:
        from src.rag.rerank import rerank
        retrievers["hybrid+rerank"] = lambda q, k: rerank(
            q, hybrid.search(q, k=max(k, 8), threshold=0.0), k=k)

    result = run_comparison(test_set, retrievers, top_k=args.top_k, ks=(1, 3, 5))

    print()
    print("=" * 62)
    print("  多路检索对比评测")
    print("=" * 62)
    print(f"  查询数 {len(test_set)}   Top-K={args.top_k}")
    print(f"  {'方法':<16}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    print("-" * 62)
    for name, r in result.items():
        s = r["summary"]
        print(f"  {name:<16}{s['hit@1']*100:>7.1f}%{s['hit@3']*100:>7.1f}%"
              f"{s['hit@5']*100:>7.1f}%{s['mrr']:>8.4f}")
    print("-" * 62)
    print("  逐查询排名（— 表示未命中）：")
    methods = list(result.keys())
    print(f"  {'查询':<28}" + "".join(f"{m:>14}" for m in methods))
    for i, case in enumerate(test_set):
        q = case["query"][:26]
        cols = ""
        for m in methods:
            r = result[m]["per_query"][i]["rank"]
            cols += f"{('第' + str(r) + '位') if r else '—':>14}"
        print(f"  {q:<28}{cols}")
    print()


if __name__ == "__main__":
    main()
