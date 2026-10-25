"""
检索命中率评测：加载标注测试集，逐条检索，输出 hit@K 与 MRR。

用法：
  python eval_retrieval.py                       # 默认 data/eval/sample_queries.json
  python eval_retrieval.py --test-set path.json --top-k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import settings
from src.rag import vector_store
from src.rag.eval import load_test_set, run_eval

DEFAULT_TEST_SET = "data/eval/sample_queries.json"


def ensure_index():
    try:
        if vector_store.load_index() is None:
            print("未找到向量索引，请先运行 python build_index.py")
            sys.exit(1)
    except Exception as e:
        print(f"无法加载向量索引（检查 embedding 是否已配置）: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="检索命中率评测")
    parser.add_argument("--test-set", default=DEFAULT_TEST_SET, help="标注测试集 JSON 路径")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0, help="相似度阈值，默认 0=纯排名评测")
    parser.add_argument("--index-dir", default=None, help="向量索引目录（默认用配置的向量库路径）")
    args = parser.parse_args()

    if args.index_dir:
        settings.vector_store_path = args.index_dir
    ensure_index()
    test_set = load_test_set(args.test_set)
    result = run_eval(test_set, top_k=args.top_k, threshold=args.threshold)

    m = result["metrics"]
    print()
    print("=" * 62)
    print("  检索命中率评测结果")
    print("=" * 62)
    print(f"  查询数 {m['queries']}   Top-K={m['k']}   命中 {m['hits']} 条")
    print(f"  hit@{m['k']} = {m['hit_at_k'] * 100:.1f}%")
    print(f"  MRR        = {m['mrr']}")
    print()
    for q in result["per_query"]:
        if q["error"]:
            mark, rank = "⚠️", f"出错: {q['error']}"
        elif q["hit"]:
            mark, rank = "✅", f"第{q['rank']}位命中"
        else:
            mark, rank = "❌", "未命中"
        print(f"  {mark} {q['query']}  →  {rank}")
    print()


if __name__ == "__main__":
    main()
