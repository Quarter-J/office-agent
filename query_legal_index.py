"""命令行查询法律知识库索引。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag.legal.retriever import citation, search, source_label


def main() -> int:
    parser = argparse.ArgumentParser(description="查询法律知识库")
    parser.add_argument("query", help="自然语言问题或法条关键词")
    parser.add_argument("--top-k", type=int, default=5, help="返回条数，1-20")
    args = parser.parse_args()

    try:
        hits = search(args.query, top_k=args.top_k, raise_errors=True)
    except Exception as exc:
        print(f"❌ 检索失败: {exc}")
        return 1

    if not hits:
        print("未找到相关法律条文")
        return 0

    output = []
    for rank, hit in enumerate(hits, start=1):
        output.append({
            "rank": rank,
            "score": round(hit.score, 4),
            "source": source_label(hit.meta),
            "citation": citation(hit.meta),
            "content": hit.text,
        })
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
