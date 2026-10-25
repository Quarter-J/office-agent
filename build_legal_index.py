"""构建中国法律法规独立知识库索引。

只处理 Markdown 语料，先按类型/时效状态过滤，再按「第X条」聚合切块。
默认只写语料 JSON；加 --embed 才会调用 embedding 并构建 FAISS 索引。

示例：
  python build_legal_index.py --raw-dir C:\\data\\china-laws\\markdown
  python build_legal_index.py --raw-dir C:\\data\\china-laws\\markdown --embed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import settings
from src.rag.corpus import dump_chunks
from src.rag.legal.pipeline import CORE_TYPES, DEFAULT_STATUSES, ingest_legal


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="中国法律法规 RAG 入库")
    parser.add_argument("--raw-dir", default=settings.legal_raw_dir, help="法律 Markdown 根目录")
    parser.add_argument("--types", default=",".join(CORE_TYPES), help="逗号分隔的法规类型")
    parser.add_argument("--statuses", default=",".join(DEFAULT_STATUSES), help="逗号分隔的时效状态")
    parser.add_argument("--chunk-size", type=int, default=settings.legal_chunk_size, help="每块最大字符数")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 篇，便于试跑")
    parser.add_argument("--embed", action="store_true", help="构建 FAISS 向量索引（调用 embedding API）")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"❌ 法律语料目录不存在: {raw_dir}")
        return 1

    chunks, metrics, details = ingest_legal(
        raw_dir,
        types=_csv(args.types),
        statuses=_csv(args.statuses),
        chunk_size=args.chunk_size,
        limit=args.limit,
    )

    if not chunks:
        print("❌ 没有可入库的法规内容")
        return 1

    n_corpus = dump_chunks(chunks, settings.legal_corpus_path)
    report = {
        **metrics,
        "raw_dir": str(raw_dir),
        "types": list(_csv(args.types)),
        "statuses": list(_csv(args.statuses)),
        "chunk_size": args.chunk_size,
        "corpus_path": settings.legal_corpus_path,
        "vector_store_path": settings.legal_vector_store_path,
        "embedded": bool(args.embed),
    }
    report_path = Path(settings.legal_corpus_path).with_name("legal_ingest_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 64)
    print("  中国法律法规入库结果")
    print("=" * 64)
    print(f"  扫描文档       {metrics['total']}")
    print(f"  类型过滤       {metrics['skipped_type']}")
    print(f"  状态过滤       {metrics['skipped_status']}")
    print(f"  去重           {metrics['duplicates']}")
    print(f"  命中/入库      {metrics['matched']} / {metrics['selected']}")
    print(f"  有条文文档     {metrics['documents_with_articles']}")
    print(f"  无条文文档     {metrics['documents_without_articles']}")
    print(f"  条数           {metrics['articles']}")
    print(f"  知识块         {n_corpus}")
    print(f"  字符数         {metrics['chars']}")
    print(f"  语料文件       {settings.legal_corpus_path}")
    print(f"  入库报告       {report_path}")
    print("-" * 64)

    if not args.embed:
        print("  未执行向量化；加 --embed 可构建 FAISS 索引")
        print()
        return 0

    try:
        from src.rag import vector_store

        vector_store.build_index(
            chunks,
            save=True,
            save_dir=settings.legal_vector_store_path,
        )
        print(f"  ✅ FAISS 索引: {settings.legal_vector_store_path}")
        print()
        return 0
    except Exception as exc:
        print(f"  ❌ 向量化失败: {exc}")
        print("  语料 JSON 已保留，可修复 embedding 配置后重试 --embed")
        print()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
