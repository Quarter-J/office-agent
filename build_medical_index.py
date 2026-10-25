"""
医学知识库入库：规则归一化 + 术语映射 + 脏数据拦截 + 分段向量化。

用法：
  python build_medical_index.py                 # 入库并打印指标（不向量化，零成本）
  python build_medical_index.py --embed         # 额外构建 FAISS 向量索引（调用 embedding API）
  python build_medical_index.py --raw-dir ... --terminology ...
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import settings
from src.rag.corpus import dump_chunks
from src.rag.medical.pipeline import ingest_medical
from src.rag.medical.terminology import TerminologyMapper

DEFAULT_RAW_DIR = "data/medical/raw"
DEFAULT_TERMS = "data/medical/terminology.json"


def main():
    parser = argparse.ArgumentParser(description="医学知识库入库")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="原始医学文档目录")
    parser.add_argument("--terminology", default=DEFAULT_TERMS, help="术语映射表 JSON 路径")
    parser.add_argument("--embed", action="store_true", help="执行 FAISS 向量化（调用 embedding API）")
    args = parser.parse_args()

    with open(args.terminology, encoding="utf-8") as f:
        table = json.load(f)
    mapper = TerminologyMapper(table)

    chunks, metrics, details = ingest_medical(
        args.raw_dir, mapper,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    n_corpus = dump_chunks(chunks, settings.medical_corpus_path)

    print()
    print("=" * 62)
    print("  医学知识库入库结果")
    print("=" * 62)
    print(f"  总文档数     {metrics['total']}")
    print(f"  脏数据拦截   {metrics['intercepted']}  ({_fmt_reasons(metrics['reasons'])})")
    print(f"  拦截率       {metrics['intercept_rate'] * 100:.1f}%")
    print(f"  入库文档     {metrics['ingested']}")
    print(f"  知识块       {metrics['chunks']}")
    print("-" * 62)
    for d in details:
        if d["status"] == "ingested":
            print(f"  ✅ {d['source']:<22} 入库  {d['chunks']} 块")
        else:
            print(f"  ⛔ {d['source']:<22} 拦截  {d['reason']}")
    print()

    if args.embed:
        from src.rag import vector_store
        vector_store.build_index(chunks, save=True, save_dir=settings.medical_vector_store_path)
        print(f"✅ 已向量化到 {settings.medical_vector_store_path}")
    else:
        print("（未执行向量化，加 --embed 可构建 FAISS 索引）")
    print()


def _fmt_reasons(reasons):
    return ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))


if __name__ == "__main__":
    main()
