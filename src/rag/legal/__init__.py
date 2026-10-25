"""中国法律法规 RAG 入库与检索子系统。"""

from src.rag.legal.pipeline import (
    CORE_TYPES,
    DEFAULT_STATUSES,
    ingest_legal,
    split_frontmatter,
)

__all__ = [
    "CORE_TYPES",
    "DEFAULT_STATUSES",
    "ingest_legal",
    "split_frontmatter",
]
