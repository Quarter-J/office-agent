"""
文档加载与检索（兼容层）

实际实现已迁移到 src.rag，此处保留旧函数签名以兼容现有 import。
"""

from pathlib import Path
from typing import List, Optional

from src.utils.logger import logger


def load_local_documents(kb_dir: Optional[str] = None) -> List[dict]:
    """加载知识库目录中的文档（兼容旧返回格式，仅用于文本场景）"""
    from src.rag.parsers.base import load_documents
    docs = load_documents(kb_dir)
    return [
        {
            "file": d.source,
            "content": "\n\n".join(b.text for b in d.blocks),
            "path": d.path,
        }
        for d in docs
    ]


def build_vector_store(kb_dir: Optional[str] = None) -> bool:
    """从知识库文档构建向量索引"""
    from src.rag.indexer import build_index
    return build_index(kb_dir)


def semantic_search(query: str, top_k: int = 3) -> List[dict]:
    """语义检索（兼容旧返回格式：content/source/score）"""
    from src.rag import vector_store
    hits = vector_store.search(query, top_k=top_k)
    return [
        {
            "rank": i + 1,
            "content": h.text[:500],
            "source": h.meta.get("source", "未知"),
            "score": round(h.score, 4),
        }
        for i, h in enumerate(hits)
    ]
