"""
统一 FAISS 向量存储封装

- embedding 实例单例（避免每次检索重建）
- 索引构建 / 加载 / 增量合并
- 阈值语义检索（L2 距离归一化为相似度）
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from src.config.settings import settings
from src.utils.logger import logger
from src.rag.validation import validate_top_k


@dataclass
class ScoredChunk:
    text: str
    meta: dict
    score: float


_embeddings = None
_embeddings_lock = threading.Lock()


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                from langchain_openai import OpenAIEmbeddings
                kwargs = dict(
                    model=settings.embedding_model,
                    openai_api_key=settings.embedding_api_key or settings.openai_api_key,
                    # 非 OpenAI 端点（如千问/DashScope）需发送原始文本而非 token ID
                    check_embedding_ctx_length=False,
                    request_timeout=settings.embedding_timeout_seconds,
                    max_retries=0,
                    # 千问/DashScope embedding 单次批量上限 10，超过会 400
                    chunk_size=settings.embedding_batch_size,
                )
                if settings.embedding_base_url:
                    kwargs["openai_api_base"] = settings.embedding_base_url
                _embeddings = OpenAIEmbeddings(**kwargs)
    return _embeddings


def _vector_path() -> Path:
    return Path(settings.vector_store_path)


def build_index(chunks, save: bool = True, save_dir: str = None) -> bool:
    """全量构建并保存 FAISS 索引（save_dir 可指定独立目录，默认用配置路径）"""
    from langchain_community.vectorstores import FAISS
    texts = [c.text for c in chunks]
    metadatas = [c.meta for c in chunks]
    store = FAISS.from_texts(texts, get_embeddings(), metadatas=metadatas)
    if save:
        path = Path(save_dir) if save_dir else _vector_path()
        path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(path))
        logger.info(f"FAISS 索引已构建: {len(texts)} 块 -> {path}")
    return True


def load_index():
    from langchain_community.vectorstores import FAISS
    path = _vector_path()
    if not path.exists():
        return None
    return FAISS.load_local(
        str(path), get_embeddings(), allow_dangerous_deserialization=True
    )


def merge_index(chunks) -> bool:
    """增量合并新 chunk（索引不存在时退化为全量构建）"""
    path = _vector_path()
    if not path.exists():
        return build_index(chunks, save=True)
    store = load_index()
    if store is None:
        return build_index(chunks, save=True)

    from langchain_community.vectorstores import FAISS
    small = FAISS.from_texts(
        [c.text for c in chunks],
        get_embeddings(),
        metadatas=[c.meta for c in chunks],
    )
    store.merge_from(small)
    store.save_local(str(path))
    logger.info(f"FAISS 索引增量合并: +{len(chunks)} 块")
    return True


def _l2_to_similarity(score: float) -> float:
    """FAISS 默认 IndexFlatL2 返回 L2 距离，映射到 (0,1] 相似度"""
    return 1.0 / (1.0 + float(score))


def source_label(meta: dict) -> str:
    """把 chunk 元数据拼成可读的源头定位：文件名 + 标题路径 + 页码/行列。

    例：`spring_boot.md · Spring Boot 使用说明 / 自动装配`
        `报告.pdf · 第3页`
        `通讯录.xlsx · Sheet1 · 第2行`
    """
    parts = [meta.get("source") or "未知"]
    heading = meta.get("heading_path")
    if heading:
        parts.append(str(heading))
    if meta.get("page"):
        parts.append(f"第{meta['page']}页")
    elif meta.get("slide"):
        parts.append(f"第{meta['slide']}页")
    elif meta.get("row"):
        sheet = f"{meta['sheet']}·" if meta.get("sheet") else ""
        parts.append(f"{sheet}第{meta['row']}行")
    return " · ".join(parts)


def search(query: str, top_k: int = 3, threshold: float = None, *, raise_errors: bool = False) -> list:
    """语义检索，返回 ScoredChunk 列表；低于阈值过滤"""
    validate_top_k(top_k)
    threshold = settings.similarity_threshold if threshold is None else threshold
    try:
        store = load_index()
        if store is None:
            raise FileNotFoundError("向量索引不存在，请先构建（python build_index.py）")
        docs_with_scores = store.similarity_search_with_score(query, k=top_k)
        results: list = []
        for doc, score in docs_with_scores:
            sim = _l2_to_similarity(score)
            if sim < threshold:
                continue
            results.append(ScoredChunk(text=doc.page_content, meta=doc.metadata, score=sim))
        return results
    except Exception as e:
        logger.warning(f"语义检索失败: {e}")
        if raise_errors:
            raise
        return []
