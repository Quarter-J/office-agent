"""
ingestion 编排：parse → chunk → extract → embed → 双写（FAISS + entities）
"""

from __future__ import annotations

from pathlib import Path

from src.config.settings import settings
from src.rag.parsers.base import load_documents, load_file
from src.rag.chunker import structure_chunker
from src.rag import vector_store, extractor, entity_store
from src.utils.logger import logger


def _chunk_doc(doc) -> list:
    chunks = structure_chunker(
        doc.blocks,
        source=doc.source,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    for c in chunks:
        c.meta["source"] = doc.source
        c.meta["format"] = doc.fmt
    return chunks


def _extract_and_store(chunks) -> int:
    try:
        entities = extractor.extract_entities(chunks)
        if entities:
            return entity_store.upsert_entities(entities)
    except Exception as e:
        logger.warning(f"实体抽取失败（不影响语义检索）: {e}")
    return 0


def build_index(kb_dir: str = None) -> bool:
    """全量构建：解析知识库目录 → 切分 → 建 FAISS → 抽实体写库"""
    docs = load_documents(kb_dir)
    if not docs:
        logger.warning("知识库为空，跳过索引构建")
        return False

    all_chunks: list = []
    for doc in docs:
        all_chunks.extend(_chunk_doc(doc))

    if not all_chunks:
        logger.warning("没有可索引的内容")
        return False

    vector_store.build_index(all_chunks, save=True)
    n_entities = _extract_and_store(all_chunks)
    logger.info(f"索引构建完成: {len(docs)} 文档, {len(all_chunks)} 块, {n_entities} 实体")
    return True


def incremental_index(file_paths) -> bool:
    """新增文档后增量索引：只处理指定文件并合并进现有索引"""
    kb_dir = Path(settings.knowledge_base_dir)
    new_chunks: list = []
    for fp in file_paths:
        p = Path(fp)
        doc = load_file(p, kb_dir)
        if doc is None:
            continue
        new_chunks.extend(_chunk_doc(doc))

    if not new_chunks:
        logger.info("增量索引：无新内容")
        return False

    vector_store.merge_index(new_chunks)
    _extract_and_store(new_chunks)
    logger.info(f"增量索引完成: +{len(new_chunks)} 块")
    return True
