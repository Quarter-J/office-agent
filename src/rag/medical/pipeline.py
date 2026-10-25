"""
医学入库管线编排：读取 → 规则归一化 → 脏数据拦截 → 术语映射 → 分段切块。

本模块只做「入库」并产出指标（文档数/拦截数/拦截率/块数），不做向量化；
向量化是独立可选项，见 build_medical_index.py 的 --embed。
"""

from __future__ import annotations

import re
from pathlib import Path

from src.rag.chunker import structure_chunker
from src.rag.parsers.base import Block
from src.rag.medical.normalize import normalize_text
from src.rag.medical.dirty import detect_dirty

SUPPORTED_SUFFIXES = (".txt", ".md", ".markdown")


def read_text(path: Path) -> str:
    """按常见编码顺序读取；全部失败则抛出（视为读入错误）。"""
    for enc in ("utf-8", "gb18030", "utf-8-sig"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法以常见编码读取，疑似损坏")


def _text_to_blocks(text: str) -> list:
    """简易 Markdown 文本 → Block 列表（# 标题 + 段落），复用结构切分器。"""
    blocks = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            blocks.append(Block(type="heading", text=m.group(2).strip(), level=len(m.group(1))))
        else:
            blocks.append(Block(type="paragraph", text=line))
    return blocks


def ingest_medical(raw_dir, terminology, *, chunk_size=500, chunk_overlap=50):
    """扫描原始目录，逐份：归一化 → 拦截 → 映射 → 切块。

    返回 (all_chunks, metrics, details)。metrics 含 total/intercepted/ingested/
    chunks/reasons/intercept_rate；details 为逐文件明细。
    """
    raw_dir = Path(raw_dir)
    metrics = {
        "total": 0,
        "intercepted": 0,
        "ingested": 0,
        "chunks": 0,
        "reasons": {},
        "intercept_rate": 0.0,
    }
    all_chunks = []
    details = []

    for fp in sorted(raw_dir.rglob("*.*")):
        if not fp.is_file() or fp.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        metrics["total"] += 1
        source = fp.name

        try:
            raw = read_text(fp)
        except Exception:
            _mark_intercept(metrics, details, source, "read_error")
            continue

        normalized = normalize_text(raw)
        dirty, reason = detect_dirty(normalized)
        if dirty:
            _mark_intercept(metrics, details, source, reason)
            continue

        mapped = terminology.map_text(normalized)
        blocks = _text_to_blocks(mapped)
        chunks = structure_chunker(blocks, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            _mark_intercept(metrics, details, source, "no_content")
            continue

        for c in chunks:
            c.meta["source"] = source
            c.meta["domain"] = "medical"
        all_chunks.extend(chunks)
        metrics["ingested"] += 1
        metrics["chunks"] += len(chunks)
        details.append({"source": source, "status": "ingested", "chunks": len(chunks)})

    if metrics["total"]:
        metrics["intercept_rate"] = round(metrics["intercepted"] / metrics["total"], 4)

    return all_chunks, metrics, details


def _mark_intercept(metrics, details, source, reason):
    metrics["intercepted"] += 1
    metrics["reasons"][reason] = metrics["reasons"].get(reason, 0) + 1
    details.append({"source": source, "status": "intercepted", "reason": reason})
