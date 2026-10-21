"""
结构感知切分器

输入 Block 列表，输出 Chunk。规则：
- 按标题层级构建 section，正文 chunk 携带 heading_path（父子上下文自包含）
- 表格/代码块原子化，不跨 chunk 切碎
- 无结构文档回退固定滑窗
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rag.parsers.base import Block


@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: dict = field(default_factory=dict)


def structure_chunker(blocks, *, source, chunk_size=500, chunk_overlap=50) -> list:
    chunks: list = []
    counter = 0
    for section in _build_sections(blocks):
        heading_text = " / ".join(t for _, t in section["headings"])
        atomic = [b for b in section["blocks"] if b.type in ("table", "code")]
        prose = [b for b in section["blocks"] if b.type not in ("table", "code")]

        prose_text = "\n\n".join(b.text for b in prose if b.text.strip())
        for piece in _split_long_text(prose_text, chunk_size, chunk_overlap):
            full = f"{heading_text}\n\n{piece}" if heading_text else piece
            if not full.strip():
                continue
            meta = {"heading_path": heading_text}
            if section["headings"]:
                meta["section"] = section["headings"][-1][1]
            chunks.append(Chunk(chunk_id=f"{source}#{counter}", text=full, meta=meta))
            counter += 1

        for b in atomic:
            full = f"{heading_text}\n\n{b.text}" if heading_text else b.text
            if not full.strip():
                continue
            meta = dict(b.meta)
            meta["heading_path"] = heading_text
            chunks.append(Chunk(chunk_id=f"{source}#{counter}", text=full, meta=meta))
            counter += 1

    return chunks


def _build_sections(blocks: list) -> list:
    sections: list = []
    heading_stack: list = []
    current: list = []

    for b in blocks:
        if b.type == "heading":
            if current:
                sections.append({"headings": list(heading_stack), "blocks": current})
                current = []
            while heading_stack and heading_stack[-1][0] >= b.level:
                heading_stack.pop()
            heading_stack.append((b.level, b.text))
        else:
            current.append(b)

    if current:
        sections.append({"headings": list(heading_stack), "blocks": current})
    return sections


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    pieces: list = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return pieces
