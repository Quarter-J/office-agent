"""中国法律法规入库管线。

数据格式：一篇法规一个 Markdown，文件头为 YAML frontmatter，正文按
「第X章 / 第X节 / 第X条」组织。入库时只保留指定类型和时效状态，
按条文边界聚合切块，避免把一条法条从中间截断。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml

from src.rag.chunker import Chunk

CORE_TYPES = (
    "宪法",
    "法律",
    "行政法规",
    "司法解释",
    "法律解释",
    "监察法规",
)
DEFAULT_STATUSES = ("有效",)
DEFAULT_CHUNK_SIZE = 1200

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.S)
_ARTICLE_RE = re.compile(
    r"^[\t \u3000]*(第[零〇一二三四五六七八九十百千万两0-9]+条"
    r"(?:之[零〇一二三四五六七八九十百千万两0-9]+)?)(?:[\t \u3000]+(.*))?$"
)
_HEADING_RE = re.compile(
    r"^[\t \u3000]*(第[零〇一二三四五六七八九十百千万两0-9]+(?:编|章|节))"
    r"[\t \u3000]*(.*)$"
)
_MD_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_TOC_RE = re.compile(r"^[\t \u3000]*目[\t \u3000]*录[\t \u3000]*$")


@dataclass
class LegalDocument:
    path: Path
    source: str
    meta: dict
    body: str


@dataclass
class LegalSegment:
    label: str
    text: str
    heading_path: str = ""


def split_frontmatter(text: str) -> tuple[dict, str]:
    """拆分 YAML frontmatter 与正文；缺失或损坏时返回空元数据。"""
    text = text.lstrip("\ufeff")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return {str(k): _as_text(v) for k, v in meta.items()}, text[match.end():]


def _as_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_document(path: Path, root: Path) -> LegalDocument:
    text = path.read_text(encoding="utf-8-sig")
    meta, body = split_frontmatter(text)
    return LegalDocument(
        path=path,
        source=path.relative_to(root).as_posix(),
        meta=meta,
        body=_clean_legal_text(body),
    )


def _clean_legal_text(text: str) -> str:
    """清除网页转换产生的无意义链接标记，保留法条正文。"""
    text = re.sub(r"\[([^\]]+)\]\(javascript:[^)]*\)", r"\1", text, flags=re.I)
    return text.replace("\ufeff", "").replace("\u200b", "")


def _date_key(meta: dict) -> tuple[int, int, int]:
    for field in ("effective_date", "publish"):
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(meta.get(field) or ""))
        if match:
            return tuple(int(part) for part in match.groups())
    return (0, 0, 0)


def _heading_level(label: str) -> int:
    if label.endswith("编"):
        return 1
    if label.endswith("章"):
        return 2
    return 3


def extract_segments(body: str, title: str) -> tuple[list[LegalSegment], int]:
    """把正文拆成前言 + 条文段落；返回 (segments, article_count)。"""
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[0].strip().startswith("# ") and lines[0].strip()[2:].strip() == title:
        lines = lines[1:]

    heading_stack: list[tuple[int, str]] = []
    preamble_lines: list[str] = []
    segments: list[LegalSegment] = []
    current_label: Optional[str] = None
    current_lines: list[str] = []
    current_heading = ""

    def heading_path() -> str:
        return " / ".join(text for _, text in heading_stack if text)

    def flush_article() -> None:
        nonlocal current_label, current_lines
        if current_label is not None:
            text = "\n".join(line for line in current_lines if line.strip()).strip()
            if text:
                segments.append(LegalSegment(current_label, text, current_heading))
        current_label = None
        current_lines = []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if current_label is not None:
                current_lines.append("")
            continue

        article = _ARTICLE_RE.match(line)
        if article:
            flush_article()
            current_label = article.group(1)
            current_heading = heading_path()
            current_lines = [stripped]
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_article()
            label, text = heading.group(1), heading.group(2).strip()
            level = _heading_level(label)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, f"{label} {text}".strip()))
            continue

        md_heading = _MD_HEADING_RE.match(line)
        if md_heading:
            flush_article()
            level = len(md_heading.group(1))
            text = md_heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            continue

        if current_label is not None:
            current_lines.append(stripped)
        elif not _TOC_RE.match(line):
            preamble_lines.append(stripped)

    flush_article()

    article_count = len(segments)
    if article_count:
        # 有条文时前言通常只是标题、目录和发布通知，作为 RAG 块噪声较大。
        return segments, article_count

    preamble = "\n".join(preamble_lines).strip()
    if not preamble:
        preamble = body.strip()
    if not preamble:
        return [], 0
    return _paragraph_segments(preamble), 0


def _paragraph_segments(text: str) -> list[LegalSegment]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    return [LegalSegment("", paragraph, "") for paragraph in paragraphs]


def _split_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    pieces: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_hard_split(paragraph, max_chars))
        elif not buffer:
            buffer = paragraph
        elif len(buffer) + len(paragraph) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{paragraph}"
        else:
            pieces.append(buffer)
            buffer = paragraph
    if buffer:
        pieces.append(buffer)
    return pieces or _hard_split(text, max_chars)


def _hard_split(text: str, max_chars: int) -> list[str]:
    sentences = [part for part in re.split(r"(?<=[。；;！？!?])", text) if part]
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars))
        elif not buffer:
            buffer = sentence
        elif len(buffer) + len(sentence) <= max_chars:
            buffer += sentence
        else:
            pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return pieces


def _base_label(label: str) -> str:
    return re.sub(r"（续\d+）$", "", label)


def _group_segments(
    segments: Iterable[LegalSegment],
    *,
    title: str,
    source: str,
    max_chars: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[str] = []
    labels: list[str] = []
    current_heading: Optional[str] = None
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, labels, buffer_len, current_heading
        if not buffer:
            return
        heading = current_heading or ""
        header = f"《{title}》"
        if heading:
            header += f"\n{heading}"
        meta = {
            "source": source,
            "format": ".md",
            "domain": "legal",
            "title": title,
            "heading_path": heading,
            "article_start": _base_label(labels[0]) if labels else "",
            "article_end": _base_label(labels[-1]) if labels else "",
            "kind": "article" if any(label.startswith("第") for label in labels) else "preamble",
        }
        chunks.append(Chunk(
            chunk_id=f"{source}#{len(chunks)}",
            text=f"{header}\n\n" + "\n\n".join(buffer),
            meta=meta,
        ))
        buffer = []
        labels = []
        buffer_len = 0

    for segment in segments:
        for label, text in _split_segment(segment, max_chars):
            if buffer and current_heading != segment.heading_path:
                flush()
            if buffer and buffer_len + len(text) + 2 > max_chars:
                flush()
            if not buffer:
                current_heading = segment.heading_path
            buffer.append(text)
            labels.append(label)
            buffer_len += len(text) + 2
    flush()
    return chunks


def _split_segment(segment: LegalSegment, max_chars: int) -> list[tuple[str, str]]:
    if len(segment.text) <= max_chars:
        return [(segment.label, segment.text)]

    body = segment.text
    if segment.label and body.startswith(segment.label):
        body = body[len(segment.label):].lstrip()
    pieces = _split_text(body, max_chars)
    output: list[tuple[str, str]] = []
    for index, piece in enumerate(pieces):
        label = segment.label if index == 0 else f"{segment.label}（续{index + 1}）"
        output.append((label, f"{label}\n{piece}" if label else piece))
    return output


def _chunk_document(doc: LegalDocument, max_chars: int) -> tuple[list[Chunk], int]:
    segments, article_count = extract_segments(doc.body, str(doc.meta.get("title") or ""))
    chunks = _group_segments(
        segments,
        title=str(doc.meta.get("title") or doc.path.stem),
        source=doc.source,
        max_chars=max_chars,
    )
    for chunk in chunks:
        for key in ("type", "status", "office", "publish", "effective_date", "url"):
            chunk.meta[key] = str(doc.meta.get(key) or "")
        chunk.meta["region"] = "全国"
    return chunks, article_count


def ingest_legal(
    raw_dir,
    *,
    types: Iterable[str] = CORE_TYPES,
    statuses: Iterable[str] = DEFAULT_STATUSES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: Optional[int] = None,
) -> tuple[list[Chunk], dict, list[dict]]:
    """扫描法律 Markdown 目录，过滤、去重并切块。

    返回 (chunks, metrics, details)。不调用 embedding，便于先检查入库质量。
    """
    raw_root = Path(raw_dir)
    if not raw_root.exists():
        raise FileNotFoundError(f"法律语料目录不存在: {raw_root}")
    allowed_types = {str(item).strip() for item in types if str(item).strip()}
    allowed_statuses = {str(item).strip() for item in statuses if str(item).strip()}

    metrics = {
        "total": 0,
        "matched": 0,
        "selected": 0,
        "skipped_type": 0,
        "skipped_status": 0,
        "skipped_frontmatter": 0,
        "skipped_empty": 0,
        "duplicates": 0,
        "documents_with_articles": 0,
        "documents_without_articles": 0,
        "articles": 0,
        "chunks": 0,
        "chars": 0,
    }
    selected: list[LegalDocument] = []
    details: list[dict] = []

    for path in sorted(raw_root.rglob("*.md")):
        metrics["total"] += 1
        try:
            doc = read_document(path, raw_root)
        except Exception as exc:
            metrics["skipped_empty"] += 1
            details.append({"source": path.name, "status": "skipped", "reason": f"read_error: {exc}"})
            continue

        doc_type = str(doc.meta.get("type") or "").strip()
        status = str(doc.meta.get("status") or "").strip()
        title = str(doc.meta.get("title") or "").strip()
        if not title:
            metrics["skipped_frontmatter"] += 1
            continue
        if allowed_types and doc_type not in allowed_types:
            metrics["skipped_type"] += 1
            continue
        if allowed_statuses and status not in allowed_statuses:
            metrics["skipped_status"] += 1
            continue
        if not doc.body.strip():
            metrics["skipped_empty"] += 1
            continue
        selected.append(doc)
        metrics["matched"] += 1

    # 同一标题、同一类型只保留最新版本，避免新旧版本同时命中。
    latest: dict[tuple[str, str], LegalDocument] = {}
    for doc in selected:
        key = (str(doc.meta.get("type") or ""), str(doc.meta.get("title") or ""))
        previous = latest.get(key)
        if previous is None or _date_key(doc.meta) > _date_key(previous.meta):
            if previous is not None:
                metrics["duplicates"] += 1
            latest[key] = doc
        else:
            metrics["duplicates"] += 1
    selected = sorted(latest.values(), key=lambda doc: doc.source)
    if limit is not None:
        selected = selected[:max(0, limit)]
    metrics["selected"] = len(selected)

    chunks: list[Chunk] = []
    for doc in selected:
        doc_chunks, article_count = _chunk_document(doc, chunk_size)
        if not doc_chunks:
            metrics["skipped_empty"] += 1
            details.append({"source": doc.source, "status": "skipped", "reason": "no_content"})
            continue
        chunks.extend(doc_chunks)
        metrics["chunks"] += len(doc_chunks)
        metrics["chars"] += sum(len(chunk.text) for chunk in doc_chunks)
        metrics["articles"] += article_count
        if article_count:
            metrics["documents_with_articles"] += 1
        else:
            metrics["documents_without_articles"] += 1
        details.append({
            "source": doc.source,
            "title": str(doc.meta.get("title") or ""),
            "type": str(doc.meta.get("type") or ""),
            "status": str(doc.meta.get("status") or ""),
            "articles": article_count,
            "chunks": len(doc_chunks),
        })

    return chunks, metrics, details
