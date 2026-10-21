from __future__ import annotations

import re
from pathlib import Path

from src.rag.parsers.base import Block, DocumentParser, register
from src.utils.logger import logger


_HEADING_STYLE_RE = re.compile(r"heading\s*(\d)|标题\s*(\d)", re.IGNORECASE)


@register
class DocxParser(DocumentParser):
    supported = (".docx",)

    def parse(self, path: Path) -> list:
        try:
            import docx
            from docx.text.paragraph import Paragraph
            from docx.table import Table
        except ImportError:
            logger.warning("缺少 python-docx 依赖，跳过 .docx 解析")
            return []

        document = docx.Document(str(path))
        blocks: list = []

        # 按文档顺序遍历正文段落与表格
        for child in document.element.body.iterchildren():
            tag = child.tag.split("}")[-1]
            if tag == "p":
                para = Paragraph(child, document)
                text = para.text.strip()
                if not text:
                    continue
                level = self._heading_level(para)
                if level:
                    blocks.append(Block(type="heading", text=text, level=level))
                else:
                    blocks.append(Block(type="paragraph", text=text))
            elif tag == "tbl":
                table = Table(child, document)
                md = self._table_to_markdown(table)
                if md:
                    blocks.append(Block(type="table", text=md, meta={"is_table": True}))

        return blocks

    def _heading_level(self, para) -> int:
        """优先用 style 名识别标题，缺失时用加粗+短文本启发式兜底"""
        style_name = (para.style.name or "") if para.style is not None else ""
        m = _HEADING_STYLE_RE.search(style_name)
        if m:
            return int(m.group(1) or m.group(2))
        if "title" in style_name.lower():
            return 1
        # 兜底：短文本 + 全加粗
        text = para.text.strip()
        if 0 < len(text) <= 40:
            runs = para.runs
            if runs and all(r.bold for r in runs):
                return 1
        return 0

    @staticmethod
    def _table_to_markdown(table) -> str:
        rows = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
        if not rows:
            return ""
        ncols = len(table.rows[0].cells) if table.rows else 0
        sep = "| " + " | ".join(["---"] * ncols) + " |"
        return rows[0] + "\n" + sep + "\n" + "\n".join(rows[1:])
