from __future__ import annotations

import re
from pathlib import Path

from src.rag.parsers.base import Block, DocumentParser, register


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^```")


def _flush(blocks: list, buf: list):
    text = "\n".join(buf).strip()
    if text:
        blocks.append(Block(type="paragraph", text=text))
    buf.clear()


@register
class MarkdownParser(DocumentParser):
    supported = (".md", ".markdown")

    def parse(self, path: Path) -> list:
        content = path.read_text(encoding="utf-8")
        blocks: list = []
        buf: list = []
        in_code = False
        code_lang = ""
        code_buf: list = []

        for line in content.split("\n"):
            if in_code:
                if _FENCE_RE.match(line.strip()):
                    blocks.append(Block(type="code", text="\n".join(code_buf), meta={"lang": code_lang}))
                    in_code = False
                    code_buf = []
                else:
                    code_buf.append(line)
                continue

            stripped = line.strip()
            if _FENCE_RE.match(stripped):
                _flush(blocks, buf)
                in_code = True
                code_lang = stripped[3:].strip()
                continue

            m = _HEADING_RE.match(line)
            if m:
                _flush(blocks, buf)
                level = len(m.group(1))
                blocks.append(Block(type="heading", text=m.group(2).strip(), level=level))
                continue

            buf.append(line)

        if in_code:
            blocks.append(Block(type="code", text="\n".join(code_buf), meta={"lang": code_lang}))
        _flush(blocks, buf)
        return blocks
