from __future__ import annotations

from pathlib import Path

from src.rag.parsers.base import Block, DocumentParser, register


TEXT_EXTENSIONS = (".txt", ".py", ".java", ".yaml", ".yml", ".json", ".xml", ".csv")


@register
class TextParser(DocumentParser):
    supported = TEXT_EXTENSIONS

    def parse(self, path: Path) -> list:
        content = path.read_text(encoding="utf-8")
        blocks = []
        for para in content.split("\n\n"):
            para = para.strip()
            if para:
                blocks.append(Block(type="paragraph", text=para))
        return blocks
