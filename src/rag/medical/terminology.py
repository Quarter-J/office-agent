"""
医学术语映射：术语变体 → 标准词。

映射表 JSON 结构：
  {"标准词": ["变体1", "变体2", ...], ...}
匹配忽略大小写与空白，长变体优先，避免子串误替换。
"""

from __future__ import annotations

import re


class TerminologyMapper:
    def __init__(self, table: dict = None):
        self._variant_to_standard = {}
        self._key_to_standard = {}
        self._pattern = None
        if table:
            self.load(table)

    @staticmethod
    def _key(term: str) -> str:
        return re.sub(r"\s+", "", (term or "")).lower()

    def load(self, table: dict) -> None:
        for standard, variants in table.items():
            for v in variants:
                self._variant_to_standard[v] = standard
                self._key_to_standard[self._key(v)] = standard
        self._compile()

    def _compile(self) -> None:
        if not self._variant_to_standard:
            self._pattern = None
            return
        variants = sorted(self._variant_to_standard, key=len, reverse=True)
        self._pattern = re.compile("|".join(re.escape(v) for v in variants), re.IGNORECASE)

    def map_term(self, term: str) -> str:
        """单个术语映射；未收录返回原词。"""
        return self._key_to_standard.get(self._key(term), term)

    def map_text(self, text: str) -> str:
        """整段文本替换所有已收录变体为标准词。"""
        if self._pattern is None:
            return text

        def repl(match):
            return self._key_to_standard.get(self._key(match.group(0)), match.group(0))

        return self._pattern.sub(repl, text)
