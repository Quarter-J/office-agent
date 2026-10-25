"""
BM25 关键词检索（自包含实现）。

中文分词走 jieba（默认），tokenizer 可注入以便离线测试。
"""

from __future__ import annotations

import math
from collections import Counter

from src.rag.vector_store import ScoredChunk


def jieba_tokenize(text: str) -> list:
    import jieba
    return [w for w in jieba.cut_for_search(text) if w.strip()]


class BM25Index:
    def __init__(self, tokenizer=None, k1=1.5, b=0.75):
        self.tokenizer = tokenizer or jieba_tokenize
        self.k1 = k1
        self.b = b
        self._docs = []      # list[dict] {text, meta}
        self._tokens = []    # list[list[str]]
        self._df = Counter()  # term -> 文档频
        self._avgdl = 0.0

    def index(self, items):
        """items：每项含 text 与 meta（dict 或带 .text/.meta 的对象）。"""
        self._docs = [
            {"text": it["text"], "meta": it["meta"]} if isinstance(it, dict)
            else {"text": it.text, "meta": it.meta}
            for it in items
        ]
        self._tokens = []
        self._df = Counter()
        for doc in self._docs:
            toks = self.tokenizer(doc["text"])
            self._tokens.append(toks)
            for term in set(toks):
                self._df[term] += 1
        n = len(self._docs)
        self._avgdl = sum(len(t) for t in self._tokens) / n if n else 0.0

    def search(self, query, k=10):
        q = self.tokenizer(query)
        n = len(self._docs)
        if n == 0 or not q:
            return []
        scored = []
        for i, toks in enumerate(self._tokens):
            dl = len(toks)
            if dl == 0:
                continue
            tf = Counter(toks)
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                score += idf * (tf[term] * (self.k1 + 1)) / (
                    tf[term] + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, i in scored[:k]:
            d = self._docs[i]
            out.append(ScoredChunk(text=d["text"], meta=d["meta"], score=score))
        return out
