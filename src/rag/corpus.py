"""
知识块语料持久化：入库时落盘 JSON，供 BM25/混合检索离线加载。
"""

from __future__ import annotations

import json
from pathlib import Path


def dump_chunks(chunks, path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"text": c.text, "meta": c.meta} for c in chunks]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return len(data)


def load_chunks(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
