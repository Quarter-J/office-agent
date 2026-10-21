"""
实体结构化存储与精确查找

联系方式等低语义事实走精确匹配，不走向量。同名不同人（如不同部门的张三）
通过 label 区分，绝不静默合并，保证 lookup 时能返回候选要求用户确认。
"""

from __future__ import annotations

import json
import re

from src.storage.database import query, insert, update


def canonicalize(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip().lower())


def make_label(name: str, attrs: dict) -> str:
    """展示名：附带区分字段（部门/职位），用于同名消歧"""
    dept = (attrs.get("dept") or "").strip()
    title = (attrs.get("title") or "").strip()
    if dept and title:
        return f"{name}（{dept}·{title}）"
    if dept:
        return f"{name}（{dept}）"
    if title:
        return f"{name}（{title}）"
    return name


def upsert_entities(entities) -> int:
    count = 0
    for e in entities:
        key = canonicalize(e.name)
        if not key:
            continue
        label = make_label(e.name, e.attrs)
        existing = query(
            "entities",
            where="entity_type = ? AND canonical_name = ? AND label = ?",
            params=(e.entity_type, key, label),
            limit=1,
        )
        if existing:
            row = existing[0]
            merged = _merge_attrs(json.loads(row["attrs"] or "{}"), e.attrs)
            update("entities", {
                "attrs": json.dumps(merged, ensure_ascii=False),
                "source_file": e.source_file or row.get("source_file"),
                "source_page": e.source_page if e.source_page is not None else row.get("source_page"),
            }, "id = ?", (row["id"],))
        else:
            insert("entities", {
                "entity_type": e.entity_type,
                "name": e.name,
                "label": label,
                "canonical_name": key,
                "attrs": json.dumps(e.attrs, ensure_ascii=False),
                "source_file": e.source_file,
                "source_page": e.source_page,
            })
        count += 1
    return count


def lookup_entity(entity_type: str, name: str, field: str = None) -> list:
    """按规范化名精确查找；返回全部同名实体（含不同 label）"""
    key = canonicalize(name)
    rows = query(
        "entities",
        where="entity_type = ? AND canonical_name = ?",
        params=(entity_type, key),
        order_by="id ASC",
        limit=50,
    )
    return [_to_match(r) for r in rows]


def search_entities(query_text: str, top_k: int = 5) -> list:
    """模糊兜底：姓名/规范化名包含匹配"""
    like = f"%{query_text}%"
    rows = query(
        "entities",
        where="name LIKE ? OR canonical_name LIKE ?",
        params=(like, like),
        order_by="id ASC",
        limit=top_k,
    )
    return [_to_match(r) for r in rows]


def _to_match(row: dict) -> dict:
    return {
        "name": row["name"],
        "label": row.get("label", row["name"]),
        "canonical_name": row["canonical_name"],
        "attrs": json.loads(row["attrs"] or "{}"),
        "source_file": row.get("source_file"),
        "source_page": row.get("source_page"),
    }


def _merge_attrs(a: dict, b: dict) -> dict:
    merged = dict(a)
    for k, v in (b or {}).items():
        if k in ("phone", "email") and isinstance(v, list) and isinstance(merged.get(k), list):
            merged[k] = list(dict.fromkeys(merged[k] + v))
        elif v not in (None, "", []):
            merged[k] = v
    return merged
