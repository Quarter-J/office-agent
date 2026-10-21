"""
实体抽取：正则 + LLM 混合

正则抓取手机/座机/邮箱等精确值；LLM 判断姓名归属与部门/职位等区分字段。
低语义事实（联系方式）靠精确查找，不靠向量检索。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.utils.logger import logger


_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_LANDLINE_RE = re.compile(r"(?<!\d)(0\d{2,3}-?\d{7,8})(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclass
class Entity:
    entity_type: str
    name: str
    canonical_name: str
    attrs: dict = field(default_factory=dict)   # {"phone":[...],"email":[...],"dept":"","title":""}
    source_file: str = ""
    source_page: int = None


def canonicalize(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip().lower())


def _regex_hits(text: str) -> dict:
    return {
        "phones": list(dict.fromkeys(_PHONE_RE.findall(text) + _LANDLINE_RE.findall(text))),
        "emails": list(dict.fromkeys(_EMAIL_RE.findall(text))),
    }


def _llm_extract(text: str) -> list:
    from src.api.llm import get_llm
    from src.tools.prompts import ENTITY_EXTRACTION_TEMPLATE
    llm = get_llm()
    prompt = ENTITY_EXTRACTION_TEMPLATE.format(text=text[:3000])
    response = llm.invoke(prompt)
    return _parse_json_array(response.content)


def _parse_json_array(text: str) -> list:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                data = json.loads(text[start:end + 1])
                return data if isinstance(data, list) else []
            except (json.JSONDecodeError, ValueError):
                pass
        return []


def extract_entities(chunks) -> list:
    """对含联系方式的 chunk 抽实体；姓名↔值绑定以 LLM 为准，正则只做门控"""
    entities: list = []
    for c in chunks:
        text = c.text
        hits = _regex_hits(text)
        if not hits["phones"] and not hits["emails"]:
            continue
        source = c.meta.get("source", "")
        page = c.meta.get("page")
        records = _llm_extract(text)
        for r in records:
            name = (r.get("name") or "").strip()
            if not name:
                continue
            phones = [p for p in (r.get("phones") or []) if p]
            emails = [e for e in (r.get("emails") or []) if e]
            if not phones and not emails:
                continue
            entities.append(Entity(
                entity_type="person",
                name=name,
                canonical_name=canonicalize(name),
                attrs={
                    "phone": phones,
                    "email": emails,
                    "dept": (r.get("dept") or "").strip(),
                    "title": (r.get("title") or "").strip(),
                },
                source_file=source,
                source_page=page,
            ))
    return entities
