"""
检索后重排序：用 LLM 对候选片段按相关性重排（「检索 + 重排序」链路的重排序段）。

重排序走一个专用 client（temperature=0、较大输出预算），并对单条调用做异常回退：
解析失败或调用异常时退回候选原顺序，保证评测流程不会因单条失败中断。
"""

from __future__ import annotations

import re

from src.config.settings import settings

PROMPT_TEMPLATE = (
    "给定查询和若干候选医学文本片段，按与查询的相关性从高到低排序。"
    "只输出片段编号（用逗号分隔，如 2,1,3），不要输出任何解释。\n\n"
    "查询：{query}\n\n"
    "{candidates}"
)

_RERANK_LLM = None


def _get_rerank_llm():
    global _RERANK_LLM
    if _RERANK_LLM is None:
        from langchain_openai import ChatOpenAI

        _RERANK_LLM = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model_name,
            temperature=0,
            max_tokens=8192,  # 推理模型需给 reasoning 留余量
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
    return _RERANK_LLM


def build_prompt(query: str, candidates) -> str:
    lines = [f"[{i}] {c.text.strip()[:200]}" for i, c in enumerate(candidates, start=1)]
    return PROMPT_TEMPLATE.format(query=query, candidates="\n".join(lines))


def parse_ranking(text) -> list:
    """从 LLM 输出解析编号序列（去重、保序）；解析失败返回空列表。"""
    seen = []
    for n in re.findall(r"\d+", text or ""):
        v = int(n)
        if v >= 1 and v not in seen:
            seen.append(v)
    return seen


def rerank(query, candidates, k=None):
    """用 LLM 重排序候选，返回重排后的候选列表；失败则退回原顺序。"""
    if not candidates:
        return []
    order = []
    try:
        resp = _get_rerank_llm().invoke(build_prompt(query, candidates))
        order = parse_ranking(resp.content)
    except Exception:
        order = []

    ranked = [candidates[i - 1] for i in order if 1 <= i <= len(candidates)]
    seen = set(order)
    for i, c in enumerate(candidates, start=1):
        if i not in seen:
            ranked.append(c)
    return ranked[:k] if k is not None else ranked
