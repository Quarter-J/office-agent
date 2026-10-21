"""Helpers shared by routing and task agents; all decisions use the current turn."""

import re
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def current_turn_messages(state: dict) -> list:
    messages = list(state.get("messages") or [])
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return []


def latest_user_input(state: dict) -> str:
    messages = current_turn_messages(state)
    if not messages:
        return ""
    content = messages[0].content
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))


def latest_tool_batch(state: dict) -> list:
    """Match results to the latest assistant tool request, regardless of result order."""
    messages = current_turn_messages(state)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            calls = {call["id"]: call for call in msg.tool_calls}
            return [(calls[result.tool_call_id], result) for result in messages[i + 1:]
                    if isinstance(result, ToolMessage) and result.tool_call_id in calls]
    return []


def delivery_forbidden(text: str) -> bool:
    return bool(re.search(
        r"(?:不需要|不用|不要|无需|暂时不|暂不|先不|别|不)\s*(?:再|实际|对外)?\s*"
        r"(?:发送|发给|发到|发出去|发消息|通知|转发)|"
        r"(?:只|仅)(?:需|要)?(?:生成|起草|写)|\b(?:do not|don't|never)\s+send\b",
        text, re.IGNORECASE,
    ))


def wants_delivery(text: str) -> bool:
    if delivery_forbidden(text):
        return False
    return bool(re.search(
        r"发送|发给|发到|发往|发出去|转发|发消息|发通知|"
        r"通知(?:给|大家|团队|所有人)|(?:通知|告诉)\s*[^\s，。:：]+\s*[:：]|"
        r"\b(?:send|notify|deliver)\b", text, re.IGNORECASE,
    ))


def is_complex_task(text: str) -> bool:
    return wants_delivery(text) and bool(re.search(
        r"写|生成|起草|创作|编写|拟定|日报|报告|总结|\b(?:write|generate|draft|create)\b",
        text, re.IGNORECASE,
    ))


def business_intent(text: str):
    """Specific office actions precede broad writing/search keywords."""
    if any(word in text for word in ("添加到知识库", "保存文档", "存入知识库", "加入知识库")):
        return "add_knowledge"
    if "知识库" in text and any(word in text for word in ("查", "搜", "检索")):
        return "knowledge_search"
    if "日报" in text:
        if any(word in text for word in ("查看", "查询", "列表", "列出")) and not is_complex_task(text):
            return "list_reports"
        if any(word in text for word in ("生成", "写", "制作")) or is_complex_task(text):
            return "generate_report"
        if any(word in text for word in ("最近", "历史")):
            return "list_reports"
    return None
