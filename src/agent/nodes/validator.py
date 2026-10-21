"""
节点2：参数校验节点

校验指令是否缺失关键信息。
优先从用户原始消息中提取，提取不到再触发人机询问兜底。
"""

import re
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.utils.logger import logger


REQUIRED_PARAMS = {
    "send_message": {
        "platform": "请指定消息发送平台（QQ/微信/企业微信/钉钉/邮件）",
        "receiver": "请指定消息接收人",
        "message_content": "请填写要发送的消息内容",
    },
    "generate_report": {
        "report_name": "请填写您的姓名",
        "report_today_work": "请描述今日完成的工作内容",
    },
    "web_search": {
        "search_query": "请填写搜索关键词",
    },
    "knowledge_search": {
        "search_query": "请填写要检索的关键词",
    },
    "add_knowledge": {
        "knowledge_file_name": "请指定文档名称",
        "knowledge_file_content": "请提供文档内容",
    },
    "list_reports": {},
    "chat": {},
}


def validate_params(state: AgentState) -> dict:
    intent = state.get("intent", "chat")
    logger.info(f"校验参数: intent={intent}")

    user_input = _get_user_input(state)
    logger.info(f"用户原始输入: {user_input[:80] if user_input else '空'}")

    required = REQUIRED_PARAMS.get(intent, {})
    updates = {}
    missing = []

    for param, question in required.items():
        value = state.get(param)
        if not value:
            extracted = _try_extract(param, intent, user_input)
            if extracted:
                updates[param] = extracted
            else:
                missing.append({"param": param, "question": question})

    result = {
        "missing_params": missing if missing else None,
        "needs_clarification": bool(missing),
        "clarification_question": "；".join([m["question"] for m in missing]) if missing else None,
        "task_status": "pending" if missing else "running",
    }

    if updates:
        logger.info(f"从用户消息中提取到参数: {updates}")
        result.update(updates)

    if missing:
        logger.info(f"仍需用户补充: {result['clarification_question']}")
    else:
        logger.info("所有参数已就绪")

    return result


def _get_user_input(state: AgentState) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _try_extract(param: str, intent: str, user_input: str) -> str:
    if not user_input:
        return ""

    if intent == "web_search" and param == "search_query":
        return _extract_search_query(user_input)

    if intent == "knowledge_search" and param == "search_query":
        return _extract_search_query(user_input)

    if intent == "send_message":
        if param == "platform":
            return _extract_platform(user_input)
        if param == "receiver":
            return _extract_receiver(user_input)
        if param == "message_content":
            return _extract_message_content(user_input)

    if intent == "generate_report":
        if param == "report_name":
            return _extract_name(user_input)
        if param == "report_today_work":
            return _extract_today_work(user_input)

    return ""


def _extract_search_query(text: str) -> str:
    patterns = [
        r"(?:搜索|搜一下|查一下|查找|百度)\s*(.*)",
        r"(?:搜索|搜一下|查一下|查找|百度)[：:]\s*(.*)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return text


def _extract_platform(text: str) -> str:
    text_lower = text.lower()
    if "qq" in text_lower or "q群" in text or "qq群" in text_lower:
        return "qq"
    if "微信" in text:
        return "wechat"
    if "钉钉" in text:
        return "dingtalk"
    if "企业微信" in text or "企业微" in text:
        return "wecom"
    if "邮件" in text or "邮箱" in text or "mail" in text_lower:
        return "email"
    return ""


def _extract_receiver(text: str) -> str:
    patterns = [
        r"(?:发给|给|发送到|通知)\s*(\S+?)(?:[，,。\s]|$)",
        r"(?:QQ好友|QQ|微信|钉钉|邮件)\s*(\S+?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return ""


def _extract_message_content(text: str) -> str:
    patterns = [
        r"(?:消息|内容)[：:]\s*(.*)",
        r"(?:通知|告诉)[：:]?\s*(.*)",
        r"说[：:]\s*(.*)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""


def _extract_name(text: str) -> str:
    patterns = [
        r"(?:我是|我叫|姓名)[：:]?\s*(\S+)",
        r"(\S+?)的日报",
        r"(\S+?)的工作日报",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return ""


def _extract_today_work(text: str) -> str:
    patterns = [
        r"(?:完成|做了|完成了)[：:]\s*(.*)",
        r"工作内容[：:]\s*(.*)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""
