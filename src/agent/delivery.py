"""Shared handoff from generated content/reports to the message agent."""

import json
import re
from langchain_core.messages import HumanMessage
from src.agent.context import wants_delivery
from src.utils.logger import logger


EXTRACT_SEND_PROMPT = """从用户原始指令中提取发送目标，不能猜测缺失信息。
返回 JSON：{"platform":"qq/wechat/dingtalk/wecom/email 或空字符串",
"receiver":"收件人或空字符串","group_id":"群号或 null"}。
QQ群发送只需 group_id，不需要 receiver。只返回 JSON。"""


def extract_send_params(user_input: str) -> dict:
    params = {"platform": "", "receiver": "", "group_id": None}
    try:
        from src.api.llm import get_llm
        response = get_llm().invoke([HumanMessage(content=EXTRACT_SEND_PROMPT + "\n用户输入: " + user_input)])
        data = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        for key in params:
            value = data.get(key)
            if value is not None:
                params[key] = str(value).strip()
    except Exception as exc:
        logger.warning(f"发送目标提取失败，尝试明确的目标格式: {exc}")
    if not params["platform"]:
        for keyword, platform in (("qq", "qq"), ("企业微信", "wecom"), ("微信", "wechat"),
                                  ("钉钉", "dingtalk"), ("邮件", "email")):
            if keyword in user_input.lower():
                params["platform"] = platform
                break
    group = re.search(r"(?:QQ群|qq\s*群)\s*(\d+)", user_input, re.IGNORECASE)
    if group:
        params.update(platform="qq", group_id=group.group(1))
    if not params["receiver"] and not params["group_id"]:
        match = re.search(r"(?:发给|发送给|给|通知)\s*(?:QQ(?:好友)?|微信)?\s*"
                          r"([^\s，。:：]+?)(?=发消息|发送|发通知|[\s，。:：]|$)", user_input, re.IGNORECASE)
        if match:
            params["receiver"] = match.group(1)
    return params


def attach_delivery(state: dict, user_input: str, content: str, output: str) -> dict:
    result = {"generated_content": content, "final_output": output,
              "task_status": "success", "next_agent": None}
    if not wants_delivery(user_input):
        return result
    params = {key: state.get(key) for key in ("platform", "receiver", "group_id")}
    target = params["group_id"] if params["platform"] == "qq" and params["group_id"] else params["receiver"]
    if not params["platform"] or not target:
        extracted = extract_send_params(user_input)
        params = {key: value or extracted.get(key) for key, value in params.items()}
    result.update(params)
    result.update(message_content=content, next_agent="send_message_agent")
    return result
