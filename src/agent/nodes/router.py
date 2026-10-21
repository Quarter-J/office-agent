"""
节点3：任务路由节点

根据意图将请求分发至对应的执行节点。
"""

from typing import Literal
from src.agent.state import AgentState
from src.utils.logger import logger


def route_by_intent(state: AgentState) -> Literal[
    "send_message_handler",
    "generate_report_handler",
    "list_reports_handler",
    "web_search_handler",
    "knowledge_search_handler",
    "add_knowledge_handler",
    "chat_handler",
]:
    intent = state.get("intent", "chat")
    logger.info(f"路由分发: {intent}")
    return f"{intent}_handler"
