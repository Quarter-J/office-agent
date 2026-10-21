# -*- coding: utf-8 -*-
"""Tool Registry - Central registration of all tools for Tool Calling mode.

All tools listed in ROUTER_TOOLS are directly available to the Router Agent
via LLM bind_tools(). The LLM selects and calls tools based on user input,
removing the need for explicit intent classification on simple tasks.
"""

import json
from langchain_core.tools import tool
from src.tools.search import (
    web_search,
    knowledge_base_search,
    rag_search,
    lookup_entity,
    add_to_knowledge_base,
)
from src.tools.message import (
    send_qq_message,
    send_wechat_message,
    get_qq_number_by_nickname,
)
from src.tools.report import generate_daily_report, list_recent_reports
from src.utils.logger import logger


@tool
def delegate_to_supervisor(task_description: str) -> str:
    """Delegate complex multi-step tasks that require orchestration across multiple tools.

    Use this ONLY when the task genuinely requires multiple sequential steps
    that no single tool call can handle. Examples:
    - "Write a resignation letter and send it to Zhang San" (generate + send)
    - "Generate today's report and notify the team" (generate + send)
    - Any task that needs sequential execution of different capabilities

    Do NOT use for simple single-step tasks. For those, call the appropriate tool directly.

    Args:
        task_description: Brief description of the complex multi-step task for the supervisor
    """
    logger.info(f"[Delegation] Complex task delegated: {task_description}")
    return json.dumps({
        "status": "delegated",
        "description": task_description,
    })


# All tools available for direct Tool Calling by the Router Agent
ROUTER_TOOLS = [
    web_search,
    knowledge_base_search,
    rag_search,
    lookup_entity,
    add_to_knowledge_base,
    send_qq_message,
    send_wechat_message,
    get_qq_number_by_nickname,
    generate_daily_report,
    list_recent_reports,
    delegate_to_supervisor,
]
