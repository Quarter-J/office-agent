"""
节点6：记忆更新节点

保存本次操作记录，更新用户习惯记忆。
"""

import json
from datetime import datetime
from src.agent.state import AgentState
from src.utils.logger import logger


def update_memory(state: AgentState) -> dict:
    intent = state.get("intent", "chat")
    task_status = state.get("task_status", "unknown")
    final_output = state.get("final_output", "")

    logger.info(f"更新记忆: intent={intent}, status={task_status}")

    # TODO: 接入数据库后启用长期记忆
    # from src.storage.database import insert, update, query
    #
    # # 记录任务日志
    # insert("task_logs", {
    #     "task_type": intent,
    #     "intent": intent,
    #     "status": task_status,
    #     "input_data": json.dumps(_get_input_snapshot(state), ensure_ascii=False),
    #     "output_data": final_output[:500] if final_output else "",
    # })
    #
    # if task_status == "success" and intent in ("send_message",):
    #     _update_send_memory(state)
    # elif task_status == "success" and intent in ("generate_report",):
    #     _update_report_memory(state)

    logger.info(f"记忆已更新: {intent}/{task_status}")
    return {"final_output": final_output}


def _get_input_snapshot(state: AgentState) -> dict:
    keys = [
        "intent", "platform", "receiver", "message_content",
        "search_query", "report_name", "report_date",
    ]
    return {k: state.get(k) for k in keys if state.get(k)}
