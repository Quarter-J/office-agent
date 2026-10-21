"""
节点7：结果输出节点

返回最终执行结果，记录日志，结束流程。
"""

from src.agent.state import AgentState
from src.utils.logger import logger


def output_result(state: AgentState) -> dict:
    final_output = state.get("final_output", "")
    error = state.get("error")
    task_status = state.get("task_status", "unknown")

    logger.info(f"输出结果: status={task_status}")

    if error:
        logger.error(f"任务执行失败: {error}")

    return {
        "final_output": final_output,
        "error": error,
        "task_status": task_status,
    }
