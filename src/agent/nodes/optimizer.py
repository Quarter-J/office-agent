"""
节点5：结果优化节点

对工具执行结果进行润色、精简、格式化。
"""

from src.agent.state import AgentState
from src.utils.logger import logger


def optimize_result(state: AgentState) -> dict:
    final_output = state.get("final_output", "")
    intent = state.get("intent", "chat")
    task_status = state.get("task_status", "success")

    if not final_output:
        return {"final_output": "执行完成，但未生成输出内容"}

    logger.info(f"优化输出: intent={intent}, status={task_status}")

    if task_status == "failed":
        optimized = f"❌ {final_output}"
    elif task_status == "success":
        optimized = _format_by_intent(intent, final_output)
    else:
        optimized = final_output

    return {"final_output": optimized}


def _format_by_intent(intent: str, output: str) -> str:
    intent_prefixes = {
        "send_message": "📨 ",
        "generate_report": "📊 ",
        "list_reports": "📋 ",
        "web_search": "🔍 ",
        "knowledge_search": "📚 ",
        "add_knowledge": "📝 ",
        "chat": "",
    }

    prefix = intent_prefixes.get(intent, "")
    if prefix and not output.startswith(prefix):
        return f"{prefix}{output}"

    return output
