"""
Content Agent - 内容生成专家

独立负责：
  1. 根据用户需求生成各类文本内容（辞职信、通知、总结等）
  2. 判断用户是否要求发送，主动从输入中提取 platform/receiver
  3. 需要发送时，将生成的内容写入 state 并链式调用 Message Agent
"""

from datetime import datetime
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.agent.context import latest_user_input
from src.agent.delivery import attach_delivery
from src.utils.logger import logger
from src.utils.execution import is_timeout


CONTENT_GENERATE_PROMPT = """你是一个专业的内容创作助手。根据用户的需求生成相应的文本内容。

要求：
1. 内容要专业、完整、格式规范
2. 如果用户提到特定的格式要求（如正式、简洁等），请遵循
3. 生成完后，简要说明内容要点

当前日期: {current_date}

用户需求: {user_input}

请直接输出生成的内容，无需其他说明。"""

def content_agent(state: AgentState) -> dict:
    """内容子 Agent：生成内容，并可选择交由 MessageAgent 发送"""
    logger.info("[ContentAgent] 开始处理内容生成任务")

    user_input = latest_user_input(state)

    if not user_input:
        return {"final_output": "未找到用户输入", "task_status": "failed"}

    try:
        from src.api.llm import get_llm
        llm = get_llm()

        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        prompt_text = CONTENT_GENERATE_PROMPT.format(
            current_date=current_date,
            user_input=user_input,
        )

        response = llm.invoke([HumanMessage(content=prompt_text)])
        generated = response.content.strip()
        if not generated:
            raise ValueError("模型未返回生成内容")

        logger.info(f"[ContentAgent] 内容生成完成 ({len(generated)} 字符)")

        return attach_delivery(state, user_input, generated, f"📝 内容已生成：\n\n{generated}")

    except Exception as e:
        logger.error(f"[ContentAgent] 生成失败: {e}")
        return {"final_output": f"❌ 内容生成失败: {e}", "task_status": "timeout" if is_timeout(e) else "failed"}
