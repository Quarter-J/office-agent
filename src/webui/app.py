"""
Gradio Web 聊天界面

提供类似 ChatGPT 的对话式交互界面，支持：
  - 多轮对话
  - 消息发送、日报生成、搜索等所有 Agent 功能
  - 执行结果显示（成功/失败/追问）
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.runner import run_agent
from src.utils.logger import logger


def _build_state_from_history(message: str, history: list) -> dict:
    """从 Gradio 对话历史和当前消息构建 Agent 状态"""
    messages = []

    for entry in history:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            user_msg, ai_msg = entry[0], entry[1]
            if user_msg:
                messages.append(HumanMessage(content=user_msg))
            if ai_msg:
                messages.append(AIMessage(content=ai_msg))
        elif isinstance(entry, dict):
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    return {
        "messages": messages,
        "intent": None,
        "intent_reason": None,
        "confidence": None,
        "current_agent": None,
        "task_status": "pending",
        "retry_count": 0,
        "max_retries": 2,
        "sender": None,
        "receiver": None,
        "message_content": None,
        "platform": None,
        "group_id": None,
        "scheduled_time": None,
        "search_query": None,
        "search_results": None,
        "search_source": None,
        "report_name": None,
        "report_department": None,
        "report_date": None,
        "report_content": None,
        "report_today_work": None,
        "report_tomorrow_plan": None,
        "report_issues": None,
        "report_notes": None,
        "report_file_path": None,
        "report_template": None,
        "knowledge_results": None,
        "knowledge_file_name": None,
        "knowledge_file_content": None,
        "missing_params": None,
        "needs_clarification": False,
        "clarification_question": None,
        "agent_history": [],
        "supervisor_decision": None,
        "next_agent": None,
        "session_id": "webui_default",
        "user_id": "default",
        "memory_context": None,
        "memory_loaded": False,
        "final_output": None,
        "error": None,
    }


def respond(message: str, history: list) -> str:
    """处理用户消息，返回 Agent 回复（自动继承历史对话）"""
    if not message.strip():
        return ""

    logger.info(f"WebUI 收到消息: {message[:60]}...")

    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        state = _build_state_from_history(message, history)
        result = run_agent(state)

        final_output = result.get("final_output")
        error = result.get("error")

        if error:
            error_msg = final_output or str(error)
            return f"❌ {error_msg}\n\n*({timestamp})*"
        elif final_output:
            return f"{final_output}\n\n*({timestamp})*"
        else:
            return f"未产生输出\n\n*({timestamp})*"

    except Exception as e:
        logger.error(f"WebUI 处理异常: {e}")
        return f"处理请求时出现错误: {e}"


# ============================================================
# 界面主题配置
# ============================================================

CUSTOM_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
)

CUSTOM_CSS = """
#chatbot { min-height: 500px; }
#title { text-align: center; font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem; }
#subtitle { text-align: center; color: #666; margin-bottom: 1rem; }
.footer { text-align: center; color: #999; font-size: 0.8rem; margin-top: 1rem; }
"""

# ============================================================
# 构建 Gradio 界面
# ============================================================

def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Office Agent") as demo:
        gr.Markdown(
            "# 🤖 Office Agent",
            elem_id="title",
        )
        gr.Markdown(
            "智能办公助手 · 消息发送 · 工作日报 · 知识库检索 · 网页搜索",
            elem_id="subtitle",
        )

        chatbot = gr.ChatInterface(
            fn=respond,
            description=None,
            examples=[
                "给QQ好友10086发消息：明天下午三点开会",
                "生成今天的日报，我是张三，完成了项目开发",
                "查看我最近的日报",
                "在知识库中搜索Spring Boot相关文档",
                "搜索一下Python 3.12的新特性",
                "帮我搜索最新的AI新闻",
                "你好，今天天气怎么样",
            ],
            cache_examples=False,
        )

        gr.Markdown(
            "💡 提示：输入 `clear` 清空对话 | 输入 `exit` 退出",
            elem_classes="footer",
        )

    return demo


demo = create_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
    )
