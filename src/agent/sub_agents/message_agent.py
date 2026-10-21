"""
Message Agent - 消息发送专家

独立负责：
  1. 解析消息发送指令（平台、接收人、内容）
  2. QQ 支持按昵称/备注名查询号码后发送
  3. 调用对应渠道工具发送
  4. 处理发送结果和重试
"""

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.state import AgentState
from src.agent.context import latest_user_input, delivery_forbidden
from src.tools.message import send_qq_message, send_wechat_message, get_qq_number_by_nickname
from src.tools.prompts import SEND_MESSAGE_EXTRACT_TEMPLATE
from src.utils.logger import logger
from src.utils.execution import is_timeout
from src.agent.conflicts import conflict_result


def message_agent(state: AgentState) -> dict:
    """Consume a send continuation on every exit, preserving generated content."""
    try:
        result = _send_message(state)
    except Exception as exc:
        logger.error(f"[MessageAgent] 异常: {exc}")
        result = {"final_output": f"❌ 消息发送异常: {exc}", "error": str(exc), "task_status": "failed"}
    if state.get("generated_content") and result.get("task_status") in ("failed", "timeout", "unknown", "conflict"):
        label = "发送结果未确认" if result["task_status"] == "unknown" else "尚未成功发送"
        result["final_output"] = (f"📝 内容已生成，{label}：\n\n{state['generated_content']}"
                                  f"\n\n{result['final_output']}")
        if state.get("report_file_path"):
            result["final_output"] += f"\n\n日报文件：{state['report_file_path']}"
    return {**result, "next_agent": None}


def _send_message(state: AgentState) -> dict:
    """消息子 Agent：解析指令 → 调用工具 → 返回结果"""
    logger.info("[MessageAgent] 开始处理消息发送任务")

    user_input = latest_user_input(state)

    if not user_input:
        return {"final_output": "未找到用户输入", "task_status": "failed"}
    if delivery_forbidden(user_input):
        result = conflict_result("user_constraint", "用户限定不发送，消息未发送", "send_message", user_input)
        return {"final_output": result["error"], "task_status": "conflict", "error": result["error"]}

    platform = state.get("platform")
    receiver = state.get("receiver")
    content = state.get("message_content")
    group_id = state.get("group_id")
    target = group_id if platform == "qq" and group_id else receiver

    if not all([platform, target, content]):
        try:
            from src.api.llm import get_llm
            llm = get_llm()
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            prompt = SEND_MESSAGE_EXTRACT_TEMPLATE.format_messages(
                user_input=user_input + f"\n当前日期: {current_date}"
            )
            response = llm.invoke(prompt)
            info = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            platform = platform or info.get("platform")
            receiver = receiver or info.get("receiver")
            # A generated artifact must never be replaced by another extraction.
            content = content or info.get("content")
            group_id = group_id or info.get("group_id")
        except Exception as e:
            if is_timeout(e):
                return {"final_output": "发送参数解析超时，尚未发送消息。", "task_status": "timeout"}
            logger.warning(f"[MessageAgent] LLM 解析失败，使用已有参数: {e}")

    platform = str(platform or "").strip().lower()
    receiver = str(receiver or "").strip()
    group_id = str(group_id or "").strip() or None
    target = group_id if platform == "qq" and group_id else receiver
    if not platform or not target or not content:
        return {
            "final_output": "⚠️ 信息不完整，请指定消息接收人和发送内容",
            "task_status": "pending",
            "needs_clarification": True,
            "clarification_question": "请指定发送平台、接收人（QQ群发送请提供群号）和消息内容",
        }
    if platform not in ("qq", "wechat"):
        return {"final_output": f"暂不支持 {platform} 消息发送", "task_status": "failed"}
    if platform == "qq" and group_id and not group_id.isdigit():
        return {"final_output": "QQ群号必须是纯数字", "task_status": "pending",
                "needs_clarification": True, "clarification_question": "请提供正确的数字QQ群号"}

    logger.info(f"[MessageAgent] 准备发送: platform={platform}, to={receiver}")

    qq_id = receiver

    if platform == "qq" and not group_id and not receiver.isdigit():
        logger.info(f"[MessageAgent] QQ 号非数字，尝试按昵称查询: {receiver}")
        raw = get_qq_number_by_nickname.invoke({"nickname": receiver})
        result = json.loads(raw)
        if result.get("success"):
            qq_id = result["user_id"]
            logger.info(f"[MessageAgent] 昵称查询成功: {receiver} → {qq_id}")
        else:
            error_msg = result.get("error", "未找到该QQ好友")
            if result.get("status") == "timeout":
                return {"final_output": error_msg, "task_status": "timeout"}
            return {"final_output": f"❌ {error_msg}\n\n💡 提示：请直接输入 QQ 号发送，例如：\n   \"给QQ{receiver}发消息：内容\"", "task_status": "failed"}

    try:
        if platform == "qq":
            raw = send_qq_message.invoke({
                "user_id": qq_id,
                "message": content,
                "group_id": group_id,
            })
        elif platform == "wechat":
            raw = send_wechat_message.invoke({"who": receiver, "message": content})
        else:
            return {"final_output": f"暂不支持 {platform} 消息发送", "task_status": "failed"}

        result = json.loads(raw)
        if result.get("status") in ("timeout", "unknown", "conflict"):
            return {"final_output": result["error"], "error": result["error"], "task_status": result["status"]}
        if result.get("success"):
            logger.info(f"[MessageAgent] 发送成功")
            return {"final_output": f"✅ 消息已成功发送到 {platform} 的 {target}", "task_status": "success"}
        else:
            error = result.get("error", "未知错误")
            logger.error(f"[MessageAgent] 发送失败: {error}")
            return {"final_output": f"❌ 消息发送失败: {error}", "error": error, "task_status": "failed"}
    except Exception as e:
        logger.error(f"[MessageAgent] 异常: {e}")
        return {"final_output": f"❌ 消息发送异常: {e}", "error": str(e), "task_status": "failed"}
