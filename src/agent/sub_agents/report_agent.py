"""
Report Agent - 日报专家

独立负责：
  1. 解析日报生成指令
  2. 调用 LLM 结构化工作内容
  3. 生成 Markdown 日报文件
  4. 查询历史日报
  5. 周报/月报汇总（预留）
"""

import json
from datetime import datetime
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.agent.context import latest_user_input
from src.agent.delivery import attach_delivery
from src.tools.report import generate_daily_report, list_recent_reports, get_report_by_date
from src.tools.prompts import REPORT_EXTRACT_TEMPLATE
from src.utils.logger import logger
from src.utils.execution import is_timeout


def report_agent(state: AgentState) -> dict:
    """日报子 Agent：生成日报或查询历史"""
    logger.info("[ReportAgent] 开始处理日报任务")

    user_input = latest_user_input(state)

    if not user_input:
        return {"final_output": "未找到用户输入", "task_status": "failed"}

    intent = state.get("intent", "")

    if intent == "list_reports":
        return _handle_list_reports(state, user_input)
    else:
        return _handle_generate_report(state, user_input)


def _handle_generate_report(state: AgentState, user_input: str) -> dict:
    """生成日报"""
    name = state.get("report_name")
    today_work = state.get("report_today_work")
    department = state.get("report_department") or "未知"
    tomorrow_plan = state.get("report_tomorrow_plan") or "暂无"
    issues = state.get("report_issues") or "无"
    notes = state.get("report_notes") or "无"

    if not name or not today_work:
        try:
            from src.api.llm import get_llm
            llm = get_llm()
            current_date = datetime.now().strftime("%Y-%m-%d")
            prompt = REPORT_EXTRACT_TEMPLATE.format_messages(
                user_input=f"{user_input}\n当前日期: {current_date}"
            )
            response = llm.invoke(prompt)
            info = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            name = name or info.get("name", "")
            department = info.get("department", department)
            today_work = today_work or info.get("today_work", "")
            tomorrow_plan = info.get("tomorrow_plan", tomorrow_plan)
            issues = info.get("issues", issues)
            notes = info.get("notes", notes)
        except Exception as e:
            if is_timeout(e):
                return {"final_output": "日报参数解析超时，尚未生成或发送日报。", "task_status": "timeout"}
            logger.warning(f"[ReportAgent] LLM 解析失败: {e}")

    if not name or not today_work:
        return {
            "final_output": "⚠️ 信息不完整，请提供姓名和今日工作内容",
            "task_status": "pending",
            "needs_clarification": True,
            "clarification_question": "请提供您的姓名和今日工作内容",
        }

    try:
        raw = generate_daily_report.invoke({
            "name": name,
            "department": department,
            "today_work": today_work,
            "tomorrow_plan": tomorrow_plan,
            "issues": issues,
            "notes": notes,
            "report_date": state.get("report_date"),
        })
        result = json.loads(raw)
        if result.get("status") in ("conflict", "unknown", "timeout"):
            return {"final_output": result["error"], "error": result["error"], "task_status": result["status"]}
        if result.get("success"):
            content = result["content"]
            output = f"📊 {result['summary']}\n\n文件：{result['file_path']}"
            return {**attach_delivery(state, user_input, content, output),
                    "report_content": content, "report_file_path": result["file_path"]}
        else:
            return {"final_output": f"❌ 日报生成失败: {result.get('error', '未知错误')}", "task_status": "failed"}
    except Exception as e:
        logger.error(f"[ReportAgent] 异常: {e}")
        return {"final_output": f"❌ 日报生成异常: {e}", "task_status": "failed"}


def _handle_list_reports(state: AgentState, user_input: str) -> dict:
    """查看历史日报"""
    name = state.get("report_name")
    if not name:
        import re
        m = re.search(r"(?:我|(\S+))的日报", user_input)
        if m and m.group(1):
            name = m.group(1)

    try:
        raw = list_recent_reports.invoke({"name": name or None})
        result = json.loads(raw)
        if result.get("success"):
            files = result.get("files", [])
            if not files:
                return {"final_output": "📋 暂无日报文件", "task_status": "success"}
            file_list = "\n".join([f"  📄 {f['name']} (修改于 {f['modified'][:10]})" for f in files])
            return {"final_output": f"📋 最近日报:\n{file_list}", "task_status": "success"}
        else:
            return {"final_output": f"❌ 查询失败: {result.get('error', '未知错误')}", "task_status": "failed"}
    except Exception as e:
        logger.error(f"[ReportAgent] 异常: {e}")
        return {"final_output": f"❌ 查询异常: {e}", "task_status": "failed"}
