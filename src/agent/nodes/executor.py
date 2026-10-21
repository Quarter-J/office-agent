"""
节点4：工具执行节点（LangGraph 职责）

只负责：
  1. 根据意图路由结果，调用 LangChain 封装的工具
  2. 处理工具返回结果，写入 State
  3. 处理异常（失败标记、重试逻辑）
不负责：Prompt 内容、LLM 调用细节
"""

import json
from src.agent.state import AgentState, TaskStatus
from src.utils.logger import logger
from src.tools.message import send_qq_message, send_wechat_message
from src.tools.report import generate_daily_report, list_recent_reports, get_report_by_date
from src.tools.search import web_search, knowledge_base_search, add_to_knowledge_base
from src.storage.database import insert


INTENT_HANDLER_MAP = {
    "send_message": "_execute_send_message",
    "generate_report": "_execute_generate_report",
    "list_reports": "_execute_list_reports",
    "web_search": "_execute_web_search",
    "knowledge_search": "_execute_knowledge_search",
    "add_knowledge": "_execute_add_knowledge",
}


def execute_tool(state: AgentState) -> dict:
    intent = state.get("intent", "chat")
    logger.info(f"执行工具: intent={intent}")

    if intent == "chat":
        return _execute_chat(state)

    handler_name = INTENT_HANDLER_MAP.get(intent)
    if not handler_name:
        return {
            "final_output": f"未知意图: {intent}，请重新描述您的需求",
            "error": f"未知意图: {intent}",
            "task_status": TaskStatus.FAILED,
        }

    handler = globals().get(handler_name)
    if not handler:
        return {
            "final_output": f"未实现 {intent} 的处理逻辑",
            "error": f"未实现 {intent} 的处理逻辑",
            "task_status": TaskStatus.FAILED,
        }

    result = handler(state)

    _log_execution(state, result)

    return result


def _execute_send_message(state: AgentState) -> dict:
    platform = state.get("platform", "qq")
    receiver = state.get("receiver", "")
    content = state.get("message_content", "")
    group_id = state.get("group_id")

    logger.info(f"发送消息: platform={platform}, receiver={receiver}")

    try:
        if platform == "qq":
            raw = send_qq_message.invoke({
                "user_id": receiver,
                "message": content,
                "group_id": group_id,
            })
        elif platform == "wechat":
            raw = send_wechat_message.invoke({
                "who": receiver,
                "message": content,
            })
        else:
            return _fail(f"暂不支持 {platform} 平台的消息发送")

        result = json.loads(raw)
        if result.get("success"):
            return _ok(f"消息已成功发送到 {platform} 的 {receiver}")
        else:
            return _fail(f"消息发送失败: {result.get('error', '未知错误')}", result.get("error"))

    except Exception as e:
        logger.error(f"消息发送异常: {e}")
        return _fail(f"消息发送异常: {e}", str(e))


def _execute_generate_report(state: AgentState) -> dict:
    logger.info("生成日报")
    try:
        raw = generate_daily_report.invoke({
            "name": state.get("report_name", "未知"),
            "department": state.get("report_department", "未知"),
            "today_work": state.get("report_today_work", ""),
            "tomorrow_plan": state.get("report_tomorrow_plan", "暂无"),
            "issues": state.get("report_issues", "无"),
            "notes": state.get("report_notes", "无"),
            "report_date": state.get("report_date"),
        })
        result = json.loads(raw)
        if result.get("success"):
            return {
                "final_output": result["summary"],
                "report_file_path": result["file_path"],
                "task_status": TaskStatus.SUCCESS,
            }
        else:
            return _fail(f"日报生成失败: {result.get('error', '未知错误')}", result.get("error"))
    except Exception as e:
        logger.error(f"日报生成异常: {e}")
        return _fail(f"日报生成异常: {e}", str(e))


def _execute_list_reports(state: AgentState) -> dict:
    logger.info("查看历史日报")
    try:
        raw = list_recent_reports.invoke({"name": state.get("report_name") or None})
        result = json.loads(raw)

        if result.get("success"):
            files = result.get("files", [])
            if not files:
                return _ok("暂无日报文件")
            file_list = "\n".join([
                f"  - {f['name']} (修改于 {f['modified'][:10]})"
                for f in files
            ])
            return _ok(f"最近日报:\n{file_list}")
        else:
            return _fail(f"查询失败: {result.get('error', '未知错误')}")
    except Exception as e:
        logger.error(f"查看日报异常: {e}")
        return _fail(f"查看日报异常: {e}")


def _execute_web_search(state: AgentState) -> dict:
    query = state.get("search_query", "")
    logger.info(f"网页搜索: {query}")
    try:
        raw = web_search.invoke({"query": query})
        result = json.loads(raw)

        if result.get("success"):
            results = result.get("results", [])
            if not results:
                return _ok(f"未找到关于「{query}」的信息")

            formatted = []
            for r in results:
                formatted.append(
                    f"📄 {r['title']}\n   {r['snippet'][:150]}...\n   🔗 {r['url']}"
                )
            return {
                "final_output": f"🔍 搜索「{query}」结果:\n\n" + "\n\n".join(formatted),
                "search_results": results,
                "task_status": TaskStatus.SUCCESS,
            }
        else:
            return _fail(f"搜索失败: {result.get('error', '未知错误')}")
    except Exception as e:
        logger.error(f"网页搜索异常: {e}")
        return _fail(f"搜索异常: {e}")


def _execute_knowledge_search(state: AgentState) -> dict:
    query = state.get("search_query", "")
    logger.info(f"知识库检索: {query}")
    try:
        raw = knowledge_base_search.invoke({"query": query})
        result = json.loads(raw)

        if result.get("success"):
            results = result.get("results", [])
            message = result.get("message", "")
            if message:
                return _ok(message)
            if not results:
                return _ok(f"知识库中未找到与「{query}」相关的内容")

            formatted = []
            for r in results:
                formatted.append(
                    f"[{r['rank']}] 📁 {r['file']} (相关度: {r['relevance']:.2%})\n   {r['content'][:200]}"
                )
            return {
                "final_output": f"📚 知识库检索「{query}」结果:\n\n" + "\n\n".join(formatted),
                "knowledge_results": results,
                "task_status": TaskStatus.SUCCESS,
            }
        else:
            return _fail(f"知识库检索失败: {result.get('error', '未知错误')}")
    except Exception as e:
        logger.error(f"知识库检索异常: {e}")
        return _fail(f"知识库检索异常: {e}")


def _execute_add_knowledge(state: AgentState) -> dict:
    file_name = state.get("knowledge_file_name", "untitled.md")
    content = state.get("knowledge_file_content", "")
    logger.info(f"添加知识库: {file_name}")
    try:
        raw = add_to_knowledge_base.invoke({"file_path": file_name, "content": content})
        result = json.loads(raw)
        if result.get("success"):
            return _ok(result["message"])
        else:
            return _fail(f"添加失败: {result.get('error', '未知错误')}")
    except Exception as e:
        logger.error(f"添加知识库异常: {e}")
        return _fail(f"添加知识库异常: {e}")


def _execute_chat(state: AgentState) -> dict:
    logger.info("处理普通对话")

    try:
        from src.api.llm import get_llm
        from src.tools.prompts import CHAT_SYSTEM_TEMPLATE
        from langchain_core.messages import SystemMessage
        from datetime import datetime

        llm = get_llm()
        messages = state.get("messages", [])
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        system_content = CHAT_SYSTEM_TEMPLATE.format(current_date=current_date)
        response = llm.invoke([
            SystemMessage(content=system_content),
            *messages,
        ])
        return _ok(response.content)
    except Exception as e:
        logger.error(f"对话异常: {e}")
        return _fail(f"对话处理失败: {e}")


def _ok(output: str) -> dict:
    return {"final_output": output, "task_status": TaskStatus.SUCCESS}


def _fail(output: str, error: str = None) -> dict:
    return {"final_output": output, "error": error or output, "task_status": TaskStatus.FAILED}


def _log_execution(state: AgentState, result: dict):
    """记录任务执行日志到数据库"""
    try:
        insert("task_logs", {
            "task_type": state.get("intent", "unknown"),
            "intent": state.get("intent"),
            "status": result.get("task_status", "unknown"),
            "input_data": json.dumps({
                k: state.get(k) for k in [
                    "platform", "receiver", "message_content",
                    "search_query", "report_name", "report_date",
                ] if state.get(k)
            }, ensure_ascii=False),
            "output_data": (result.get("final_output") or "")[:500],
            "error_message": result.get("error", ""),
        })
    except Exception as e:
        logger.warning(f"记录执行日志失败: {e}")
