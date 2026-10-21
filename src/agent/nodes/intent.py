"""
节点1：意图识别节点（LangGraph 职责）

只负责：
  1. 接收用户指令
  2. 调用 LangChain 的 PromptTemplate 分类意图
  3. 返回分类结果
不负责：LLM 调用细节、Prompt 内容
"""

import json
from datetime import datetime
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.utils.logger import logger
from src.tools.prompts import INTENT_CLASSIFIER_TEMPLATE


def classify_intent(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {
            "intent": "chat",
            "intent_reason": "无用户输入，默认对话模式",
            "confidence": 1.0,
        }

    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content
    else:
        return {
            "intent": "chat",
            "intent_reason": "上一条消息非用户输入，默认对话模式",
            "confidence": 1.0,
        }

    logger.info(f"正在分析意图: {user_input[:80]}...")

    try:
        from src.api.llm import get_llm
        llm = get_llm()
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        prompt = INTENT_CLASSIFIER_TEMPLATE.format_messages(
            user_input=user_input,
            current_date=current_date,
        )
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        return {
            "intent": result.get("intent", "chat"),
            "intent_reason": result.get("reason", ""),
            "confidence": result.get("confidence", 0.8),
            "final_output": None,
            "error": None,
            "task_status": "running",
        }
    except Exception as e:
        logger.warning(f"LLM 意图分类失败，使用规则降级: {e}")

    intent, reason, confidence = _rule_based_classify(user_input)

    return {
        "intent": intent,
        "intent_reason": reason,
        "confidence": confidence,
        "final_output": None,
        "error": None,
        "task_status": "running",
    }


def _rule_based_classify(user_input: str):
    text = user_input.lower()

    if any(kw in text for kw in ["发消息", "发送", "告诉", "通知", "qq", "微信", "邮件"]):
        return "send_message", "检测到消息发送关键词", 0.6

    if any(kw in text for kw in ["日报", "工作日报", "每日汇报", "工作总结"]):
        if any(kw in text for kw in ["查看", "最近", "历史", "列表"]):
            return "list_reports", "检测到查看日报关键词", 0.7
        return "generate_report", "检测到生成日报关键词", 0.7

    if any(kw in text for kw in ["搜索", "查一下", "查找", "搜一下", "百度"]):
        return "web_search", "检测到搜索关键词", 0.6

    if any(kw in text for kw in ["知识库", "本地文档", "文档中查找"]):
        return "knowledge_search", "检测到知识库检索关键词", 0.7

    if any(kw in text for kw in ["添加到知识库", "保存文档", "存入知识库"]):
        return "add_knowledge", "检测到添加知识库关键词", 0.7

    return "chat", "未匹配到特定意图，默认对话", 0.5
