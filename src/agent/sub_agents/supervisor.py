"""
Supervisor Agent - 主管调度者

职责：
  1. 分析用户意图（Classify）
  2. 路由到正确的子 Agent（Route）
  3. 收集子 Agent 的执行结果
  4. 异常时降级或追问
"""

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.state import AgentState
from src.agent.context import latest_user_input, business_intent, is_complex_task, wants_delivery, delivery_forbidden
from src.tools.prompts import INTENT_CLASSIFIER_TEMPLATE
from src.utils.logger import logger
from src.agent.conflicts import action_context, conflict_result
from src.api.context import ContextLimitError, ModelOutputLimitError, is_context_overflow

VALID_INTENTS = {"send_message", "generate_content", "generate_report", "list_reports",
                 "knowledge_search", "add_knowledge", "web_search", "chat"}


def supervisor_agent(state: AgentState) -> dict:
    """主管 Agent：分析意图，分派给子 Agent"""
    user_input = latest_user_input(state)
    if not user_input:
        return {
            "intent": "chat",
            "intent_reason": "无输入，默认对话",
            "confidence": 1.0,
            "task_status": "running",
        }

    logger.info(f"[Supervisor] 分析意图: {user_input[:80]}...")

    candidate = None
    try:
        from src.api.llm import get_llm

        llm = get_llm()
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        task_input = user_input
        if state.get("delegated_task"):
            task_input += f"\n\n委派补充说明（以用户原始指令为准）: {state['delegated_task']}"
        prompt = INTENT_CLASSIFIER_TEMPLATE.format_messages(
            user_input=task_input,
            current_date=current_date,
        )
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        intent = result.get("intent", "chat")
        candidate = intent
        if intent not in VALID_INTENTS:
            raise ValueError("无效的分类意图")
        reason = result.get("reason", "")
        confidence = result.get("confidence", 0.8)

        logger.info(f"[Supervisor] 初始判定: {intent} (置信度: {confidence})")

        # Post-classification verification: catch chat queries that need real-time data
        verified_intent, verify_reason = _verify_intent(intent, user_input)
        conflicts = list(state.get("conflicts") or [])
        if verify_reason:
            with action_context(state, "Supervisor") as context:
                conflict_result("routing", verify_reason, intent, verified_intent, resolution="rules")
                conflicts.extend(context["records"])
            logger.info(f"[Supervisor] 验证修正: {intent} -> {verified_intent} ({verify_reason})")
            intent = verified_intent
            reason = verify_reason

        logger.info(f"[Supervisor] 最终判定: {intent} (置信度: {confidence})")

        return {
            "intent": intent,
            "conflicts": conflicts,
            "intent_reason": reason,
            "confidence": confidence,
            "task_status": "running",
            "final_output": None,
            "error": None,
        }
    except Exception as e:
        if isinstance(e, (ContextLimitError, ModelOutputLimitError)) or is_context_overflow(e):
            return {"task_status": "context_error", "error": str(e), "final_output": str(e)}
        logger.warning(f"[Supervisor] LLM 分类失败，规则降级: {e}")
        intent, reason, confidence = _rule_based_classify(user_input)

        # Apply same verification to rule-based results
        verified_intent, verify_reason = _verify_intent(intent, user_input)
        if verify_reason:
            logger.info(f"[Supervisor] 规则验证修正: {intent} -> {verified_intent}")
            intent = verified_intent
            reason = verify_reason

        with action_context(state, "Supervisor") as context:
            conflict_result("routing", "分类结果无效或模型不可用，采用业务规则。",
                            candidate, intent, resolution="rules")
            conflicts = [*(state.get("conflicts") or []), *context["records"]]
        return {
            "intent": intent,
            "conflicts": conflicts,
            "intent_reason": reason,
            "confidence": confidence,
            "task_status": "running",
            "final_output": None,
            "error": None,
        }


def supervisor_router(state: AgentState) -> str:
    """主管路由：决定交给哪个子 Agent"""
    intent = state.get("intent", "chat")
    logger.info(f"[Supervisor] 路由 → {intent}")
    return f"{intent}_agent"


def _needs_realtime_data(text: str) -> bool:
    """Check if a query needs real-time/external data to answer accurately.
    Returns True if the query likely requires web search, False if it can be
    answered with general knowledge alone."""
    realtime_patterns = [
        # Weather
        "天气", "气温", "下雨", "下雪", "晴天", "阴天", "台风", "weather",
        # News & current events
        "新闻", "最新消息", "热搜", "头条", "最新进展", "最新动态",
        # Prices & finance
        "价格", "多少钱", "汇率", "股价", "股市", "行情", "涨跌", "基金",
        # Real-time info
        "现在几点", "今天几号", "今天是什么", "当前时间",
        # Timely queries
        "最新的", "现在的", "今天的", "最近的", "现在怎么样", "现在什么情况",
        # Events & schedules
        "比赛", "赛事", "直播", "上映", "排片",
        # Product/service lookups
        "哪里有卖", "附近", "推荐一个", "哪个好", "评分", "评价",
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in realtime_patterns)


def _verify_intent(intent: str, user_input: str) -> tuple:
    """Post-classification verification: catch queries misclassified as chat
    that actually need real-time data. Returns (intent, reason)."""
    specific = business_intent(user_input)
    if specific:
        return specific, "具体业务规则修正" if specific != intent else None
    if is_complex_task(user_input) and intent != "generate_content":
        return "generate_content", "先生成内容，再发送"
    if intent == "send_message" and delivery_forbidden(user_input):
        return ("generate_content" if any(w in user_input for w in ("写", "生成", "起草")) else "chat"), "用户限定不发送"
    if intent == "chat" and _needs_realtime_data(user_input):
        return "web_search", "Query needs real-time data, rerouted from chat"
    return intent, None


def _rule_based_classify(user_input: str):
    text = user_input.lower()
    specific = business_intent(user_input)
    if specific:
        return specific, "具体业务关键词匹配", 0.8
    if is_complex_task(user_input):
        return "generate_content", "检测到先写内容再发送的模式", 0.7
    if wants_delivery(user_input):
        return "send_message", "发送动作匹配", 0.7
    if any(kw in text for kw in ["写", "生成", "帮我写", "帮我生成", "创作", "起草"]):
        return "generate_content", "检测到内容生成需求", 0.7
    if any(kw in text for kw in ["日报", "工作日报", "每日汇报", "工作总结"]):
        if any(kw in text for kw in ["查看", "最近", "历史", "列表"]):
            return "list_reports", "关键词匹配", 0.7
        return "generate_report", "关键词匹配", 0.7
    if any(kw in text for kw in [
        "搜索网页", "搜索互联网", "网上搜索", "搜索最新", "搜索新闻", "网络搜索",
        "帮我搜索一下", "帮我搜索", "搜索一下最新", "网上查一下", "网上搜一下",
        "天气", "气温", "下雨", "下雪", "台风", "天气预报",
        "新闻", "热搜", "头条", "最新消息",
        "价格", "多少钱", "汇率", "股价", "股市",
        "最新的", "现在的", "今天的", "最近的",
        "比赛", "赛事", "直播", "上映",
        "推荐一个", "哪个好", "评分", "附近",
    ]):
        return "web_search", "关键词匹配", 0.7
    if any(kw in text for kw in ["知识库", "知识", "本地文档", "文档中查找", "搜索", "查一下", "查找", "搜一下"]):
        return "knowledge_search", "关键词匹配", 0.7
    return "chat", "默认对话", 0.5
