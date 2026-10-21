"""
Hybrid Agent Architecture: Tool Calling + Supervisor Orchestration

Simple tasks:  Router Agent (LLM + bind_tools) -> ToolNode -> Output
Complex tasks: Router Agent -> Supervisor -> Sub-agents -> Output

Router Agent uses Function Calling to directly select and execute tools.
Supervisor handles complex multi-step tasks that require orchestration
(e.g., generate content then send via message).
"""

import json
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from src.agent.state import AgentState
from src.agent.context import (
    latest_user_input, current_turn_messages, latest_tool_batch,
    is_complex_task, delivery_forbidden,
)
from src.tools.tool_registry import ROUTER_TOOLS
from src.agent.sub_agents.supervisor import supervisor_agent
from src.agent.sub_agents.message_agent import message_agent
from src.agent.sub_agents.report_agent import report_agent
from src.agent.sub_agents.search_agent import search_agent
from src.agent.sub_agents.content_agent import content_agent
from src.memory.manager import memory_manager
from src.utils.logger import logger
from src.utils.execution import is_timeout
from src.api.context import conversation_view, ContextLimitError, ModelOutputLimitError, is_context_overflow
from src.agent.conflicts import action_context, check_batch, conflict_result, describe_action, prepare_task, ACTION_TOOLS

# Maximum number of tool call loops before forcing output
MAX_TOOL_CALL_LOOPS = 3



# ============================================================
# Router Agent (Tool Calling entry point)
# ============================================================

ROUTER_SYSTEM_PROMPT = """You are an intelligent office assistant with access to tools.

Current date and time: {current_date}

Available tools:
- web_search: Search the internet for real-time information
- knowledge_base_search: Semantic search over local knowledge base documents (returns raw snippets)
- rag_search: Semantic search + LLM-generated answer with citations
- lookup_entity: Precisely look up a person/org's contact info (phone/email) from the knowledge base
- add_to_knowledge_base: Add documents to knowledge base
- send_qq_message: Send messages via QQ
- send_wechat_message: Send messages via WeChat
- get_qq_number_by_nickname: Look up QQ number by nickname
- generate_daily_report: Generate work daily reports
- list_recent_reports: List recent daily reports
- delegate_to_supervisor: Delegate complex multi-step tasks

=== MANDATORY TOOL CALL RULES ===

You MUST call web_search when the user message contains ANY of these keywords or topics:

WEATHER: weather, temperature, rain, snow, sunny, cloudy, typhoon,
  or Chinese: 天气, 气温, 下雨, 下雪, 晴天, 阴天, 台风, 热不热, 冷不冷

NEWS: news, latest, current events, headlines,
  or Chinese: 新闻, 最新, 热搜, 头条, 最新消息, 最新进展

PRICES: price, cost, exchange rate, stock,
  or Chinese: 价格, 多少钱, 汇率, 股价, 股市, 涨跌

SPORTS: scores, match results, game,
  or Chinese: 比赛, 赛事, 比分, 直播

REAL-TIME: requests for current external facts, not dates in local office tasks.
  Generating today's report or listing recent reports does not require web_search.

CRITICAL: Even if the message also contains a greeting (like "hello" or "hi"),
if it ALSO contains any of the above keywords, you MUST call web_search.
A greeting combined with a real-time query is NOT a pure chat message.

NEVER answer real-time queries from your training data. ALWAYS call web_search.

=== TOOL USAGE GUIDELINES ===

1. Single-step tasks: Call the appropriate tool directly.
2. Multi-step sequential tasks: Call tools one at a time. After seeing each tool result, decide the next step.
3. Complex tasks (write content THEN send it, generate report THEN notify): Use delegate_to_supervisor.
4. Pure greetings with no request ("hello", "hi"): Respond naturally WITHOUT tools.
5. General knowledge that doesn't change (math, history, how-to): Respond naturally WITHOUT tools.
6. Creative writing that stays local (write a poem, tell a joke): Respond naturally WITHOUT tools.

=== CONTACT / PRECISE FACT LOOKUP RULES ===

For questions about a specific person's or organization's contact info (phone, email,
etc.), you MUST call lookup_entity — never answer from knowledge_base_search snippets,
and never invent a number or guess.

- If lookup_entity returns exactly one match (status=unique), report it WITH its source.
- If it returns multiple matches (status=ambiguous), DO NOT pick one silently. Show the
  candidates and ask the user which one they mean.
- If it returns not_found, tell the user honestly and suggest checking the name.

=== CRITICAL: AFTER TOOL RESULTS ===

After receiving ANY tool result, you MUST:
1. Read the tool result carefully
2. Summarize the information for the user in a clear, helpful response
3. NEVER call the same tool again with a similar query
4. NEVER say "let me search again" or "let me try another query"
5. If the result has useful data, present it to the user
6. If the result shows an error or no data, tell the user honestly
7. If you already have search results with data, USE THEM - do not search again

You have a LIMITED number of tool rounds. Continue only when another step is required
to complete the user's request; otherwise summarize the results and finish.
If the user says not to send or only to generate content, never call a sending tool.

Always respond in the same language as the user's input."""


def router_agent(state: AgentState) -> dict:
    """Router Agent: LLM with bound tools handles tasks directly.

    For simple tasks, the LLM picks the right tool and calls it.
    For complex multi-step tasks, it calls delegate_to_supervisor.
    For chat, it responds without any tool call.
    """
    logger.info("[Router] Processing user input...")
    messages = state.get("messages", [])
    user_input = latest_user_input(state)
    if not state.get("tool_rounds") and (is_complex_task(user_input) or delivery_forbidden(user_input)):
        return {"supervisor_decision": "rules", "task_status": "running"}

    try:
        from src.api.llm import get_llm
        from datetime import datetime
        llm = get_llm()
        llm_with_tools = llm.bind_tools(ROUTER_TOOLS)

        # Inject current date/time into the system prompt
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        dated_prompt = ROUTER_SYSTEM_PROMPT.replace("{current_date}", current_date)

        full_messages = conversation_view(dated_prompt, state)
        response = llm_with_tools.invoke(full_messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info(f"[Router] Tool Calling: {tool_names}")
        else:
            logger.info("[Router] No tool call, responding directly (chat)")

        return {"messages": [response]}

    except Exception as e:
        if isinstance(e, (ContextLimitError, ModelOutputLimitError)) or is_context_overflow(e):
            return {"task_status": "context_error", "error": str(e)}
        if is_timeout(e):
            return {"task_status": "timeout", "error": "模型调用超时，已停止后续执行。"}
        logger.error(f"[Router] Error: {e}, falling back to Supervisor")
        return {"task_status": "error", "error": str(e)}


# ============================================================
# Routing Logic
# ============================================================

def _get_user_input(state: AgentState) -> str:
    """Extract the latest user input from messages."""
    return latest_user_input(state)


def _is_complex_task(user_input: str) -> bool:
    """Detect tasks that need multi-step orchestration.

    This is a safety net for cases where the LLM doesn't call delegate_to_supervisor
    but the task clearly requires multiple steps (e.g., write then send).
    """
    return is_complex_task(user_input)


def route_after_router(state: AgentState) -> str:
    """Decide next step after Router Agent."""
    if state.get("task_status") in ("timeout", "unknown", "conflict", "context_error"):
        return "output_result"
    if state.get("task_status") == "error":
        return "output_result" if state.get("tool_rounds") else "supervisor_agent"
    if state.get("supervisor_decision") == "rules":
        return "supervisor_agent"

    messages = state.get("messages", [])
    if not messages:
        return "output_result"

    last_msg = messages[-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_node"

    user_input = _get_user_input(state)
    if not state.get("tool_rounds") and _is_complex_task(user_input):
        logger.info("[Router] Complex task detected, routing to Supervisor")
        return "supervisor_agent"

    return "output_result"


def route_after_tool(state: AgentState) -> str:
    """Decide next step after tool execution."""
    if state.get("task_status") in ("timeout", "unknown", "conflict"):
        return "output_result"
    if any(call["name"] == "delegate_to_supervisor" and result.status != "error"
           for call, result in latest_tool_batch(state)):
        logger.info("[Tool] Delegation detected, routing to Supervisor")
        return "supervisor_agent"

    if state.get("tool_rounds", 0) >= MAX_TOOL_CALL_LOOPS:
        logger.warning(f"[Tool] Max tool calls ({MAX_TOOL_CALL_LOOPS}) reached, forcing output")
        return "output_result"

    return "router_agent"


_tool_node = ToolNode(ROUTER_TOOLS)


def execute_tools(state: AgentState, config: RunnableConfig) -> dict:
    with action_context(state, "Router") as context:
        updates = _execute_tools(state, config)
        updates["conflicts"] = [*(state.get("conflicts") or []), *context["records"]]
        return updates


def _execute_tools(state: AgentState, config: RunnableConfig) -> dict:
    """Execute one batch and record delegation explicitly in graph state.

    Delegation takes precedence over other calls in the same batch, so a send
    cannot race with the content generation it depends on. Every call still
    receives a matching ToolMessage to keep the conversation protocol valid.
    """
    messages = current_turn_messages(state)
    request = messages[-1]
    calls = request.tool_calls
    delegates = [call for call in calls if call["name"] == "delegate_to_supervisor"]
    updates = {"tool_rounds": (state.get("tool_rounds") or 0) + 1}
    if delegates:
        if len(delegates) != len(calls):
            conflict_result("routing", "委派主管优先，其余工具未执行。",
                            [call["name"] for call in calls], "delegate_to_supervisor", resolution="rules")
        task = delegates[0]["args"].get("task_description", "")
        updates["delegated_task"] = task if isinstance(task, str) else ""
        updates["messages"] = [ToolMessage(
            content=json.dumps({"status": "delegated", "description": updates["delegated_task"]}
                               if call["name"] == "delegate_to_supervisor" else
                               {"success": False, "error": "任务已转交主管，此工具未执行"}, ensure_ascii=False),
            tool_call_id=call["id"], name=call["name"],
            status="success" if call["name"] == "delegate_to_supervisor" else "error",
        ) for call in calls]
        return updates
    schemas = {tool.name: tool.args_schema for tool in ROUTER_TOOLS}
    checked = {}
    for call in calls:
        if call["name"] in ACTION_TOOLS:
            try:
                checked[call["id"]] = {**call, "args": schemas[call["name"]].model_validate(call["args"]).model_dump()}
            except (ValueError, TypeError):
                # ToolNode supplies the normal per-call validation error.
                continue
    blocked = check_batch(list(checked.values()))
    selected, omitted, aliases, unique = [], [], {}, {}
    for call in calls:
        if call["id"] in blocked:
            omitted.append(ToolMessage(content=json.dumps(blocked[call["id"]], ensure_ascii=False),
                                       tool_call_id=call["id"], name=call["name"], status="error"))
            continue
        if call["id"] in checked:
            try:
                fingerprint = describe_action(call["name"], checked[call["id"]]["args"])[1]
            except (ValueError, OSError):
                fingerprint = None
            if fingerprint and fingerprint in unique:
                aliases[call["id"]] = unique[fingerprint]
                continue
            if fingerprint:
                unique[fingerprint] = call["id"]
        selected.append(call)
    if selected:
        tool_state = {**state, "messages": [*state["messages"][:-1], request.model_copy(update={"tool_calls": selected})]}
        updates.update(_tool_node.invoke(tool_state, config=config))
    results = {msg.tool_call_id: msg for msg in [*updates.get("messages", []), *omitted]}
    for alias, original in aliases.items():
        results[alias] = results[original].model_copy(update={"tool_call_id": alias, "id": None})
    updates["messages"] = [results[call["id"]] for call in calls]
    outcomes = []
    for msg in updates.get("messages", []):
        try:
            result = json.loads(msg.content)
            if isinstance(result, dict):
                outcomes.append(result.get("status"))
        except (ValueError, TypeError):
            continue
    if "unknown" in outcomes:
        updates["task_status"] = "unknown"
    elif "timeout" in outcomes:
        updates["task_status"] = "timeout"
    elif "conflict" in outcomes:
        updates["task_status"] = "conflict"
    return updates


# ============================================================
# Supervisor Routing (for complex tasks)
# ============================================================

def _route_to_sub_agent(state: AgentState) -> str:
    """Supervisor routes to the appropriate sub-agent."""
    if state.get("task_status") == "context_error":
        return "output_result"
    intent = state.get("intent", "chat")
    logger.info(f"[Supervisor] Routing to: {intent}")

    route_map = {
        "generate_content": "generate_content_agent",
        "send_message": "send_message_agent",
        "generate_report": "generate_report_agent",
        "list_reports": "list_reports_agent",
        "knowledge_search": "knowledge_search_agent",
        "add_knowledge": "add_knowledge_agent",
        "web_search": "web_search_agent",
        "chat": "chat_agent",
    }
    return route_map.get(intent, "chat_agent")


# ============================================================
# Sub-agent Helpers
# ============================================================

def _record_agent(state: AgentState, agent_name: str) -> dict:
    history = [*(state.get("agent_history") or []), agent_name]
    return {"current_agent": agent_name, "agent_history": history}


def _wrap_sub_agent(agent_func, agent_name: str):
    def wrapped(state: AgentState) -> dict:
        logger.info(f"[{agent_name}] Executing task")
        record = _record_agent(state, agent_name)
        # Consume the previous node's continuation through an explicit update.
        with action_context(state, agent_name) as context:
            result = {"next_agent": None, **agent_func(state)}
            result["conflicts"] = [*(state.get("conflicts") or []), *context["records"]]
        result.update(record)
        return result
    return wrapped


CONTINUATIONS = {"send_message_agent": "send_message_agent"}


def _check_agent_result(state: AgentState) -> str:
    status = state.get("task_status", "")
    needs_clarify = state.get("needs_clarification", False)
    next_agent = state.get("next_agent")

    if needs_clarify or status == "pending":
        return "clarify_handler"

    if next_agent and status == "success":
        return CONTINUATIONS.get(next_agent, "output_result")

    return "output_result"


# ============================================================
# Common Nodes
# ============================================================

def clarify_handler(state: AgentState) -> dict:
    question = state.get("clarification_question", "Please provide more information")
    generated = state.get("generated_content")
    prefix = f"📝 内容已生成：\n\n{generated}\n\n" if generated else ""
    return {
        "final_output": prefix + f"⚠️ 信息不完整，需要您补充以下信息：\n\n{question}\n\n请重新输入完整指令。",
        "needs_clarification": False,
        "next_agent": None,
        "task_status": "pending",
    }


def chat_agent(state: AgentState) -> dict:
    """Chat handler for the Supervisor path (fallback)."""
    logger.info("[ChatAgent] Processing chat")
    messages = state.get("messages", [])

    try:
        from src.api.llm import get_llm
        from src.tools.prompts import CHAT_SYSTEM_TEMPLATE
        from datetime import datetime

        llm = get_llm()
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        system_content = CHAT_SYSTEM_TEMPLATE.format(current_date=current_date)

        response = llm.invoke(conversation_view(system_content, state))
        return {"final_output": response.content, "task_status": "success"}
    except Exception as e:
        logger.error(f"[ChatAgent] Error: {e}")
        return {"final_output": f"对话处理失败: {e}", "task_status": "timeout" if is_timeout(e) else "failed"}


def memory_load_node(state: AgentState) -> dict:
    """图入口：加载短期 + 长期记忆上下文"""
    session_id = state.get("session_id", "default")
    user_id = state.get("user_id", "default")

    # 从最后一条用户消息获取当前输入
    current_input = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            current_input = msg.content
            break

    ctx = memory_manager.load_context(user_id, session_id, current_input)
    return {
        **prepare_task(state),
        "tool_rounds": 0,
        "delegated_task": None,
        "supervisor_decision": None,
        "next_agent": None,
        "generated_content": None,
        "final_output": None,
        "error": None,
        "task_status": "pending",
        "needs_clarification": False,
        "agent_history": [],
        "conflicts": [],
        "memory_context": ctx.formatted_context,
        "memory_summarized": bool(ctx.conversation_summary),
        "memory_recent_turns": sum(turn.get("role") == "user" for turn in ctx.recent_turns),
        "memory_loaded": True,
        "session_id": session_id,
        "user_id": user_id,
    }


def memory_save_node(state: AgentState) -> dict:
    """图出口：保存交互到记忆"""
    session_id = state.get("session_id", "default")
    user_id = state.get("user_id", "default")
    intent = state.get("intent", "chat")

    user_input = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    final_output = state.get("final_output", "")

    memory_manager.save_interaction(
        user_id, session_id, user_input, final_output, intent
    )
    return {}


def output_result(state: AgentState) -> dict:
    """Extract and return the final output from whichever path produced it."""
    if state.get("next_agent"):
        return {"final_output": "无法继续执行指定的后续任务", "task_status": "failed", "next_agent": None}
    if state.get("final_output"):
        return {
            "final_output": state["final_output"],
            "error": state.get("error"),
            "task_status": state.get("task_status", "success"),
        }

    messages = current_turn_messages(state)

    # Find the last AIMessage that is NOT a tool call (i.e., a real response)
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            break
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return {"final_output": msg.content, "task_status": "success"}

    # At the round limit (or on a model failure), preserve this turn's actual
    # tool outputs. They may be reports, contacts or messages as well as search.
    rendered = []
    failed = bool(state.get("error"))
    outcome = state.get("task_status")
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            result = json.loads(msg.content)
        except (ValueError, TypeError):
            result = None
        if isinstance(result, dict):
            if result.get("status") == "unknown":
                outcome = "unknown"
            elif result.get("status") == "timeout" and outcome != "unknown":
                outcome = "timeout"
            elif result.get("status") == "conflict" and outcome not in ("unknown", "timeout"):
                outcome = "conflict"
            failed = failed or result.get("success") is False or result.get("status") == "error"
            body = result.get("answer") or result.get("message") or result.get("summary") or result.get("error")
            if not body:
                body = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            body = msg.content
        failed = failed or msg.status == "error"
        rendered.append(f"**{msg.name or '工具结果'}**\n{body}")
    if rendered:
        if state.get("error"):
            rendered.append(state["error"])
        return {"final_output": "\n\n".join(rendered),
                "task_status": outcome if outcome in ("timeout", "unknown", "conflict", "context_error") else ("failed" if failed else "success")}
    return {"final_output": state.get("error") or "未能获取到有效信息",
            "task_status": outcome if outcome in ("timeout", "unknown", "conflict", "context_error") else "failed"}


# ============================================================
# Graph Construction
# ============================================================

def build_hybrid_agent():
    """Build the hybrid agent graph.

    Entry: Router Agent (Tool Calling)
      - Simple tasks -> ToolNode -> Output (or loop back for multi-tool)
      - Complex tasks -> Supervisor -> Sub-agents -> Output
      - Chat -> Output directly
    """
    workflow = StateGraph(AgentState)

    # Router Agent + Tools (simple tasks)
    workflow.add_node("router_agent", router_agent)
    workflow.add_node("tool_node", execute_tools)

    # Supervisor + Sub-agents (complex tasks)
    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("generate_content_agent", _wrap_sub_agent(content_agent, "ContentAgent"))
    workflow.add_node("send_message_agent", _wrap_sub_agent(message_agent, "MessageAgent"))
    workflow.add_node("generate_report_agent", _wrap_sub_agent(report_agent, "ReportAgent"))
    workflow.add_node("list_reports_agent", _wrap_sub_agent(report_agent, "ReportAgent"))
    workflow.add_node("knowledge_search_agent", _wrap_sub_agent(search_agent, "SearchAgent"))
    workflow.add_node("add_knowledge_agent", _wrap_sub_agent(search_agent, "SearchAgent"))
    workflow.add_node("web_search_agent", _wrap_sub_agent(search_agent, "SearchAgent"))
    workflow.add_node("chat_agent", chat_agent)

    # Common nodes
    workflow.add_node("clarify_handler", clarify_handler)
    workflow.add_node("output_result", output_result)

    # Memory nodes
    workflow.add_node("memory_load", memory_load_node)
    workflow.add_node("memory_save", memory_save_node)

    # Entry point: 先加载记忆
    workflow.set_entry_point("memory_load")
    workflow.add_edge("memory_load", "router_agent")

    # Router Agent -> next step
    workflow.add_conditional_edges("router_agent", route_after_router, {
        "tool_node": "tool_node",
        "supervisor_agent": "supervisor_agent",
        "output_result": "output_result",
    })

    # Tool Node -> next step
    workflow.add_conditional_edges("tool_node", route_after_tool, {
        "router_agent": "router_agent",
        "supervisor_agent": "supervisor_agent",
        "output_result": "output_result",
    })

    # Supervisor -> sub-agent
    workflow.add_conditional_edges("supervisor_agent", _route_to_sub_agent, {
        "output_result": "output_result",
        "generate_content_agent": "generate_content_agent",
        "send_message_agent": "send_message_agent",
        "generate_report_agent": "generate_report_agent",
        "list_reports_agent": "list_reports_agent",
        "knowledge_search_agent": "knowledge_search_agent",
        "add_knowledge_agent": "add_knowledge_agent",
        "web_search_agent": "web_search_agent",
        "chat_agent": "chat_agent",
    })

    # Sub-agent result checking
    for agent in [
        "generate_content_agent",
        "send_message_agent",
        "generate_report_agent",
        "list_reports_agent",
        "knowledge_search_agent",
        "add_knowledge_agent",
        "web_search_agent",
        "chat_agent",
    ]:
        workflow.add_conditional_edges(agent, _check_agent_result, {
            **CONTINUATIONS,
            "clarify_handler": "clarify_handler",
            "output_result": "output_result",
        })

    # Exit through memory_save
    workflow.add_edge("clarify_handler", "memory_save")
    workflow.add_edge("output_result", "memory_save")
    workflow.add_edge("memory_save", END)

    return workflow.compile()


multi_agent_app = build_hybrid_agent()
