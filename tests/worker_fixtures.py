"""Spawn-safe local tasks for process supervision tests; no external services."""

import os
from pathlib import Path
import time
from src.utils.execution import operation, publish, timed_tool

_calls = 0


@timed_tool(effect="send")
def send_qq_message():
    return '{"success": true, "status": "success"}'


def worker_task(state):
    global _calls
    _calls += 1
    action = state.get("action", "echo")
    result = {**state, "worker_pid": os.getpid(), "calls": _calls,
              "task_status": "success", "final_output": "完成"}
    if action == "hung_send":
        publish({"type": "state", "state": {**result, "generated_content": "已经生成的完整正文",
                                               "report_file_path": "saved-report.md"}})
        with operation("send_wechat_message", timeout=0.08, effect="send"):
            time.sleep(1)
            Path(state["late_marker"]).write_text("should never happen", encoding="utf-8")
    elif action == "hung_task":
        publish({"type": "state", "state": {**result, "generated_content": "此前已生成的内容"}})
        time.sleep(2)
    elif action == "confirmed_then_hang":
        send_qq_message()
        time.sleep(2)
    elif action == "pause":
        time.sleep(0.25)
    elif action == "crash":
        os._exit(3)
    elif action == "late_done":
        time.sleep(0.5)
    return result


def graph_worker_task(state):
    """Exercise real LangGraph worker threads without any external connection."""
    import logging
    logging.getLogger("office_agent").addHandler(logging.NullHandler())
    from unittest.mock import patch
    from langchain_core.messages import AIMessage, HumanMessage
    from src.agent.runner import _run_graph
    from src.agent.graph import memory_manager
    from src.api import llm
    from src.config.settings import settings
    from src.tools import message

    if state.get("action") == "warm":
        return {"task_status": "success", "worker_pid": os.getpid()}

    class LocalModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content='{"intent":"generate_content"}' if self.calls == 1 else "真实工作流生成的正文")

    @timed_tool(effect="send")
    def send_wechat_message(who: str, message: str):
        time.sleep(2)
        return '{"success": true}'

    with patch.object(llm, "get_llm", return_value=LocalModel()), \
         patch.object(memory_manager, "_enabled", False), \
         patch.object(settings, "tool_timeout_seconds", 0.08), \
         patch.object(message.send_wechat_message, "func", send_wechat_message), \
         patch("socket.socket.connect", side_effect=AssertionError("offline integration test")):
        return _run_graph({**state, "messages": [HumanMessage(content="写通知然后发送给微信张三")],
                           "platform": "wechat", "receiver": "张三"})


def ledger_worker_task(state):
    import json
    from unittest.mock import patch
    from src.agent.conflicts import action_context
    from src.config.settings import settings
    if state.get("action") == "warm":
        return {**state, "task_status": "success"}

    @timed_tool(effect="send")
    def send_wechat_message(who: str, message: str):
        time.sleep(state.get("delay", 0))
        with Path(state["marker"]).open("a", encoding="utf-8") as stream:
            stream.write("sent\n")
        return '{"success": true, "status": "success"}'

    with patch.object(settings, "tool_timeout_seconds", 0.2), action_context(state, "MessageAgent"):
        receipt = json.loads(send_wechat_message("张三", "正文"))
    return {**state, "task_status": receipt.get("status", "success"), "receipt": receipt}
