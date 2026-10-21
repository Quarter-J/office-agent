from copy import deepcopy
import json
from unittest.mock import Mock, patch

from tests.support import OfflineTestCase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from src.api.context import (ContextLimitError, ModelOutputLimitError, conversation_view,
                             fit_payload, input_cost, summary_chunks)
from src.api.llm import ChatDeepSeek
from src.config.settings import settings, Settings
from src.memory.manager import MemoryManager
from src.memory.short_term import ShortTermMemory
from src.agent import graph
from src.utils.execution import execution_context
from src.tools.prompts import RAG_QA_TEMPLATE, WEB_SEARCH_SUMMARY_TEMPLATE


class ContextRejection(Exception):
    status_code = 400
    code = "context_length_exceeded"


def raw_reply(content="完成", calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    response = Mock()
    response.headers = {}
    response.parse.return_value = {"id": "offline", "model": "deepseek-chat", "choices": [
        {"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    return response


def offline_model(*responses):
    client = Mock()
    client.with_raw_response.create.side_effect = responses
    model = ChatDeepSeek(api_key="offline", base_url="http://example.invalid", model="deepseek-chat",
                        client=client, async_client=Mock(), max_retries=0, use_responses_api=False)
    return model, client.with_raw_response.create


class PayloadBudgetTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.multiple(settings, llm_context_window_tokens=4096, llm_output_tokens=512,
                                         llm_context_safety_tokens=256, context_observation_tokens=1024))

    def test_old_turns_shrink_without_changing_latest_request_or_source(self):
        history = [{"role": "system", "content": "遵循用户明确约束"}]
        for i in range(12):
            history += [{"role": "user", "content": f"旧问题{i}" + "旧" * 300},
                        {"role": "assistant", "content": "旧回答" * 100}]
        history.append({"role": "user", "content": "不要发送，只查询状态"})
        payload = {"messages": history, "tools": [{"type": "function", "function": {"name": "query", "description": "查询" * 100}}]}
        original = deepcopy(payload)
        fitted, stats = fit_payload(payload)
        self.assertLessEqual(input_cost(fitted), 3328)
        self.assertGreater(stats["dropped_turns"], 0)
        self.assertEqual(fitted["messages"][-1], history[-1])
        self.assertEqual(fitted["tools"], payload["tools"])
        self.assertEqual(payload, original)

    def test_tool_receipt_compaction_keeps_status_ids_and_json(self):
        receipt = {"success": False, "status": "unknown", "retryable": False, "message_id": "receipt-123",
                   "error": "结果未确认", "results": [{"source": "guide.md", "rank": 1, "content": "资料" * 5000}]}
        call = {"id": "tool-1", "type": "function", "function": {"name": "query", "arguments": '{"query":"原参数"}'}}
        payload = {"messages": [{"role": "user", "content": "查询资料"},
                                 {"role": "assistant", "content": None, "tool_calls": [call], "reasoning_content": "opaque-reasoning"},
                                 {"role": "tool", "tool_call_id": "tool-1", "content": json.dumps(receipt, ensure_ascii=False)}]}
        fitted, _ = fit_payload(payload)
        result = json.loads(fitted["messages"][-1]["content"])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["retryable"])
        self.assertEqual(result["message_id"], "receipt-123")
        self.assertEqual(result["results"][0]["source"], "guide.md")
        self.assertEqual(fitted["messages"][1], payload["messages"][1])
        self.assertEqual(fitted["messages"][-1]["tool_call_id"], "tool-1")
        self.assertGreater(len(payload["messages"][-1]["content"]), len(fitted["messages"][-1]["content"]))

    def test_oversized_current_instruction_is_rejected_without_truncation(self):
        payload = {"messages": [{"role": "user", "content": "必须保留的正文" * 3000}]}
        with self.assertRaises(ContextLimitError):
            fit_payload(payload)
        self.assertEqual(len(payload["messages"][0]["content"]), len("必须保留的正文") * 3000)

    def test_oversized_tool_schema_or_execution_arguments_cannot_be_trimmed(self):
        with self.assertRaises(ContextLimitError):
            fit_payload({"messages": [{"role": "user", "content": "查询"}], "tools": [{"description": "x" * 10000}]})
        call = {"id": "s", "function": {"name": "send", "arguments": json.dumps({"message": "x" * 10000})}}
        with self.assertRaises(ContextLimitError):
            fit_payload({"messages": [{"role": "user", "content": "发送"},
                                      {"role": "assistant", "tool_calls": [call]},
                                      {"role": "tool", "tool_call_id": "s", "content": '{"success":true}'}]})

    def test_current_orphan_tools_are_rejected_and_old_orphans_removed_by_turn(self):
        orphan = {"role": "tool", "tool_call_id": "missing", "content": "回执"}
        with self.assertRaises(ContextLimitError):
            fit_payload({"messages": [{"role": "user", "content": "当前问题"}, orphan]})
        fitted, _ = fit_payload({"messages": [{"role": "user", "content": "旧问题"}, orphan,
                                               {"role": "user", "content": "当前问题"}]})
        self.assertEqual(fitted["messages"], [{"role": "user", "content": "当前问题"}])

    def test_valid_tool_groups_are_removed_as_whole_historical_turns(self):
        call = {"id": "old-call", "function": {"name": "query", "arguments": "{}"}}
        messages = [{"role": "user", "content": "旧" * 3000},
                    {"role": "assistant", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "old-call", "content": "成功"},
                    {"role": "user", "content": "新问题"}]
        fitted, _ = fit_payload({"messages": messages})
        self.assertEqual(fitted["messages"], [messages[-1]])

    def test_intermediate_system_constraint_survives_history_removal(self):
        system = {"role": "system", "content": "禁止自动发送"}
        fitted, _ = fit_payload({"messages": [{"role": "user", "content": "旧" * 3000}, system,
                                               {"role": "user", "content": "当前请求"}]})
        self.assertIn(system, fitted["messages"])

    def test_stored_summary_only_replaces_already_covered_turns(self):
        messages = []
        for i in range(10):
            messages.extend([HumanMessage(content=f"问题{i}"), AIMessage(content=f"回答{i}")])
        messages.append(HumanMessage(content="当前请求"))
        state = {"messages": messages, "memory_context": "已摘要的信息", "memory_summarized": True, "memory_recent_turns": 5}
        view = conversation_view("系统约束", state)
        users = [message.content for message in view if isinstance(message, HumanMessage)]
        self.assertEqual(users, [f"问题{i}" for i in range(5, 10)] + ["当前请求"])
        self.assertEqual(len(state["messages"]), 21)
        self.assertEqual(view[1].name, "context_memory")

    def test_settings_reject_impossible_budget(self):
        with self.assertRaises(ValueError):
            Settings(_env_file=None, llm_context_window_tokens=1024, llm_output_tokens=1024)


class ModelContextTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(graph.memory_manager, "_enabled", False))

    def test_actual_sdk_payload_honors_bound_tools_and_output_reserve(self):
        model, transport = offline_model(raw_reply())
        tool = {"type": "function", "function": {"name": "query", "description": "查询", "parameters": {"type": "object", "properties": {}}}}
        model.bind_tools([tool]).invoke([HumanMessage(content="查询资料")])
        payload = transport.call_args.kwargs
        self.assertEqual(payload["tools"][0]["function"]["name"], "query")
        self.assertEqual(payload["max_tokens"], settings.llm_output_tokens)
        self.assertNotIn("_context_scale", payload)
        self.assertLessEqual(input_cost(payload), settings.llm_context_window_tokens - settings.llm_output_tokens - settings.llm_context_safety_tokens)

    def test_reference_templates_can_shrink_without_losing_question(self):
        model, transport = offline_model(raw_reply(), raw_reply())
        for template, arguments in ((RAG_QA_TEMPLATE, {"context": "【来源：guide.md】\n" + "资料" * 30000}),
                                    (WEB_SEARCH_SUMMARY_TEMPLATE, {"search_results": "来源 https://example.invalid\n" + "资料" * 30000})):
            model.invoke(template.format_messages(query="最新问题", **arguments))
            payload = transport.call_args.kwargs
            self.assertEqual(payload["messages"][-1]["content"], "最新问题")
            self.assertEqual(payload["messages"][1]["name"], "context_reference")
            self.assertIn("来源", payload["messages"][1]["content"])
            self.assertLess(len(payload["messages"][1]["content"]), 3000)

    def test_provider_overflow_retries_once_with_smaller_input(self):
        self.enterContext(patch.multiple(settings, llm_context_window_tokens=8000, llm_output_tokens=512, llm_context_safety_tokens=256))
        model, transport = offline_model(ContextRejection("maximum context length"), raw_reply())
        messages = [SystemMessage(content="约束")]
        for _ in range(12):
            messages.extend([HumanMessage(content="历史" * 100), AIMessage(content="回答" * 100)])
        messages.append(HumanMessage(content="当前问题"))
        self.assertEqual(model.invoke(messages).content, "完成")
        first, second = [call.kwargs for call in transport.call_args_list]
        self.assertLess(input_cost(second), input_cost(first))
        self.assertLessEqual(second["timeout"], first["timeout"])

    def test_repeated_overflow_and_non_context_errors_have_bounded_attempts(self):
        model, transport = offline_model(ContextRejection("overflow"), ContextRejection("overflow"), raw_reply())
        with self.assertRaises(ContextRejection):
            model.invoke([HumanMessage(content="请求")])
        self.assertEqual(transport.call_count, 2)
        model, transport = offline_model(RuntimeError("auth unavailable"), raw_reply())
        with self.assertRaises(RuntimeError):
            model.invoke([HumanMessage(content="请求")])
        self.assertEqual(transport.call_count, 1)

    def test_retry_cannot_exceed_original_time_budget(self):
        model, transport = offline_model()
        clock = [100.0]
        def reject(**kwargs):
            clock[0] = 102.0
            raise ContextRejection("too many tokens")
        transport.side_effect = reject
        with patch("src.utils.execution.time.monotonic", side_effect=lambda: clock[0]), execution_context(101):
            with self.assertRaises(TimeoutError):
                model.invoke([HumanMessage(content="请求")])
        self.assertEqual(transport.call_count, 1)

    def test_current_reasoning_and_tool_arguments_survive_serialization(self):
        model, transport = offline_model(raw_reply())
        request = AIMessage(content="", additional_kwargs={"reasoning_content": "opaque-execution-context"},
                            tool_calls=[{"id": "q", "name": "query", "args": {"query": "原始参数"}}])
        model.invoke([HumanMessage(content="问题"), request, ToolMessage(tool_call_id="q", content='{"success":true}')])
        sent = transport.call_args.kwargs["messages"][1]
        self.assertEqual(sent["reasoning_content"], "opaque-execution-context")
        self.assertEqual(json.loads(sent["tool_calls"][0]["function"]["arguments"]), {"query": "原始参数"})

    def test_output_length_limit_is_reported_as_incomplete(self):
        model, transport = offline_model(raw_reply("被截断的正文", finish="length"))
        with self.assertRaises(ModelOutputLimitError):
            model.invoke([HumanMessage(content="写内容")])
        self.assertEqual(transport.call_count, 1)

    def test_router_context_failure_does_not_redispatch(self):
        scripted = self.model(ContextLimitError("需要分段输入"))
        result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="查询资料")]})
        self.assertEqual(len(scripted.prompts), 1)
        self.assertIn("需要分段输入", result["final_output"])
        self.assertEqual(result["task_status"], "context_error")
        self.assertEqual(result["agent_history"], [])

    def test_supervisor_context_failure_stops_before_generation_or_send(self):
        scripted = self.model(ContextLimitError("当前输入过长"))
        with patch("requests.post") as send:
            result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="写通知然后发给QQ12345")]})
        self.assertEqual(len(scripted.prompts), 1)
        self.assertIn("当前输入过长", result["final_output"])
        self.assertEqual(result["task_status"], "context_error")
        send.assert_not_called()

    def test_model_only_retry_after_send_does_not_repeat_side_effect(self):
        self.enterContext(patch.multiple(settings, llm_context_window_tokens=65536, llm_output_tokens=1024, llm_context_safety_tokens=512))
        call = {"id": "send", "type": "function", "function": {"name": "send_qq_message", "arguments": '{"user_id":"12345","message":"正文"}'}}
        model, transport = offline_model(raw_reply("", calls=[call], finish="tool_calls"), ContextRejection("too many tokens"), raw_reply("已发送"))
        self.enterContext(patch("src.api.llm.get_llm", return_value=model))
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ok", "retcode": 0}
        with patch("requests.post", return_value=response) as send:
            result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="给QQ12345发消息：正文")]})
        self.assertEqual(result["task_status"], "success", result["final_output"])
        self.assertEqual(send.call_count, 1)
        self.assertEqual(transport.call_count, 3)


class RollingMemoryTests(OfflineTestCase):
    def manager(self):
        manager = MemoryManager()
        manager.short_term = ShortTermMemory(redis_url="", max_turns=20)
        return manager

    def add_pairs(self, memory, start, end, content="内容"):
        for number in range(start, end):
            memory.add_turn("s", "user", f"问题{number}:{content}")
            memory.add_turn("s", "assistant", f"回答{number}")

    def test_memory_fallback_keeps_recent_messages_when_limit_is_exceeded(self):
        memory = ShortTermMemory(redis_url="", max_turns=4)
        self.add_pairs(memory, 0, 4)
        self.assertEqual([item["content"] for item in memory.get_conversation("s")],
                         ["问题2:内容", "回答2", "问题3:内容", "回答3"])

    def test_rolling_summary_merges_and_removes_only_covered_messages(self):
        manager = self.manager()
        model = self.model(AIMessage(content="第一份摘要"), AIMessage(content="合并后的新摘要"))
        self.add_pairs(manager.short_term, 0, 5)
        manager._compress_conversation("s")
        self.assertEqual(manager.short_term.get_turn_count("s"), 6)
        self.assertEqual(manager.short_term.get_conversation("s")[0]["content"], "问题2:内容")
        self.add_pairs(manager.short_term, 5, 7)
        manager._compress_conversation("s")
        self.assertEqual(manager.short_term.get_summary("s"), "合并后的新摘要")
        self.assertEqual(manager.short_term.get_turn_count("s"), 6)
        prompt = "\n".join(message.content for message in model.prompts[1])
        self.assertIn("第一份摘要", prompt)
        self.assertNotIn("问题0", prompt)
        self.assertIn("问题2", prompt)

    def test_summary_failure_keeps_old_summary_and_raw_history(self):
        manager = self.manager()
        self.add_pairs(manager.short_term, 0, 5)
        manager.short_term.update_summary("s", "旧摘要")
        before = manager.short_term.get_conversation("s")
        self.model(RuntimeError("model unavailable"))
        manager._compress_conversation("s")
        self.assertEqual(manager.short_term.get_conversation("s"), before)
        self.assertEqual(manager.short_term.get_summary("s"), "旧摘要")

    def test_summary_length_is_enforced_and_no_repeat_below_threshold(self):
        manager = self.manager()
        self.add_pairs(manager.short_term, 0, 5)
        model = self.model(AIMessage(content="新" * 3000))
        manager._compress_conversation("s")
        manager._compress_conversation("s")
        self.assertEqual(len(manager.short_term.get_summary("s")), settings.memory_summary_max_chars)
        self.assertEqual(len(model.prompts), 1)

    def test_atomic_summary_commit_preserves_concurrent_appends(self):
        memory = self.manager().short_term
        self.add_pairs(memory, 0, 5)
        snapshot = memory.get_conversation("s")
        self.add_pairs(memory, 5, 6)
        self.assertTrue(memory.commit_summary("s", snapshot, "", "摘要", 4))
        self.assertEqual(memory.get_turn_count("s"), 8)
        self.assertEqual(memory.get_conversation("s")[-1]["content"], "回答5")

    def test_stale_or_expired_summary_snapshot_cannot_delete_new_history(self):
        memory = self.manager().short_term
        self.add_pairs(memory, 0, 5)
        snapshot = memory.get_conversation("s")
        memory.update_summary("s", "另一个请求已更新")
        self.assertFalse(memory.commit_summary("s", snapshot, "", "过期摘要", 4))
        memory.clear("s")
        self.add_pairs(memory, 10, 15)
        self.assertFalse(memory.commit_summary("s", snapshot, "", "过期摘要", 4))
        self.assertEqual(memory.get_conversation("s")[0]["content"], "问题10:内容")

    def test_chunking_preserves_unicode_and_summarizer_failure_is_atomic(self):
        text = "长文档🙂" * 3000
        self.assertEqual("".join(summary_chunks(text, 512)), text)
        manager = self.manager()
        self.add_pairs(manager.short_term, 0, 5, content=text)
        before = manager.short_term.get_conversation("s")
        model = self.model(AIMessage(content="第一段摘要"), RuntimeError("second chunk failed"))
        manager._compress_conversation("s")
        self.assertEqual(len(model.prompts), 2)
        self.assertEqual(manager.short_term.get_conversation("s"), before)
        self.assertEqual(manager.short_term.get_summary("s"), "")
