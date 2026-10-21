import json
from unittest.mock import Mock, patch
from tests.support import OfflineTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from src.agent import graph
from src.agent.context import latest_user_input, is_complex_task
from src.agent.sub_agents.supervisor import supervisor_agent, _rule_based_classify
from src.agent.sub_agents.message_agent import message_agent
from src.tools import message as message_tools
from src.tools import report as report_tools
from src.tools.prompts import INTENT_CLASSIFIER_TEMPLATE


def call(name, identifier, **args):
    return {"name": name, "id": identifier, "args": args}


def classification(intent):
    return AIMessage(content=json.dumps({"intent": intent, "confidence": 0.9}))


@tool
def local_echo(value: str) -> str:
    """Return a local result for offline graph tests."""
    return json.dumps({"success": True, "summary": value})


class RoutingTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(graph.memory_manager, "_enabled", False))

    def test_classifier_prompt_formats_without_json_key_error(self):
        messages = INTENT_CLASSIFIER_TEMPLATE.format_messages(user_input="生成日报", current_date="2026-09-15")
        self.assertIn('"intent"', messages[0].content)

    def test_supervisor_reads_human_before_tool_results(self):
        model = self.model(classification("generate_report"))
        state = {"messages": [HumanMessage(content="生成日报"),
                              AIMessage(content="", tool_calls=[call("delegate_to_supervisor", "d")]),
                              ToolMessage(content="delegated", tool_call_id="d")],
                 "delegated_task": "补充说明"}
        self.assertEqual(supervisor_agent(state)["intent"], "generate_report")
        self.assertEqual(latest_user_input(state), "生成日报")
        self.assertIn("补充说明", model.prompts[0][-1].content)
        self.assertEqual(len(model.prompts), 1)

    def test_specific_rules_and_negative_delivery(self):
        for text, expected in [("生成日报，完成了开发工作，不发送", "generate_report"),
                               ("添加到知识库：报告.md", "add_knowledge"),
                               ("查看最近的日报", "list_reports"),
                               ("帮我写一份通知", "generate_content"),
                               ("在知识库搜索如何生成日报", "knowledge_search")]:
            with self.subTest(text=text):
                self.assertEqual(_rule_based_classify(text)[0], expected)
        self.assertFalse(is_complex_task("生成日报，完成了开发工作"))
        self.assertFalse(is_complex_task("写通知，不要发送给QQ12345"))
        self.assertTrue(is_complex_task("写通知，然后发送给QQ12345"))

    def test_invalid_model_intent_uses_business_fallback(self):
        self.model(classification("unknown_agent"))
        result = supervisor_agent({"messages": [HumanMessage(content="添加到知识库：说明.md")]})
        self.assertEqual(result["intent"], "add_knowledge")

    def test_delegation_is_detected_in_any_result_position(self):
        requests = [call("delegate_to_supervisor", "d"), call("local_echo", "e", value="result")]
        results = [ToolMessage(content="delegated", tool_call_id="d"), ToolMessage(content="ok", tool_call_id="e")]
        for batch in (results, results[::-1]):
            state = {"messages": [HumanMessage(content="处理请求"), AIMessage(content="", tool_calls=requests), *batch]}
            self.assertEqual(graph.route_after_tool(state), "supervisor_agent")
        state["messages"].append(HumanMessage(content="新的请求"))
        self.assertEqual(graph.route_after_tool(state), "router_agent")

    def test_mixed_delegation_does_not_execute_premature_send(self):
        model = self.model(
            AIMessage(content="", tool_calls=[call("delegate_to_supervisor", "d", task_description="起草内容"),
                                               call("send_qq_message", "s", user_id="12345", message="premature")]),
            classification("generate_content"), AIMessage(content="生成的正文"),
        )
        with patch.object(graph._tool_node, "invoke") as ordinary_tools:
            result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="请起草会议通知")]})
        ordinary_tools.assert_not_called()
        self.assertIn("生成的正文", result["final_output"])
        self.assertEqual(result["tool_rounds"], 1)
        self.assertEqual(result["delegated_task"], "起草内容")
        outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual({m.tool_call_id for m in outputs}, {"d", "s"})
        self.assertFalse(model.responses)

    def test_content_then_send_executes_once(self):
        model = self.model(classification("generate_content"), AIMessage(content="完整会议通知"))
        with patch.object(message_tools.send_qq_message, "func", return_value='{"success":true}') as send:
            result = graph.multi_agent_app.invoke({
                "messages": [HumanMessage(content="写通知，然后发送给QQ12345")],
                "platform": "qq", "receiver": "12345",
            })
        send.assert_called_once_with(user_id="12345", message="完整会议通知", group_id=None)
        self.assertEqual(result["agent_history"], ["ContentAgent", "MessageAgent"])
        self.assertEqual(result["task_status"], "success")
        self.assertIsNone(result["next_agent"])
        self.assertFalse(model.responses)

    def test_report_then_group_send_preserves_complete_artifact(self):
        path = self.temporary_directory() / "report.md"
        original = report_tools.generate_daily_report.func
        def generate(**args):
            return original(**{**args, "output_path": str(path)})
        self.model(classification("generate_content"))  # business rule corrects to report
        with patch.object(report_tools.generate_daily_report, "func", side_effect=generate), \
             patch.object(message_tools.send_qq_message, "func", return_value='{"success":true}') as send, \
             patch.object(message_tools.get_qq_number_by_nickname, "func") as lookup:
            result = graph.multi_agent_app.invoke({
                "messages": [HumanMessage(content="生成日报，然后发送到QQ群123456")],
                "platform": "qq", "group_id": "123456", "report_name": "测试",
                "report_today_work": "完成开发工作。" * 100,
            })
        self.assertEqual(result["task_status"], "success", result["final_output"])
        sent = send.call_args.kwargs
        self.assertEqual(sent["message"], path.read_text(encoding="utf-8"))
        self.assertGreater(len(sent["message"]), 200)
        self.assertEqual(sent["group_id"], "123456")
        self.assertEqual(send.call_count, 1)
        lookup.assert_not_called()
        self.assertEqual(result["agent_history"], ["ReportAgent", "MessageAgent"])
        self.assertIsNone(result["next_agent"])

    def test_send_failure_keeps_generated_text_and_stops(self):
        self.model(classification("generate_content"), AIMessage(content="应保留的正文"))
        with patch.object(message_tools.send_qq_message, "func", return_value='{"success":false,"error":"offline"}') as send:
            result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="写通知并发给QQ12345")],
                                                  "platform": "qq", "receiver": "12345"})
        self.assertEqual(send.call_count, 1)
        self.assertEqual(result["task_status"], "failed")
        self.assertIn("应保留的正文", result["final_output"])
        self.assertIn("发送失败", result["final_output"])
        self.assertIsNone(result["next_agent"])

    def test_missing_target_clarifies_without_losing_content(self):
        self.model(classification("generate_content"), AIMessage(content="已生成的正文"),
                   AIMessage(content='{}'), AIMessage(content='{}'))
        with patch.object(message_tools.send_qq_message, "func") as send:
            result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="写通知然后发送")]})
        send.assert_not_called()
        self.assertEqual(result["task_status"], "pending")
        self.assertIn("已生成的正文", result["final_output"])
        self.assertIn("发送平台", result["final_output"])
        self.assertIsNone(result["next_agent"])

    def test_no_send_instruction_overrides_model_classification(self):
        path = self.temporary_directory() / "local-report.md"
        original = report_tools.generate_daily_report.func
        self.model(classification("send_message"))
        with patch.object(report_tools.generate_daily_report, "func", side_effect=lambda **args: original(**{**args, "output_path": str(path)})), \
             patch.object(message_tools.send_qq_message, "func") as send:
            result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="生成日报，完成了开发工作，不发送")],
                                                  "report_name": "测试", "report_today_work": "开发工作"})
        send.assert_not_called()
        self.assertEqual(result["task_status"], "success", result["final_output"])
        self.assertTrue(path.exists())
        self.assertEqual(result["agent_history"], ["ReportAgent"])

    def test_group_tool_accepts_no_private_receiver(self):
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ok", "retcode": 0}
        with patch("requests.post", return_value=response) as post:
            result = json.loads(message_tools.send_qq_message.invoke({"group_id": "123456", "message": "通知"}))
        self.assertTrue(result["success"])
        self.assertEqual(post.call_args.kwargs["json"]["group_id"], 123456)
        self.assertNotIn("user_id", post.call_args.kwargs["json"])

    def test_new_turn_resets_batch_count_and_ignores_old_delegation(self):
        self.enterContext(patch.object(graph, "_tool_node", ToolNode([local_echo])))
        self.model(AIMessage(content="", tool_calls=[call("local_echo", "e1", value="first")]), AIMessage(content="第一轮"),
                   AIMessage(content="", tool_calls=[call("local_echo", "e2", value="second")]), AIMessage(content="第二轮"))
        first = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="第一次查询")]})
        second = graph.multi_agent_app.invoke({**first, "messages": [*first["messages"], HumanMessage(content="第二次查询")],
                                               "tool_rounds": 99, "delegated_task": "旧任务"})
        self.assertEqual(first["tool_rounds"], 1)
        self.assertEqual(second["tool_rounds"], 1)
        self.assertIsNone(second["delegated_task"])
        self.assertEqual(second["final_output"], "第二轮")

    def test_batch_limit_counts_rounds_and_preserves_current_results(self):
        self.enterContext(patch.object(graph, "_tool_node", ToolNode([local_echo])))
        self.model(*(AIMessage(content="", tool_calls=[call("local_echo", f"e{i}-{j}", value=f"本轮结果{i}-{j}")
                                                       for j in range(2)]) for i in range(3)))
        result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="旧问题"), AIMessage(content="旧回答"),
                                                            HumanMessage(content="新问题")]})
        self.assertEqual(result["tool_rounds"], 3)
        self.assertIn("本轮结果2-1", result["final_output"])
        self.assertNotIn("旧回答", result["final_output"])

    def test_model_error_after_tools_does_not_redispatch(self):
        self.enterContext(patch.object(graph, "_tool_node", ToolNode([local_echo])))
        self.model(AIMessage(content="", tool_calls=[call("local_echo", "e", value="已完成的结果")]), RuntimeError("model down"))
        result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="处理请求")]})
        self.assertEqual(result["tool_rounds"], 1)
        self.assertEqual(result["agent_history"], [])
        self.assertIn("已完成的结果", result["final_output"])
        self.assertEqual(result["task_status"], "failed")
