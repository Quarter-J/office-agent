import json
import threading
import time
from unittest.mock import Mock, patch
import requests
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from tests.support import OfflineTestCase
from tests.worker_fixtures import worker_task, graph_worker_task
from src.agent import graph
from src.agent.runner import AgentRunner
from src.api.llm import llm
from src.config.settings import settings
from src.tools import message, search
from src.rag import vector_store
from src.utils.execution import execution_context, read_request, operation, remaining_seconds


class TransportTimeoutTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(graph.memory_manager, "_enabled", False))

    def test_qq_read_timeout_is_unknown_and_does_not_try_another_endpoint(self):
        with patch("requests.post", side_effect=requests.ReadTimeout("response delayed")) as post:
            result = json.loads(message.send_qq_message.invoke({"user_id": "12345", "message": "正文"}))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["retryable"])
        self.assertIn("未自动重复发送", result["error"])

    def test_qq_connect_timeout_is_not_sent(self):
        with patch("requests.post", side_effect=requests.ConnectTimeout("connection delayed")) as post:
            result = json.loads(message.send_qq_message.invoke({"group_id": "123456", "message": "正文"}))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["delivery_status"], "not_sent")

    def test_qq_ambiguous_responses_never_retry(self):
        invalid_json = Mock(status_code=200)
        invalid_json.json.side_effect = ValueError("invalid json")
        for response in (Mock(status_code=500), invalid_json):
            with self.subTest(status=response.status_code), patch("requests.post", return_value=response) as post:
                result = json.loads(message.send_qq_message.invoke({"user_id": "12345", "message": "正文"}))
                self.assertEqual(result["status"], "unknown")
                self.assertEqual(post.call_count, 1)

    def test_qq_only_switches_endpoint_for_explicit_unsupported_response(self):
        success = Mock(status_code=200)
        success.json.return_value = {"status": "ok", "retcode": 0}
        with patch("requests.post", side_effect=[Mock(status_code=404), success]) as post:
            result = json.loads(message.send_qq_message.invoke({"user_id": "12345", "message": "正文"}))
        self.assertTrue(result["success"])
        self.assertEqual(post.call_count, 2)
        self.assertTrue(post.call_args_list[0].args[0].endswith("send_private_msg"))
        self.assertTrue(post.call_args_list[1].args[0].endswith("send_msg"))

    def test_expired_budget_never_starts_send(self):
        with execution_context(time.monotonic() - 1), patch("requests.post") as post:
            with self.assertRaises(TimeoutError):
                message.send_qq_message.invoke({"user_id": "12345", "message": "正文"})
        post.assert_not_called()

    def test_read_retry_shares_budget_and_has_a_limit(self):
        request = Mock(side_effect=requests.ReadTimeout("slow read"))
        with patch.object(settings, "readonly_timeout_retries", 1):
            with self.assertRaises(requests.ReadTimeout):
                read_request(request, "https://example.invalid")
        self.assertEqual(request.call_count, 2)
        clock = [100.0]
        def exceed_budget(*args, **kwargs):
            clock[0] = 102
            raise requests.ReadTimeout("slow")
        request = Mock(side_effect=exceed_budget)
        with patch("src.utils.execution.time.monotonic", side_effect=lambda: clock[0]), execution_context(101):
            with self.assertRaises(TimeoutError):
                read_request(request, "https://example.invalid")
        self.assertEqual(request.call_count, 1)

    def test_expired_monitor_acknowledgement_does_not_start_operation(self):
        clock = [100.0]
        events = []
        executed = Mock()
        def slow_monitor(event):
            events.append(event["type"])
            if event["type"] == "start":
                clock[0] = 102
        with patch("src.utils.execution.time.monotonic", side_effect=lambda: clock[0]), execution_context(110, slow_monitor):
            with self.assertRaises(TimeoutError):
                with operation("send", 1, "send"):
                    executed()
            self.assertEqual(remaining_seconds(10), 8)
        executed.assert_not_called()
        self.assertEqual(events, ["start", "end"])

    def test_web_search_returns_timeout_after_bounded_retry(self):
        with patch("requests.Session.get", side_effect=requests.ReadTimeout("slow")) as get:
            result = json.loads(search.web_search.invoke({"query": "测试"}))
        self.assertEqual(get.call_count, 2)
        self.assertEqual(result["status"], "timeout")

    def test_embedding_timeout_falls_back_to_keyword_search(self):
        directory = self.temporary_directory()
        (directory / "guide.md").write_text("Spring Boot 自动装配使用配置与条件注解，帮助开发者管理组件。", encoding="utf-8")
        with patch.object(settings, "knowledge_base_dir", str(directory)), \
             patch.object(vector_store, "load_index", side_effect=TimeoutError("embedding timeout")):
            result = json.loads(search.knowledge_base_search.invoke({"query": "Spring Boot"}))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["method"], "keyword")

    def test_model_request_timeout_respects_remaining_task_budget(self):
        with execution_context(time.monotonic() + 0.5), \
             patch.object(ChatOpenAI, "invoke", return_value=AIMessage(content="answer")) as invoke:
            llm.invoke([HumanMessage(content="question")])
        self.assertGreater(invoke.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(invoke.call_args.kwargs["timeout"], 0.5)
        self.assertEqual(llm.max_retries, 0)

    def test_router_does_not_retry_an_uncertain_send(self):
        model = self.model(AIMessage(content="", tool_calls=[{
            "name": "send_qq_message", "id": "send", "args": {"user_id": "12345", "message": "正文"},
        }]))
        with patch("requests.post", side_effect=requests.ReadTimeout("slow")) as post:
            result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="给QQ12345发消息：正文")]})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(model.prompts), 1)
        self.assertEqual(result["task_status"], "unknown")
        self.assertIn("发送结果未确认", result["final_output"])

    def test_generated_content_survives_uncertain_send(self):
        self.model(AIMessage(content='{"intent":"generate_content"}'), AIMessage(content="已经生成的正文"))
        with patch("requests.post", side_effect=requests.ReadTimeout("slow")) as post:
            result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="写通知然后发给QQ12345")],
                                                  "platform": "qq", "receiver": "12345"})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["task_status"], "unknown")
        self.assertIn("已经生成的正文", result["final_output"])
        self.assertIn("发送结果未确认", result["final_output"])
        self.assertIsNone(result["next_agent"])

    def test_router_model_timeout_stops_without_supervisor_retry(self):
        model = self.model(TimeoutError("model slow"))
        result = graph.multi_agent_app.invoke({"messages": [HumanMessage(content="你好")]})
        self.assertEqual(len(model.prompts), 1)
        self.assertEqual(result["task_status"], "timeout")
        self.assertEqual(result["agent_history"], [])


class WorkerTimeoutTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.runner = AgentRunner(task=worker_task)
        self.addCleanup(self.runner.close)
        # Warm up separately so OS spawn/import costs do not make short deadline
        # tests flaky; production still counts startup inside the task budget.
        self.warm = self.runner.invoke({}, timeout=10)
        self.assertEqual(self.warm["task_status"], "success", self.warm)

    def test_normal_requests_reuse_worker(self):
        result = self.runner.invoke({}, timeout=2)
        self.assertEqual(result["worker_pid"], self.warm["worker_pid"])
        self.assertEqual(result["calls"], self.warm["calls"] + 1)

    def test_tool_timeout_kills_execution_and_preserves_artifact(self):
        marker = self.temporary_directory() / "late-send.txt"
        start = time.monotonic()
        result = self.runner.invoke({"action": "hung_send", "late_marker": str(marker)}, timeout=3)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertEqual(result["task_status"], "unknown")
        self.assertIn("已经生成的完整正文", result["final_output"])
        self.assertIn("saved-report.md", result["final_output"])
        self.assertIn("发送结果未确认", result["final_output"])
        self.assertIsNone(self.runner._process)
        time.sleep(1.05)
        self.assertFalse(marker.exists(), "Timed-out process continued executing")
        recovered = self.runner.invoke({}, timeout=10)
        self.assertNotEqual(recovered["worker_pid"], self.warm["worker_pid"])
        self.assertEqual(recovered["task_status"], "success")

    def test_total_deadline_also_stops_unguarded_blocking_calls(self):
        result = self.runner.invoke({"action": "hung_task"}, timeout=0.15)
        self.assertEqual(result["task_status"], "timeout")
        self.assertIn("此前已生成的内容", result["final_output"])
        self.assertIsNone(self.runner._process)

    def test_confirmed_send_remains_confirmed_when_later_work_times_out(self):
        result = self.runner.invoke({"action": "confirmed_then_hang"}, timeout=0.15)
        self.assertEqual(result["task_status"], "timeout")
        self.assertIn("消息发送已确认成功", result["final_output"])
        self.assertNotIn("发送结果未确认", result["final_output"])

    def test_waiting_request_can_timeout_without_cancelling_active_request(self):
        results = []
        thread = threading.Thread(target=lambda: results.append(self.runner.invoke({"action": "pause"}, timeout=2)))
        thread.start()
        until = time.monotonic() + 1
        while not self.runner._lock.locked() and time.monotonic() < until:
            time.sleep(0.001)
        waiting = self.runner.invoke({}, timeout=0.03)
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(waiting["task_status"], "timeout")
        self.assertIn("尚未开始执行", waiting["final_output"])
        self.assertEqual(results[0]["task_status"], "success")

    def test_worker_crash_is_reported_and_next_request_can_start(self):
        result = self.runner.invoke({"action": "crash"}, timeout=2)
        self.assertEqual(result["task_status"], "failed")
        self.assertEqual(self.runner.invoke({}, timeout=10)["task_status"], "success")

    def test_real_graph_propagates_tool_deadline_from_worker_threads(self):
        runner = AgentRunner(task=graph_worker_task)
        self.addCleanup(runner.close)
        self.assertEqual(runner.invoke({"action": "warm"}, timeout=15)["task_status"], "success")
        start = time.monotonic()
        result = runner.invoke({}, timeout=3)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertEqual(result["task_status"], "unknown")
        self.assertIn("真实工作流生成的正文", result["final_output"])
        self.assertIn("发送结果未确认", result["final_output"])
