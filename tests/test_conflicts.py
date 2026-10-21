import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from tests.support import OfflineTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from src.agent import graph
from src.agent.conflicts import action_context, prepare_task
from src.agent.runner import AgentRunner
from src.tools import message, report
from src.tools.search import add_to_knowledge_base
from src.utils.action_ledger import ActionLedger
from tests.worker_fixtures import ledger_worker_task
from tests.test_routing import local_echo


def request(identifier, text="给QQ12345发消息：正文", **values):
    return {"task_id": identifier, "messages": [HumanMessage(content=text)], **values}


def send_call(identifier, message_text="正文", target="12345"):
    return {"name": "send_qq_message", "id": identifier, "args": {"user_id": target, "message": message_text}}


def success_response():
    result = Mock(status_code=200)
    result.json.return_value = {"status": "ok", "retcode": 0}
    return result


class ConflictTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(graph.memory_manager, "_enabled", False))

    def invoke_send(self, state, text="正文", actor="MessageAgent", **kwargs):
        with action_context(state, actor):
            return json.loads(message.send_qq_message.invoke({"user_id": "12345", "message": text, **kwargs}))

    def test_tool_guard_blocks_no_send_and_missing_permission(self):
        with patch("requests.post") as post:
            for text in ("写通知，不发送", "查询天气"):
                result = self.invoke_send(request(text, text))
                self.assertEqual(result["status"], "conflict")
                self.assertEqual(result["conflict"]["kind"], "user_constraint")
        post.assert_not_called()

    def test_tool_guard_checks_generation_dependency_and_body(self):
        state = request("generated", "写通知然后发给QQ12345")
        with patch("requests.post") as post:
            self.assertEqual(self.invoke_send(state)["conflict"]["kind"], "dependency")
            state["generated_content"] = "完整正文"
            self.assertEqual(self.invoke_send(state)["conflict"]["kind"], "content")
        post.assert_not_called()
        self.assertEqual(len(ActionLedger().conflicts("generated")), 2)

    def test_target_disagreement_cannot_send_to_another_person(self):
        with patch("requests.post") as post:
            result = self.invoke_send(request("wrong-target"), user_id="67890")
        self.assertEqual(result["conflict"]["kind"], "target")
        post.assert_not_called()

    def test_wechat_explicit_contact_is_checked_before_client_creation(self):
        with action_context(request("wechat-target", "微信通知张三：正文"), "Router"):
            result = json.loads(message.send_wechat_message.invoke({"who": "李四", "message": "正文"}))
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(result["conflict"]["kind"], "target")

    def test_conflicting_writes_preserve_file_and_independent_read_result(self):
        path = self.temporary_directory() / "report.md"
        path.write_text("原版本", encoding="utf-8")
        args = {"name": "测试", "department": "测试", "output_path": str(path)}
        self.model(AIMessage(content="", tool_calls=[
            {"id": "write-one", "name": "generate_daily_report", "args": {**args, "today_work": "版本一"}},
            {"id": "write-two", "name": "generate_daily_report", "args": {**args, "today_work": "版本二"}},
            {"id": "read", "name": "local_echo", "args": {"value": "独立查询结果"}},
        ]))
        with patch.object(graph, "_tool_node", ToolNode([report.generate_daily_report, local_echo])):
            result = graph.multi_agent_app.invoke(request("conflicting-writes", "生成日报并查询资料"))
        self.assertEqual(result["task_status"], "conflict")
        self.assertIn("独立查询结果", result["final_output"])
        self.assertEqual(path.read_text(encoding="utf-8"), "原版本")

    def test_invalid_action_parameters_do_not_hide_independent_results(self):
        state = request("invalid-parameters", "处理文档")
        self.model(AIMessage(content="", tool_calls=[
            {"id": "invalid", "name": "add_to_knowledge_base", "args": {"file_path": None, "content": "正文"}},
            {"id": "read", "name": "local_echo", "args": {"value": "独立查询结果"}},
        ]), AIMessage(content="查询完成，写入参数有误"))
        with patch.object(graph, "_tool_node", ToolNode([add_to_knowledge_base, local_echo])):
            result = graph.multi_agent_app.invoke(state)
        outputs = {msg.tool_call_id: msg for msg in result["messages"] if isinstance(msg, ToolMessage)}
        self.assertEqual(outputs["invalid"].status, "error")
        self.assertEqual(json.loads(outputs["read"].content)["summary"], "独立查询结果")

    def test_both_graph_paths_use_the_guard(self):
        self.model(AIMessage(content="", tool_calls=[send_call("bad", target="67890")]))
        with patch("requests.post") as post:
            router_result = graph.multi_agent_app.invoke(request("router-guard"))
        self.assertEqual(router_result["task_status"], "conflict")
        post.assert_not_called()
        self.model(AIMessage(content='{"intent":"generate_content"}'), AIMessage(content="已生成的正文"))
        with patch("requests.post") as post:
            result = graph.multi_agent_app.invoke(request("specialist-guard", "写通知然后发给QQ12345",
                                                           platform="qq", receiver="67890"))
        self.assertEqual(result["task_status"], "conflict")
        self.assertIn("已生成的正文", result["final_output"])
        self.assertEqual(result["conflicts"][-1]["actor"], "MessageAgent")
        post.assert_not_called()

    def test_contradictory_batch_is_rejected_before_first_send(self):
        self.model(AIMessage(content="", tool_calls=[send_call("first", "版本一"), send_call("second", "版本二")]))
        with patch("requests.post") as post:
            result = graph.multi_agent_app.invoke(request("batch-conflict"))
        self.assertEqual(result["task_status"], "conflict")
        self.assertEqual(result["conflicts"][-1]["kind"], "batch")
        self.assertEqual({msg.tool_call_id for msg in result["messages"] if isinstance(msg, ToolMessage)}, {"first", "second"})
        post.assert_not_called()

    def test_identical_batch_and_later_round_send_only_once(self):
        self.model(AIMessage(content="", tool_calls=[send_call("one"), send_call("two")]),
                   AIMessage(content="", tool_calls=[send_call("again")]), AIMessage(content="发送完成"))
        with patch("requests.post", return_value=success_response()) as post:
            result = graph.multi_agent_app.invoke(request("duplicates"))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["task_status"], "success")
        outputs = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        self.assertEqual(len(outputs), 3)
        self.assertTrue(json.loads(outputs[-1].content)["replayed"])

    def test_success_receipt_survives_another_agent_and_ledger_instance(self):
        state = request("receipt")
        with patch("requests.post", return_value=success_response()) as post:
            self.assertTrue(self.invoke_send(state)["success"])
            again = self.invoke_send(state, actor="Router")
        self.assertTrue(again["success"])
        self.assertTrue(again["replayed"])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(ActionLedger().actions("receipt")[0]["status"], "success")

    def test_changed_proposal_does_not_replace_completed_action(self):
        state = request("changed")
        with patch("requests.post", return_value=success_response()) as post:
            self.invoke_send(state)
            conflict = self.invoke_send(state, text="另一份正文", actor="Router")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(conflict["status"], "conflict")
        self.assertIn("此前已成功", conflict["error"])
        record = ActionLedger().conflicts("changed")[0]
        self.assertEqual(record["accepted"]["arguments"]["message"], "正文")
        self.assertEqual(record["proposed"]["arguments"]["message"], "另一份正文")

    def test_unknown_and_failed_receipts_are_not_retried(self):
        import requests
        for identifier, response in (("unknown", requests.ReadTimeout("slow")), ("failed", Mock(status_code=403))):
            with self.subTest(identifier=identifier):
                options = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
                with patch("requests.post", **options) as post:
                    self.invoke_send(request(identifier))
                    second = self.invoke_send(request(identifier))
                self.assertEqual(post.call_count, 1)
                self.assertEqual(second["status"], identifier)

    def test_concurrent_claims_cannot_send_twice(self):
        entered, release = threading.Event(), threading.Event()
        def transport(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test transport was not released")
            return success_response()
        with patch("requests.post", side_effect=transport) as post, ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.invoke_send, request("concurrent"))
            try:
                self.assertTrue(entered.wait(2))
                second = pool.submit(self.invoke_send, request("concurrent")).result(timeout=2)
                self.assertEqual(second["status"], "unknown")
            finally:
                release.set()
            self.assertTrue(first.result(timeout=2)["success"])
        self.assertEqual(post.call_count, 1)

    def test_new_turn_gets_new_task_and_may_repeat_an_explicit_action(self):
        first = prepare_task(request("first"))
        repeated = prepare_task(first)
        self.assertEqual(first["task_id"], repeated["task_id"])
        next_turn = prepare_task({**first, "messages": [*first["messages"], HumanMessage(content="给QQ12345发消息：正文")]})
        self.assertNotEqual(first["task_id"], next_turn["task_id"])
        with patch("requests.post", return_value=success_response()) as post:
            self.invoke_send(first)
            self.invoke_send(next_turn)
        self.assertEqual(post.call_count, 2)

    def test_unavailable_ledger_fails_before_transport(self):
        with patch.object(ActionLedger, "claim", side_effect=OSError("disk unavailable")), patch("requests.post") as post:
            with self.assertRaises(OSError):
                self.invoke_send(request("storage-error"))
        post.assert_not_called()

    def test_budget_exhausted_during_claim_prevents_action(self):
        from src.utils.execution import execution_context
        original = ActionLedger.claim
        clock = [100.0]
        def slow_claim(ledger, *args, **kwargs):
            receipt = original(ledger, *args, **kwargs)
            clock[0] = 102.0
            return receipt
        with patch("src.utils.execution.time.monotonic", side_effect=lambda: clock[0]), execution_context(101), \
             patch.object(ActionLedger, "claim", slow_claim), patch("requests.post") as post:
            result = self.invoke_send(request("claim-timeout"))
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["delivery_status"], "not_sent")
        post.assert_not_called()

    def test_report_replay_does_not_overwrite_newer_file(self):
        path = self.temporary_directory() / "report.md"
        args = {"name": "测试", "department": "测试", "today_work": "完成测试", "output_path": str(path)}
        state = request("report", "生成日报")
        with action_context(state, "ReportAgent"):
            initial = json.loads(report.generate_daily_report.invoke(args))
        path.write_text("后来保存的版本", encoding="utf-8")
        with action_context(state, "Router"):
            again = json.loads(report.generate_daily_report.invoke(args))
        self.assertTrue(again["replayed"])
        self.assertEqual(again["content"], initial["content"])
        self.assertEqual(path.read_text(encoding="utf-8"), "后来保存的版本")

    def test_file_version_disagreement_preserves_existing_file(self):
        path = self.temporary_directory() / "report.md"
        path.write_text("别人的新版本", encoding="utf-8")
        state = request("version", "生成日报", resource_versions={os.path.normcase(str(path.resolve())): hashlib.sha256(b"old").hexdigest()})
        with action_context(state, "ReportAgent"):
            result = json.loads(report.generate_daily_report.invoke({"name": "测试", "department": "测试",
                               "today_work": "新正文", "output_path": str(path)}))
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(path.read_text(encoding="utf-8"), "别人的新版本")

    def test_shared_resource_and_unknown_claim_remain_blocked_until_verified(self):
        ledger = ActionLedger()
        self.assertIsNone(ledger.claim("one", "write", "a", {}, "owner", ["file", "index"]))
        self.assertEqual(ledger.claim("two", "write", "b", {}, "another", ["other-file", "index"])["status"], "resource_busy")
        ledger.interrupt("one", "owner")
        record = ledger.actions("one")[0]
        self.assertEqual(record["status"], "unknown")
        self.assertFalse(ledger.resolve("one", "write", expected_updated=0, success=False, evidence="检查磁盘", result={}))
        self.assertEqual(ledger.claim("two", "write", "b", {}, "another", "file")["status"], "resource_busy")
        self.assertTrue(ledger.resolve("one", "write", expected_updated=record["updated"], success=False,
                                       evidence="检查文件与索引，确认未写入", result={"error": "确认未写入"}))
        self.assertIsNone(ledger.claim("two", "write", "b", {}, "another", ["file", "index"]))

    def test_routing_override_has_durable_reason(self):
        self.model(AIMessage(content='{"intent":"send_message"}'), AIMessage(content="本地正文"))
        result = graph.multi_agent_app.invoke(request("routing-rule", "只生成通知，不发送"))
        records = ActionLedger().conflicts("routing-rule")
        self.assertEqual(result["task_status"], "success")
        self.assertEqual(records[0]["proposed"], "send_message")
        self.assertEqual(records[0]["accepted"], "generate_content")
        self.assertEqual(records[0]["resolution"], "rules")


class ProcessConflictTests(OfflineTestCase):
    def test_killed_send_is_persisted_and_not_restarted(self):
        runner = AgentRunner(task=ledger_worker_task)
        self.addCleanup(runner.close)
        self.assertEqual(runner.invoke({"action": "warm"}, timeout=10)["task_status"], "success")
        marker = self.temporary_directory() / "sent.txt"
        state = request("killed", "微信通知张三：正文", marker=str(marker), delay=2)
        first = runner.invoke(state, timeout=3)
        self.assertEqual(first["task_status"], "unknown")
        self.assertEqual(ActionLedger().actions("killed")[0]["status"], "unknown")
        second = runner.invoke({**state, "delay": 0}, timeout=10)
        self.assertEqual(second["task_status"], "unknown")
        self.assertFalse(marker.exists())

    def test_success_is_replayed_after_worker_restart(self):
        runner = AgentRunner(task=ledger_worker_task)
        self.addCleanup(runner.close)
        marker = self.temporary_directory() / "sent.txt"
        state = request("restart", "微信通知张三：正文", marker=str(marker), delay=0)
        first = runner.invoke(state, timeout=10)
        self.assertEqual(first["task_status"], "success")
        runner.close()
        second = runner.invoke(state, timeout=10)
        self.assertEqual(second["task_status"], "success")
        self.assertTrue(second["receipt"]["replayed"])
        self.assertEqual(marker.read_text(encoding="utf-8"), "sent\n")
