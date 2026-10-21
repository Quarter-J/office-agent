"""Bounded application entry point with a reusable, killable worker process.

The raw LangGraph in graph.py remains available for composition/unit testing.
CLI, Web UI and other applications should call run_agent for hard time limits.
"""

import atexit
import multiprocessing
import threading
import time
import uuid
from pathlib import Path

from src.config.settings import settings
from src.utils.execution import execution_context, publish
from src.agent.conflicts import prepare_task
from src.utils.action_ledger import ActionLedger


def _run_graph(state):
    from src.agent.graph import multi_agent_app
    result = state
    for result in multi_agent_app.stream(state, stream_mode="values"):
        publish({"type": "state", "state": result})
    return result


def _worker_main(connection, task):
    send_lock = threading.Lock()
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            identifier = request["id"]

            def sink(event):
                with send_lock:
                    connection.send({**event, "request_id": identifier})
                    if event["type"] == "start":
                        # A side effect must not begin before the parent knows
                        # it is in flight. This closes the deadline/event race.
                        acknowledgement = connection.recv()
                        if acknowledgement.get("ack") != event["id"]:
                            raise RuntimeError("执行监控确认失败")

            try:
                with execution_context(request["deadline"], sink):
                    result = task(request["state"])
                sink({"type": "done", "state": result, "at": time.monotonic()})
            except Exception as exc:
                sink({"type": "error", "error": str(exc), "at": time.monotonic()})
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        connection.close()


def _interrupted_result(state, completed, active, reason, *, timed_out=True):
    """Report what is known; terminating a client cannot undo a remote send."""
    result = dict(state)
    uncertain_send = any(item.get("effect") == "send" for item in active.values())
    uncertain_write = any(item.get("effect") == "write" for item in active.values())
    status = "unknown" if uncertain_send or uncertain_write else ("timeout" if timed_out else "failed")
    parts = []
    seen = set()
    for item in completed:
        data = item.get("result")
        if not isinstance(data, dict):
            continue
        if data.get("status") == "unknown":
            status = "unknown"
        if data.get("success"):
            if data.get("content") and item["name"] == "generate_daily_report":
                result["generated_content"] = data["content"]
                result["report_file_path"] = data.get("file_path")
            text = data.get("summary") or data.get("answer") or data.get("message")
            if item["name"] in ("send_qq_message", "send_wechat_message"):
                text = "消息发送已确认成功。"
            if not text and data.get("results"):
                text = "\n".join(str(hit.get("content") or hit.get("snippet") or hit.get("title") or "")
                                 for hit in data["results"] if isinstance(hit, dict))
        else:
            text = data.get("error")
        if text and text not in seen:
            parts.append(text)
            seen.add(text)
    if result.get("generated_content"):
        parts.append("已生成的正文：\n\n" + result["generated_content"])
    if result.get("report_file_path"):
        parts.append("日报文件：" + result["report_file_path"])
    parts.append(reason)
    if uncertain_send:
        parts.append("发送结果未确认，请核实接收方是否收到。未自动重复发送。")
    elif uncertain_write:
        parts.append("文件操作结果未确认，请检查目标文件。未自动重复执行。")
    result.update(final_output="\n\n".join(parts), task_status=status, error=reason,
                  next_agent=None, needs_clarification=False)
    return result


class AgentRunner:
    def __init__(self, task=_run_graph):
        self._task = task
        self._lock = threading.Lock()
        self._process = None
        self._connection = None

    def _start(self):
        if self._process is not None and self._process.is_alive():
            return
        self._stop()
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_worker_main, args=(child, self._task), daemon=True)
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self._connection, self._process = parent, process

    def _stop(self):
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.5)
            self._process.close()
        if self._connection is not None:
            self._connection.close()
        self._process = self._connection = None

    def close(self):
        with self._lock:
            self._stop()

    def _interrupt_actions(self, state):
        if Path(settings.action_ledger_path).exists() and state.get("task_id") and state.get("execution_id"):
            try:
                ActionLedger().interrupt(state["task_id"], state["execution_id"])
            except Exception:
                # A durable running claim also blocks retries. Never release a
                # claim just because recovery storage is temporarily unavailable.
                pass

    def invoke(self, state, *, timeout=None):
        seconds = settings.task_timeout_seconds if timeout is None else timeout
        if seconds <= 0:
            raise ValueError("任务时限必须大于 0")
        deadline = time.monotonic() + seconds
        identifier = uuid.uuid4().hex
        snapshot = {**state, **prepare_task(state), "execution_id": identifier,
                    "final_output": None, "generated_content": None,
                    "report_file_path": None, "error": None, "next_agent": None}
        completed, active = [], {}
        if not self._lock.acquire(timeout=seconds):
            return _interrupted_result(snapshot, completed, active, "任务等待超时，尚未开始执行。")
        try:
            self._start()
            self._connection.send({"id": identifier, "deadline": deadline, "state": snapshot})
            while True:
                end = min([deadline] + [item["deadline"] for item in active.values()])
                remaining = end - time.monotonic()
                if remaining <= 0:
                    reason = "任务总时限已到，已停止后续执行。" if end == deadline else "操作执行超时，已停止后续执行。"
                    self._stop()
                    self._interrupt_actions(snapshot)
                    return _interrupted_result(snapshot, completed, active, reason)
                if not self._connection.poll(min(remaining, 0.1)):
                    if not self._process.is_alive():
                        raise RuntimeError("工作进程已退出")
                    continue
                event = self._connection.recv()
                if event.get("request_id") != identifier:
                    continue
                kind = event["type"]
                if kind == "start":
                    active[event["id"]] = event
                    self._connection.send({"ack": event["id"]})
                elif kind == "end":
                    active.pop(event["id"], None)
                elif kind == "tool_result":
                    completed.append(event)
                    # A returned receipt is stronger evidence than a still-queued
                    # scope-end event; do not label a confirmed send as unknown.
                    active.pop(event["id"], None)
                elif kind == "state":
                    snapshot = event["state"]
                elif kind == "done":
                    return event["state"]
                elif kind == "error":
                    raise RuntimeError(event["error"])
        except Exception as exc:
            self._stop()
            self._interrupt_actions(snapshot)
            return _interrupted_result(snapshot, completed, active, f"执行中断：{exc}", timed_out=False)
        except BaseException:
            self._stop()
            self._interrupt_actions(snapshot)
            raise
        finally:
            self._lock.release()


_runner = AgentRunner()
atexit.register(_runner.close)


def run_agent(state, *, timeout=None):
    return _runner.invoke(state, timeout=timeout)
