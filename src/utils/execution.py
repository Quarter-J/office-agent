"""Execution budgets and progress events for the supervised worker process.

Network timeouts handle recoverable failures. The parent process enforces hard
deadlines, including calls that do not cooperate with cancellation.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import inspect
import time
import uuid

from src.config.settings import settings

_deadline = ContextVar("execution_deadline", default=None)
_sink = ContextVar("execution_event_sink", default=None)


def publish(event: dict):
    sink = _sink.get()
    if sink is not None:
        sink({**event, "at": time.monotonic()})


@contextmanager
def execution_context(deadline, sink=None):
    deadline_token = _deadline.set(deadline)
    sink_token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(sink_token)
        _deadline.reset(deadline_token)


def remaining_seconds(cap: float) -> float:
    deadline = _deadline.get()
    seconds = min(cap, deadline - time.monotonic()) if deadline is not None else cap
    if seconds <= 0:
        raise TimeoutError("执行时限已到，未启动后续操作")
    return seconds


@contextmanager
def operation(name: str, timeout: float, effect: str = ""):
    seconds = remaining_seconds(timeout)
    deadline = time.monotonic() + seconds
    token = _deadline.set(deadline)
    identifier = uuid.uuid4().hex
    try:
        publish({"type": "start", "id": identifier, "name": name,
                 "deadline": deadline, "effect": effect})
        # Waiting for the monitor's acknowledgement also consumes the budget.
        remaining_seconds(timeout)
        yield identifier
    finally:
        try:
            publish({"type": "end", "id": identifier})
        finally:
            _deadline.reset(token)


def timed_tool(effect: str = ""):
    """Keep LangChain's tool schema while publishing start/result/end events."""
    def decorate(func):
        signature = inspect.signature(func)
        @wraps(func)
        def wrapped(*args, **kwargs):
            with operation(func.__name__, settings.tool_timeout_seconds, effect) as identifier:
                if effect in ("send", "write"):
                    from src.agent.conflicts import execute_action
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    def invoke(values):
                        bound.arguments.update(values)
                        return func(*bound.args, **bound.kwargs)
                    result = execute_action(func.__name__, bound.arguments, effect,
                                            invoke)
                else:
                    result = func(*args, **kwargs)
                try:
                    data = json.loads(result) if isinstance(result, str) else result
                except (ValueError, TypeError):
                    data = None
                publish({"type": "tool_result", "id": identifier, "name": func.__name__, "result": data})
                return result
        return wrapped
    return decorate


def http_timeout():
    return (remaining_seconds(settings.http_connect_timeout_seconds),
            remaining_seconds(settings.http_read_timeout_seconds))


def read_request(request, *args, **kwargs):
    """Only read operations may retry; all attempts share the current budget."""
    import requests
    for attempt in range(settings.readonly_timeout_retries + 1):
        try:
            return request(*args, **kwargs, timeout=http_timeout())
        except (requests.Timeout, requests.ConnectionError):
            if attempt >= settings.readonly_timeout_retries:
                raise
            remaining_seconds(settings.http_read_timeout_seconds)


def is_timeout(exc: Exception) -> bool:
    import requests
    # SDK timeout exceptions use these names; avoid importing optional SDKs here.
    return isinstance(exc, (TimeoutError, requests.Timeout)) or type(exc).__name__ in {
        "APITimeoutError", "ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout",
    }


def timeout_result(message: str) -> dict:
    return {"success": False, "status": "timeout", "error_type": "timeout",
            "retryable": False, "error": message}


def unknown_delivery(message: str) -> dict:
    return {"success": False, "status": "unknown", "delivery_status": "unknown",
            "retryable": False, "error": message}
