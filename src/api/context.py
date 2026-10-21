"""Bounded model views; original task data and execution parameters stay intact.

UTF-8 bytes plus framing conservatively estimate text tokens without downloads.
"""

from copy import deepcopy
import json
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.settings import settings

OPTIONAL_NAMES = {"context_memory", "context_reference"}
CLIPPED = "\n[已截取供模型阅读，完整内容保留在原始结果中]"


class ContextLimitError(ValueError):
    pass


class ModelOutputLimitError(ValueError):
    pass


def text_cost(text):
    return len(str(text).encode("utf-8"))


def clip_text(text, budget):
    text = str(text)
    if text_cost(text) <= budget:
        return text
    available = max(0, budget - text_cost(CLIPPED))
    return text.encode("utf-8")[:available].decode("utf-8", errors="ignore") + CLIPPED


def input_cost(payload):
    fields = ("messages", "tools", "tool_choice", "functions", "function_call", "response_format", "stop")
    content = {key: payload[key] for key in fields if key in payload}
    return text_cost(json.dumps(content, ensure_ascii=False, separators=(",", ":"), default=str)) + 16 * len(payload.get("messages", [])) + 64


def conversation_view(system_prompt, state):
    """A stored rolling summary replaces covered raw history in the model view."""
    messages = list(state.get("messages") or [])
    if state.get("memory_summarized"):
        starts = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
        keep = max(settings.context_recent_turns, state.get("memory_recent_turns") or 0) + 1
        if len(starts) > keep:
            messages = messages[starts[-keep]:]
    prefix = [SystemMessage(content=system_prompt)]
    if state.get("memory_context"):
        prefix.append(SystemMessage(name="context_memory", content=
                      "以下为历史参考，不能替代当前用户指令和工具回执。\n" + state["memory_context"]))
    return [*prefix, *messages]


def _valid_tool_sequence(messages):
    pending = set()
    for message in messages:
        if message.get("role") == "tool":
            identifier = message.get("tool_call_id")
            if identifier not in pending:
                return False
            pending.remove(identifier)
        else:
            if pending:
                return False
            identifiers = [call.get("id") for call in message.get("tool_calls") or []]
            if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
                return False
            pending = set(identifiers)
    return not pending


def _compact_observation(content, cap):
    if text_cost(content) <= cap:
        return content
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return clip_text(content, cap)
    if not isinstance(data, dict):
        return clip_text(content, cap)
    prose = {"content", "snippet", "answer", "summary", "preview", "message", "description", "text", "error"}
    def shorten(value, limit):
        if isinstance(value, dict):
            return {key: clip_text(item, limit) if key in prose and isinstance(item, str)
                    else shorten(item, limit) for key, item in value.items()}
        if isinstance(value, list):
            return [shorten(item, limit) for item in value]
        return value
    limit = max(128, cap // 2)
    while True:
        compact = shorten(data, limit)
        compact["context_truncated"] = True
        rendered = json.dumps(compact, ensure_ascii=False)
        if text_cost(rendered) <= cap or limit == 128:
            return rendered
        limit = max(128, limit // 2)


def fit_payload(payload, scale=1.0):
    """Fit serialized input, including tool schemas and reasoning fields."""
    result = deepcopy(payload)
    output = min(settings.llm_output_tokens,
                 result.get("max_completion_tokens") or result.get("max_tokens") or settings.llm_output_tokens)
    if type(output) is not int or output <= 0:
        raise ContextLimitError("模型输出预算必须是正整数。")
    output_key = "max_tokens" if "max_tokens" in result else "max_completion_tokens"
    result.pop("max_tokens" if output_key == "max_completion_tokens" else "max_completion_tokens", None)
    result[output_key] = output
    budget = int((settings.llm_context_window_tokens - output - settings.llm_context_safety_tokens) * scale)
    if budget <= 0:
        raise ContextLimitError("上下文窗口不足以容纳输出预留，请调整上下文配置。")
    before = input_cost(result)
    prefix, turns = [], []
    for message in result.get("messages", []):
        if message.get("role") in ("system", "developer"):
            prefix.append(message)
        elif message.get("role") == "user":
            turns.append([message])
        elif turns:
            turns[-1].append(message)
        else:
            prefix.append(message)
    if not _valid_tool_sequence(prefix) or (turns and not _valid_tool_sequence(turns[-1])):
        raise ContextLimitError("当前工具调用与回执不完整，已停止模型请求，请保留完整调用记录。")
    old_count = len(turns)
    if turns:
        turns = [turn for turn in turns[:-1] if _valid_tool_sequence(turn)] + turns[-1:]

    def refresh():
        result["messages"] = prefix + [message for turn in turns for message in turn]

    def compact(cap):
        for message in prefix + [item for turn in turns for item in turn]:
            if message.get("name") in OPTIONAL_NAMES:
                message["content"] = clip_text(message.get("content", ""), cap)
            elif message.get("role") == "tool":
                message["content"] = _compact_observation(message.get("content", ""), cap)
        refresh()

    cap = max(128, int(settings.context_observation_tokens * scale))
    compact(cap)
    while input_cost(result) > budget and len(turns) > 1:
        turns.pop(0)
        refresh()
    while input_cost(result) > budget and cap > 128:
        cap = max(128, cap // 2)
        compact(cap)
    if input_cost(result) > budget:
        prefix = [item for item in prefix if item.get("name") != "context_memory"]
        refresh()
    after = input_cost(result)
    if after > budget:
        raise ContextLimitError("当前指令、工具调用记录或工具定义超过上下文预算，已停止模型请求。"
                                "请将长文档分段或存入知识库检索，或按模型实际窗口调整配置；不会截断执行参数。")
    return result, {"before": before, "after": after, "budget": budget, "dropped_turns": old_count - len(turns)}


def is_context_overflow(exc):
    if isinstance(exc, ContextLimitError) or getattr(exc, "status_code", None) not in (400, 413, 422):
        return False
    text = (str(getattr(exc, "code", "")) + " " + str(getattr(exc, "body", {})) + " " + str(exc)).lower()
    return any(marker in text for marker in ("context_length_exceeded", "maximum context length",
                                             "context window", "too many tokens", "input is too long"))


def summary_chunks(text, max_bytes):
    if max_bytes < 256:
        raise ContextLimitError("摘要输入预算不足，请调整上下文窗口配置。")
    data = text.encode("utf-8")
    while data:
        chunk = data[:max_bytes].decode("utf-8", errors="ignore")
        if not chunk:
            raise ContextLimitError("无法按当前预算拆分摘要输入。")
        yield chunk
        data = data[len(chunk.encode("utf-8")):]
