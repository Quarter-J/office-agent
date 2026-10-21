"""Deterministic arbitration shared by Router tools and specialist agents.

Tool schemas never expose task/step IDs: models cannot evade a claim by choosing
another ID. The current graph supports one delivery step per user task.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from langchain_core.messages import HumanMessage
from src.agent.context import latest_user_input, wants_delivery, delivery_forbidden, is_complex_task
from src.config.settings import settings
from src.utils.action_ledger import ActionLedger

_context = ContextVar("agent_action_context", default=None)
SEND_TOOLS = {"send_qq_message", "send_wechat_message"}
ACTION_TOOLS = SEND_TOOLS | {"generate_daily_report", "add_to_knowledge_base"}


def prepare_task(state):
    """Keep an explicit retry ID; rotate it when returned state gets a new turn."""
    messages = list(state.get("messages") or [])
    turn_id = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            turn_id = messages[i].id or uuid.uuid4().hex
            messages[i] = messages[i].model_copy(update={"id": turn_id})
            break
    previous = state.get("task_turn_id")
    task_id = state.get("task_id") if not previous or previous == turn_id else None
    return {"messages": messages, "task_id": task_id or uuid.uuid4().hex, "task_turn_id": turn_id}


@contextmanager
def action_context(state, actor):
    context = {"state": state, "actor": actor, "task_id": state.get("task_id") or uuid.uuid4().hex,
               "owner": state.get("execution_id") or uuid.uuid4().hex, "records": []}
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def conflict_result(kind, reason, proposed=None, accepted=None, *, resolution="blocked"):
    context = _context.get()
    record = {"kind": kind, "actor": context["actor"] if context else "execution",
              "proposed": proposed, "accepted": accepted, "reason": reason, "resolution": resolution}
    if context:
        ActionLedger().record_conflict(context["task_id"], **record)
        context["records"].append(record)
    return {"success": False, "status": "conflict", "retryable": False,
            "error": reason, "conflict": record}


def describe_action(name, arguments):
    """Canonicalize transport defaults so equivalent calls share a fingerprint."""
    args = dict(arguments)
    resource = None
    if name == "send_qq_message":
        group = str(args.get("group_id") or "").strip()
        target = group or str(args.get("user_id") or "").strip()
        target = str(int(target)) if target.isdigit() else target
        args = {"platform": "qq", "kind": "group" if group else "private",
                "target": target, "message": args.get("message", "")}
    elif name == "send_wechat_message":
        args = {"platform": "wechat", "kind": "private", "target": str(args.get("who") or "").strip(),
                "message": args.get("message", "")}
    elif name == "generate_daily_report":
        args = {"tomorrow_plan": "暂无", "issues": "无", "notes": "无", **args}
        args["report_date"] = args.get("report_date") or date.today().isoformat()
        target = args.get("output_path") or str(Path(settings.report_output_dir) / f"日报_{args.get('name', '')}_{args['report_date']}.md")
        resource = os.path.normcase(str(Path(target).resolve()))
        args["output_path"] = resource
    elif name == "add_to_knowledge_base":
        source = Path(args.get("file_path", ""))
        resource = os.path.normcase(str((Path(settings.knowledge_base_dir) /
                                        (source.name if source.is_absolute() else source)).resolve()))
        # Batch preflight must not read files. execute_action snapshots source
        # content before claiming the actual write.
        if source.is_absolute() and not args.get("content"):
            args["source_path"] = os.path.normcase(str(source.resolve()))
        args["file_path"] = resource
    proposal = {"tool": name, "arguments": args}
    fingerprint = hashlib.sha256(json.dumps(proposal, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    step = "delivery" if name in SEND_TOOLS else name + (":" + resource if resource else "")
    return step, fingerprint, proposal, resource


def check_send(name, arguments):
    context = _context.get()
    if not context or name not in SEND_TOOLS:
        return None
    state = context["state"]
    text = latest_user_input(state)
    proposed = describe_action(name, arguments)[2]
    args = proposed["arguments"]
    if delivery_forbidden(text) or not wants_delivery(text):
        return conflict_result("user_constraint", "用户未要求发送或明确限定不发送，发送已拦截。", proposed, text)
    if is_complex_task(text) and not state.get("generated_content"):
        return conflict_result("dependency", "正文尚未生成，不能执行发送。", proposed, "先完成内容生成")
    expected_content = state.get("generated_content") or state.get("message_content")
    if expected_content and args["message"] != expected_content:
        return conflict_result("content", "发送内容与任务中已确定的正文不一致，请确认要发送的版本。", proposed,
                               {"message": expected_content})
    if not args["target"] or not str(args["message"]).strip():
        return conflict_result("parameters", "发送目标或正文缺失，请补充后再执行。", proposed)
    # Use only explicit syntax as independent evidence; do not guess names with an LLM.
    prefix = re.split(r"[：:]", text, maxsplit=1)[0]
    groups = re.findall(r"QQ\s*群\s*(\d+)", prefix, re.IGNORECASE)
    users = re.findall(r"QQ\s*(?:好友|用户)?\s*(\d+)", prefix, re.IGNORECASE)
    explicit = groups or users
    if explicit and (args["platform"] != "qq" or args["kind"] != ("group" if groups else "private")
                     or set(map(lambda value: str(int(value)), explicit)) != {args["target"]}):
        return conflict_result("target", "发送目标与用户明确指定的 QQ 目标不一致，请确认接收人。", proposed, explicit)
    wechat = re.findall(r"(?:发送给|发给|给)\s*微信\s*([^\s，。:：]+)|"
                        r"微信\s*(?:通知|告诉|发送给|发给|给)\s*([^\s，。:：]+)", prefix)
    contacts = {left or right for left, right in wechat}
    if contacts and (args["platform"] != "wechat" or contacts != {args["target"]}):
        return conflict_result("target", "发送目标与用户明确指定的微信联系人不一致，请确认接收人。", proposed, sorted(contacts))
    # Parameters supplied by the caller or a completed generation handoff are constraints too.
    platform = state.get("platform")
    target = state.get("group_id") if platform == "qq" and state.get("group_id") else state.get("receiver")
    if platform and platform != args["platform"]:
        return conflict_result("target", "发送平台与任务中已确定的平台不一致。", proposed, platform)
    if target and (platform == "wechat" or str(target).isdigit()):
        expected = str(int(str(target))) if platform == "qq" else str(target).strip()
        kind = "group" if platform == "qq" and state.get("group_id") else "private"
        if expected != args["target"] or kind != args["kind"]:
            return conflict_result("target", "发送接收人与任务中已确定的目标不一致。", proposed, target)
    return None


def check_batch(calls):
    """Reject conflicting action groups before any member starts; keep independent calls."""
    blocked = {}
    sends = [call for call in calls if call["name"] in SEND_TOOLS]
    for call in sends:
        result = check_send(call["name"], call["args"])
        if result:
            blocked.update({item["id"]: result for item in sends})
            break
    groups = {}
    for call in calls:
        if call["name"] not in ACTION_TOOLS or call["id"] in blocked:
            continue
        try:
            step, fingerprint, proposal, resource = describe_action(call["name"], call["args"])
        except (ValueError, OSError):
            # Let the individual tool report unusable filesystem paths.
            continue
        groups.setdefault(resource or step, []).append((call, fingerprint, proposal))
    for group in groups.values():
        if len({fingerprint for _, fingerprint, _ in group}) > 1:
            result = conflict_result("batch", "同一步骤或目标文件出现不同执行方案，相关动作已暂停，请确认目标和内容。",
                                     [proposal for _, _, proposal in group])
            blocked.update({call["id"]: result for call, _, _ in group})
    return blocked


def execute_action(name, arguments, effect, invoke):
    """Claim before side effects; uncertain or completed steps are never re-run."""
    context = _context.get()
    if not context or effect not in ("send", "write"):
        return invoke(arguments)
    arguments = dict(arguments)
    if name == "generate_daily_report":
        arguments["report_date"] = arguments.get("report_date") or date.today().isoformat()
    elif name == "add_to_knowledge_base":
        source = Path(arguments["file_path"])
        try:
            if source.is_absolute() and not arguments.get("content") and source.is_file():
                arguments["content"] = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return json.dumps({"success": False, "status": "failed", "retryable": False,
                               "error": f"读取源文件失败，尚未写入知识库：{exc}"}, ensure_ascii=False)
    blocked = check_send(name, arguments)
    if blocked:
        return json.dumps(blocked, ensure_ascii=False)
    step, fingerprint, proposal, resource = describe_action(name, arguments)
    ledger = ActionLedger()
    resources = [resource] if resource else []
    if name == "add_to_knowledge_base":
        resources.append("knowledge_index:" + os.path.normcase(str(Path(settings.vector_store_path).resolve())))
    existing = ledger.claim(context["task_id"], step, fingerprint, proposal, context["owner"], resources)
    if existing:
        if existing["status"] == "resource_busy":
            result = conflict_result("resource", "目标文件仍被其他动作占用或写入结果未知，请核实后再处理。",
                                     proposal, json.loads(existing["proposal"]))
        elif existing["fingerprint"] != fingerprint:
            prior = "该步骤此前已成功执行。" if existing["status"] == "success" else ""
            result = conflict_result("step", prior + "同一任务步骤出现不同执行方案，已停止重复执行，请确认目标和内容。",
                                     proposal, json.loads(existing["proposal"]))
            if existing["result"]:
                result["previous_result"] = json.loads(existing["result"])
        elif existing["status"] in ("running", "unknown"):
            result = {"success": False, "status": "unknown", "retryable": False,
                      "error": "该动作正在执行或此前结果未确认。未自动重复执行，请先核实结果。"}
        else:
            result = json.loads(existing["result"])
            result["replayed"] = True
        return json.dumps(result, ensure_ascii=False)
    try:
        # A caller may pin the version it read before asking an agent to edit.
        # Resource claims serialize cooperating writers across runner instances.
        versions = {os.path.normcase(str(Path(key).resolve())): value
                    for key, value in (context["state"].get("resource_versions") or {}).items()}
        if resource and resource in versions:
            actual = hashlib.sha256(Path(resource).read_bytes()).hexdigest() if Path(resource).exists() else None
            if actual != versions[resource]:
                data = conflict_result("version", "目标文件在任务读取后已变化，未覆盖新版本。", proposal,
                                       {"expected": versions[resource], "actual": actual})
                ledger.finish(context["task_id"], step, context["owner"], "failed", data)
                return json.dumps(data, ensure_ascii=False)
        from src.utils.execution import remaining_seconds, timeout_result
        try:
            remaining_seconds(settings.tool_timeout_seconds)
        except TimeoutError:
            data = {**timeout_result("执行时限已到，动作尚未开始。"), "delivery_status": "not_sent"}
            ledger.finish(context["task_id"], step, context["owner"], "failed", data)
            return json.dumps(data, ensure_ascii=False)
        raw = invoke(arguments)
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            data = None
        if not isinstance(data, dict):
            data = {"success": False, "status": "unknown", "error": "动作返回结果无法确认，未自动重复执行。"}
        status = "success" if data.get("success") is True else (
            "unknown" if data.get("status") == "unknown" or effect == "write" else "failed")
        if status == "unknown":
            data.update(status="unknown", retryable=False)
        ledger.finish(context["task_id"], step, context["owner"], status, data)
        return json.dumps(data, ensure_ascii=False)
    except BaseException:
        ledger.finish(context["task_id"], step, context["owner"], "unknown",
                      {"success": False, "status": "unknown", "error": "动作执行中断，结果未确认。"})
        raise
