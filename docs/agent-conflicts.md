# Agent 冲突处理

当前工作流通过确定性规则、动作记录和文件占用处理执行冲突。Router 的工具路径与 Supervisor 的子 Agent 路径使用同一套工具包装层。无需增加仲裁 Agent。

## 已实现的行为

- 用户未要求发送、明确禁止发送、生成正文尚未完成、发送正文与生成正文不一致时，拦截发送。
- 对明确写出的 QQ 号码、QQ群号、支持格式的微信联系人，以及任务中已确定的平台和目标进行核对。自然语言检查使用有限规则，不能保证理解所有表达。
- 同一批工具对同一个发送步骤或文件提出不同方案时，执行前拦截该组动作；无关查询仍可完成。相同的调用合并执行，并为每个工具调用 ID 返回结果。
- 同一任务的同一步骤只能取得一次执行权。成功时缓存原回执；明确失败时也不自动重试；执行中或结果未知时阻止再次执行。修改目标或正文不能绕过去重。
- 文件写入使用 SQLite 资源占用记录。知识库写入同时占用共享向量索引，避免不同文档并发更新同一索引。支持调用方提供文件版本，拒绝覆盖已变化的版本。
- 冲突双方的方案、来源 Agent、原因和裁决写入 SQLite。规则可以纠正的路由冲突继续执行；需要确认的动作冲突返回 `task_status="conflict"`。已有正文和独立工具结果保留在输出中。
- 进程被终止时，已取得执行权但没有回执的动作标记为 `unknown`；重启不会自动重做。如果整个应用异常退出，残留的 `running` 记录同样会阻止重做。

## 任务身份和重试

默认每轮用户输入分配新的 `task_id`，发送步骤固定为 `delivery`。步骤编号由代码确定，不向模型开放。当前任务只支持一个发送方案，多收件人或不同平台的批量发送需要先扩展显式任务计划。

应用调用 `run_agent` 时，可主动指定一个稳定的 `task_id`，并在重复提交同一任务时保留它：

```python
from uuid import uuid4
from langchain_core.messages import HumanMessage
from src.agent.runner import run_agent

if __name__ == "__main__":
    request = {
        "task_id": uuid4().hex,
        "messages": [HumanMessage(content="生成今天的日报，我是张三，完成了接口开发")],
    }
    result = run_agent(request)
    # 若调用方丢失响应，重复提交 request 可复用已完成工具的回执。
```

调用方应将任务 ID 与业务请求一起保存。重新创建任务 ID 会被视为用户新请求，允许再次执行。向返回状态追加新的 HumanMessage 会生成新任务 ID；重试时应保留原始消息与任务 ID。这里只复用工具执行结果，模型推理仍可能重新执行；模型改写了已执行步骤的参数时会报告冲突。

数据库路径由 `ACTION_LEDGER_PATH` 配置，默认 `data/db/action_ledger.db`。多个进程应使用同一个本地数据库路径。数据库包含消息正文和执行回执，按业务数据管理。

底层工具单独调用时没有用户任务上下文，保持原有直接调用行为。需要保护的独立集成必须用 `action_context(state, actor)` 包裹工具调用；CLI、Web UI 和原始 LangGraph 已接入此上下文。原始 LangGraph 仍不提供硬超时，应使用 `run_agent`。

## 文件版本与未知结果核实

`resource_versions` 可传入调用方先前读取的文件版本：键为文件路径，值为文件字节的 SHA-256；文件原本不存在时值为 `None`。检查在获得资源占用权后、写入前执行。没有提供版本时，只提供协作进程间的写入互斥，后续独立任务仍可更新已有文件。外部编辑器不遵循资源占用，无法由此数据库完全协调。

未知结果不会因等待一段时间而自动变成失败。先检查接收方、文件或通道回执，再使用不向模型暴露的维护接口核实记录：

```python
from src.utils.action_ledger import ActionLedger

ledger = ActionLedger()
records = ledger.actions(task_id)
# 找到需要核实的 unknown 记录；确认执行进程已停止且核实外部结果后：
record = next(item for item in records if item["step_id"] == step_id)
resolved = ledger.resolve(
    task_id, step_id,
    expected_updated=record["updated"],
    success=True,
    evidence="已核实的通道回执编号或文件检查结果",
    result={"message": "已核实动作完成"},
)
```

维护接口仅接受状态为 `unknown` 且版本未变化的记录，避免覆盖新的执行结果。应用整体崩溃留下 `running` 时，必须先确认对应进程已停止，再调用 `ledger.interrupt(task_id, owner)`。核实完成后释放文件资源。即使确认失败，也不会自动重试原步骤；再次执行需要用户明确提出新任务。

## 边界与验证

本实现处理执行约束、路由、参数和文件写入冲突。尚未实现不同资料之间的事实矛盾识别、通用多步骤任务规划或自然语言澄清后的自动续跑。

本地动作记录不能让远端消息获得严格的 exactly-once 语义。远端接受消息而本地尚未写入成功回执时，仍只能返回未知；不会自动补发。清空记录或使用新的任务 ID 会失去原任务的去重保护。

测试使用模型与消息替身，不向真实联系人发送消息：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -t . -v
```
