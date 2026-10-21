# 执行时限与超时结果

CLI 和 Web UI 均通过 `src.agent.runner.run_agent(state)` 执行任务。它使用一个可复用的工作进程，正常请求保留进程中的记忆与客户端；工具、模型或整轮任务超过时限时，父进程终止工作进程。下一次请求会创建新进程。

直接调用 `src.agent.graph.multi_agent_app` 是底层工作流接口，适合组合和单元测试，**不提供进程级硬超时**。应用集成应使用 `run_agent`。Windows 脚本应在 `if __name__ == "__main__":` 内调用，以支持进程启动。

```python
from langchain_core.messages import HumanMessage
from src.agent.runner import run_agent

if __name__ == "__main__":
    result = run_agent({"messages": [HumanMessage(content="查询知识库里的部署步骤")]})
    print(result["final_output"])
```

可以在 `.env` 中设置以下参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `TASK_TIMEOUT_SECONDS` | 120 | 整轮时限，包含排队、工作进程启动、记忆处理、模型、工具和重试 |
| `TOOL_TIMEOUT_SECONDS` | 30 | 单次工具执行时限，包含内部重试和模型调用 |
| `LLM_TIMEOUT_SECONDS` | 30 | 单次模型调用时限，SDK 自动重试已关闭 |
| `EMBEDDING_TIMEOUT_SECONDS` | 10 | 嵌入请求网络超时，SDK 自动重试已关闭 |
| `HTTP_CONNECT_TIMEOUT_SECONDS` | 5 | HTTP 建立连接的等待时间 |
| `HTTP_READ_TIMEOUT_SECONDS` | 10 | HTTP 等待响应数据的时间 |
| `READONLY_TIMEOUT_RETRIES` | 1 | 查询请求最多额外重试次数，受剩余预算限制 |

所有工具/模型时限都受剩余整轮预算约束。HTTP 连接和读取超时并非完整请求的墙钟时限，因此还需要父进程监督。进程终止与资源回收会有少量调度开销。当前工作进程串行处理请求；排队超时不会中断正在运行的另一个请求。

## 返回状态

- `success`：操作明确完成。
- `failed`：操作明确失败。
- `timeout`：等待或执行超时。已经完成的结果、资料和生成正文会保留。
- `unknown`：发送或文件写入过程中被中断，无法确认最终结果。不会自动重做。
- `conflict`：执行方案违反任务约束或与同一步骤的已有方案不一致，相关动作暂停。详见 [Agent 冲突处理](agent-conflicts.md)。
- `context_error`：Router 或 Supervisor 无法在上下文预算内完成模型请求，或模型输出不完整，已停止继续分派。详见 [长上下文处理](context-management.md)。

QQ 连接建立超时返回 `timeout` / `delivery_status=not_sent`；等待发送响应超时、连接中断、异常响应返回 `unknown`。只有 HTTP 404/405 明确表示接口不可用时，才尝试兼容接口。微信自动化卡住时由工作进程监督器结束本地执行，远端是否已收到仍需核实。

当前 OneBot/微信通道没有统一可靠的请求幂等标识和按该标识查询回执的接口，因此不实施自动补发。终止本地进程无法撤销已被远端接收的请求。超时后内存回退中的短期缓存会随进程重启而丢失；前端提供的对话历史、已保存的文件及 SQLite 数据仍保留。

向量检索网络超时后尝试关键词检索；已有片段但回答生成超时时保留检索片段。硬时限触发时结束本轮任务，不在已终止的工作进程中继续降级。

## 离线验证

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -t . -v
```

测试包含实际进程终止、工作进程复用/恢复、整轮与工具时限、队列等待时限、LangGraph 工具线程中的预算传播，以及网络响应超时不重复发送。消息和模型使用替身，不向真实联系人发送消息。
