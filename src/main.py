"""
Office Agent - 基于 LangGraph 的多功能办公助手

四层架构：
  - 交互层：CLI 控制台 / Web 管理界面（预留）
  - 智能调度层：LangGraph 状态机 + 7 节点工作流
  - 能力工具层：消息 / 日报 / 检索 / 记忆
  - 数据存储层：SQLite / 向量库 / 文件系统
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from langchain_core.messages import HumanMessage, AIMessage
from rich.markdown import Markdown
from src.agent.runner import run_agent
from src.utils.logger import logger
from src.utils.startup import check_environment

console = Console()


WELCOME_TEXT = """
[bold cyan]🤖 Office Agent - 智能办公助手 v2.0[/bold cyan]

[green]可用功能:[/green]
  📨  [bold]消息发送[/bold]  - 发送 QQ / 微信消息
    例: "给QQ好友12345发消息：明天下午开会"
    例: "微信通知张三：项目已上线"
    例: "在QQ群67890发通知：下午放假"

  📊  [bold]工作日报[/bold]  - 生成/查看日报
    例: "生成今天的日报，我是张三，完成了项目开发"
    例: "查看我最近的日报"

  📚  [bold]知识库检索[/bold]  - 在本地知识库中查找
    例: "在知识库中搜索Spring Boot相关文档"
    例: "把这个文档添加到知识库：..."

  🌐  [bold]网页搜索[/bold]  - 搜索互联网实时信息
    例: "搜索一下Python 3.12的新特性"
    例: "帮我搜索最新的AI新闻"

  💬  [bold]普通对话[/bold]  - 闲聊或咨询（支持多轮记忆）

[dim]输入 'exit' 退出，输入 'clear' 清屏并重置对话[/dim]
"""


def print_divider():
    console.print("─" * 60, style="dim")


def main():
    check_environment()

    console.print(Panel(WELCOME_TEXT, border_style="cyan"))

    conversation_history = []

    while True:
        try:
            user_input = console.input("\n[bold yellow]👤 你: [/bold yellow]").strip()

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("\n[bold cyan]👋 再见！[/bold cyan]")
                break
            if user_input.lower() == "clear":
                conversation_history = []
                os.system("cls" if os.name == "nt" else "clear")
                console.print(Panel(WELCOME_TEXT, border_style="cyan"))
                continue

            print_divider()
            console.print("[dim]🤔 思考中...[/dim]")

            messages = list(conversation_history)
            messages.append(HumanMessage(content=user_input))

            state = {
                "messages": messages,
                # 意图与路由
                "intent": None,
                "intent_reason": None,
                "confidence": None,
                "current_agent": None,
                # 任务状态
                "task_status": "pending",
                "retry_count": 0,
                "max_retries": 2,
                # 消息参数
                "sender": None,
                "receiver": None,
                "message_content": None,
                "platform": None,
                "group_id": None,
                "scheduled_time": None,
                # 搜索参数
                "search_query": None,
                "search_results": None,
                "search_source": None,
                # 日报参数
                "report_name": None,
                "report_department": None,
                "report_date": None,
                "report_content": None,
                "report_today_work": None,
                "report_tomorrow_plan": None,
                "report_issues": None,
                "report_notes": None,
                "report_file_path": None,
                "report_template": None,
                # 知识库参数
                "knowledge_results": None,
                "knowledge_file_name": None,
                "knowledge_file_content": None,
                # 校验与兜底
                "missing_params": None,
                "needs_clarification": False,
                "clarification_question": None,
                # 多 Agent 协作
                "agent_history": [],
                "supervisor_decision": None,
                "next_agent": None,
                # 执行结果
                "final_output": None,
                "error": None,
            }

            result = run_agent(state)

            final_output = result.get("final_output")
            error = result.get("error")

            conversation_history.append(HumanMessage(content=user_input))
            if final_output:
                conversation_history.append(AIMessage(content=final_output))
            elif error:
                conversation_history.append(AIMessage(content=f"错误: {error}"))

            console.print()

            if error:
                console.print(Panel(
                    f"[red]❌ {final_output}[/red]",
                    title="[red]执行结果",
                    border_style="red",
                ))
            elif final_output:
                # 检测是否包含追问信息（人机兜底）
                if "⚠️" in final_output:
                    console.print(Panel(
                        Markdown(final_output),
                        title="[yellow]⚠️ 需要补充信息",
                        border_style="yellow",
                    ))
                else:
                    console.print(Panel(
                        Markdown(final_output),
                        title="[green]✅ 执行结果",
                        border_style="green",
                    ))
            else:
                console.print("[yellow]未产生输出[/yellow]")

            print_divider()

        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]👋 再见！[/bold cyan]")
            break
        except Exception as e:
            logger.error(f"运行时错误: {e}")
            console.print(Panel(
                f"[red]发生错误: {e}[/red]",
                title="[red]错误",
                border_style="red",
            ))
            print_divider()


if __name__ == "__main__":
    main()
