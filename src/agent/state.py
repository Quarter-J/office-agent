"""
Agent 全局状态定义

依据需求文档设计，包含：
  - 用户原始指令与解析参数
  - 任务执行进度与状态
  - 工具调用结果与异常信息
  - 上下文记忆与历史快照
"""

from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages


class TaskStatus:
    """任务状态常量"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    CONTEXT_ERROR = "context_error"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class AgentState(TypedDict):
    # ============================================================
    # 对话基础
    # ============================================================
    messages: Annotated[list, add_messages]

    # ============================================================
    # 意图与路由（Supervisor 产出）
    # ============================================================
    intent: Optional[str]              # 分类意图
    intent_reason: Optional[str]       # 意图分类原因
    confidence: Optional[float]        # 意图置信度
    current_agent: Optional[str]       # 当前执行的子 Agent 名称

    # ============================================================
    # 任务状态管理
    # ============================================================
    task_status: Optional[str]         # pending / running / success / failed / timeout / unknown / conflict / context_error
    retry_count: int                   # 当前重试次数
    max_retries: int                   # 最大重试次数
    tool_rounds: int                  # 本轮工具批次计数（不含历史消息）
    delegated_task: Optional[str]     # 本轮委派的补充说明
    task_id: str                      # 同一任务重试时保持；新用户轮次重新分配
    task_turn_id: Optional[str]
    execution_id: str                 # 单次执行拥有者，用于进程中断后的记录恢复
    conflicts: Optional[list]         # 冲突双方、依据与裁决
    resource_versions: Optional[dict] # 调用方已读取文件的绝对路径 -> SHA-256（不存在为 None）

    # ============================================================
    # 消息发送参数（MessageAgent 使用）
    # ============================================================
    sender: Optional[str]
    receiver: Optional[str]
    message_content: Optional[str]
    platform: Optional[str]
    group_id: Optional[str]
    scheduled_time: Optional[str]

    # ============================================================
    # 搜索参数（SearchAgent 使用）
    # ============================================================
    search_query: Optional[str]
    search_results: Optional[list]
    search_source: Optional[str]

    # ============================================================
    # 日报参数（ReportAgent 使用）
    # ============================================================
    report_name: Optional[str]
    report_department: Optional[str]
    report_date: Optional[str]
    report_content: Optional[str]
    report_today_work: Optional[str]
    report_tomorrow_plan: Optional[str]
    report_issues: Optional[str]
    report_notes: Optional[str]
    report_file_path: Optional[str]
    report_template: Optional[str]

    # ============================================================
    # 知识库参数（SearchAgent 使用）
    # ============================================================
    knowledge_results: Optional[list]
    knowledge_file_name: Optional[str]
    knowledge_file_content: Optional[str]

    # ============================================================
    # 校验与兜底
    # ============================================================
    missing_params: Optional[list]
    needs_clarification: bool
    clarification_question: Optional[str]

    # ============================================================
    # 多 Agent 协作记录
    # ============================================================
    agent_history: Optional[list]      # 已执行的子 Agent 列表
    supervisor_decision: Optional[str] # 主管决策记录
    next_agent: Optional[str]          # 链式调用：下一个要执行的 Agent（如 content → message）
    generated_content: Optional[str]  # 完整生成结果，发送失败/追问时保留

    # ============================================================
    # 记忆系统
    # ============================================================
    session_id: Optional[str]           # 会话标识（短期记忆）
    user_id: Optional[str]              # 用户标识（长期记忆）
    memory_context: Optional[str]       # 注入的格式化记忆上下文
    memory_loaded: bool                 # 记忆是否已加载
    memory_summarized: bool             # 模型视图只带摘要及近期完整轮次
    memory_recent_turns: int            # 尚未摘要的用户轮次数，避免过早裁剪

    # ============================================================
    # 执行结果
    # ============================================================
    final_output: Optional[str]
    error: Optional[str]
