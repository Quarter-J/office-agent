"""
工具层（LangChain 职责）

提供：
  - 消息发送工具（QQ / 微信 / 预留钉钉/企业微信/邮件）
  - 日报生成工具（生成 / 查询 / 汇总）
  - 信息检索工具（知识库检索 / 语义检索 / RAG 问答）
  - Prompt 模板管理（所有 LLM 交互的 Prompt）
  - 文档加载与向量检索（本地文件 / FAISS）
"""

from src.tools.message import (
    send_qq_message,
    send_wechat_message,
    send_dingtalk_message,
    send_wecom_message,
    send_email,
    get_qq_number_by_nickname,
)

from src.tools.report import (
    generate_daily_report,
    list_recent_reports,
    get_report_by_date,
    aggregate_weekly_report,
)

from src.tools.search import (
    knowledge_base_search,
    rag_search,
    lookup_entity,
    add_to_knowledge_base,
)

from src.tools.prompts import (
    INTENT_CLASSIFIER_TEMPLATE,
    SEND_MESSAGE_EXTRACT_TEMPLATE,
    REPORT_EXTRACT_TEMPLATE,
    SEARCH_EXTRACT_TEMPLATE,
    KNOWLEDGE_EXTRACT_TEMPLATE,
    ADD_KNOWLEDGE_EXTRACT_TEMPLATE,
    NAME_EXTRACT_TEMPLATE,
    CHAT_SYSTEM_TEMPLATE,
    DAILY_REPORT_MD_TEMPLATE,
    RAG_QA_TEMPLATE,
)

from src.tools.retriever import (
    load_local_documents,
    build_vector_store,
    semantic_search as vector_semantic_search,
)

__all__ = [
    # 消息工具
    "send_qq_message",
    "send_wechat_message",
    "send_dingtalk_message",
    "send_wecom_message",
    "send_email",
    "get_qq_number_by_nickname",
    # 日报工具
    "generate_daily_report",
    "list_recent_reports",
    "get_report_by_date",
    "aggregate_weekly_report",
    # 检索工具
    "knowledge_base_search",
    "rag_search",
    "lookup_entity",
    "add_to_knowledge_base",
    # Prompt 模板
    "INTENT_CLASSIFIER_TEMPLATE",
    "SEND_MESSAGE_EXTRACT_TEMPLATE",
    "REPORT_EXTRACT_TEMPLATE",
    "SEARCH_EXTRACT_TEMPLATE",
    "KNOWLEDGE_EXTRACT_TEMPLATE",
    "ADD_KNOWLEDGE_EXTRACT_TEMPLATE",
    "NAME_EXTRACT_TEMPLATE",
    "CHAT_SYSTEM_TEMPLATE",
    "DAILY_REPORT_MD_TEMPLATE",
    "RAG_QA_TEMPLATE",
    # 文档检索
    "load_local_documents",
    "build_vector_store",
    "vector_semantic_search",
]
