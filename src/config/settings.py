from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ============================================================
    # LLM 配置（请先在 .env 中配置，或在 src/api/llm.py 中直接指定）
    # ============================================================
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_context_window_tokens: int = Field(default=32768, ge=1024)
    llm_output_tokens: int = Field(default=4096, ge=128)
    llm_context_safety_tokens: int = Field(default=1024, ge=128)
    context_observation_tokens: int = Field(default=2048, ge=128)
    context_recent_turns: int = Field(default=3, ge=1)

    # 总时限包含排队、模型、工具与记忆操作；应用入口通过工作进程强制执行。
    task_timeout_seconds: float = Field(default=120, gt=0)
    tool_timeout_seconds: float = Field(default=30, gt=0)
    llm_timeout_seconds: float = Field(default=30, gt=0)
    embedding_timeout_seconds: float = Field(default=10, gt=0)
    http_connect_timeout_seconds: float = Field(default=5, gt=0)
    http_read_timeout_seconds: float = Field(default=10, gt=0)
    readonly_timeout_retries: int = Field(default=1, ge=0, le=2)

    # ============================================================
    # 消息渠道配置
    # ============================================================
    # QQ 机器人 (OneBot 协议)
    qq_bot_ws_url: str = "ws://127.0.0.1:6700"
    qq_bot_http_url: str = "http://127.0.0.1:5700"
    qq_bot_access_token: str = ""

    # 企业微信机器人（预留）
    wecom_webhook_url: str = ""
    wecom_corp_id: str = ""
    wecom_agent_id: str = ""
    wecom_secret: str = ""

    # 钉钉机器人（预留）
    dingtalk_webhook_url: str = ""
    dingtalk_access_token: str = ""

    # 邮件配置（预留）
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""

    # ============================================================
    # 工作日报配置
    # ============================================================
    daily_report_template: str = "./src/config/templates/daily_report_simple.md"
    report_output_dir: str = "./data/reports"

    # ============================================================
    # 知识库配置
    # ============================================================
    knowledge_base_dir: str = "./data/knowledge_base"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = ""  # 嵌入模型 base_url，空则走 OpenAI 默认
    embedding_api_key: str = ""  # 嵌入模型独立 key（如千问/DashScope），空则复用 openai_api_key
    chunk_size: int = 500
    chunk_overlap: int = 50
    similarity_threshold: float = 0.3  # 语义检索相似度阈值，低于则视为未命中
    enable_ocr: bool = False  # 扫描 PDF 是否启用 OCR（需安装 rapidocr-onnxruntime）

    # ============================================================
    # 存储配置
    # ============================================================
    database_path: str = "data/db/office_agent.db"
    action_ledger_path: str = "data/db/action_ledger.db"
    vector_store_path: str = "data/db/vector_store"

    # ============================================================
    # 记忆系统配置
    # ============================================================
    redis_url: str = ""  # redis://localhost:6379/0，留空则使用内存回退
    memory_enabled: bool = True
    memory_short_term_ttl: int = 1800  # 短期记忆 TTL（秒）
    memory_short_term_max_turns: int = Field(default=20, ge=4)  # 兼容旧名称，单位为消息条数
    memory_summary_trigger_turns: int = Field(default=10, ge=4)  # 单位为消息条数
    memory_summary_max_chars: int = Field(default=800, ge=100)
    memory_long_term_top_k: int = 5
    memory_fact_confidence_threshold: float = 0.6
    memory_vector_store_path: str = "data/db/memory_vectors"

    # ============================================================
    # 定时任务配置（预留）
    # ============================================================
    scheduler_enabled: bool = False
    scheduler_log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_context_settings(self):
        if self.llm_output_tokens + self.llm_context_safety_tokens >= self.llm_context_window_tokens:
            raise ValueError("模型窗口必须大于输出预留与安全余量之和")
        if self.memory_summary_trigger_turns > self.memory_short_term_max_turns:
            raise ValueError("摘要触发消息数不能大于短期记忆消息上限")
        return self


settings = Settings()
