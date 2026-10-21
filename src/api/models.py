"""
API 数据模型 - 定义与 LLM 交互的请求/响应数据结构
"""

from typing import Optional
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """LLM 响应统一封装"""
    content: str
    finish_reason: str = "stop"
    usage: Optional[dict] = None
    error: Optional[str] = None
