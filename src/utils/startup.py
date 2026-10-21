"""
启动检查模块

在 Agent 启动时检查运行环境和依赖。
"""

import warnings
from pathlib import Path
from src.utils.logger import logger
from src.config.settings import settings


def check_environment():
    """检查运行环境，输出状态信息"""
    logger.info("=" * 40)
    logger.info("Office Agent v2.0 启动中...")
    logger.info("=" * 40)

    # 检查 LLM 配置
    _check_llm_config()

    # 创建必要目录
    _ensure_directories()

    # 检查向量索引
    _check_vector_store()

    logger.info("环境检查完成")


def _check_llm_config():
    """检查 LLM 配置状态"""
    api_key = settings.openai_api_key
    base_url = settings.openai_base_url
    model = settings.llm_model_name

    if not api_key:
        logger.warning("API Key 未设置！请在 .env 文件中配置 OPENAI_API_KEY")
        logger.warning("部分功能将使用规则降级运行")
        return

    logger.info(f"LLM 配置已加载: model={model}, base_url={base_url}")
    logger.info("API Key 已设置 ✅")


def _check_vector_store():
    """检查向量索引状态"""
    vector_path = Path(settings.vector_store_path)
    if vector_path.exists() and any(vector_path.iterdir()):
        logger.info("知识库向量索引已就绪 ✅")
    else:
        logger.info("知识库向量索引未构建，运行 python build_index.py 构建")


def _ensure_directories():
    """确保运行所需目录存在"""
    directories = [
        "data/db",
        "data/logs",
        "data/reports",
        "data/knowledge_base",
        "logs",
    ]
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.debug(f"目录已确认: {d}")
