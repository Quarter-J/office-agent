"""
解析器基类与注册表

将不同格式文档统一解析为带类型/层级/元数据的 Block 列表。
切分器（chunker）只认 Block，不关心原始格式；新增格式 = 新增一个 parser，
切分/索引/检索层都不需要改动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.logger import logger


@dataclass
class Block:
    """文档结构化块：切分的最小语义单元"""
    type: str                     # heading / paragraph / table / list / code
    text: str
    level: int = 0                # 标题层级，正文 = 0
    meta: dict = field(default_factory=dict)  # page/slide/row/heading_path 等


@dataclass
class ParsedDocument:
    """解析后的单个文档"""
    source: str                   # 相对知识库目录的路径
    path: str                     # 绝对路径
    fmt: str                      # 文件后缀（含点），如 .md/.docx
    blocks: list


class DocumentParser(ABC):
    """文档解析器抽象基类：子类声明 supported 后缀并实现 parse"""
    supported: tuple = ()

    @abstractmethod
    def parse(self, path: Path) -> list:
        ...


_PARSERS: dict = {}


def register(parser_cls: type) -> type:
    """装饰器：把 parser 实例注册进全局表（parser 需无参构造）"""
    instance = parser_cls()
    for ext in parser_cls.supported:
        _PARSERS[ext.lower()] = instance
    return parser_cls


def get_parser(suffix: str) -> Optional[DocumentParser]:
    return _PARSERS.get(suffix.lower())


def load_file(path: Path, base_dir: Path) -> Optional[ParsedDocument]:
    """解析单个文件为 ParsedDocument（无对应 parser 或解析失败返回 None）"""
    parser = get_parser(path.suffix)
    if parser is None:
        return None
    try:
        blocks = parser.parse(path)
    except Exception as e:
        logger.warning(f"解析文档失败: {path} - {e}")
        return None
    if not blocks:
        return None
    try:
        source = str(path.relative_to(base_dir))
    except ValueError:
        source = path.name
    return ParsedDocument(
        source=source, path=str(path), fmt=path.suffix.lower(), blocks=blocks
    )


def load_documents(kb_dir: Optional[str] = None) -> list:
    """加载知识库目录下的所有支持文档"""
    from src.config.settings import settings
    target = Path(kb_dir or settings.knowledge_base_dir)
    if not target.exists():
        logger.warning(f"知识库目录不存在: {target}")
        return []

    docs: list = []
    for fp in sorted(target.rglob("*.*")):
        if not fp.is_file():
            continue
        doc = load_file(fp, target)
        if doc is not None:
            docs.append(doc)

    logger.info(f"共加载 {len(docs)} 个文档")
    return docs
