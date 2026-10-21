"""导入即注册所有内置解析器"""

from src.rag.parsers.text import TextParser
from src.rag.parsers.markdown import MarkdownParser
from src.rag.parsers.docx import DocxParser
from src.rag.parsers.pdf import PdfParser
from src.rag.parsers.pptx import PptxParser
from src.rag.parsers.xlsx import XlsxParser

__all__ = [
    "TextParser",
    "MarkdownParser",
    "DocxParser",
    "PdfParser",
    "PptxParser",
    "XlsxParser",
]
