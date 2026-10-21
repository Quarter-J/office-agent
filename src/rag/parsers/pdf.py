from __future__ import annotations

from pathlib import Path

from src.rag.parsers.base import Block, DocumentParser, register
from src.utils.logger import logger


@register
class PdfParser(DocumentParser):
    supported = (".pdf",)

    def parse(self, path: Path) -> list:
        try:
            import pypdf
        except ImportError:
            logger.warning("缺少 pypdf 依赖，跳过 .pdf 解析")
            return []

        reader = pypdf.PdfReader(str(path))
        blocks: list = []
        has_text = False
        for idx, page in enumerate(reader.pages):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                text = ""
            if text:
                has_text = True
                blocks.append(Block(type="paragraph", text=text, meta={"page": idx + 1}))

        if has_text:
            return blocks
        return self._ocr(path)

    def _ocr(self, path: Path) -> list:
        """无文字层 → OCR 回退（仅当 enable_ocr 且装了 OCR 库）"""
        from src.config.settings import settings
        if not settings.enable_ocr:
            logger.warning(f"PDF 无文字层且未启用 OCR（ENABLE_OCR=False），跳过: {path.name}")
            return []
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
            result, _ = ocr(str(path))
            if not result:
                return []
            text = "\n".join(line[1] for line in result if line and line[1])
            return [Block(type="paragraph", text=text)] if text else []
        except ImportError:
            logger.warning("未安装 rapidocr-onnxruntime，跳过扫描 PDF OCR")
            return []
        except Exception as e:
            logger.warning(f"PDF OCR 失败: {e}")
            return []
