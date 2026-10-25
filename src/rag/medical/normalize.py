"""
规则归一化：纯规则清洗，无外部依赖、可离线测试。

- 全角字符 → 半角（中文医疗文档常见于从 Word/网页复制粘贴）
- 行内空白折叠、每行去首尾空白
- 常见剂量单位别名 → 符号（毫克→mg 等）
"""

from __future__ import annotations

import re

_UNIT_ALIASES = [
    ("摄氏度", "℃"),
    ("毫克", "mg"),
    ("微克", "μg"),
    ("毫升", "ml"),
]


def full_to_half(text: str) -> str:
    """全角字母/数字/标点与全角空格转半角。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:  # 全角 ASCII 可打印区
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def normalize_units(text: str) -> str:
    for src, dst in _UNIT_ALIASES:
        text = text.replace(src, dst)
    return text


def collapse_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    """规范化入口：全半角 → 单位 → 空白折叠。"""
    text = full_to_half(text)
    text = normalize_units(text)
    return collapse_whitespace(text).strip()
