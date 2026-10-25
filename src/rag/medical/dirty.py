"""
脏数据检测与拦截（纯规则，可离线测试）。

判定口径（满足任一即判脏拦截）：
  empty      规范化后为空或纯空白
  too_short  规范化后不足 MIN_CONTENT_LEN 字符，无知识价值
  garbled    含替换符 U+FFFD / 空字节，或非法字符占比过高
"""

from __future__ import annotations

MIN_CONTENT_LEN = 20

_PUNCT = set("，。、；：？！…—·（）()[]{}《》<>\"'“”‘’/\\|-_+=%℃.,;:!?@#$%^&*~·")


def _is_valid_char(ch: str) -> bool:
    return ch.isspace() or ch.isalnum() or ch in _PUNCT


def detect_dirty(text: str):
    """返回 (是否脏, 原因)。原因 None 表示干净。"""
    if not text or not text.strip():
        return True, "empty"
    if len(text.strip()) < MIN_CONTENT_LEN:
        return True, "too_short"
    if "�" in text or "\x00" in text:
        return True, "garbled"
    bad = sum(1 for ch in text if not _is_valid_char(ch))
    if bad / len(text) > 0.3:
        return True, "garbled"
    return False, None
