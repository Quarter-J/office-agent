"""Top-k constraints shared by tool schemas and direct retrieval calls."""

from typing import Annotated
from pydantic import Field

MAX_TOP_K = 20
TopK = Annotated[int, Field(strict=True, ge=1, le=MAX_TOP_K)]


def validate_top_k(top_k: int) -> int:
    if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k 必须是 1 到 {MAX_TOP_K} 之间的整数")
    return top_k
