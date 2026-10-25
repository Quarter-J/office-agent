"""
检索延迟统计：分位数与汇总。

纯函数、无外部依赖，便于离线单测。P50/P90/P95/P99 采用线性插值法
（与 numpy 默认 percentile 一致）。延迟样本单位统一为毫秒。
"""

from __future__ import annotations

from typing import Sequence


def percentile(samples: Sequence[float], p: float) -> float:
    """返回第 p 百分位数（线性插值）。p 取值 0~100，空样本抛 ValueError。"""
    if not samples:
        raise ValueError("样本为空，无法计算分位数")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"分位点 p 必须在 0~100 之间，收到 {p}")
    xs = sorted(float(x) for x in samples)
    if p == 0.0:
        return xs[0]
    if p == 100.0:
        return xs[-1]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = lo + 1
    if hi >= len(xs):
        return xs[-1]
    frac = rank - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def latency_summary(samples_ms: Sequence[float]) -> dict:
    """把一串耗时（毫秒）聚合成 {count, mean/min/max, p50/p90/p95/p99}。"""
    xs = [float(x) for x in samples_ms]
    if not xs:
        return {"count": 0}
    return {
        "count": len(xs),
        "mean_ms": round(sum(xs) / len(xs), 2),
        "min_ms": round(min(xs), 2),
        "max_ms": round(max(xs), 2),
        "p50_ms": round(percentile(xs, 50.0), 2),
        "p90_ms": round(percentile(xs, 90.0), 2),
        "p95_ms": round(percentile(xs, 95.0), 2),
        "p99_ms": round(percentile(xs, 99.0), 2),
    }
