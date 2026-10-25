"""
FAISS IndexFlatL2 检索延迟随语料规模 N 的变化曲线（合成向量，维度取当前索引 1024）。

测两段：build+search（在线检索）与 read_index（磁盘加载，即 vector_store 每次 search 前
的 load_index）。embedding 往返 ~150ms 与 N 无关，作为常数单列，最终 P95 ≈ 150 + 载入 + 搜索。
"""

import time
import numpy as np
import faiss

D = 1024
SIZES = [1_000, 10_000, 50_000, 100_000, 200_000]
K = 5
NQ = 20


def bench(N):
    rng = np.random.default_rng(0)
    xb = rng.random((N, D), dtype=np.float32)
    xq = rng.random((NQ, D), dtype=np.float32)

    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(D)
    index.add(xb)
    t_build = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for i in range(NQ):
        index.search(xq[i:i + 1], K)
    t_search = (time.perf_counter() - t0) / NQ * 1000

    faiss.write_index(index, "data/db/_bench_scale.faiss")
    t0 = time.perf_counter()
    faiss.read_index("data/db/_bench_scale.faiss")
    t_load = (time.perf_counter() - t0) * 1000

    return t_build, t_search, t_load


if __name__ == "__main__":
    print(f"{'N(块)':>10}{'build(ms)':>10}{'search/次(ms)':>14}{'load(ms)':>10}{'≈P95(ms)':>10}")
    for N in SIZES:
        b, s, l = bench(N)
        p95 = 150 + s + l  # embedding 往返 ~150ms 常数
        print(f"{N:>10}{b:>10.1f}{s:>14.3f}{l:>10.1f}{p95:>10.0f}")
