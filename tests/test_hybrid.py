from unittest import TestCase

from src.rag.bm25 import BM25Index
from src.rag.hybrid import rrf_fuse
from src.rag.eval_compare import hit_at_k, summary, run_comparison
from src.rag.rerank import build_prompt, parse_ranking
from src.rag.vector_store import ScoredChunk


def _hit(text, source, score=1.0):
    return ScoredChunk(text=text, meta={"source": source}, score=score)


class BM25Tests(TestCase):
    def test_ranks_matching_doc_first(self):
        bm = BM25Index(tokenizer=lambda t: t.split())
        bm.index([
            {"text": "the quick brown fox", "meta": {"source": "a.md"}},
            {"text": "the lazy dog", "meta": {"source": "b.md"}},
        ])
        hits = bm.search("quick fox", k=2)
        self.assertEqual([h.meta["source"] for h in hits], ["a.md"])

    def test_ranks_both_matching_docs_by_score(self):
        bm = BM25Index(tokenizer=lambda t: t.split())
        bm.index([
            {"text": "fox fox fox", "meta": {"source": "a.md"}},
            {"text": "fox", "meta": {"source": "b.md"}},
        ])
        hits = bm.search("fox", k=2)
        self.assertEqual([h.meta["source"] for h in hits], ["a.md", "b.md"])

    def test_empty_query(self):
        bm = BM25Index(tokenizer=lambda t: t.split())
        bm.index([{"text": "hello world", "meta": {"source": "a.md"}}])
        self.assertEqual(bm.search("", k=5), [])


class RrfTests(TestCase):
    def test_fusion_ranks_shared_item_first(self):
        fused = rrf_fuse([
            [_hit("x", "shared"), _hit("y", "only_a")],
            [_hit("x", "shared"), _hit("z", "only_b")],
        ])
        self.assertEqual(fused[0].meta["source"], "shared")

    def test_weighted_fusion_can_prefer_one_route(self):
        fused = rrf_fuse([
            [_hit("x", "vector_only")],
            [_hit("y", "keyword_only")],
        ], weights=[2.0, 1.0])
        self.assertEqual(fused[0].meta["source"], "vector_only")


class MetricsTests(TestCase):
    def test_hit_at_k(self):
        self.assertAlmostEqual(hit_at_k([1, 2, None], 1), 1 / 3)
        self.assertAlmostEqual(hit_at_k([1, 2, None], 3), 2 / 3)
        self.assertEqual(hit_at_k([], 3), 0.0)

    def test_summary(self):
        s = summary([1, None], ks=(1, 3))
        self.assertEqual(s["queries"], 2)
        self.assertEqual(s["hit@1"], 0.5)
        self.assertEqual(s["mrr"], 0.5)


class RerankTests(TestCase):
    def test_parse_ranking(self):
        self.assertEqual(parse_ranking("2,1,3"), [2, 1, 3])
        self.assertEqual(parse_ranking("编号：3，1，2。"), [3, 1, 2])
        self.assertEqual(parse_ranking("无关文本"), [])
        self.assertEqual(parse_ranking("1,1,2"), [1, 2])  # 去重保序

    def test_build_prompt_contains_query_and_candidates(self):
        cands = [_hit("片段A", "a.md"), _hit("片段B", "b.md")]
        p = build_prompt("查询X", cands)
        self.assertIn("查询X", p)
        self.assertIn("[1] 片段A", p)
        self.assertIn("[2] 片段B", p)


class ComparisonTests(TestCase):
    def test_run_comparison(self):
        test_set = [
            {"query": "q1", "relevant": ["a.md"]},
            {"query": "q2", "relevant": ["b.md"]},
        ]
        retrievers = {
            "vec": lambda q, k: [_hit("a", "a.md")] if q == "q1" else [_hit("x", "x.md")],
            "hyb": lambda q, k: [_hit("a", "a.md")] if q == "q1" else [_hit("b", "b.md")],
        }
        result = run_comparison(test_set, retrievers, top_k=3, ks=(1, 3))
        self.assertEqual(result["vec"]["summary"]["hit@1"], 0.5)
        self.assertEqual(result["hyb"]["summary"]["hit@1"], 1.0)
