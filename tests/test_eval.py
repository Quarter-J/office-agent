from unittest import TestCase
from unittest.mock import patch

from src.rag.eval import (
    match_relevant,
    first_relevant_rank,
    compute_metrics,
    run_eval,
)


class MatchTests(TestCase):
    def test_string_matches_source(self):
        self.assertTrue(match_relevant("a.md", {"source": "a.md"}))
        self.assertFalse(match_relevant("a.md", {"source": "b.md"}))

    def test_dict_matches_source_only(self):
        self.assertTrue(match_relevant({"source": "a.md"}, {"source": "a.md"}))
        self.assertFalse(match_relevant({"source": "a.md"}, {"source": "b.md"}))

    def test_dict_matches_section(self):
        self.assertTrue(match_relevant(
            {"source": "a.md", "section": "用法"}, {"source": "a.md", "section": "用法"}))
        self.assertFalse(match_relevant(
            {"source": "a.md", "section": "用法"}, {"source": "a.md", "section": "安装"}))

    def test_dict_matches_heading_path(self):
        self.assertTrue(match_relevant(
            {"source": "a.md", "section": "自动装配"}, {"source": "a.md", "heading_path": "自动装配"}))
        self.assertFalse(match_relevant(
            {"source": "a.md", "section": "自动装配"}, {"source": "b.md", "heading_path": "自动装配"}))


class FirstRelevantRankTests(TestCase):
    def test_found_at_rank(self):
        metas = [{"source": "x.md"}, {"source": "a.md"}, {"source": "b.md"}]
        self.assertEqual(first_relevant_rank(["a.md"], metas, 3), 2)

    def test_not_found(self):
        self.assertIsNone(first_relevant_rank(["z.md"], [{"source": "a.md"}], 3))

    def test_respects_k(self):
        metas = [{"source": "x.md"}, {"source": "a.md"}]
        self.assertIsNone(first_relevant_rank(["a.md"], metas, 1))


class ComputeMetricsTests(TestCase):
    def test_empty(self):
        self.assertEqual(
            compute_metrics([], 3),
            {"k": 3, "queries": 0, "hits": 0, "hit_at_k": 0.0, "mrr": 0.0},
        )

    def test_perfect(self):
        self.assertEqual(
            compute_metrics([1, 1], 3),
            {"k": 3, "queries": 2, "hits": 2, "hit_at_k": 1.0, "mrr": 1.0},
        )

    def test_partial(self):
        # rank 2 -> mrr 0.5；rank None -> 未命中。hit=1/2, mrr=(0.5+0)/2=0.25
        self.assertEqual(
            compute_metrics([2, None], 3),
            {"k": 3, "queries": 2, "hits": 1, "hit_at_k": 0.5, "mrr": 0.25},
        )


class RunEvalTests(TestCase):
    def test_run_eval_uses_vector_store_and_computes_metrics(self):
        from src.rag.vector_store import ScoredChunk

        test_set = [
            {"query": "q1", "relevant": ["a.md"]},
            {"query": "q2", "relevant": ["b.md"]},
        ]

        def fake_search(query, top_k=3, threshold=None):
            if query == "q1":
                return [ScoredChunk(text="t1", meta={"source": "a.md"}, score=0.9)]
            return [ScoredChunk(text="t2", meta={"source": "x.md"}, score=0.8)]

        with patch("src.rag.vector_store.search", side_effect=fake_search):
            result = run_eval(test_set, top_k=3)

        self.assertEqual(result["metrics"]["hit_at_k"], 0.5)
        self.assertEqual(result["metrics"]["mrr"], 0.5)
        self.assertTrue(result["per_query"][0]["hit"])
        self.assertFalse(result["per_query"][1]["hit"])
