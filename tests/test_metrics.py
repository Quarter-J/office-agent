from unittest import TestCase

from src.rag.metrics import percentile, latency_summary


class PercentileTests(TestCase):
    def test_single_sample_returns_itself(self):
        self.assertEqual(percentile([42.0], 95), 42.0)

    def test_boundaries(self):
        xs = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(xs, 0), 1)
        self.assertEqual(percentile(xs, 100), 5)

    def test_median_linear_interpolation(self):
        # 1..10 的中位数 = 5.5
        self.assertEqual(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50), 5.5)

    def test_known_p95(self):
        # 1..20：rank=(20-1)*0.95=18.05 -> 19 + 0.05*(20-19)=19.05
        self.assertAlmostEqual(percentile(list(range(1, 21)), 95), 19.05)

    def test_unsorted_input(self):
        self.assertEqual(percentile([5, 1, 3], 50), 3)

    def test_invalid_p_rejected(self):
        for p in (-1, 101, 100.1):
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    percentile([1, 2, 3], p)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            percentile([], 50)


class LatencySummaryTests(TestCase):
    def test_empty(self):
        self.assertEqual(latency_summary([]), {"count": 0})

    def test_summary_values(self):
        s = latency_summary([10, 20, 30, 40])
        self.assertEqual(s["count"], 4)
        self.assertEqual(s["mean_ms"], 25.0)
        self.assertEqual(s["min_ms"], 10.0)
        self.assertEqual(s["max_ms"], 40.0)
        self.assertEqual(s["p50_ms"], 25.0)
        self.assertEqual(s["p90_ms"], 37.0)
