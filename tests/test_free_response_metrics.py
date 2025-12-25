from __future__ import annotations

import unittest

from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k


class TestFreeResponsePassK(unittest.TestCase):
    def test_compute_pass_at_k_exact(self) -> None:
        rows = [
            (0, 0, True),
            (0, 1, True),
            (1, 0, False),
            (1, 1, False),
        ]
        metrics = compute_pass_at_k(rows, (1, 2))
        self.assertEqual(metrics, {"pass@1": 0.5, "pass@2": 0.5})


class TestFreeResponseAvgK(unittest.TestCase):
    def test_compute_avg_at_k_exact(self) -> None:
        rows = [
            (0, 0, True),
            (0, 1, False),
            (1, 0, False),
            (1, 1, False),
        ]
        metrics = compute_avg_at_k(rows, (1, 2))
        self.assertEqual(metrics, {"avg@1": 0.5, "avg@2": 0.25})
