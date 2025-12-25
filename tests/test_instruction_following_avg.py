from __future__ import annotations

import unittest

from src.eval.metrics.at_k import compute_avg_at_k


class TestInstructionFollowingAvg(unittest.TestCase):
    def test_compute_avg_at_k(self) -> None:
        rows = [
            (0, 0, True),
            (0, 1, False),
            (1, 0, False),
            (1, 1, False),
        ]
        metrics = compute_avg_at_k(rows, (1, 2))
        self.assertEqual(metrics, {"avg@1": 0.5, "avg@2": 0.25})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
