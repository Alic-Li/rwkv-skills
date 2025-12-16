from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.metrics.code_generation.mbpp.evaluation import evaluate_mbpp


class TestMbppPlusEvaluation(unittest.TestCase):
    def test_mbpp_plus_deserializes_inputs_and_scores(self) -> None:
        problem = {
            "task_id": "Mbpp/106",
            "entry_point": "add_lists",
            "prompt": (
                "\"\"\"\n"
                "Write a function to append the given list to the given tuples.\n"
                "assert add_lists([5, 6, 7], (9, 10)) == (9, 10, 5, 6, 7)\n"
                "\"\"\"\n"
            ),
            "canonical_solution": "\n\ndef add_lists(test_list, test_tup):\n  return test_tup + tuple(test_list)\n",
            # JSON cannot represent tuples; they arrive as lists and must be deserialized for MBPP+.
            "base_input": [[[5, 6, 7], [9, 10]]],
            "plus_input": [[[], [1, 2, 3]]],
            "atol": 0.0,
        }
        sample = {
            "task_id": "Mbpp/106",
            "completion": problem["canonical_solution"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            problem_file = root / "mbpp_plus" / "test.jsonl"
            problem_file.parent.mkdir(parents=True, exist_ok=True)
            problem_file.write_text(json.dumps(problem) + "\n", encoding="utf-8")

            sample_file = root / "samples.jsonl"
            sample_file.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            metrics, details_path = evaluate_mbpp(
                sample_file=str(sample_file),
                k=(1,),
                n_workers=1,
                timeout=3.0,
                problem_file=str(problem_file),
            )

        self.assertEqual(float(metrics["pass@1"]), 1.0)
        self.assertEqual(float(metrics["base_pass@1"]), 1.0)
        self.assertTrue(str(details_path).endswith("_results.jsonl"))

