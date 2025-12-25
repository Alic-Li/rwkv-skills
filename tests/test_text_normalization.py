from __future__ import annotations

import unittest

from src.eval.metrics.free_response import _normalize_text
from src.eval.metrics.instruction_following.metrics import _build_loose_variants


class TestFreeResponseNormalizeText(unittest.TestCase):
    def test_normalize_text_collapses_whitespace(self) -> None:
        self.assertEqual(_normalize_text("a\tb\nc"), "a b c")

    def test_normalize_text_replaces_nbsp(self) -> None:
        self.assertEqual(_normalize_text("a\u00a0b"), "a b")

    def test_normalize_text_replaces_latex_space_escape(self) -> None:
        self.assertEqual(_normalize_text("a\\ b"), "a b")


class TestInstructionFollowingLooseVariants(unittest.TestCase):
    def test_build_loose_variants_splits_real_newlines(self) -> None:
        response = "line1\nline2\nline3"
        variants = _build_loose_variants(response)
        self.assertIn("line2\nline3", variants)
        self.assertIn("line1\nline2", variants)
        self.assertIn("line2", variants)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

