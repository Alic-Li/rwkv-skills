from __future__ import annotations

"""Prepare HumanEval+ dataset (EvalPlus release with additional test cases)."""

import os
from pathlib import Path
from typing import Iterator

from evalplus.data import get_human_eval_plus

from ..data_utils import dataset_cache_dir, write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY


def _iter_records(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("human_eval_plus 仅提供 test split")

    # EvalPlus 默认缓存到 ~/.cache，我们将其重定向到仓库下的 data/cache。
    cache_root = dataset_cache_dir(Path("data"), "human_eval_plus").parent
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    problems = get_human_eval_plus()
    for task_id, problem in problems.items():
        yield {
            "task_id": task_id,
            "prompt": problem.get("prompt", ""),
            "canonical_solution": problem.get("canonical_solution"),
            "entry_point": problem.get("entry_point"),
            "test": problem.get("test"),
            "contract": problem.get("contract"),
            "plus_input": problem.get("plus_input"),
            "atol": problem.get("atol"),
        }


@CODE_GENERATION_REGISTRY.register("human_eval_plus")
def prepare_human_eval_plus(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "human_eval_plus"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split))
    return [target]


__all__ = ["prepare_human_eval_plus"]
