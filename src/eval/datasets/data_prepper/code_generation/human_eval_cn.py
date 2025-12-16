from __future__ import annotations

"""Prepare HumanEval-CN (CodeGeeX HumanEval-X python split, bundled as fallback)."""

import re
from pathlib import Path
from typing import Iterator

from ..data_utils import dataset_cache_dir, download_file, read_jsonl, write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY

DATA_URL = "https://hf-mirror.com/datasets/zai-org/humaneval-x/resolve/main/data/python/data/humaneval.jsonl"


_DEF_RE = re.compile(r"def\s+(?P<name>[\w_]+)\s*\(")


def _extract_entry_point(payload: dict) -> str | None:
    """Try to recover entry_point from upstream fields when missing."""
    raw = payload.get("entry_point")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    for key in ("declaration", "prompt"):
        text = payload.get(key)
        if not isinstance(text, str):
            continue
        match = _DEF_RE.search(text)
        if match:
            return match.group("name")
    return None


def _iter_records(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("human_eval_cn 仅提供 test split")

    cache_dir = dataset_cache_dir(Path("data"), "human_eval_cn")
    source_path = cache_dir / "humaneval.jsonl"
    download_file(DATA_URL, source_path)
    for row in read_jsonl(source_path):
        entry_point = _extract_entry_point(row)
        yield {
            "task_id": row.get("task_id"),
            "prompt": row.get("prompt"),
            "canonical_solution": row.get("canonical_solution"),
            "entry_point": entry_point,
            "test": row.get("test"),
            "example_test": row.get("example_test"),
            "text": row.get("text"),
            "declaration": row.get("declaration"),
        }


@CODE_GENERATION_REGISTRY.register("human_eval_cn")
def prepare_human_eval_cn(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "human_eval_cn"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split))
    return [target]


__all__ = ["prepare_human_eval_cn"]
