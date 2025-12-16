from __future__ import annotations

"""Prepare HumanEvalFix (HumanEvalPack, Python split) as JSONL."""

from pathlib import Path
from typing import Iterator

from ..data_utils import configure_hf_home, write_jsonl

configure_hf_home()

from datasets import load_dataset  # type: ignore
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY


def _format_bugfix_prompt(
    prompt: str | None,
    buggy_solution: str | None,
    entry_point: str | None,
) -> str:
    prompt_text = (prompt or "").rstrip()
    buggy = (buggy_solution or "").rstrip()
    entry = (entry_point or "").strip()

    parts: list[str] = []
    if prompt_text:
        parts.append(prompt_text)
    if buggy:
        parts.append("# Buggy implementation:")
        parts.append(buggy)
    if entry:
        parts.append(f"# Fix the function `{entry}` so it passes all tests.")
    return "\n".join(parts).strip()


def _iter_records(split: str) -> Iterator[dict]:
    dataset = load_dataset("bigcode/humanevalpack", "python", split=split)
    for row in dataset:
        prompt = row.get("prompt") or row.get("instruction") or ""
        buggy_solution = row.get("buggy_solution") or ""
        if not isinstance(prompt, str):
            prompt = ""
        if not isinstance(buggy_solution, str):
            buggy_solution = str(buggy_solution)
        entry_point = row.get("entry_point") or ""
        if not isinstance(entry_point, str):
            entry_point = str(entry_point)

        prompt = _format_bugfix_prompt(prompt, buggy_solution, entry_point)
        yield {
            "task_id": row.get("task_id"),
            "prompt": prompt,
            "canonical_solution": row.get("canonical_solution"),
            "buggy_solution": row.get("buggy_solution"),
            "entry_point": row.get("entry_point"),
            "test": row.get("test"),
            "example_test": row.get("example_test"),
        }


@CODE_GENERATION_REGISTRY.register("human_eval_fix")
def prepare_human_eval_fix(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "human_eval_fix"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split))
    return [target]


__all__ = ["prepare_human_eval_fix"]
