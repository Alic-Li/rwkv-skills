from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

from evalplus.data.mbpp import get_mbpp, get_mbpp_plus

from ..data_utils import write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY

_QUESTION_REPLACE = ("    ", "\t")


def _iter_mbpp_records(problems: Mapping[str, dict], *, keep_plus_inputs: bool) -> Iterable[dict]:
    def _normalize_jsonable(value):
        if isinstance(value, complex):
            return str(value)
        if isinstance(value, dict):
            return {k: _normalize_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize_jsonable(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_normalize_jsonable(v) for v in value)
        return value

    for task_id, problem in problems.items():
        payload = _normalize_jsonable(dict(problem))
        prompt = payload.get("prompt") or payload.get("question") or ""
        if isinstance(prompt, str):
            prompt_tab = prompt.replace(*_QUESTION_REPLACE)
            payload["prompt"] = prompt_tab
            payload["question"] = prompt_tab
        else:
            payload["prompt"] = ""
            payload["question"] = ""
        payload.setdefault("task_id", str(task_id))
        if not keep_plus_inputs:
            payload.pop("base_input", None)
            payload.pop("plus_input", None)
        yield payload


@CODE_GENERATION_REGISTRY.register("mbpp")
def prepare_mbpp(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("mbpp 目前仅提供 test split")
    dataset_dir = (output_root / "mbpp").expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (output_root / "cache").expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import evalplus.data.utils as evalplus_data_utils

    evalplus_data_utils.CACHE_DIR = str(cache_root)
    target = dataset_dir / f"{split}.jsonl"
    problems = get_mbpp()
    write_jsonl(target, _iter_mbpp_records(problems, keep_plus_inputs=False))
    return [target]


@CODE_GENERATION_REGISTRY.register("mbpp_plus")
def prepare_mbpp_plus(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("mbpp_plus 目前仅提供 test split")
    dataset_dir = (output_root / "mbpp_plus").expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (output_root / "cache").expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import evalplus.data.utils as evalplus_data_utils

    evalplus_data_utils.CACHE_DIR = str(cache_root)
    target = dataset_dir / f"{split}.jsonl"
    problems = get_mbpp_plus()
    write_jsonl(target, _iter_mbpp_records(problems, keep_plus_inputs=True))
    return [target]


__all__ = ["prepare_mbpp", "prepare_mbpp_plus"]
