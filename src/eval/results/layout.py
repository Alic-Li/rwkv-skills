"""Canonical output layout for evaluation artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.eval.scheduler.config import (
    RESULTS_ROOT,
    DEFAULT_LOG_DIR,
    DEFAULT_COMPLETION_DIR,
    DEFAULT_EVAL_RESULT_DIR,
    DEFAULT_RUN_LOG_DIR,
)
from src.eval.scheduler.dataset_utils import canonical_slug, safe_slug


COMPLETIONS_ROOT = DEFAULT_COMPLETION_DIR
EVAL_RESULTS_ROOT = DEFAULT_EVAL_RESULT_DIR
SCORES_ROOT = DEFAULT_LOG_DIR
CONSOLE_LOG_ROOT = DEFAULT_RUN_LOG_DIR
PARAM_SEARCH_ROOT = RESULTS_ROOT / "param_search"


def ensure_results_structure() -> None:
    for path in (
        RESULTS_ROOT,
        COMPLETIONS_ROOT,
        EVAL_RESULTS_ROOT,
        SCORES_ROOT,
        CONSOLE_LOG_ROOT,
        PARAM_SEARCH_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _dataset_file_stem(dataset_slug: str, *, is_cot: bool) -> str:
    slug = canonical_slug(dataset_slug)
    return f"{slug}__cot" if is_cot else slug


def _model_dataset_relpath(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    model_dir = safe_slug(model_name)
    stem = _dataset_file_stem(dataset_slug, is_cot=is_cot)
    return Path(model_dir) / stem


def _materialize(base: Path, *, suffix: str, root: Path) -> Path:
    target = root / base.parent / f"{base.name}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def jsonl_path(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    ensure_results_structure()
    rel = _model_dataset_relpath(dataset_slug, is_cot=is_cot, model_name=model_name)
    return _materialize(rel, suffix=".jsonl", root=COMPLETIONS_ROOT)


def scores_path(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    ensure_results_structure()
    rel = _model_dataset_relpath(dataset_slug, is_cot=is_cot, model_name=model_name)
    return _materialize(rel, suffix=".json", root=SCORES_ROOT)


def eval_details_path(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    ensure_results_structure()
    rel = _model_dataset_relpath(dataset_slug, is_cot=is_cot, model_name=model_name)
    return _materialize(rel, suffix="_results.jsonl", root=EVAL_RESULTS_ROOT)


def param_search_dir(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    ensure_results_structure()
    rel = _model_dataset_relpath(dataset_slug, is_cot=is_cot, model_name=model_name)
    directory = PARAM_SEARCH_ROOT / rel
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def param_search_trial_path(
    dataset_slug: str,
    *,
    is_cot: bool,
    model_name: str,
    trial_index: int,
) -> Path:
    directory = param_search_dir(dataset_slug, is_cot=is_cot, model_name=model_name)
    return directory / f"trial_{trial_index:02d}.jsonl"


def param_search_records_path(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    directory = param_search_dir(dataset_slug, is_cot=is_cot, model_name=model_name)
    return directory / "records.jsonl"


__all__ = [
    "COMPLETIONS_ROOT",
    "EVAL_RESULTS_ROOT",
    "CONSOLE_LOG_ROOT",
    "SCORES_ROOT",
    "PARAM_SEARCH_ROOT",
    "ensure_results_structure",
    "jsonl_path",
    "scores_path",
    "eval_details_path",
    "param_search_dir",
    "param_search_trial_path",
    "param_search_records_path",
    "write_scores_json",
]


def write_scores_json(
    dataset_slug: str,
    *,
    is_cot: bool,
    model_name: str,
    metrics: dict,
    samples: int,
    problems: int | None = None,
    log_path: Path | str,
    task: str | None = None,
    task_details: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist aggregated metrics as JSON in the canonical scores directory.

    The payload shape intentionally mirrors the documented example:
    {
        "dataset": ..., "model": ..., "cot": bool,
        "metrics": {...}, "samples": int,
        "problems": optional int,
        "created_at": iso8601, "log_path": "results/completions/...jsonl",
        "task": "optional task name",
        "task_details": {"task specific breakdowns"},
        ...extra
    }
    """

    def _normalize_jsonable(value):  # noqa: ANN001
        import numpy as np  # local import: avoids import cost in CLI startup

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _normalize_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_normalize_jsonable(v) for v in value]
        return value

    ensure_results_structure()
    path = scores_path(dataset_slug, is_cot=is_cot, model_name=model_name)
    payload = {
        "dataset": dataset_slug,
        "model": model_name,
        "cot": bool(is_cot),
        "metrics": _normalize_jsonable(metrics),
        "samples": int(samples),
        "created_at": datetime.utcnow().replace(microsecond=False).isoformat() + "Z",
        "log_path": str(log_path),
    }
    if problems is not None:
        payload["problems"] = int(problems)
    if task:
        payload["task"] = task
    if task_details:
        payload["task_details"] = task_details
    if extra:
        payload.update(extra)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path
