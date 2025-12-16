from __future__ import annotations

"""Prepare CMMLU (Chinese MMLU) from lmlmcat/cmmlu release."""

import csv
import io
import zipfile
from pathlib import Path
from typing import Iterable

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY

DATA_URL = "https://hf-mirror.com/datasets/lmlmcat/cmmlu/resolve/main/cmmlu_v1_0_1.zip"
_CHOICE_LETTERS = ("A", "B", "C", "D")


def _iter_rows(zip_path: Path, split: str) -> Iterable[dict]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = f"{split}/"
        for member in sorted(name for name in zf.namelist() if name.startswith(prefix) and name.endswith(".csv")):
            subject = Path(member).stem
            with zf.open(member, "r") as handle:
                reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig"))
                for row in reader:
                    question = (row.get("Question") or "").strip()
                    answer = (row.get("Answer") or "").strip().upper()
                    options = [(row.get(letter) or "").strip() for letter in _CHOICE_LETTERS]
                    if not question or not answer:
                        continue
                    yield {
                        "question": question,
                        "answer": answer,
                        "subject": subject,
                        **{letter: text for letter, text in zip(_CHOICE_LETTERS, options) if text},
                    }


@MULTIPLE_CHOICE_REGISTRY.register("cmmlu")
def prepare_cmmlu(output_root: Path, split: str = "test") -> list[Path]:
    cache_dir = dataset_cache_dir(Path("data"), "cmmlu")
    zip_path = cache_dir / "cmmlu_v1_0_1.zip"
    download_file(DATA_URL, zip_path)

    dataset_dir = output_root / "cmmlu"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_rows(zip_path, split))
    return [target]


__all__ = ["prepare_cmmlu"]
