from __future__ import annotations

"""Prepare MMMLU (multilingual MMLU variant) from ncoop57/mmmlu."""

import csv
import io
import zipfile
from pathlib import Path
from typing import Iterable

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY

DATA_URL = "https://hf-mirror.com/datasets/ncoop57/mmmlu/resolve/main/data.zip"
_CHOICE_LETTERS = ("A", "B", "C", "D")


def _iter_rows(zip_path: Path, split: str) -> Iterable[dict]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = f"data/{split}/"
        for member in sorted(name for name in zf.namelist() if name.startswith(prefix) and name.endswith(".csv")):
            subject = Path(member).stem
            if subject.endswith(f"_{split}"):
                subject = subject[: -len(f"_{split}")]
            with zf.open(member, "r") as handle:
                reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8"))
                for row in reader:
                    if len(row) < 6:
                        continue
                    question = row[0].strip()
                    options = [opt.strip() for opt in row[1:1 + len(_CHOICE_LETTERS)] if opt is not None]
                    answer = row[5].strip().upper() if len(row) > 5 else ""
                    if not question or not options or not answer:
                        continue
                    yield {
                        "question": question,
                        "answer": answer,
                        "subject": subject,
                        **{letter: options[idx] for idx, letter in enumerate(_CHOICE_LETTERS) if idx < len(options)},
                    }


@MULTIPLE_CHOICE_REGISTRY.register("mmmlu")
def prepare_mmmlu(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("mmmlu 目前仅提供 test split")

    cache_dir = dataset_cache_dir(Path("data"), "mmmlu")
    zip_path = cache_dir / "data.zip"
    download_file(DATA_URL, zip_path)

    dataset_dir = output_root / "mmmlu"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_rows(zip_path, split))
    return [target]


__all__ = ["prepare_mmmlu"]
