from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


PAN25_FULLY_HUMAN_LABEL_TEXT = "fully human-written"
MASTER_FIELDNAMES = [
    "id",
    "source_id",
    "class",
    "rewrite_level",
    "text",
    "source_corpus",
    "split_hint",
    "language",
    "source_dataset",
    "split",
]


@dataclass(frozen=True)
class MasterRow:
    id: str
    source_id: str
    cls: str
    rewrite_level: str
    text: str
    source_corpus: str
    split_hint: str
    language: Optional[str] = None
    source_dataset: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "class": self.cls,
            "rewrite_level": self.rewrite_level,
            "text": self.text,
            # extra metadata columns (safe: your pipeline only requires the first 5)
            "source_corpus": self.source_corpus,
            "split_hint": self.split_hint,
            "language": self.language,
            "source_dataset": self.source_dataset,
        }


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from e


def iter_pan25_fully_human_rows(pan25_dir: Path) -> Iterator[MasterRow]:
    for split_name in ("train", "dev"):
        jsonl_path = pan25_dir / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            continue

        for i, obj in enumerate(_iter_jsonl(jsonl_path), start=1):
            if obj.get("label_text") != PAN25_FULLY_HUMAN_LABEL_TEXT:
                continue

            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            # Stable IDs: split + 1-indexed line counter among kept rows
            # (we also keep the original split_name as metadata)
            rid = f"pan25_{split_name}_human_{i:07d}"
            yield MasterRow(
                id=rid,
                source_id=rid,
                cls="human",
                rewrite_level="none",
                text=text,
                source_corpus="pan25",
                split_hint=split_name,
                language=obj.get("language"),
                source_dataset=obj.get("source_dataset"),
            )


def stable_split(id_: str, train: float, val: float) -> str:
    """
    Deterministic split based on SHA256(id).
    """
    h = hashlib.sha256(id_.encode("utf-8")).hexdigest()
    # 0..1
    r = int(h[:8], 16) / 0xFFFFFFFF
    if r < train:
        return "train"
    if r < train + val:
        return "val"
    return "test"


def _csv_dialect(delimiter: str) -> dict:
    if len(delimiter) != 1:
        raise ValueError("--delimiter must be exactly one character")
    return {
        "delimiter": delimiter,
        "quoting": csv.QUOTE_ALL,
        "lineterminator": "\n",
    }


def write_master_csv(rows: Iterable[MasterRow], out_csv: Path, delimiter: str = ";") -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDNAMES, **_csv_dialect(delimiter))
        w.writeheader()
        for r in rows:
            d = r.to_dict()
            d["split"] = ""  # filled by a second pass or by the caller
            w.writerow(d)
            n += 1
    return n


def write_master_with_split_csv(
    rows: Iterable[MasterRow],
    out_csv: Path,
    train: float,
    val: float,
    delimiter: str = ";",
) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDNAMES, **_csv_dialect(delimiter))
        w.writeheader()
        for r in rows:
            d = r.to_dict()
            d["split"] = stable_split(r.id, train=train, val=val)
            w.writerow(d)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a clean PAN25-only master CSV from fully human train/dev texts.")
    ap.add_argument(
        "--pan25-dir",
        type=Path,
        default=Path("data/pan25-generative-ai-detection-task2-train"),
        help="Directory containing train.jsonl and dev.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/master_human_texts_pan25.csv"),
        help="Output CSV path",
    )
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument(
        "--delimiter",
        choices=[",", ";", "\t"],
        default=";",
        help="CSV delimiter. Default ';' opens cleanly in many European Excel locales.",
    )
    args = ap.parse_args()

    if args.train_frac <= 0 or args.val_frac < 0 or args.train_frac + args.val_frac >= 1:
        raise SystemExit("Invalid split fractions: need train>0, val>=0, train+val < 1")

    if not args.pan25_dir.exists():
        raise SystemExit(f"PAN25 directory not found: {args.pan25_dir}")

    def iter_all() -> Iterator[MasterRow]:
        yield from iter_pan25_fully_human_rows(args.pan25_dir)

    n = write_master_with_split_csv(
        iter_all(),
        args.out,
        train=args.train_frac,
        val=args.val_frac,
        delimiter=args.delimiter,
    )
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    # Avoid letting tokenizers spawn many threads on Windows by default.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

