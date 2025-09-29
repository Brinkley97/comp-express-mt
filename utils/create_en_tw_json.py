"""Utility to convert Akuapem Twi CSV data into HuggingFace translation JSON lines.

The CSV is expected to contain at least two columns named
``English`` and ``Akuapem Twi``. Each non-empty row is transformed into a
JSON object matching the structure used throughout the repository:

    {"translation": {"en": "<source>", "twi": "<target>"}}

Usage example (from the project root):

    python utils/create_en_tw_json.py \
        --csv "data/akuapem_dataset - verified_data.csv" \
        --output-dir data/en-tw \
        --seed 2025

The script produces `train.json`, `dev.json`, and `test.json` inside the
specified output directory using default split ratios of 70/15/15. Rows with
missing values are skipped, and the shuffle is seeded for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "twi"
ENGLISH_COLUMN = "English"
AKUAPEM_COLUMN = "Akuapem Twi"
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_RANDOM_SEED = 2025
SPLIT_NAMES = ("train", "dev", "test")


def _read_parallel_rows(csv_path: Path) -> List[Tuple[str, str]]:
    """Return a list of (source, target) sentence pairs from the CSV file."""

    pairs: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file '{csv_path}' has no header row")

        missing_columns = {
            column
            for column in (ENGLISH_COLUMN, AKUAPEM_COLUMN)
            if column not in reader.fieldnames
        }
        if missing_columns:
            raise ValueError(
                "CSV file is missing required columns: " + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            source = (row.get(ENGLISH_COLUMN) or "").strip()
            target = (row.get(AKUAPEM_COLUMN) or "").strip()

            if not source or not target:
                continue

            pairs.append((source, target))

    return pairs


def _write_jsonl(
    pairs: Iterable[Tuple[str, str]],
    output_path: Path,
    source_lang: str = DEFAULT_SOURCE_LANG,
    target_lang: str = DEFAULT_TARGET_LANG,
) -> int:
    """Write translation pairs to a JSON lines file and return the number of rows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for source, target in pairs:
            record = {"translation": {source_lang: source, target_lang: target}}
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            count += 1

    return count


def _split_pairs(
    pairs: List[Tuple[str, str]],
    ratios: Sequence[float],
    seed: int,
) -> Dict[str, List[Tuple[str, str]]]:
    """Return shuffled train/dev/test splits respecting the requested ratios."""

    if len(ratios) != 3:
        raise ValueError("Exactly three ratios are required for train/dev/test splits")

    if any(r < 0 for r in ratios):
        raise ValueError("Split ratios must be non-negative")

    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be positive")

    # Normalise to protect against ratios that don't sum exactly to 1.0
    normalised = [r / ratio_sum for r in ratios]

    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * normalised[0])
    dev_end = train_end + int(total * normalised[1])

    splits = {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:],
    }

    if any(len(splits[name]) == 0 for name in SPLIT_NAMES):
        raise ValueError(
            "One of the splits ended up empty; adjust the ratios or ensure the dataset is large enough."
        )

    return splits


def convert_csv(
    csv_path: Path,
    output_dir: Path,
    source_lang: str = DEFAULT_SOURCE_LANG,
    target_lang: str = DEFAULT_TARGET_LANG,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Path]:
    """Convert the CSV file to JSON lines and return the written split paths."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file '{csv_path}' does not exist")

    pairs = _read_parallel_rows(csv_path)
    if not pairs:
        raise ValueError(f"No valid translation rows were found in '{csv_path}'")

    splits = _split_pairs(pairs, ratios=ratios, seed=seed)

    written_paths: Dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        split_pairs = splits[split_name]
        output_path = output_dir / f"{split_name}.json"
        count = _write_jsonl(split_pairs, output_path, source_lang=source_lang, target_lang=target_lang)
        print(f"Wrote {count} translation pairs to {output_path}")  # noqa: T201 (informational)
        written_paths[split_name] = output_path

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the source CSV file (expects 'English' and 'Akuapem Twi' columns)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the JSON split files should be written",
    )
    parser.add_argument(
        "--source-lang",
        default=DEFAULT_SOURCE_LANG,
        help=f"Source language code to use in JSON output (default: {DEFAULT_SOURCE_LANG})",
    )
    parser.add_argument(
        "--target-lang",
        default=DEFAULT_TARGET_LANG,
        help=f"Target language code to use in JSON output (default: {DEFAULT_TARGET_LANG})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS[0],
        help="Proportion of examples to use for the training split (default: 0.7)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS[1],
        help="Proportion of examples to use for the dev/validation split (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS[2],
        help="Proportion of examples to use for the test split (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed used when shuffling before splitting (default: {DEFAULT_RANDOM_SEED})",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    convert_csv(
        csv_path=parsed.csv,
        output_dir=parsed.output_dir,
        source_lang=parsed.source_lang,
        target_lang=parsed.target_lang,
        ratios=(parsed.train_ratio, parsed.dev_ratio, parsed.test_ratio),
        seed=parsed.seed,
    )


if __name__ == "__main__":
    main()
