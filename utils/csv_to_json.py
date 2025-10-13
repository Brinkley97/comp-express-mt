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
The script produces the corresponding json file in the specified output directory.
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
DEFAULT_RANDOM_SEED = 2025


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


def convert_csv(
    csv_path: Path,
    output_dir: Path,
    source_lang: str = DEFAULT_SOURCE_LANG,
    target_lang: str = DEFAULT_TARGET_LANG,
) -> Dict[str, Path]:
    """Convert the CSV file to JSON lines and return the written split paths."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file '{csv_path}' does not exist")

    pairs = _read_parallel_rows(csv_path)
    if not pairs:
        raise ValueError(f"No valid translation rows were found in '{csv_path}'")

    # write pairs to a single JSON file
    output_dir.mkdir(parents=True, exist_ok=True)
    # using the csv file stem as the output file name
    output_path = output_dir / f"{csv_path.stem}.json"

    count = _write_jsonl(pairs, output_path, source_lang=source_lang, target_lang=target_lang)
    print(f"Wrote {count} translation pairs to {output_path}")  # noqa: T201 (informational)

    # return None
    return None


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
    )


if __name__ == "__main__":
    main()