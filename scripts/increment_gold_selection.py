#!/usr/bin/env python3
"""Increment gold_selection values in pure selection result files."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _update_record(record: Dict[str, Any]) -> bool:
    """Increment gold_selection for every candidate in record['tgts']."""
    updated = False
    for candidate in record.get("tgts", []):
        if not isinstance(candidate, dict):
            continue
        for stats in candidate.values():
            if isinstance(stats, dict) and "gold_selection" in stats:
                try:
                    stats["gold_selection"] = int(stats["gold_selection"]) + 1
                    updated = True
                except (ValueError, TypeError):
                    raise ValueError(
                        f"gold_selection must be numeric, found {stats['gold_selection']}"
                    )
    return updated


def process_file(file_path: Path, dry_run: bool = False) -> bool:
    """Process a single JSON file and optionally write changes."""
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    updated_any = False
    for records in data.values():
        if not isinstance(records, list):
            continue
        for record in records:
            if _update_record(record):
                updated_any = True

    if updated_any and not dry_run:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    return updated_any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Increment gold_selection values in pure selection result files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="experiments/results/pure_selection_results",
        help="Root directory containing result JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be updated without writing changes.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    files = sorted(root.rglob("*.json"))
    if not files:
        print(f"No JSON files found under {root}")
        return

    updated_files = 0
    for file_path in files:
        try:
            updated = process_file(file_path, dry_run=args.dry_run)
        except ValueError as exc:
            print(f"[ERROR] {file_path}: {exc}")
            continue
        if updated:
            updated_files += 1
            status = "DRY-RUN" if args.dry_run else "UPDATED"
            print(f"[{status}] {file_path}")

    if not updated_files:
        print("No files required updating.")


if __name__ == "__main__":
    main()
