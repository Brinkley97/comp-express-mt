#!/usr/bin/env python3
"""Run COMET-based evaluation on translation triplets and log results."""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from comet import download_model, load_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing *_triples.json files with src/mt/ref entries.",
    )
    parser.add_argument(
        "--model",
        default="masakhane/africomet-stl-1.1",
        help="COMET model checkpoint to use (Hugging Face identifier or path).",
    )
    parser.add_argument(
        "--pattern",
        default="*_triples.json",
        help="Glob pattern to match evaluation files inside input_dir.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Prediction batch size for COMET scoring.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (set to 0 for CPU-only).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write aggregated metrics JSON (defaults to <input_dir>/comet_metrics.json).",
    )
    parser.add_argument(
        "--store_segment_scores",
        action="store_true",
        help="Include per-segment COMET scores in the output JSON.",
    )
    return parser.parse_args()


def discover_files(root: Path, pattern: str) -> List[Path]:
    files = sorted(root.glob(pattern))
    return [f for f in files if f.is_file()]


def load_triplets(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")

    required_keys = {"src", "mt"}
    if any("ref" in entry for entry in data):
        required_keys.add("ref")

    validated: List[Dict[str, Any]] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {idx} in {path} is not a dict: {entry!r}")
        missing = [key for key in required_keys if key not in entry]
        if missing:
            raise ValueError(f"Entry {idx} in {path} missing keys: {missing}")
        validated.append({k: entry[k] for k in entry if k in {"src", "mt", "ref"}})
    return validated


def run_comet(
    model_id: str,
    files: Iterable[Path],
    *,
    batch_size: int,
    gpus: int,
    store_segments: bool,
) -> Dict[str, Any]:
    model_path = download_model(model_id)
    model = load_from_checkpoint(model_path)

    results = []
    for path in files:
        triplets = load_triplets(path)
        if not triplets:
            continue

        prediction = model.predict(triplets, batch_size=batch_size, gpus=gpus)

        if isinstance(prediction, tuple) and len(prediction) == 2:
            segment_scores, system_score = prediction
        elif isinstance(prediction, dict):
            segment_scores = prediction.get("scores") or prediction.get("segment_scores")
            system_score = prediction.get("system_score") or prediction.get("mean_score")
            if system_score is None and segment_scores is not None:
                system_score = float(sum(segment_scores) / len(segment_scores))
        else:
            # Fallback: prediction might just be a list of segment scores
            segment_scores = prediction
            system_score = float(sum(segment_scores) / len(segment_scores)) if segment_scores else math.nan

        record = {
            "file": str(path.name),
            "num_samples": len(triplets),
            "system_score": float(system_score) if system_score is not None else math.nan,
        }

        if store_segments and segment_scores is not None:
            record["segment_scores"] = [float(score) for score in segment_scores]

        results.append(record)

    return {
        "model": model_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "files": results,
    }


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        sys.exit(f"Input directory not found: {input_dir}")

    files = discover_files(input_dir, args.pattern)
    if not files:
        sys.exit(f"No files matching {args.pattern!r} in {input_dir}")

    report = run_comet(
        args.model,
        files,
        batch_size=args.batch_size,
        gpus=args.gpus,
        store_segments=args.store_segment_scores,
    )

    output_path = Path(args.output) if args.output else input_dir / "comet_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Saved COMET metrics for {len(report['files'])} file(s) to {output_path}")


if __name__ == "__main__":
    main()
