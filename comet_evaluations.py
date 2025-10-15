#!/usr/bin/env python3
"""Run COMET-based evaluation on translation triplets and log results."""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import csv
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument(
        "--qe_model",
        default=None,
        help="Optional COMET QE checkpoint (reference-free). If provided, runs both STL (reference-based) and QE.",
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="Generate scatter and histogram plots comparing STL vs QE (when --qe_model is set).",
    )
    parser.add_argument(
        "--export_segments",
        action="store_true",
        help="Export per-segment CSVs with STL and QE scores (when --qe_model is set).",
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


def run_single_model(
    model_id: str,
    files: Iterable[Path],
    *,
    batch_size: int,
    gpus: int,
    store_segments: bool,
    drop_ref: bool = False,
) -> Dict[str, Any]:
    """Run a single COMET model across files. If drop_ref is True, remove 'ref' before scoring (QE-style)."""
    model_path = download_model(model_id)
    model = load_from_checkpoint(model_path)

    results = []
    for path in files:
        triplets = load_triplets(path)
        if not triplets:
            continue

        eval_items = []
        for ex in triplets:
            item = {"src": ex["src"], "mt": ex["mt"]}
            if not drop_ref and "ref" in ex:
                item["ref"] = ex["ref"]
            eval_items.append(item)

        prediction = model.predict(eval_items, batch_size=batch_size, gpus=gpus)

        if isinstance(prediction, tuple) and len(prediction) == 2:
            segment_scores, system_score = prediction
        elif isinstance(prediction, dict):
            segment_scores = prediction.get("scores") or prediction.get("segment_scores")
            system_score = prediction.get("system_score") or prediction.get("mean_score")
            if system_score is None and segment_scores is not None:
                system_score = float(sum(segment_scores) / len(segment_scores))
        else:
            segment_scores = prediction
            system_score = float(sum(segment_scores) / len(segment_scores)) if segment_scores else math.nan

        record = {
            "file": str(path.name),
            "num_samples": len(eval_items),
            "system_score": float(system_score) if system_score is not None else math.nan,
        }

        if store_segments and segment_scores is not None:
            record["segment_scores"] = [float(score) for score in segment_scores]

        # Keep full text for downstream CSV export if needed
        if store_segments:
            record["_examples"] = eval_items

        results.append(record)

    return {
        "model": model_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "files": results,
    }


def spearman_corr(a: List[float], b: List[float]) -> float:
    """Compute Spearman rank correlation via numpy without external deps."""
    xa = np.asarray(a)
    xb = np.asarray(b)
    ra = np.argsort(np.argsort(xa))
    rb = np.argsort(np.argsort(xb))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compare_and_visualize(stl_report: Dict[str, Any],
                          qe_report: Dict[str, Any],
                          *,
                          input_dir: Path,
                          export_segments: bool,
                          make_plots: bool) -> Dict[str, Any]:
    """Align STL and QE by file and index; write per-file CSVs and plots; return summary stats."""
    summary = {"files": [], "overall": {}}

    # Build quick lookups by filename
    stl_by_file = {f["file"]: f for f in stl_report["files"]}
    qe_by_file = {f["file"]: f for f in qe_report["files"]}

    all_stl = []
    all_qe = []

    for fname, stl_rec in stl_by_file.items():
        if fname not in qe_by_file:
            continue
        qe_rec = qe_by_file[fname]

        stl_scores = stl_rec.get("segment_scores")
        qe_scores = qe_rec.get("segment_scores")
        stl_examples = stl_rec.get("_examples", [])
        qe_examples = qe_rec.get("_examples", [])

        if not stl_scores or not qe_scores:
            continue

        n = min(len(stl_scores), len(qe_scores))
        stl_scores = stl_scores[:n]
        qe_scores = qe_scores[:n]

        # Accumulate global vectors
        all_stl.extend(stl_scores)
        all_qe.extend(qe_scores)

        # Per-file stats
        pearson = float(np.corrcoef(stl_scores, qe_scores)[0, 1]) if n > 1 else float("nan")
        spearman = spearman_corr(stl_scores, qe_scores) if n > 1 else float("nan")
        diffs = [s - q for s, q in zip(stl_scores, qe_scores)]
        mean_diff = float(mean(diffs)) if diffs else float("nan")

        file_stats = {
            "file": fname,
            "num_segments": n,
            "stl_system": stl_rec.get("system_score", float("nan")),
            "qe_system": qe_rec.get("system_score", float("nan")),
            "pearson": pearson,
            "spearman": spearman,
            "mean_diff_stl_minus_qe": mean_diff,
        }
        summary["files"].append(file_stats)

        # Optional CSV export (include text when available)
        if export_segments:
            csv_path = input_dir / f"{Path(fname).stem}_segments_stl_qe.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["idx", "src", "mt", "ref", "stl", "qe", "stl_minus_qe"])
                for i in range(n):
                    src = stl_examples[i].get("src") if i < len(stl_examples) else ""
                    mt = stl_examples[i].get("mt") if i < len(stl_examples) else ""
                    ref = stl_examples[i].get("ref") if i < len(stl_examples) else ""
                    writer.writerow([i, src, mt, ref, stl_scores[i], qe_scores[i], stl_scores[i] - qe_scores[i]])

        # Optional plots
        if make_plots and n >= 2:
            # Scatter: QE vs STL
            plt.figure()
            plt.scatter(stl_scores, qe_scores)
            plt.xlabel("STL score (reference-based)")
            plt.ylabel("QE score (reference-free)")
            plt.title(f"STL vs QE — {fname}")
            plt.tight_layout()
            (input_dir / f"{Path(fname).stem}_scatter_stl_vs_qe.png").parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(input_dir / f"{Path(fname).stem}_scatter_stl_vs_qe.png", dpi=180)
            plt.close()

            # Histogram of differences
            plt.figure()
            plt.hist([s - q for s, q in zip(stl_scores, qe_scores)], bins=30)
            plt.xlabel("STL − QE")
            plt.ylabel("Count")
            plt.title(f"Differences (STL − QE) — {fname}")
            plt.tight_layout()
            plt.savefig(input_dir / f"{Path(fname).stem}_hist_stl_minus_qe.png", dpi=180)
            plt.close()

    # Overall stats across all files
    if len(all_stl) >= 2 and len(all_qe) == len(all_stl):
        overall_pearson = float(np.corrcoef(all_stl, all_qe)[0, 1])
        overall_spearman = spearman_corr(all_stl, all_qe)
        diffs = np.asarray(all_stl) - np.asarray(all_qe)
        mean_diff = float(diffs.mean())

        # Simple bootstrap CI for mean difference
        rng = np.random.default_rng(123)
        B = 2000
        boots = []
        n = len(diffs)
        for _ in range(B):
            idxs = rng.integers(0, n, size=n)
            boots.append(float(diffs[idxs].mean()))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

        summary["overall"] = {
            "num_segments": len(all_stl),
            "pearson": overall_pearson,
            "spearman": overall_spearman,
            "mean_diff_stl_minus_qe": mean_diff,
            "mean_diff_95ci": [lo, hi],
        }

    return summary


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        sys.exit(f"Input directory not found: {input_dir}")

    files = discover_files(input_dir, args.pattern)
    if not files:
        sys.exit(f"No files matching {args.pattern!r} in {input_dir}")

    stl_report = run_single_model(
        args.model,
        files,
        batch_size=args.batch_size,
        gpus=args.gpus,
        store_segments=(args.store_segment_scores or bool(args.qe_model) or args.export_segments or args.make_plots),
        drop_ref=False,
    )

    output_path = Path(args.output) if args.output else input_dir / "comet_metrics.json"

    if args.qe_model:
        qe_report = run_single_model(
            args.qe_model,
            files,
            batch_size=args.batch_size,
            gpus=args.gpus,
            store_segments=True,
            drop_ref=True,
        )

        combined = {
            "stl": stl_report,
            "qe": qe_report,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(combined, handle, ensure_ascii=False, indent=2)

        summary = compare_and_visualize(
            stl_report, qe_report,
            input_dir=input_dir,
            export_segments=args.export_segments,
            make_plots=args.make_plots,
        )
        with (input_dir / "comet_comparison_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(
            f"Saved STL+QE metrics to {output_path}\n"
            f"Summary written to {input_dir / 'comet_comparison_summary.json'}\n"
            f"CSV/plots emitted per file (if requested)."
        )
    else:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(stl_report, handle, ensure_ascii=False, indent=2)
        print(f"Saved COMET metrics for {len(stl_report['files'])} file(s) to {output_path}")


if __name__ == "__main__":
    main()
