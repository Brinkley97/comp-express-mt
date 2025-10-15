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
    parser.add_argument(
        "--perm_samples",
        type=int,
        default=2000,
        help="Number of permutations/bootstraps for p-values and CIs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible resampling."
    )
    parser.add_argument(
        "--overall_subset",
        default=None,
        help="Comma-separated list of file basenames (e.g., '1_to_m_triples.json,m_to_1_triples.json') to include in the OVERALL aggregation. If not set, all files are used."
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


# Permutation helpers
def signflip_pvalue_mean_diff(diffs: np.ndarray, *, B: int = 2000, seed: int = 123) -> float:
    """
    Two-sided paired permutation (sign-flip) test on the mean of paired differences.
    H0: mean(diffs) == 0. Returns p-value.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = diffs.size
    if n == 0:
        return float("nan")
    obs = float(diffs.mean())
    rng = np.random.default_rng(seed)
    # Vectorized sign flips: +1/-1 with equal prob
    flips = rng.choice([-1.0, 1.0], size=(B, n))
    perm_means = (flips * diffs).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(obs)).mean())
    return p


def permutation_pvalue_corr(x: List[float], y: List[float], *, kind: str = "pearson", B: int = 2000, seed: int = 123) -> float:
    """
    Two-sided permutation test for correlation between x and y.
    kind: 'pearson' or 'spearman'. Returns p-value.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    n = xa.size
    if n < 3:
        return float("nan")

    if kind == "spearman":
        r_obs = spearman_corr(xa.tolist(), ya.tolist())
    else:
        r_obs = float(np.corrcoef(xa, ya)[0, 1])

    if not np.isfinite(r_obs):
        return float("nan")

    rng = np.random.default_rng(seed)
    # Pre-generate permutations by shuffling indices of y
    idx = np.arange(n)
    r_perm = np.empty(B, dtype=float)
    for b in range(B):
        rng.shuffle(idx)
        if kind == "spearman":
            r_perm[b] = spearman_corr(xa.tolist(), ya[idx].tolist())
        else:
            r_perm[b] = float(np.corrcoef(xa, ya[idx])[0, 1])
    p = float((np.abs(r_perm) >= abs(r_obs)).mean())
    return p


def compare_and_visualize(stl_report: Dict[str, Any],
                          qe_report: Dict[str, Any],
                          *,
                          input_dir: Path,
                          export_segments: bool,
                          make_plots: bool,
                          subset_files: List[str] | None = None,
                          perm_samples: int = 2000,
                          seed: int = 123) -> Dict[str, Any]:
    """Align STL and QE by file and index; write per-file CSVs and plots; return summary stats."""
    summary = {"files": [], "overall": {}}

    # Build quick lookups by filename
    stl_by_file = {f["file"]: f for f in stl_report["files"]}
    qe_by_file = {f["file"]: f for f in qe_report["files"]}

    # Normalize subset filter (accept basenames or stems)
    subset_names = None
    subset_stems = None
    if subset_files:
        subset_names = {name.strip() for name in subset_files if name and name.strip()}
        subset_stems = {Path(n).stem for n in subset_names}

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

        # Accumulate global vectors (respect subset if provided)
        use_for_overall = (subset_names is None) or (fname in subset_names) or (Path(fname).stem in subset_stems)
        if use_for_overall:
            all_stl.extend(stl_scores)
            all_qe.extend(qe_scores)

        # Per-file stats
        pearson = float(np.corrcoef(stl_scores, qe_scores)[0, 1]) if n > 1 else float("nan")
        spearman = spearman_corr(stl_scores, qe_scores) if n > 1 else float("nan")
        diffs = [s - q for s, q in zip(stl_scores, qe_scores)]
        mean_diff = float(mean(diffs)) if diffs else float("nan")
        p_mean = signflip_pvalue_mean_diff(np.asarray(diffs, dtype=float), B=perm_samples, seed=seed) if diffs else float("nan")
        p_pearson = permutation_pvalue_corr(stl_scores, qe_scores, kind="pearson", B=perm_samples, seed=seed) if n > 2 else float("nan")
        p_spearman = permutation_pvalue_corr(stl_scores, qe_scores, kind="spearman", B=perm_samples, seed=seed) if n > 2 else float("nan")

        file_stats = {
            "file": fname,
            "num_segments": n,
            "stl_system": stl_rec.get("system_score", float("nan")),
            "qe_system": qe_rec.get("system_score", float("nan")),
            "pearson": pearson,
            "spearman": spearman,
            "mean_diff_stl_minus_qe": mean_diff,
            "p_value_mean_diff": p_mean,
            "p_value_pearson": p_pearson,
            "p_value_spearman": p_spearman,
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

    # Track which files contributed to OVERALL aggregation
    if subset_names is None:
        used_files = sorted(list(stl_by_file.keys() & qe_by_file.keys()))
    else:
        used = []
        for fname in stl_by_file.keys():
            if fname in qe_by_file and ((fname in subset_names) or (Path(fname).stem in subset_stems)):
                used.append(fname)
        used_files = sorted(used)

    # Overall stats across all files
    if len(all_stl) >= 2 and len(all_qe) == len(all_stl):
        overall_pearson = float(np.corrcoef(all_stl, all_qe)[0, 1])
        overall_spearman = spearman_corr(all_stl, all_qe)
        diffs = np.asarray(all_stl) - np.asarray(all_qe)
        mean_diff = float(diffs.mean())

        # Simple bootstrap CI for mean difference
        rng = np.random.default_rng(seed)
        B = perm_samples
        boots = []
        n = len(diffs)
        for _ in range(B):
            idxs = rng.integers(0, n, size=n)
            boots.append(float(diffs[idxs].mean()))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

        p_mean = signflip_pvalue_mean_diff(diffs, B=perm_samples, seed=seed)
        p_pearson = permutation_pvalue_corr(all_stl, all_qe, kind="pearson", B=perm_samples, seed=seed)
        p_spearman = permutation_pvalue_corr(all_stl, all_qe, kind="spearman", B=perm_samples, seed=seed)

        summary["overall"] = {
            "num_segments": len(all_stl),
            "pearson": overall_pearson,
            "spearman": overall_spearman,
            "mean_diff_stl_minus_qe": mean_diff,
            "mean_diff_95ci": [lo, hi],
            "p_value_mean_diff": float(p_mean),
            "p_value_pearson": float(p_pearson),
            "p_value_spearman": float(p_spearman),
            "files_used": used_files,
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

        subset = None
        if args.overall_subset:
            subset = [p.strip() for p in args.overall_subset.split(",") if p.strip()]
        summary = compare_and_visualize(
            stl_report, qe_report,
            input_dir=input_dir,
            export_segments=args.export_segments,
            make_plots=args.make_plots,
            subset_files=subset,
            perm_samples=args.perm_samples,
            seed=args.seed,
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
