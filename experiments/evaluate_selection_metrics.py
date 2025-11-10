import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def _extract_label_pairs(records: Iterable[Dict]) -> Tuple[List[int], List[int]]:
    """Flatten gold/predicted selection pairs from the serialized experiment records."""
    gold_labels: List[int] = []
    pred_labels: List[int] = []

    for record in records:
        for candidate in record.get("tgts", []):
            if not isinstance(candidate, dict):
                continue
            (_, stats), = candidate.items()
            gold = stats.get("gold_selection")
            pred = stats.get("llm_selection")
            if gold is None or pred is None:
                continue
            gold_labels.append(gold)
            pred_labels.append(pred)

    return gold_labels, pred_labels


def _accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    matches = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred)
    return matches / len(y_true)


def _macro_f1_score(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0

    labels = sorted(set(y_true))
    if not labels:
        return 0.0

    f1_scores: List[float] = []
    for label in labels:
        tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def evaluate_results_file(results_path: str) -> Dict[str, Dict[str, float]]:
    with open(results_path, "r", encoding="utf-8") as f:
        results_blob = json.load(f)

    metrics: Dict[str, Dict[str, float]] = {}
    for model_name, records in results_blob.items():
        gold, preds = _extract_label_pairs(records)
        accuracy = _accuracy_score(gold, preds)
        macro_f1 = _macro_f1_score(gold, preds)
        metrics[model_name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "num_predictions": len(preds),
        }
    return metrics


def _default_output_path(results_path: str) -> str:
    stem, ext = os.path.splitext(results_path)
    return f"{stem}_metrics{ext}"


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy and macro-F1 for selection experiments.")
    parser.add_argument(
        "--results-files",
        nargs="+",
        required=True,
        help="One or more JSON files produced by experiment scripts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to place the *_metrics.json files. Defaults to same directory as inputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for results_path in args.results_files:
        metrics = evaluate_results_file(results_path)
        output_path = _default_output_path(results_path)
        if output_dir:
            output_path = os.path.join(output_dir, os.path.basename(output_path))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Wrote metrics for {results_path} → {output_path}")


if __name__ == "__main__":
    main()
