import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def _extract_label_pairs(records: Iterable[Dict]) -> Tuple[List[int], List[int]]:
    """
    Extract parallel lists of gold and predicted selection labels from an iterable of record dicts.
    Each input record is expected to contain a "tgts" key mapping to an iterable of candidate entries.
    Each candidate should be a dict with a single key-value pair; the value is expected to be another
    dict (here called "stats") that may include the keys "gold_selection" and "llm_selection". This
    function iterates over all records and all candidates, collecting pairs (gold, pred) whenever both
    "gold_selection" and "llm_selection" are present and not None.
    Args:
        records: An iterable of dictionaries representing records. For each record, record.get("tgts", [])
            should yield candidate entries. Non-dict candidates, candidates whose value is not a dict,
            or candidates missing either "gold_selection" or "llm_selection" are skipped.
    Returns:
        A tuple (gold_labels, pred_labels) where:
          - gold_labels is a list of gold selection labels (ints) in the order they were encountered.
          - pred_labels is a list of predicted selection labels (ints) in the corresponding order.
        The two lists are guaranteed to have the same length. If no valid pairs are found, both lists
        will be empty.
    Notes:
        - The function preserves input order across records and candidates.
        - It does not perform type coercion; it appends values as-is (the calling code may expect ints).
        - Candidates that are not dicts or that do not adhere to the expected single-key structure are ignored.
    """
    
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
    """
    Compute the accuracy between ground-truth labels and predicted labels.

    Parameters
    ----------
    y_true : List[int]
        Ground-truth integer labels.
    y_pred : List[int]
        Predicted integer labels.

    Returns
    -------
    float
        The fraction of elements in y_true that are equal to the corresponding
        elements in y_pred. If y_true is empty, returns 0.0.

    Notes
    -----
    - Comparison is performed pairwise using zip(y_true, y_pred); only paired
      elements are compared with equality (==).
    - The denominator is len(y_true). If y_pred is shorter than y_true, any
      unpaired true labels are effectively treated as incorrect. If y_pred is
      longer than y_true, extra predictions are ignored.
    - Time complexity: O(n) where n = len(y_true).
    """
    if not y_true:
        return 0.0
    matches = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred)
    return matches / len(y_true)


def _macro_f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute the macro-averaged F1 score for integer class labels.

    This function computes the F1 score for each class present in y_true and returns the
    unweighted (macro) average of those per-class F1 scores. Per-class precision and recall
    are computed from true positives (tp), false positives (fp) and false negatives (fn)
    over the zipped pairs of y_true and y_pred.

    Parameters
    ----------
    y_true : List[int]
        Ground-truth integer labels. If empty, the function returns 0.0.
    y_pred : List[int]
        Predicted integer labels. Only the first min(len(y_true), len(y_pred)) pairs
        are considered (zipping behavior).

    Returns
    -------
    float
        The macro-averaged F1 score in the range [0.0, 1.0]. Returns 0.0 when y_true is
        empty or when no labels are present in y_true.

    Notes
    -----
    - Only labels that occur in y_true are considered when computing per-class F1.
      Predicted labels that never appear in y_true are not treated as separate classes;
      they only contribute to false negatives for classes present in y_true.
    - For each class:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    - The returned score is the arithmetic mean of the per-class F1 scores (macro average).

    Complexity
    ----------
    Time: O(n * L) where n is the number of paired examples considered and L is the number
          of unique labels in y_true.
    Space: O(L) for storing per-class scores.

    Example
    -------
    >>> # For y_true = [0, 1, 1] and y_pred = [0, 1, 0]:
    >>> # class 0: tp=1, fp=1, fn=0  -> f1 = 2/3
    >>> # class 1: tp=1, fp=0, fn=1  -> f1 = 1/2
    >>> # macro F1 = (2/3 + 1/2) / 2 = 7/12 ≈ 0.5833
    """
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
    """
    Parse a JSON results file and compute per-model evaluation metrics.

    This function reads a JSON file at the given path, where the top-level structure is
    expected to map model identifiers (strings) to a collection of prediction records.
    Each collection of records is passed to the internal helper _extract_label_pairs, which
    must return two parallel sequences: gold labels and predicted labels. For each model,
    the function computes an accuracy score and a macro-averaged F1 score using the
    internal helpers _accuracy_score and _macro_f1_score, and records the number of
    predictions processed.

    Parameters
    ----------
    results_path : str
        Path to a JSON file containing model results. The JSON should map model names
        to their associated prediction records in a format accepted by
        _extract_label_pairs.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A mapping from model name to a metrics dictionary with the following keys:
          - "accuracy" (float): accuracy score for the model's predictions.
          - "macro_f1" (float): macro-averaged F1 score for the model's predictions.
          - "num_predictions" (int): number of predictions used to compute the metrics.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file cannot be decoded as valid JSON.
    Exception
        Any exceptions raised by the helper functions (_extract_label_pairs,
        _accuracy_score, _macro_f1_score) are propagated.

    Example
    -------
    >>> evaluate_results_file("results.json")
    {
      "modelA": {"accuracy": 0.87, "macro_f1": 0.84, "num_predictions": 1200},
      "modelB": {"accuracy": 0.91, "macro_f1": 0.89, "num_predictions": 1150}
    """
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
    """
    Generate a default output file path for metric results by appending "_metrics"
    to the stem of an existing file path while preserving its original extension.

    Args:
        results_path (str): Path to the original results file. Can be a filename
            or a full path.

    Returns:
        str: New path with "_metrics" inserted before the file extension. If the
        input has no extension, "_metrics" is appended to the end of the filename.

    Examples:
        >>> _default_output_path("results.json")
        "results_metrics.json"
        >>> _default_output_path("/data/run/results")
        "/data/run/results_metrics"
    """
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
        default="experiments/results/selection_results_metric_wise",
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
