#!/usr/bin/env python3
"""Evaluate a fine-tuned translation model on multiple splits with BLEU and chrF++."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import evaluate
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Path or identifier of the fine-tuned model.")
    parser.add_argument("--source_lang", required=True, help="Source language code (e.g. en).")
    parser.add_argument("--target_lang", required=True, help="Target language code (e.g. sw).")
    parser.add_argument("--source_prefix", default="", help="Optional prefix applied to every source sentence.")
    parser.add_argument("--forced_bos_token", help="Token forced as first decoder token for multilingual models.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME=PATH",
        required=True,
        help="Named JSONL datasets to evaluate, e.g. train=data/train.json test=data/test.json.",
    )
    parser.add_argument("--output_dir", help="Directory to save predictions/metrics (optional).")
    parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size.")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum generation length.")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam search width.")
    parser.add_argument("--device", default=None, help="Device override (cpu, cuda, cuda:0, mps, ...).")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32",
                        help="Optional mixed-precision mode for generation.")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars.")
    parser.add_argument("--save_predictions", action="store_true", help="Persist model outputs to disk.")
    parser.add_argument("--save_json", action="store_true", help="Persist src/mt/ref triples as JSON files.")
    return parser.parse_args()


def load_split(path: Path, source_lang: str, target_lang: str) -> List[Dict[str, str]]:
    # Load a JSONL translation dataset and extract source/target pairs.
    examples: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            translation = record.get("translation")
            if not translation:
                raise ValueError(f"Expected 'translation' field in {path} but found: {record.keys()}")
            try:
                source = translation[source_lang]
                target = translation[target_lang]
            except KeyError as exc:
                raise KeyError(f"Missing language '{exc.args[0]}' in translation record from {path}") from exc
            examples.append({"source": source, "target": target})
    return examples


def chunked(iterable: List[Dict[str, str]], size: int) -> Iterable[List[Dict[str, str]]]:
    """Yield chunks of a given size from the iterable."""
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def setup_tokenizer(tokenizer, source_lang: str, target_lang: str, forced_bos_token: Optional[str]) -> None:
    # Configure tokenizer for multilingual models if applicable.
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = target_lang
    if forced_bos_token and hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.lang_code_to_id.get(forced_bos_token)


def evaluate_split(
    model,
    tokenizer,
    data: List[Dict[str, str]],
    *,
    source_prefix: str,
    batch_size: int,
    max_length: int,
    num_beams: int,
    device: torch.device,
    show_progress: bool,
) -> Dict[str, float]:
    if not data:
        return {}, [], [], []

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    predictions: List[str] = []
    references: List[str] = []
    sources_all: List[str] = []

    iterator = chunked(data, batch_size)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Generating", leave=False)

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            sources = [source_prefix + item["source"] for item in batch]
            references.extend(item["target"] for item in batch)
            sources_all.extend(item["source"] for item in batch)

            tokenized = tokenizer(
                sources,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            generated = model.generate(
                **tokenized,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=1,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(decoded)

    sacrebleu = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    chrf = chrf_metric.compute(predictions=predictions, references=[[ref] for ref in references], word_order=2)

    results = {
        "bleu": round(sacrebleu["score"], 4),
        "chrfpp": round(chrf["score"], 4),
    }
    return results, predictions, references, sources_all


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.to(device)

    if args.precision == "fp16":
        model = model.half()
    elif args.precision == "bf16":
        model = model.to(dtype=torch.bfloat16)

    setup_tokenizer(tokenizer, args.source_lang, args.target_lang, args.forced_bos_token)

    datasets: Dict[str, Path] = {}
    for item in args.datasets:
        if "=" not in item:
            raise ValueError(f"Dataset specification must look like name=path.jsonl, got: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        datasets[name] = path

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_name, path in datasets.items():
        data = load_split(path, args.source_lang, args.target_lang)
        if not data:
            print(f"Split: {split_name} (empty dataset, skipping)")
            continue
        metrics, predictions, references, sources = evaluate_split(
            model,
            tokenizer,
            data,
            source_prefix=args.source_prefix,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_beams=args.num_beams,
            device=device,
            show_progress=args.progress,
        )
        summary[split_name] = metrics
        print(f"Split: {split_name}")
        print("  BLEU   :", metrics["bleu"])
        print("  chrF++ :", metrics["chrfpp"])

        if output_dir is not None:
            if args.save_predictions:
                pred_file = output_dir / f"{split_name}_predictions.txt"
                with pred_file.open("w", encoding="utf-8") as handle:
                    handle.write("\n".join(predictions))
            if args.save_json:
                json_path = output_dir / f"{split_name}_triples.json"
                records = [
                    {"src": src, "mt": hyp, "ref": ref}
                    for src, hyp, ref in zip(sources, predictions, references)
                ]
                with json_path.open("w", encoding="utf-8") as handle:
                    json.dump(records, handle, ensure_ascii=False, indent=2)

    if output_dir is not None:
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
