#!/usr/bin/env python3
"""Generate top-k translation candidates from a fine-tuned seq2seq model."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the fine-tuned model directory or model identifier on the Hub.",
    )
    parser.add_argument(
        "--inputs",
        help="Path to a text file containing one source sentence per line. Defaults to stdin if omitted.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save JSONL predictions. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--source_prefix",
        default="",
        help="Optional text prefix injected before every source sentence (e.g. T5 translation hint).",
    )
    parser.add_argument(
        "--source_lang",
        help="Tokenizer source language id (required for multilingual tokenizers such as mBART).",
    )
    parser.add_argument(
        "--target_lang",
        help="Tokenizer target language id (required for multilingual tokenizers such as mBART).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of sentences to process per batch.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum length for generated sequences.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of candidate translations to return per source sentence.",
    )
    parser.add_argument(
        "--strategy",
        choices=["beam", "sample"],
        default="beam",
        help="Generation strategy: deterministic beam search or stochastic top-k sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling cumulative probability mass (only used when --strategy sample).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for sampling (only used when --strategy sample).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Execution device override, e.g. 'cuda', 'cuda:0', 'mps', or 'cpu'. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--skip_special_tokens",
        dest="skip_special_tokens",
        action="store_true",
        default=True,
        help="Remove special tokens from decoded outputs (recommended).",
    )
    parser.add_argument(
        "--keep_special_tokens",
        dest="skip_special_tokens",
        action="store_false",
        help="Retain special tokens in decoded outputs.",
    )
    return parser.parse_args()


def load_lines(path: Optional[str]) -> List[str]:
    if path is None:
        return [line.rstrip("\n") for line in sys.stdin if line.strip()]
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def batched(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_tokenizer(tokenizer, source_lang: Optional[str], target_lang: Optional[str]) -> None:
    if source_lang is not None:
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = source_lang
        if hasattr(tokenizer, "set_src_lang_special_tokens"):
            tokenizer.set_src_lang_special_tokens(source_lang)
    if target_lang is not None:
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = target_lang
        if hasattr(tokenizer, "set_tgt_lang_special_tokens"):
            tokenizer.set_tgt_lang_special_tokens(target_lang)


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    prepare_tokenizer(tokenizer, args.source_lang, args.target_lang)

    sentences = load_lines(args.inputs)
    if not sentences:
        print("No input sentences provided.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else None
    output_handle = output_path.open("w", encoding="utf-8") if output_path is not None else None

    generation_kwargs = {
        "max_length": args.max_length,
        "num_return_sequences": args.top_k,
        "no_repeat_ngram_size": 0,
    }

    if args.strategy == "beam":
        generation_kwargs.update({
            "num_beams": args.top_k,
            "do_sample": False,
        })
    else:
        generation_kwargs.update({
            "do_sample": True,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    with torch.no_grad():
        for batch in batched(sentences, args.batch_size):
            prefixed_inputs = [f"{args.source_prefix}{line}" for line in batch]
            tokenized = tokenizer(
                prefixed_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model.generate(**tokenized, **generation_kwargs)
            outputs = outputs.view(len(batch), args.top_k, -1)

            decoded = [
                tokenizer.batch_decode(candidate_set, skip_special_tokens=args.skip_special_tokens)
                for candidate_set in outputs
            ]

            for src, candidates in zip(batch, decoded):
                record = {"source": src, "candidates": candidates}
                serialized = json.dumps(record, ensure_ascii=False)
                if output_handle is not None:
                    output_handle.write(serialized + "\n")
                else:
                    print(serialized)

    if output_handle is not None:
        output_handle.close()


if __name__ == "__main__":
    main()
