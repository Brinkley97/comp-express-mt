import json
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

notebook_dir = os.getcwd()
print(f"Notebook Directory: {notebook_dir}")
sys.path.append(os.path.join(notebook_dir, 'utils/'))

from llms import TextGenerationModelFactory
from prompting_strategies.experiment_c import (
    ZeroShotPromptFactory,
    FewShotPromptFactory,
    ChainOfThoughtPromptFactory,
)
from tag_schema import build_schema, map_values_to_schema

warnings.filterwarnings('ignore')

MAX_SELECTION_RETRIES = 3

AKAN_ALIASES = {"akan", "akuapem", "akuapem twi", "twi"}


def load_json(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


def _is_akan_language(name: Optional[str]) -> bool:
    if not name:
        return False
    return name.strip().lower() in AKAN_ALIASES


def _detect_direction(source_language: str, target_language: str) -> str:
    if _is_akan_language(source_language) and not _is_akan_language(target_language):
        return "akan_to_english"
    if not _is_akan_language(source_language) and _is_akan_language(target_language):
        return "english_to_akan"
    raise ValueError(
        f"Unsupported language pairing for experiment C: {source_language} -> {target_language}"
    )


def _derive_schema(dataset: Dict, direction: str):
    if not dataset:
        raise ValueError("Dataset is empty; cannot derive tag schema.")
    first_row = next(iter(dataset.values()))
    sample_value = next(iter(first_row.values()))
    if isinstance(sample_value, list):
        tag_length = len(sample_value)
    elif isinstance(sample_value, dict):
        tag_length = len(sample_value)
    else:
        raise ValueError("Unsupported tag structure; expected list or dict of tag values.")
    return build_schema(direction, tag_length)


def _parse_selection(raw_selection) -> Optional[int]:
    if raw_selection is None:
        return None
    text = str(raw_selection).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        match = re.search(r'\b(\d+)\b', text)
        if match:
            return int(match.group(1))
    return None


def _generate_with_retry(model, base_prompt: str) -> Tuple[int, str]:
    last_output = ""
    prompt = base_prompt
    for attempt in range(1, MAX_SELECTION_RETRIES + 1):
        print(f"\n Validating prompt schema: {prompt} \n")
        output = model.generate(prompt)
        selection = _parse_selection(output)
        if selection is not None:
            return selection, str(output)
        last_output = str(output)
        print(f"[warn] Attempt {attempt} returned non-numeric selection. Retrying...")
        prompt = (
            f"{base_prompt}\n\n"
            "Your previous response did not include a valid numeric selection. "
            "Respond again with ONLY the selection integer (e.g., '2')."
        )
    raise ValueError(
        f"Model failed to return a numeric selection after {MAX_SELECTION_RETRIES} attempts. "
        f"Last output: {last_output}"
    )


def _write_results(outputs: Dict, experiment_name: Optional[str], label: str) -> None:
    if experiment_name is None:
        return
    output_path = f"experiments/results/{experiment_name}_{label}_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)


def _coerce_tags_from_values(tag_values, schema) -> Dict[str, str]:
    if isinstance(tag_values, dict):
        return tag_values
    if isinstance(tag_values, list):
        return map_values_to_schema(tag_values, schema)
    raise ValueError("Tags must be provided as a dict or list.")


def _extract_options_and_tags(row: Dict, schema) -> Tuple[List[str], List[Dict]]:
    """Return candidate sentences and a list of tag runs (tags + expected index)."""
    if "options" in row:
        options_dict = row["options"]
    else:
        options_dict = {k: v for k, v in row.items() if k != "tags"}

    candidate_sentences = list(options_dict.keys())

    tag_runs: List[Dict] = []

    if "tags" in row:
        tags_dict = _coerce_tags_from_values(row["tags"], schema)
        target_sentence = row.get("target_sentence")
        expected_idx = None
        if target_sentence and target_sentence in candidate_sentences:
            expected_idx = candidate_sentences.index(target_sentence)
        elif "target_index" in row:
            expected_idx = row["target_index"]
        tag_runs.append({"tags": tags_dict, "expected_idx": expected_idx})
        return candidate_sentences, tag_runs

    for idx, values in enumerate(options_dict.values()):
        tags_dict = _coerce_tags_from_values(values, schema)
        tag_runs.append({"tags": tags_dict, "expected_idx": idx})

    return candidate_sentences, tag_runs


def _run_prompt_experiment(
    model_names: List[str],
    dataset: Dict,
    prompt_factory,
    tag_schema,
    label: str,
    experiment_name: Optional[str] = None,
) -> Dict:
    outputs: Dict[str, List[Dict]] = {}

    for current_model in model_names:
        print(
            "\n***********************************************************\n",
            f"Running {label.replace('_', ' ').title()} experiment with {current_model}",
            "\n***********************************************************\n",
        )
        model = TextGenerationModelFactory.create_instance(current_model)
        model_results = []
        
        for source_sentence, row in tqdm(dataset.items(), total=len(dataset)):
            candidate_sentences, tag_runs = _extract_options_and_tags(row, tag_schema)

            for run in tag_runs:
                tags = run["tags"]
                expected_idx = run.get("expected_idx")
                prompt = prompt_factory.get_base_prompt(
                    source_sentence,
                    candidate_sentences,
                    tags=tags,
                )
                selection, raw_output = _generate_with_retry(model, prompt)

                row_results = {}
                for idx, sentence in enumerate(candidate_sentences):
                    row_results[sentence] = {
                        'gold_selection': idx,
                        'llm_selection': selection,
                    }

                model_results.append({
                    'src': source_sentence,
                    'tags': tags,
                    'expected_selection': expected_idx,
                    'llm_selection': selection,
                    'raw_output': raw_output,
                    'tgts': [{sentence: output} for sentence, output in row_results.items()],
                })

        outputs[current_model] = model_results

    _write_results(outputs, experiment_name, label)
    return outputs


def run_zero_shot_experiment(
    model_names: List[str],
    dataset: Dict,
    prompt_factory: Optional[ZeroShotPromptFactory] = None,
    experiment_name: Optional[str] = None,
    source_language: str = "Akuapem Twi",
    target_language: str = "English",
    akan_variant: str = "Akuapem Twi",
) -> Dict:
    prompt_factory = prompt_factory or ZeroShotPromptFactory(
        source_language=source_language,
        target_language=target_language,
        akan_variant=akan_variant,
    )
    direction = _detect_direction(source_language, target_language)
    schema = _derive_schema(dataset, direction)
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
        tag_schema=schema,
        label="zero_shot",
        experiment_name=experiment_name,
    )


def run_few_shot_experiment(
    model_names: List[str],
    dataset: Dict,
    prompt_factory: Optional[FewShotPromptFactory] = None,
    experiment_name: Optional[str] = None,
    source_language: str = "Akuapem Twi",
    target_language: str = "English",
    akan_variant: str = "Akuapem Twi",
) -> Dict:
    prompt_factory = prompt_factory or FewShotPromptFactory(
        source_language=source_language,
        target_language=target_language,
        akan_variant=akan_variant,
    )
    direction = _detect_direction(source_language, target_language)
    schema = _derive_schema(dataset, direction)
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
        tag_schema=schema,
        label="few_shot",
        experiment_name=experiment_name,
    )


def run_chain_of_thought_experiment(
    model_names: List[str],
    dataset: Dict,
    prompt_factory: Optional[ChainOfThoughtPromptFactory] = None,
    experiment_name: Optional[str] = None,
    source_language: str = "Akuapem Twi",
    target_language: str = "English",
    akan_variant: str = "Akuapem Twi",
) -> Dict:
    prompt_factory = prompt_factory or ChainOfThoughtPromptFactory(
        source_language=source_language,
        target_language=target_language,
        akan_variant=akan_variant,
    )
    direction = _detect_direction(source_language, target_language)
    schema = _derive_schema(dataset, direction)
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
        tag_schema=schema,
        label="chain_of_thought",
        experiment_name=experiment_name,
    )


def _infer_languages_from_path(data_path: str) -> Tuple[str, str]:
    lower = data_path.lower()
    if "many_to_one" in lower:
        return "English", "Akuapem Twi"
    return "Akuapem Twi", "English"


if __name__ == "__main__":
    data_path = 'data/tagged_data/one_to_many_akan_eng_mappings_with_tags.json'
    dataset_dict = load_json(data_path)
    default_source, default_target = _infer_languages_from_path(data_path)

    print(f"Default source language: {default_source}, Default target language: {default_target}"
          "\n***************************************************************************\n")

    selected_models = [
        "gpt-oss-120b",
        "llama-3.3-70b-instruct",
    ]

    run_zero_shot_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        experiment_name="1_to_many_experiment_c",
        source_language=default_source,
        target_language=default_target,
    )
