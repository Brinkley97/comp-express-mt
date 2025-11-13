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
from tag_schema import (
    build_schema,
    build_schema_from_keys,
    map_values_to_schema,
    map_dict_to_schema,
    canonicalize_dataset_key,
)

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
    output_path = f"experiments/results/pure_selection_results/exp_c/{experiment_name}_{label}_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)


def _coerce_tags_from_values(tag_values, schema) -> Dict[str, str]:
    if isinstance(tag_values, dict):
        return map_dict_to_schema(tag_values, schema)
    if isinstance(tag_values, list):
        return map_values_to_schema(tag_values, schema)
    raise ValueError("Tags must be provided as a dict or list.")


def _extract_options_and_tags(row: Dict, direction: str) -> Tuple[List[str], List[Dict]]:
    """Return candidate sentences and tag info derived from per-option annotations."""
    candidate_sentences = list(row.keys())
    values = list(row.values())
    tag_runs: List[Dict] = []

    if not values:
        return candidate_sentences, tag_runs

    first_value = values[0]

    if isinstance(first_value, list):
        
        schema = build_schema(direction, len(first_value))
        for idx, option_values in enumerate(values):
            tags_dict = _coerce_tags_from_values(option_values, schema)
            tag_runs.append({
                "tags": tags_dict,
                "expected_idx": idx,
                "schema": schema,
            })

        return candidate_sentences, tag_runs

    if isinstance(first_value, dict):
        keys = set()
        for option_values in values:
            keys.update(canonicalize_dataset_key(k) for k in option_values.keys())

        
        schema = build_schema_from_keys(keys)
        for idx, option_values in enumerate(values):
            tags_dict = _coerce_tags_from_values(option_values, schema)
            tag_runs.append({
                "tags": tags_dict,
                "expected_idx": idx,
                "schema": schema,
            })
        return candidate_sentences, tag_runs

    raise ValueError("Unsupported tag structure; expected list or dict of tag values.")


def _run_prompt_experiment(
    model_names: List[str],
    dataset: Dict,
    prompt_factory,
    direction: str,
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
            candidate_sentences, tag_runs = _extract_options_and_tags(row, direction)

            for idx, run in enumerate(tag_runs):
                tags = run["tags"]
                schema = run.get("schema")
                prompt = prompt_factory.get_base_prompt(
                    source_sentence,
                    candidate_sentences,
                    tags=tags,
                    tag_schema=schema,
                )
            
                selection, raw_output = _generate_with_retry(model, prompt)

                tgts_entry = [{candidate_sentences[idx]: {
                    'gold_selection': idx,
                    'llm_selection': selection,
                    'raw_output': raw_output,
                }}]

                model_results.append({
                    'src': source_sentence,
                    'tags': tags,
                    'tgts': tgts_entry,
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
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
        direction=direction,
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
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
        direction=direction,
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
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
        direction=direction,
        label="chain_of_thought",
        experiment_name=experiment_name,
    )


def _infer_languages_from_path(data_path: str) -> Tuple[str, str]:
    lower = data_path.lower()
    if "many_to_one" in lower:
        return "English", "Akuapem Twi"
    return "Akuapem Twi", "English"


if __name__ == "__main__":
    data_path = 'data/tagged_data/many_to_one_akan_eng_mappings_with_tags.json'
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
        experiment_name="many_to_1_experiment_c",
        source_language=default_source,
        target_language=default_target,
    )

    run_few_shot_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        experiment_name="many_to_1_experiment_c",
        source_language=default_source,
        target_language=default_target,
    )

    run_chain_of_thought_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        experiment_name="many_to_1_experiment_c",
        source_language=default_source,
        target_language=default_target,
    )
