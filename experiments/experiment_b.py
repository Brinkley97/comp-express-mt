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
from prompting_strategies.experiment_b import (
    ZeroShotPromptFactory as TagZeroShotPromptFactory,
    FewShotPromptFactory as TagFewShotPromptFactory,
    ChainOfThoughtPromptFactory as TagChainOfThoughtPromptFactory,
)
from prompting_strategies.experiment_c import (
    ZeroShotPromptFactory as SelectionZeroShotPromptFactory,
    FewShotPromptFactory as SelectionFewShotPromptFactory,
    ChainOfThoughtPromptFactory as SelectionChainOfThoughtPromptFactory,
)
from pragmatic_tags import parse_tags, TagParseError
from tag_schema import build_schema

warnings.filterwarnings('ignore')

MAX_RETRIES = 3

AKAN_ALIASES = {"akan", "akuapem", "akuapem twi", "twi"}


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
        f"Unsupported language pairing for experiment B: {source_language} -> {target_language}"
    )


def _derive_schema(dataset: Dict, direction: str):
    if not dataset:
        raise ValueError("Dataset is empty; cannot derive tag schema.")
    first_row = next(iter(dataset.values()))
    first_value = next(iter(first_row.values()))
    if isinstance(first_value, list):
        tag_length = len(first_value)
    elif isinstance(first_value, dict):
        tag_length = len(first_value)
    else:
        raise ValueError("Unsupported tag structure; expected list or dict of tag values.")
    return build_schema(direction, tag_length)


def load_json(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


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


def _generate_tags_with_retry(model, base_prompt: str) -> Tuple[Dict, str]:
    last_error = ""
    prompt = base_prompt
    for attempt in range(1, MAX_RETRIES + 1):
        
        output = model.generate(prompt)
        # print(f"[debug] Tag generation attempt {attempt} output:\n{output}\n")
        try:
            tags = parse_tags(output)
            return tags.model_dump(), str(output)
        except TagParseError as exc:
            last_error = str(exc)
            print(f"[warn] Attempt {attempt} failed to parse tags: {exc}")
            prompt = (
                f"{base_prompt}\n\n"
                f"Your previous response was invalid because: {last_error}. AVOID MAKING THE SAME MISTAKE.\n"
                "Regenerate ONLY the TAGS line using the specified format and valid context you have. Don't make things up."
            )

    raise ValueError(
        f"Model failed to return valid TAGS after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def _generate_selection_with_retry(model, base_prompt: str) -> Tuple[int, str]:
    last_output = ""
    prompt = base_prompt
    for attempt in range(1, MAX_RETRIES + 1):
        output = model.generate(prompt)
        selection = _parse_selection(output)
        if selection is not None:
            return selection, str(output)
        last_output = str(output)
        print(f"[warn] Attempt {attempt} returned non-numeric selection. Retrying...")
        prompt = (
            f"{base_prompt}\n\n"
            "All options were validated by native speakers. You must choose one of them. "
            "Respond with ONLY the selection integer (e.g., '2'). Do not provide explanations."
        )

    raise ValueError(
        f"Model failed to return a numeric selection after {MAX_RETRIES} attempts. "
        f"Last output: {last_output}"
    )


def _write_results(outputs: Dict, experiment_name: Optional[str], label: str) -> None:
    if experiment_name is None:
        return
    output_path = f"experiments/results/pure_selection_results/exp_b/{experiment_name}_{label}_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)


def _run_prompt_experiment(
    model_names: List[str],
    dataset: Dict,
    tag_prompt_factory,
    selection_prompt_factory,
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
            candidate_sentences = list(row.keys())

            tag_prompt = tag_prompt_factory.get_base_prompt(
                source_sentence,
                candidate_sentences,
                tag_dimensions=tag_schema,
                fallback_values={
                    "AUDIENCE": "INDIVIDUAL",
                    "AGE": "PEER",
                    "FORMALITY": "FORMAL",
                    "SPEECH_ACT": "STATEMENT",
                },
            )
            tags_dict, tag_raw_output = _generate_tags_with_retry(model, tag_prompt)

            selection_prompt = selection_prompt_factory.get_base_prompt(
                source_sentence,
                candidate_sentences,
                tags=tags_dict,
            )
            selection, selection_raw_output = _generate_selection_with_retry(model, selection_prompt)

            row_results = {}
            for idx, sentence in enumerate(candidate_sentences):
                row_results[sentence] = {
                    'gold_selection': idx,
                    'llm_selection': selection,
                    'raw_output': selection_raw_output,
                    'predicted_tags': tags_dict,
                }

            model_results.append({
                'src': source_sentence,
                'tgts': [{sentence: output} for sentence, output in row_results.items()],
            })

        outputs[current_model] = model_results

    _write_results(outputs, experiment_name, label)
    return outputs


def run_zero_shot_experiment(
    model_names: List[str],
    dataset: Dict,
    tag_prompt_factory: Optional[TagZeroShotPromptFactory] = None,
    selection_prompt_factory: Optional[SelectionZeroShotPromptFactory] = None,
    experiment_name: Optional[str] = None,
    source_language: str = "Akuapem Twi",
    target_language: str = "English",
    akan_variant: str = "Akuapem Twi",
) -> Dict:
    tag_prompt_factory = tag_prompt_factory or TagZeroShotPromptFactory(
        source_language=source_language,
        target_language=target_language,
        akan_variant=akan_variant,
    )
    selection_prompt_factory = selection_prompt_factory or SelectionZeroShotPromptFactory(
        source_language=source_language,
        target_language=target_language,
        akan_variant=akan_variant,
        tags_source_description="model-predicted pragmatic context",
    )
    direction = _detect_direction(source_language, target_language)
    schema = _derive_schema(dataset, direction)
    
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        tag_prompt_factory=tag_prompt_factory,
        selection_prompt_factory=selection_prompt_factory,
        tag_schema=schema,
        label="zero_shot",
        experiment_name=experiment_name,
    )


def run_few_shot_experiment(
    model_names: List[str],
    dataset: Dict,
    tag_prompt_factory: Optional[TagFewShotPromptFactory] = None,
    selection_prompt_factory: Optional[SelectionFewShotPromptFactory] = None,
    experiment_name: Optional[str] = None,
    source_language: str = "Akuapem Twi",
    target_language: str = "English",
    akan_variant: str = "Akuapem Twi",
) -> Dict:
    tag_prompt_factory = tag_prompt_factory or TagFewShotPromptFactory(
        source_language=source_language,
        target_language=target_language,
        akan_variant=akan_variant,
    )
    selection_prompt_factory = selection_prompt_factory or SelectionFewShotPromptFactory(
        source_language=source_language,
        target_language=target_language,
        akan_variant=akan_variant,
        tags_source_description="model-predicted pragmatic context",
    )
    direction = _detect_direction(source_language, target_language)
    schema = _derive_schema(dataset, direction)
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        tag_prompt_factory=tag_prompt_factory,
        selection_prompt_factory=selection_prompt_factory,
        tag_schema=schema,
        label="few_shot",
        experiment_name=experiment_name,
    )


def run_chain_of_thought_experiment(
    model_names: List[str],
    dataset: Dict,
    tag_prompt_factory: Optional[TagChainOfThoughtPromptFactory] = None,
    selection_prompt_factory: Optional[SelectionChainOfThoughtPromptFactory] = None,
    experiment_name: Optional[str] = None,
    source_language: str = "Akuapem Twi",
    target_language: str = "English",
    akan_variant: str = "Akuapem Twi",
) -> Dict:
    tag_prompt_factory = tag_prompt_factory or TagChainOfThoughtPromptFactory(
        source_language=source_language,
        target_language=target_language,
        akan_variant=akan_variant,
    )
    selection_prompt_factory = selection_prompt_factory or SelectionChainOfThoughtPromptFactory(
        source_language=source_language,
        target_language=target_language,
        akan_variant=akan_variant,
        tags_source_description="model-predicted pragmatic context",
    )
    direction = _detect_direction(source_language, target_language)
    schema = _derive_schema(dataset, direction)
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        tag_prompt_factory=tag_prompt_factory,
        selection_prompt_factory=selection_prompt_factory,
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
    data_path = 'data/tagged_data/many_to_one_akan_eng_mappings_with_tags.json'
    dataset_dict = load_json(data_path)
    default_source, default_target = _infer_languages_from_path(data_path)

    print(f"Default source language: {default_source}, Default target language: {default_target}"
          "\n***************************************************************************\n")

    selected_models = [
        "llama-3.3-70b-instruct",
        "gpt-oss-120b",
    ]

    # run_zero_shot_experiment(
    #     model_names=selected_models,
    #     dataset=dataset_dict,
    #     experiment_name="many_to_1_experiment_b",
    #     source_language=default_source,
    #     target_language=default_target,
    # )

    # run_few_shot_experiment(
    #     model_names=selected_models,
    #     dataset=dataset_dict,
    #     experiment_name="many_to_1_experiment_b",
    #     source_language=default_source,
    #     target_language=default_target,
    # )

    run_chain_of_thought_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        experiment_name="many_to_1_experiment_b",
        source_language=default_source,
        target_language=default_target,
    )
