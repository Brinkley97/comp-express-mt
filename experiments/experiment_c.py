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

warnings.filterwarnings('ignore')

MAX_SELECTION_RETRIES = 3


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


def _generate_with_retry(model, prompt: str) -> Tuple[int, str]:
    last_output = ""
    for attempt in range(1, MAX_SELECTION_RETRIES + 1):
        output = model.generate(prompt)
        selection = _parse_selection(output)
        if selection is not None:
            return selection, str(output)
        last_output = str(output)
        print(f"[warn] Attempt {attempt} returned non-numeric selection. Retrying...")
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


TAG_KEYS = ["Gender", "Animacy", "Status", "Age", "Formality", "Audience", "Speech_Act"]


def _coerce_tags(tag_values) -> Dict[str, str]:
    if isinstance(tag_values, dict):
        return tag_values
    if isinstance(tag_values, list):
        if len(tag_values) != len(TAG_KEYS):
            raise ValueError(f"Expected {len(TAG_KEYS)} tag values, got {len(tag_values)}")
        return {key: value for key, value in zip(TAG_KEYS, tag_values)}
    raise ValueError("Tags must be provided as a dict or list.")


def _extract_options_and_tags(row: Dict) -> Tuple[List[str], Dict[str, str]]:
    """Support rows of the form {'options': {...}, 'tags': {...}} or legacy dicts."""
    if "options" in row:
        options_dict = row["options"]
    else:
        options_dict = {k: v for k, v in row.items() if k != "tags"}

    tags_data = row.get("tags")
    if tags_data is None:
        raise ValueError("Experiment C requires 'tags' entry per sentence.")

    tags = _coerce_tags(tags_data)
    return list(options_dict.keys()), tags


def _run_prompt_experiment(
    model_names: List[str],
    dataset: Dict,
    prompt_factory,
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
            candidate_sentences, tags = _extract_options_and_tags(row)
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
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
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
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
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
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
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

    selected_models = [
        "gpt-oss-120b",
        "llama-3.3-70b-instruct",
        "mistral-small-3.1",
        "granite-3.3-8b-instruct",
    ]

    run_zero_shot_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        experiment_name="experiment_c",
        source_language=default_source,
        target_language=default_target,
    )
