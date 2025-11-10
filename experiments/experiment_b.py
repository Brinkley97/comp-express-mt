import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

notebook_dir = os.getcwd()
print(f"Notebook Directory: {notebook_dir}")
sys.path.append(os.path.join(notebook_dir, 'utils/'))

from llms import TextGenerationModelFactory
from prompting_strategies.base import BasePromptFactory
from prompting_strategies.experiment_b import (
    ZeroShotPromptFactory,
    FewShotPromptFactory,
    ChainOfThoughtPromptFactory,
)
from pragmatic_tags import parse_tags_and_selection, TagParseError

warnings.filterwarnings('ignore')

MAX_SELECTION_RETRIES = 3


def load_json(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


def _generate_with_tags(model, prompt: str) -> Tuple[Dict, int, str]:
    last_error = ""
    for attempt in range(1, MAX_SELECTION_RETRIES + 1):
        output = model.generate(prompt)
        try:
            tags, selection = parse_tags_and_selection(output)
            return tags.model_dump(), selection, str(output)
        except TagParseError as exc:
            last_error = str(exc)
            print(f"[warn] Attempt {attempt} failed to parse tags/selection: {exc}")

    raise ValueError(
        f"Model failed to return valid TAGS/SELECTION after {MAX_SELECTION_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def _write_results(outputs: Dict, experiment_name: Optional[str], label: str) -> None:
    if experiment_name is None:
        return
    output_path = f"experiments/results/{experiment_name}_{label}_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)


def _run_prompt_experiment(
    model_names: List[str],
    dataset: Dict,
    prompt_factory: BasePromptFactory,
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

        for idx_key, row in tqdm(dataset.items(), total=len(dataset)):
            candidate_sentences = list(row.keys())
            prompt = prompt_factory.get_base_prompt(idx_key, candidate_sentences)
            tags_dict, selection, raw_output = _generate_with_tags(model, prompt)

            row_results = {}
            for option_index, sentence in enumerate(candidate_sentences):
                row_results[sentence] = {
                    'gold_selection': option_index,
                    'llm_selection': selection,
                }

            model_results.append({
                'src': idx_key,
                'tags': tags_dict,
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
        "gemma-3-27b-it",
    ]

    run_zero_shot_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        experiment_name="experiment_b",
        source_language=default_source,
        target_language=default_target,
    )
