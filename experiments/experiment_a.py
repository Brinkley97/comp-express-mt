import json
import os
import sys
import warnings
from typing import Dict, List, Optional

from tqdm import tqdm

# Get the current working directory of the notebook
notebook_dir = os.getcwd()
print(f"Notebook Directory: {notebook_dir}")

# Add the parent directory to the system path for utils imports
sys.path.append(os.path.join(notebook_dir, 'utils/'))

from llms import TextGenerationModelFactory
from prompting_strategies.base import BasePromptFactory
from prompting_strategies.experiment_a import (
    ZeroShotPromptFactory,
    FewShotPromptFactory,
    ChainOfThoughtPromptFactory,
)

warnings.filterwarnings('ignore')


def load_json(filepath: str) -> Dict:
    """Load a JSON file and return its contents as a dictionary."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


def _coerce_selection(raw_selection) -> int:
    """Ensure the model output is parsed into an integer option index."""
    try:
        return int(str(raw_selection).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Model output must be an integer selection, got: {raw_selection}") from exc


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
            "\n***********************************************************",
            f"Running {label.replace('_', ' ').title()} experiment with {current_model}",
            "\n***********************************************************\n",
        )
        model = TextGenerationModelFactory.create_instance(current_model)
        model_results = []

        for idx_key, row in tqdm(dataset.items(), total=len(dataset)):
            sentence_list = list(row.keys())
            row_results = {}

            for idx, sentence in enumerate(sentence_list):
                prompt = prompt_factory.get_base_prompt(idx_key, sentence_list)
                print(f"Prompt: {prompt}")
                output = model.generate(prompt)
                row_results[sentence] = {
                    'gold_selection': idx,
                    'llm_selection': _coerce_selection(output),
                }

            model_results.append({
                'src': idx_key,
                'tgts': [{sentence: output} for sentence, output in row_results.items()]
            })

        outputs[current_model] = model_results

    _write_results(outputs, experiment_name, label)
    return outputs


def run_zero_shot_experiment(
    model_names: List[str],
    dataset: Dict,
    prompt_factory: Optional[ZeroShotPromptFactory] = None,
    experiment_name: Optional[str] = None,
) -> Dict:
    prompt_factory = prompt_factory or ZeroShotPromptFactory()
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
) -> Dict:
    prompt_factory = prompt_factory or FewShotPromptFactory()
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
) -> Dict:
    prompt_factory = prompt_factory or ChainOfThoughtPromptFactory()
    return _run_prompt_experiment(
        model_names=model_names,
        dataset=dataset,
        prompt_factory=prompt_factory,
        label="chain_of_thought",
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    data_path = 'data/tagged_data/one_to_many_akan_eng_mappings_with_tags.json'
    dataset_dict = load_json(data_path)

    selected_models = [
        "llama-3.3-70b-instruct",
        "mistral-small-3.1",
        "gpt-oss-120b",
        "gemma-3-27b-it",
    ]

    zero_shot_prompt = ZeroShotPromptFactory()
    few_shot_prompt = FewShotPromptFactory()
    cot_prompt = ChainOfThoughtPromptFactory()

    run_zero_shot_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        prompt_factory=zero_shot_prompt,
        experiment_name="1_to_many_experiment_a",
    )

    run_few_shot_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        prompt_factory=few_shot_prompt,
        experiment_name="1_to_many_experiment_a",
    )

    run_chain_of_thought_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        prompt_factory=cot_prompt,
        experiment_name="1_to_many_experiment_a",
    )
