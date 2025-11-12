import json
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

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

MAX_SELECTION_RETRIES = 3


def load_json(filepath: str) -> Dict:
    """Load a JSON file and return its contents as a dictionary."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


def _parse_selection(raw_selection) -> Optional[int]:
    """
    Parse a raw selection value and return an integer when one can be reliably extracted.
    Parameters
    ----------
    raw_selection : Any
        The raw input to parse. May be None, an int, a string, or any object with a string
        representation.
    Returns
    -------
    Optional[int]
        The parsed integer, or None if no integer can be obtained.
    Behavior
    --------
    - If raw_selection is None, returns None.
    - Converts raw_selection to str and strips leading/trailing whitespace.
      - If the resulting string is empty, returns None.
      - Attempts to parse the entire stripped string as an int; if successful, returns that int.
      - If full-string parsing fails, searches for the first contiguous sequence of digits
        bounded by word boundaries using the regex r'\b(\d+)\b' and returns that number if found.
      - If no suitable digits are found, returns None.
    Notes
    -----
    - The regex uses word boundaries, so digit sequences embedded inside alphanumeric words
      (e.g., 'abc123def') will not match, whereas digits separated by non-word characters or
      string boundaries (e.g., ' 123 ', '(123)') will match.
    Examples
    --------
    >>> _parse_selection(None)
    None
    >>> _parse_selection(' 42 ')
    42
    >>> _parse_selection('choice 7 selected')
    7
    >>> _parse_selection('abc123def')
    None
    """
    
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
    """
    Attempt to generate a numeric selection from a model, retrying on non-numeric outputs.
    This function calls model.generate(prompt) repeatedly (up to the module-level
    MAX_SELECTION_RETRIES) and uses the module-level helper _parse_selection to
    extract a numeric selection from the model's output. If a numeric selection is
    returned by _parse_selection, the function returns a tuple containing the parsed
    selection and the string form of the model output.
    Parameters
    ----------
    model
        An object exposing a generate(prompt: str) -> Any method. The returned value
        will be converted to a string and passed to _parse_selection.
    prompt : str
        The prompt to provide to model.generate.
    Returns
    -------
    tuple[int, str]
        (selection, output_str) where `selection` is the integer parsed from the
        model output and `output_str` is the string representation of the model's
        output that produced that selection.
    Raises
    ------
    ValueError
        If no numeric selection can be parsed from the model output after
        MAX_SELECTION_RETRIES attempts. The raised error message includes the last
        model output (as a string).
    Side effects
    ------------
    - Calls model.generate(prompt) on each attempt.
    - Calls _parse_selection(output) to determine if the output contains a numeric
      selection.
    - Prints a warning message to stdout for each failed attempt that returns a
      non-numeric selection.
    - May propagate exceptions raised by model.generate or _parse_selection.
    Notes
    -----
    - Relies on the presence of module-level names MAX_SELECTION_RETRIES and
      _parse_selection.
    - The function returns the first successful numeric selection encountered; it
      does not attempt to refine or validate numeric ranges beyond what
      _parse_selection provides.
    """
    
    last_output = ""
    for attempt in range(1, MAX_SELECTION_RETRIES + 1):
        output = model.generate(prompt)
        selection = _parse_selection(output)
        if selection is not None:
            return selection, str(output)
        last_output = str(output)
        print(
            f"[warn] Attempt {attempt} returned non-numeric selection. Retrying..."
        )

    raise ValueError(
        f"Model failed to return a numeric selection after "
        f"{MAX_SELECTION_RETRIES} attempts. Last output: {last_output}"
    )


def _write_results(outputs: Dict, experiment_name: Optional[str], label: str) -> None:
    """
    Write the provided outputs to a JSON file named by the experiment and label.

    Parameters
    ----------
    outputs : Dict
        A JSON-serializable mapping (e.g., dict) containing the results to be written.
    experiment_name : Optional[str]
        The experiment name used to construct the output filename. If None, the function
        does nothing and returns immediately.
    label : str
        A short label appended to the filename (for example, "train" or "eval").

    Returns
    -------
    None

    Side effects
    ------------
    - Ensures the target directory ("experiments/results") exists (creates it if necessary).
    - Writes the JSON-serialized `outputs` to a file at
      "experiments/results/{experiment_name}_{label}_results.json" using UTF-8 encoding,
      pretty-printed with indentation and ensure_ascii=False.

    Raises
    ------
    TypeError
        If `outputs` contains objects that are not JSON-serializable.
    OSError
        If directory creation or file writing fails due to filesystem errors or permissions.

    Examples
    --------
    >>> _write_results({"accuracy": 0.92}, "my_experiment", "eval")
    # Creates "experiments/results/my_experiment_eval_results.json" with the JSON contents.
    """
    if experiment_name is None:
        return
    output_path = f"experiments/results/pure_selection_results/exp_a/{experiment_name}_{label}_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)


def _run_prompt_experiment( model_names: List[str], dataset: Dict, prompt_factory: BasePromptFactory, 
                           label: str, experiment_name: Optional[str] = None,) -> Dict:
    
    """
    Run a prompt-based selection experiment across multiple text-generation models and collect results.
    This function iterates over a list of model names, instantiates each model via
    TextGenerationModelFactory, and for each item in the provided dataset runs a
    prompt-generation/evaluation loop using the provided prompt factory. For every
    sentence in a dataset row it generates a prompt (via prompt_factory.get_base_prompt),
    queries the model using _generate_with_retry, and records the model output along
    with a "gold" selection index. Results for all models are written out with
    _write_results and also returned.
    Parameters
    ----------
    model_names : List[str]
        Iterable of model identifiers (strings) that will be passed to
        TextGenerationModelFactory.create_instance to create model instances.
    dataset : Dict
        Mapping of source identifiers to row dictionaries. Each row is expected to
        be a mapping whose keys are candidate sentence strings. The source identifier
        (the dataset key) is used as 'src' in the output. Iteration is done via
        dataset.items().
    prompt_factory : BasePromptFactory
        Factory object used to construct prompts. Must implement:
            get_base_prompt(src_id, sentence_list) -> str
        where src_id is the current dataset key and sentence_list is the list of
        candidate sentence strings for that row.
    label : str
        Short label for the experiment (used in printed status messages and when
        writing results). Underscores in this label are converted to spaces in the
        printed status header.
    experiment_name : Optional[str], default=None
        Optional experiment name used by _write_results to determine the output
        location or filename. If None, _write_results will be called with the
        experiment_name set to None (behavior depends on _write_results).
    Returns
    -------
    Dict[str, List[Dict]]
        A mapping from model name to a list of result entries. Each result entry is
        a dictionary with keys:
            - 'src': the dataset key for the row (source identifier)
            - 'tgts': a list of per-sentence mappings; each element is a dict mapping
            the sentence string to a result object with the following fields:
                - 'gold_selection' (int): the index of the sentence in the row's
                sentence list (as produced by enumerate)
                - 'llm_selection' (Any): the selection/decision produced by the
                language model (value returned as `selection` from
                _generate_with_retry)
                - 'raw_output' (str): the raw text output produced by the model
    Side effects
    ------------
    - Prints progress headers to stdout for each model.
    - Uses tqdm to display a progress bar while iterating dataset rows.
    - Calls TextGenerationModelFactory.create_instance to instantiate models.
    - Calls _generate_with_retry(model, prompt) for model outputs.
    - Writes final aggregated results via _write_results(outputs, experiment_name, label).
    Raises
    ------
    Any exceptions raised by TextGenerationModelFactory.create_instance,
    prompt_factory.get_base_prompt, _generate_with_retry, or _write_results are
    propagated to the caller.
    Notes
    -----
    - The function assumes that row.keys() order (converted to list) is the
        canonical ordering used to determine gold_selection indices.
    - The prompt produced by prompt_factory.get_base_prompt is the same for every
        sentence in a given row (the prompt factory receives the full sentence list
        and the row identifier). The loop still records a separate result per sentence.
    """
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
            sentence_list = list(row.keys())
            row_results = {}

            for idx, sentence in enumerate(sentence_list):
                prompt = prompt_factory.get_base_prompt(idx_key, sentence_list)
                selection, raw_output = _generate_with_retry(model, prompt)
                row_results[sentence] = {
                    'gold_selection': idx,
                    'llm_selection': selection,
                    'raw_output': raw_output,
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
    ]

    run_zero_shot_experiment(
        model_names=selected_models,
        dataset=dataset_dict,
        experiment_name="many_to_1_experiment_a",
        source_language=default_source,
        target_language=default_target,
    )

    # run_few_shot_experiment(
    #     model_names=selected_models,
    #     dataset=dataset_dict,
    #     experiment_name="1_to_many_experiment_a",
    #     source_language=default_source,
    #     target_language=default_target,
    # )

    # run_chain_of_thought_experiment(
    #     model_names=selected_models,
    #     dataset=dataset_dict,
    #     experiment_name="1_to_many_experiment_a",
    #     source_language=default_source,
    #     target_language=default_target,
    # )
