import os
import sys
import warnings

import pandas as pd

from tqdm import tqdm
from itertools import islice
import json

# Get the current working directory of the notebook
notebook_dir = os.getcwd()

print(f"Notebook Directory: {notebook_dir}")
# Add the parent directory to the system path
sys.path.append(os.path.join(notebook_dir, 'utils/'))

from metrics import EvaluationMetric
from data_processing import DataProcessing
from llms import TextGenerationModelFactory
from prompting_strategies import ZeroShotPromptFactory, FewShotPromptFactory, ChainOfThoughtPrompt

from typing import List, Dict

warnings.filterwarnings('ignore')

def load_json(filepath: str) -> Dict:
    """Load a JSON file and return its contents as a dictionary."""
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def run_zero_shot_experiment(model_name: List[str], dataset: Dict, source_lang='akan', target_lang='english', prompts: [ZeroShotPromptFactory, FewShotPromptFactory, ChainOfThoughtPrompt]=ZeroShotPromptFactory, experiment_name=None) ->Dict:
    outputs = {}

    # Initialize the text generation model
    for current_model in model_name:
        print(f"\n***********************************************************",
              f"Running Zero Shot experiment with {current_model}",
              f"\n***********************************************************\n")
        llm_instance = TextGenerationModelFactory
        model = llm_instance.create_instance(current_model)

        results = []
        

        for index, row in tqdm(dataset.items(), total=len(dataset)):
            row_results = {}
            sentence_list = list(row.keys())
            for idx, sentence in enumerate(sentence_list):
                prompt = prompts.get_base_prompt(index, sentence_list)

                print(f"Prompt: {prompt}")
                output = model.generate(prompt)

                # write to row results
                row_results[sentence] = {'gold_selection': idx, 'llm_selection': output}

            # write to results
            # {
            #     'src': index,
            #     'tgts': [{ 'sentence': output }, ...]
            # }
            results.append({
                'src': index,
                'tgts': [{sentence: output} for sentence, output in row_results.items()]
            })

            break
    
        # initialize output dicts with results for each model
        outputs[current_model] = results
        
    # write to json file
    if experiment_name is not None:
        output_path = f"experiments/results/{experiment_name}_zero_shot_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    return outputs

    #     # Generate the model's response
    #     response = model.generate_text(prompt)

    #     # Evaluate the response
    #     metric = EvaluationMetric(target_text, response)
    #     bleu_score = metric.compute_bleu()
    #     rouge_score = metric.compute_rouge()

    #     results.append({
    #         'source_text': source_text,
    #         'target_text': target_text,
    #         'model_response': response,
    #         'bleu_score': bleu_score,
    #         'rouge_score': rouge_score
    #     })

    # return pd.DataFrame(results)


if __name__ == "__main__":

    model_name = 'llama-3.1-70b-instruct'
    data_path = 'data/tagged_data/one_to_many_akan_eng_mappings_with_tags.json'
    dataset_dict = load_json(data_path)

    selected_models = ["llama-3.3-70b-instruct", "mistral-small-3.1", "gpt-oss-120b", "gemma-3-27b-it"]

    zero_shot_direct_prompt = ZeroShotPromptFactory("direct")

    run_zero_shot_experiment(model_name=selected_models, dataset=dataset_dict, prompts=zero_shot_direct_prompt, experiment_name="1_to_many_experiment_a")