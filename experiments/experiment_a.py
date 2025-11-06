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

def run_zero_shot_experiment(model_name: str, dataset: Dict, source_lang='akan', target_lang='english', prompts: [ZeroShotPromptFactory, FewShotPromptFactory, ChainOfThoughtPrompt]=ZeroShotPromptFactory) ->Dict:
    # Initialize the text generation model
    llm_instance = TextGenerationModelFactory
    model = llm_instance.create_instance(model_name)

    results = []
    outputs = {}

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
    

    print(f"Sample results: {results}")

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

    zero_shot_direct_prompt = ZeroShotPromptFactory("direct")

    run_zero_shot_experiment(model_name=model_name, dataset=dataset_dict, prompts=zero_shot_direct_prompt)