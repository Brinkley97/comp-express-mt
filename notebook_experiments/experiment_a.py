# %% [markdown]
# # Experiment A: Direct LLM Selection (Control)

# %%
print("Experiment A: Direct LLM Selection (Control)")

# %%
import os
import sys
import warnings

import pandas as pd

from tqdm import tqdm
from itertools import islice

# Get the current working directory of the notebook
notebook_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(notebook_dir, '../utils'))

from metrics import EvaluationMetric
from data_processing import DataProcessing
from llms import TextGenerationModelFactory
from prompting_strategies import ZeroShotPromptFactory, FewShotPromptFactory, ChainOfThoughtPrompt

# %%
pd.set_option('max_colwidth', 800)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


warnings.filterwarnings('ignore')

# %% [markdown]
# ## Load Data

# %%
file_name = "akuapem_with_tags_dataset-verified_data.xlsx"
path = os.path.join("../data/", file_name)

# %% [markdown]
# ### Load One to Many

# %%
print("####### LOAD DATASET #######")

# %%
one_to_many_df = pd.read_excel(path, sheet_name="1-M_tags")
akan_source_one = "Akan (Source, One)"
eng_target_many = "English (Target, Many)"
one_many_cols_to_rename = {"Akuapem Twi": akan_source_one, 
                  "English": eng_target_many
                  }
one_to_many_df.rename(columns=one_many_cols_to_rename, inplace=True)
one_to_many_df

# %%
print(f"Subset of dataset: {one_to_many_df.head(7)}")

# %%
one_to_many_df = one_to_many_df.loc[:33, :]
one_to_many_df

# %%
akan_one_to_eng_many_mappings = one_to_many_df.groupby(akan_source_one)[eng_target_many].apply(list).to_dict()
# for akan, e_list in akan_one_to_eng_many_mappings.items():
#     print(f"Key: {akan}")
#     print(f"Values: {e_list}\n")

# %% [markdown]
# ## Initialize Models + Propmt Models

# %%
print("####### INITIALIZE MODELS + PROMPTS #######")

# %%
tgmf = TextGenerationModelFactory
llama_31_70b_instruct = tgmf.create_instance('llama-3.1-70b-instruct')
llama_33_70b_instruct = tgmf.create_instance('llama-3.3-70b-instruct')
models = [llama_31_70b_instruct, llama_33_70b_instruct]
# models = [llama_33_70b_instruct]

# %%
print(" ### INITIALIZE MODELS ###")
print(f" 1. {models[0].model_name} \n 2. {models[1].model_name}")

# %%
def generate_data(prompt: str, model):
    model_output = model.generate(prompt)
    # model_outputs[model.model_name] = model_output
    return model_output

# %%
def get_llm_labels(df, prompt_model_col_name):
    # print(prompt_model_col_name)
    filt_llm_name = (df['llm_name'] == prompt_model_col_name)
    # print(filt_llm_name)
    filt_df = df[filt_llm_name]
    # print(filt_df)
    return filt_df['akan_sentence'], filt_df['true_label'], filt_df['llm_label']

# %% [markdown]
# ### Load Prompts

# %%
zero_shot_direct_prompt = ZeroShotPromptFactory("direct")
few_shot_direct_prompt = FewShotPromptFactory("direct")
chain_of_thought_prompt = ChainOfThoughtPrompt("direct")
prompt_names = [zero_shot_direct_prompt, few_shot_direct_prompt, chain_of_thought_prompt]

# %%
print(" ### INITIALIZE PROMPTS ###")
print(f" 1. {prompt_names[0].get_name()} \n 2. {prompt_names[1].get_name()} \n 3. {prompt_names[2].get_name()}")

# %%
mappings = akan_one_to_eng_many_mappings.items()
# mappings
# mappings = islice(akan_one_to_eng_many_mappings.items(), 3)

# %%
idx = 0
results = {}
for prompt_name in prompt_names:
    prompt_name_and_type = prompt_name.get_name()
    # print(f"\n\t{prompt_name_and_type} : {len(mappings)}")
    print(f"\n ### PROMPT NAME: {prompt_name_and_type} ###")
    prompt_results = []
    # for source_idx, (src, tgts) in enumerate(mappings):
    for src, tgts in tqdm(mappings):
        # print(f" Source ({source_idx}): {src}")
        for tgt_idx, tgt in enumerate(tgts):
            # print(f"\t\ttgt ({tgt_idx}): {tgt}")
            # print(f"  Target idx: ({tgt_idx})")
            prompt = prompt_name.get_base_prompt(akan_sentence=src, english_sentences=tgts)
            if idx == 0:
                # print(prompt_name_and_type)
                print(f"\n   PROMPT: {prompt}")
                idx = idx + 1
            for model in models:
                llm_result = generate_data(prompt, model)
                llm_result_to_sentence = tgts[int(llm_result)]
                # print(f"\t\tModel: {model.__name__()}\tGenerated: {llm_result}\n")
                result = (src, tgt, tgt_idx, int(llm_result), llm_result_to_sentence, f"{prompt_name_and_type}-{model.__name__()}", prompt_name_and_type)
                prompt_results.append(result)
    idx = 0
    results[prompt_name_and_type] = prompt_results
results

# %%
zero_shot_results = results[list(results.keys())[0]]
few_shot_results = results[list(results.keys())[1]]
chain_of_thought_results = results[list(results.keys())[2]]
col_names = ['akan_sentence', 'english_sentences', 'true_label', 'llm_label', 'llm_sentence', 'llm_name', 'prompt_name']

# %%
zero_shot_df = pd.DataFrame(zero_shot_results, columns=col_names)
few_shot_df = pd.DataFrame(few_shot_results, columns=col_names)
chain_of_thought_df = pd.DataFrame(chain_of_thought_results, columns=col_names)
print(f"Subset of COT: {chain_of_thought_df.head(7)}")

# %% [markdown]
# ## Multiple LLMs

# %%
print("####### FILTER BY PROMPT TYPE x LLM #######")

# %%
def realign_results(new_df, results_df, prompt_type, models):
    for model in models:
        
        model_name = model.__name__()
        col_prefix = f"{prompt_type}-{model_name}"
        print(f" PROMPT TYPE x LLM: {col_prefix}")

        akan_sentences, true_labels, llm_labels = get_llm_labels(results_df, col_prefix)
        # print(len(akan_sentences), len(true_labels), len(llm_labels))
        new_df[f"{col_prefix}-akan_sentence"] = akan_sentences.values
        new_df[f"true_label"] = true_labels.values
        new_df[f"{col_prefix}-llm_label"] = llm_labels.values
        
    return new_df

# %%
full_results_df = pd.DataFrame()
realign_results(full_results_df, zero_shot_df, prompt_names[0].get_name(), models)
realign_results(full_results_df, few_shot_df, prompt_names[1].get_name(), models)
realign_results(full_results_df, chain_of_thought_df, prompt_names[2].get_name(), models)

# %%
# print(f" Subset of results by prompt x llm: {full_results_df.head(7)}")

# %%
print("\n####### EVALUATION METRICS #######")

# %%
get_metrics = EvaluationMetric()

actual_label = full_results_df['true_label'].values
for model in models:
    for prompt_name in prompt_names:
        llm_labels_col_name = f"{prompt_name.get_name()}-{model.__name__()}-llm_label"
        # print(f" col_prefix: {llm_labels_col_name}")
        print(f" TRUE LABELS: {actual_label}")
        model_predictions = full_results_df[llm_labels_col_name].values
        # print(f"{llm_labels_col_name}: {model_predictions}")
        print(f" PROMPT TYPE x LLM NAME: {llm_labels_col_name}")
        print(f"\tWITH LABELS {model_predictions}")
        get_metrics.eval_classification_report(actual_label, model_predictions)
        print(" ============================================================================================")
        print()


# %%
# save_zero_shot_results_dir = os.path.join('../data/', "experiement_a-zero_fewresults_df.csv")
# print(save_zero_shot_results_dir)
# DataProcessing.save_data(full_results_df, save_zero_shot_results_dir)

# %%
# save_zero_shot_results_dir = os.path.join('../data/', "zero_shot_results_df.csv")
# print(save_zero_shot_results_dir)
# DataProcessing.save_data(zero_shot_results_df, save_zero_shot_results_dir)


