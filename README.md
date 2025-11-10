# CompExpress-MT

African machine translation research codebase with a focus on low-resource African languages (LAFAND-MT). The repo now bundles fine-tuning, evaluation, data analysis, and LLM-assisted prompt selection experiments in one place. Core fine-tuning and evaluation scripts live under `baseline_codebase/`.

- Paper: A Few Thousand Translations Go a Long Way (LAFAND-MT)
- Supported models: MT5, ByT5, MBART50, M2M100, Afri variants, + external LLMs via prompt factories

## Repository layout

- Training & generation scripts (`baseline_codebase/`)
  - `baseline_codebase/run_translation.py` (HF fine-tuning/inference with experiment logging)
  - `baseline_codebase/generate_topk_translations.py` (top-k candidate generation via beam or sampling)
- Evaluation (`baseline_codebase/`)
  - `baseline_codebase/evaluate_translations.py` (BLEU/chrF++/GLEU with JSONL inputs, batch options, export utilities)
  - `baseline_codebase/comet_evaluations.py` (COMET STL + optional QE analysis, statistics, plots)
- Data & artifacts
  - `data/finetune_data/<src-tgt>/` (training JSONL/CSV pairs)
  - `data/tagged_data/` + `data/misc_data/` (gold standards, context tags, aggregation exports)
  - `evals/` (archived metrics, triples, comparison summaries)
  - `models/` (HF checkpoints e.g. `m2m100_en_tw_418M/`, `baselines/`)
- Experiments & notebooks
  - `experiments/` (analysis scripts)
  - `experiments/notebooks/` (pandas + viz workflows for creating labels, QA, etc.)
- Utilities
  - `utils/data_processing.py` (pandas helpers for context datasets)
  - `utils/prompting_strategies.py` (prompt factories: zero/few-shot, chain-of-thought)
  - `utils/llms.py` (LLM factory for Groq/OpenAI-compatible deployments)
  - `utils/metrics.py` (classification metrics wrappers)
- Project config
  - `requirements.txt`
  - `.env` (optional, for API keys consumed by `utils/llms.py`)

## Setup

- Python 3.9–3.11 recommended
- Create an isolated environment (conda or `python -m venv .venv && source .venv/bin/activate`)
- Install core dependencies: `pip install -r requirements.txt`
- Optional HF stack pin: `pip install "transformers==4.31.0" datasets sentencepiece sacrebleu accelerate torch`
- Enable distributed/Accelerate: `accelerate config`
- For LLM experiments add environment keys in `.env` (`GROQ_CLOUD_API_KEY`, `NAVI_GATOR_API_KEY`, `HUGGING_FACE_API_KEY`)

### Quickstart

1. Clone and enter the repo:
   ```bash
   git clone https://github.com/Brinkley97/comp_express_mt.git
   cd comp_express_mt
   ```
2. Set up Python (3.9–3.11) and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Configure Hugging Face and (optionally) Weights & Biases:
   ```bash
   huggingface-cli login          # needed for gated checkpoints
   wandb login                    # optional if using W&B logging
   ```
4. Run `accelerate config` once, then you can launch trainer/eval scripts with `accelerate launch` when using multi-GPU or mixed precision.

### Environment variables

- Copy `.env.example` (if present) to `.env`, or create it manually with the API keys you need. The scripts read via `python-dotenv`.
- Recommended variables:
  - `GROQ_CLOUD_API_KEY`, `NAVI_GATOR_API_KEY`, `HUGGING_FACE_API_KEY` for LLM endpoints
  - `WANDB_PROJECT`, `WANDB_ENTITY` if you consistently log to W&B

### Data preparation

- Ensure fine-tuning JSON files follow the structure in `data/finetune_data/<src-tgt>/`.
- Convert CSV gold data to JSON using `utils/csv_to_json.py`:
  ```bash
  python utils/csv_to_json.py \
    --input_csv data/tagged_data/akuapem_with_tags_dataset-verified_data.csv \
    --src_col en --tgt_col twi \
    --output_json data/finetune_data/en-tw/train.json
  ```
- Use `utils/create_en_tw_json.py` or other helpers in `utils/data_processing.py` to standardize schema (see docstrings for options).
- After creating training/validation/test splits, place them under `data/finetune_data/<src-tgt>/`. Sample splits already exist for Twi ↔ English.
- For evaluation artifacts, create a folder under `evals/<experiment_name>/` where metrics and predictions will be written.

### GPU requirements

- Scripts leverage PyTorch; NVIDIA GPU with >=12GB VRAM recommended for training M2M100/MT5 variants.
- Install CUDA-enabled `torch` that matches your system (pip may install CPU by default).
- For Apple Silicon, install `torch` with `pip install torch==<version> --extra-index-url https://download.pytorch.org/whl/cpu` and rely on CPU or Metal (depending on support).

### Running experiments end-to-end

1. **Fine-tune / run inference** with `baseline_codebase/run_translation.py`. Include `--predict_with_generate` to output translations during validation.
2. **Evaluate** with `baseline_codebase/evaluate_translations.py`, pointing to the saved checkpoint directory.
3. **Generate top-k candidates** when you need n-best lists for human annotation via `baseline_codebase/generate_topk_translations.py`.
4. **Score with COMET** by feeding the exported triples to `baseline_codebase/comet_evaluations.py`.
5. **Prompting experiments** can be executed via custom scripts that import `utils/prompting_strategies.py` and `utils/llms.py`.

Example fine-tune + eval sequence:
```bash
# Train / finetune
accelerate launch baseline_codebase/run_translation.py \
  --model_name_or_path facebook/m2m100_418M \
  --source_lang en --target_lang twi \
  --train_file data/finetune_data/en-tw/train.json \
  --validation_file data/finetune_data/en-tw/dev.json \
  --test_file data/finetune_data/en-tw/test.json \
  --output_dir models/m2m100_en_tw_ep10 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --predict_with_generate --fp16

# Evaluate saved checkpoint
accelerate launch baseline_codebase/evaluate_translations.py \
  --model_path models/m2m100_en_tw_ep10 \
  --source_lang en --target_lang twi \
  --datasets test=data/finetune_data/en-tw/test.json \
  --output_dir evals/m2m100_en_tw_eval \
  --batch_size 16 --num_beams 5 --precision fp16 \
  --save_predictions --save_json --progress

# Optional COMET scoring
python baseline_codebase/comet_evaluations.py \
  --input_dir evals/m2m100_en_tw_eval \
  --model masakhane/africomet-stl-1.1 \
  --batch_size 16 --gpus 1
```

### Data format

Primary training/eval data uses HuggingFace JSON lines:

```json
{"translation": {"en": "source text", "twi": "target text"}}
```

- Language codes follow ISO 639-3 (e.g. `twi`, `ewe`, `fon`)
- Directory naming: `{src}-{tgt}/` (e.g. `en-tw/`)
- Additional CSV assets for human labels live under `data/tagged_data/`

## Training with `baseline_codebase/run_translation.py`

`baseline_codebase/run_translation.py` wraps the HF Seq2Seq trainer and adds experiment logging controls.

```bash
python baseline_codebase/run_translation.py \
  --model_name_or_path facebook/m2m100_418M \
  --source_lang en --target_lang twi \
  --train_file data/finetune_data/en-tw/train.json \
  --validation_file data/finetune_data/en-tw/dev.json \
  --test_file data/finetune_data/en-tw/test.json \
  --output_dir models/m2m100_en_tw_ep10 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --predict_with_generate --fp16 \
  --log_backend wandb --wandb_project comp-express --wandb_entity research
```

Key additions:

- `--log_backend {file,wandb}` toggles JSONL logging vs. direct W&B logging
- `--metrics_log_file` customizes the JSONL sink (defaults to `<output_dir>/metrics_log.jsonl`)
- `--wandb_project` / `--wandb_entity` control W&B targets when `--log_backend wandb`
- `--epochs` overrides trainer epochs and appends `_ep{N}` suffix to `output_dir` and `run_name`

Tips:

- Reduce `--per_device_train_batch_size` or add `--gradient_accumulation_steps` for low-memory cards
- For ByT5-style models include `--source_prefix "translate English to Twi: "`
- Use `--forced_bos_token twi` for MBART/M2M bilingual decoding

## Batch evaluation (`baseline_codebase/evaluate_translations.py`)

`baseline_codebase/evaluate_translations.py` loads HF checkpoints, runs generation on named JSONL splits, and reports BLEU, sacreBLEU, chrF++, and Google BLEU. Predictions and src/mt/ref triples can be persisted for downstream COMET analysis.

```bash
python baseline_codebase/evaluate_translations.py \
  --model_path models/m2m100_en_tw_ep10 \
  --source_lang en --target_lang twi \
  --datasets dev=data/finetune_data/en-tw/dev.json test=data/finetune_data/en-tw/test.json \
  --output_dir evals/m2m100_en_tw_eval \
  --batch_size 16 --num_beams 5 --precision fp16 \
  --save_predictions --save_json --progress
```

- Accepts multiple splits via `NAME=PATH` pairs
- `--precision {fp32,bf16,fp16}` controls generation dtype
- Outputs `metrics.json`, optional `<split>_predictions.txt`, and `<split>_triples.json`

## Top-k candidate generation (`baseline_codebase/generate_topk_translations.py`)

Generate n-best lists from a fine-tuned model using either deterministic beam search or stochastic sampling.

```bash
python baseline_codebase/generate_topk_translations.py \
  --model_path models/m2m100_en_tw_ep10 \
  --inputs data/misc_data/akuapem_dataset - verified_data.csv \
  --output evals/m2m100_en_tw_eval/test_topk.jsonl \
  --source_prefix "translate English to Twi: " \
  --source_lang en --target_lang twi \
  --top_k 5 --strategy beam --batch_size 8
```

- Supports stdin/stdout streaming when `--inputs`/`--output` are omitted
- Switch to sampling with `--strategy sample --top_p 0.9 --temperature 0.7`

## COMET & statistical analysis (`baseline_codebase/comet_evaluations.py`)

Run reference-based COMET STL scoring, optional QE scoring, and paired statistical tests on exported triples.

```bash
python baseline_codebase/comet_evaluations.py \
  --input_dir evals/m2m100_en_tw_eval \
  --model masakhane/africomet-stl-1.1 \
  --qe_model Unbabel/wmt22-cometkiwi-da \
  --batch_size 16 --gpus 1 \
  --store_segment_scores --export_segments --make_plots \
  --overall_subset test_triples.json \
  --perm_samples 5000
```

- Aggregates STL/QE metrics per file plus OVERALL, 1→M, M→1 buckets
- Writes `comet_metrics.json`, optional CSV exports, and comparison plots
- Includes permutation/sign-flip tests for system differences and correlations

## Prompting experiments

- `utils/prompting_strategies.py` supplies zero-shot, few-shot, and chain-of-thought prompt builders.
- `three_prompt_strategies.md` outlines the evaluation protocol.
- `utils/llms.py` provides a Factory Method wrapper for Groq/OpenAI-compatible chat endpoints; configure API keys in `.env` before use.

## Data processing helpers

- `utils/data_processing.py` streamlines CSV loading, column renaming, pivot construction, and NaN handling for the Akan pragmatics datasets.
- `utils/metrics.py` adds quick accuracy/precision/recall/F1 helpers for classification experiments that complement prompt evaluations.

## Notebooks and analyses

- `experiments/notebooks/create_correct_labels.ipynb` – correct/augment gold labels
- `experiments/notebooks/data_analysis.ipynb` – exploratory data analysis for Akan ↔ English pairs
- `experiments/notebooks/data_analysis_play.ipynb` – scratchpad for hypothesis testing and plotting

## Troubleshooting

- Repetitive or looping generations: adjust decoding (`--no_repeat_ngram_size 3 --repetition_penalty 1.2 --length_penalty 1.0 --top_p 0.9 --top_k 50`)
- Verify ISO language codes and `--source_prefix` formatting when switching models
- Ensure COMET inputs are JSON lists of `{src, mt, ref}` objects; use `evaluate_translations.py --save_json` to export

## Citation

We are thank for LAFAND-MT paper for giving us the base code to finetune our models.

```
@inproceedings{adelani-etal-2022-thousand,
    title = "A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for {A}frican News Translation",
    author = "Adelani, David  and
      Alabi, Jesujoba  and
      Fan, Angela  and
      Kreutzer, Julia  and
      Shen, Xiaoyu  and
      Reid, Machel  and
      Ruiter, Dana  and
      Klakow, Dietrich  and
      Nabende, Peter  and
      Chang, Ernie  and
      Gwadabe, Tajuddeen  and
      Sackey, Freshia  and
      Dossou, Bonaventure F. P.  and
      Emezue, Chris  and
      Leong, Colin  and
      Beukman, Michael  and
      Muhammad, Shamsuddeen  and
      Jarso, Guyo  and
      Yousuf, Oreen  and
      Niyongabo Rubungo, Andre  and
      Hacheme, Gilles  and
      Wairagala, Eric Peter  and
      Nasir, Muhammad Umair  and
      Ajibade, Benjamin  and
      Ajayi, Tunde  and
      Gitau, Yvonne  and
      Abbott, Jade  and
      Ahmed, Mohamed  and
      Ochieng, Millicent  and
      Aremu, Anuoluwapo  and
      Ogayo, Perez  and
      Mukiibi, Jonathan  and
      Ouoba Kabore, Fatoumata  and
      Kalipe, Godson  and
      Mbaye, Derguene  and
      Tapo, Allahsera Auguste  and
      Memdjokam Koagne, Victoire  and
      Munkoh-Buabeng, Edwin  and
      Wagner, Valencia  and
      Abdulmumin, Idris  and
      Awokoya, Ayodele  and
      Buzaaba, Happy  and
      Sibanda, Blessing  and
      Bukula, Andiswa  and
      Manthalu, Sam",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.223",
    doi = "10.18653/v1/2022.naacl-main.223",
    pages = "3053--3070"
}
```
