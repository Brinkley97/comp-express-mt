# CompExpress-MT

African machine translation research codebase focused on low-resource African languages (LAFAND-MT). Provides training, evaluation, and generation pipelines using HuggingFace Transformers and JoeyNMT.

- Paper: A Few Thousand Translations Go a Long Way (LAFAND-MT)
- Models: MT5, ByT5, MBART50, M2M100 (plus Afri variants)

## Repository layout

- Training and generation
  - [run_translation.py](run_translation.py) — main HF script (fine-tuning and inference)
  - [generate_topk_translations.py](generate_topk_translations.py) — top-k candidates generation
- Evaluation
  - [evaluate_translations.py](evaluate_translations.py)
  - [comet_evaluations.py](comet_evaluations.py)
- Data and artifacts
  - [data/](data) — CSVs/JSONs and pairs (e.g., `en-tw/`)
  - [lafand-mt/](lafand-mt) — extended pipeline, datasets, predictions, JoeyNMT, pretraining
  - [lafand-mt/model_predictions/](lafand-mt/model_predictions) — saved model outputs
  - [models/](models) — checkpoints
  - [evals/](evals) — evaluation outputs and configs
- Project config
  - [requirements.txt](requirements.txt)

## Setup

- Python: 3.9–3.11
- Install deps:
  - Preferred: `pip install -r requirements.txt`
  - Or (pinned HF stack): `pip install "transformers==4.31.0" datasets sentencepiece sacrebleu accelerate torch`
- Optional distributed:
  - Configure Accelerate: `accelerate config`

## Data format

HuggingFace JSON lines:
```json
{"translation": {"en": "source text", "twi": "target text"}}
```
- Language codes: ISO 639-3 (e.g., `twi`, `ewe`, `fon`)
- Directory naming: `{src}-{tgt}/` (e.g., `en-twi/`)
- Typical locations:
  - [data/](data)
  - lafand-mt-style: `lafand-mt/data/json_files/{src}-{tgt}/` (train.json, dev.json, test.json)

## Quickstart: HuggingFace training

- MT5 / ByT5 (use source_prefix)
```bash
python run_translation.py \
  --model_name_or_path google/byt5-base \
  --source_lang en --target_lang twi \
  --source_prefix "translate English to Twi: " \
  --train_file data/json_files/en-twi/train.json \
  --validation_file data/json_files/en-twi/dev.json \
  --test_file data/json_files/en-twi/test.json \
  --output_dir models/byt5_en_twi \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --num_train_epochs 5 --learning_rate 3e-4 \
  --predict_with_generate --fp16
```

- MBART50 / M2M100 (use forced_bos_token; use closest/fake code if unsupported)
```bash
python run_translation.py \
  --model_name_or_path facebook/m2m100_418M \
  --source_lang en --target_lang twi \
  --forced_bos_token twi \
  --train_file data/json_files/en-twi/train.json \
  --validation_file data/json_files/en-twi/dev.json \
  --test_file data/json_files/en-twi/test.json \
  --output_dir models/m2m100_en_twi \
  --predict_with_generate --fp16
```

Tips:
- Low memory: reduce `--per_device_train_batch_size`, increase `--gradient_accumulation_steps`
- African languages often benefit from ByT5 (character-level) vs. MT5 (subword)

## Generate predictions

- Standard generation via `--predict_with_generate` during eval/test saves outputs to `output_dir`
- Top-k candidates:
```bash
python generate_topk_translations.py \
  --model_dir models/byt5_en_twi \
  --input_json data/json_files/en-twi/test.json \
  --k 5 \
  --out_file lafand-mt/model_predictions/twi/byt5_en_twi_topk.txt
```

## Evaluate

- BLEU (SacreBLEU integrated when using `--predict_with_generate`)
- Custom evaluation scripts:
  - See [evaluate_translations.py](evaluate_translations.py)
  - See [comet_evaluations.py](comet_evaluations.py)

## JoeyNMT alternative

Workflow (see configs/scripts under [lafand-mt/](lafand-mt)):
1) Train SentencePiece tokenizer  
2) Apply tokenization  
3) Create config  
4) Train model

## Troubleshooting

- Repetitive/looping outputs (seen in some files under [lafand-mt/model_predictions/](lafand-mt/model_predictions)):
  - Decode params: `--no_repeat_ngram_size 3 --repetition_penalty 1.2 --length_penalty 1.0 --early_stopping true --top_p 0.9 --top_k 50`
  - Data issues: verify language codes and prefix formatting
  - Model choice: try ByT5 if subword models repeat

## Pretrained African models

- AfriMT5: `masakhane/afri-mt5-base`
- AfriByT5: `masakhane/afri-byt5-base`
- AfriMBART: `masakhane/afri-mbart50`

## Citation

LAFAND-MT paper:
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
    pages = "3053--3070",
    abstract = "Recent advances in the pre-training for language models leverage large-scale datasets to create multilingual models. However, low-resource languages are mostly left out in these datasets. This is primarily because many widely spoken languages that are not well represented on the web and therefore excluded from the large-scale crawls for datasets. Furthermore, downstream users of these models are restricted to the selection of languages originally chosen for pre-training. This work investigates how to optimally leverage existing pre-trained models to create low-resource translation systems for 16 African languages. We focus on two questions: 1) How can pre-trained models be used for languages not included in the initial pretraining? and 2) How can the resulting translation models effectively transfer to new domains? To answer these questions, we create a novel African news corpus covering 16 languages, of which eight languages are not part of any existing evaluation dataset. We demonstrate that the most effective strategy for transferring both additional languages and additional domains is to leverage small quantities of high-quality translation data to fine-tune large pre-trained models.",
}
```
