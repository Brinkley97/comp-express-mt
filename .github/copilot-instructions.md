# CompExpress-MT AI Coding Assistant Instructions

## Project Overview
This is an **African machine translation research codebase** focusing on low-resource African languages. The project implements the [LAFAND-MT](https://arxiv.org/abs/2205.02022) (A Few Thousand Translations Go a Long Way) paper, providing MT systems for 16+ African languages using pre-trained models.

## Architecture & Data Flow

### Core Components
- **`lafand-mt/run_translation.py`**: Main HuggingFace-based training script for MT5, ByT5, MBART50, M2M100 models
- **`lafand-mt/data/`**: Multi-format datasets (JSON/TSV/text) for 20+ language pairs (en-twi, fr-ewe, etc.)
- **`lafand-mt/joeytrainer/`**: JoeyNMT-based alternative training pipeline
- **`lafand-mt/mt5_byt5_pre_training/`**: Custom pre-training scripts for African language adaptation

### Data Pipeline Architecture
```
TSV/CSV → csv_to_json.py → JSON format → run_translation.py → Fine-tuned Models
                     ↘                ↗
                      joeytrainer/ (alternative path)
```

## Key Development Patterns

### Dataset Format Convention
All training data uses HuggingFace JSON format:
```json
{"translation": {"en": "source text", "twi": "target text"}}
```
Language codes follow ISO 639-3 (twi, ewe, fon, etc.). Use **exact language pairs** in directory names: `en-twi/`, `fr-ewe/`.

### Model Training Workflows

#### Primary: HuggingFace Transformers
```bash
python run_translation.py \
    --model_name_or_path google/byt5-base \
    --source_lang en --target_lang twi \
    --source_prefix "translate English to Twi: " \
    --train_file data/json_files/en-twi/train.json \
    --validation_file data/json_files/en-twi/dev.json \
    --test_file data/json_files/en-twi/test.json
```

#### Alternative: JoeyNMT Pipeline
1. `bash setup.sh` - Install JoeyNMT environment
2. `bash train_sp.sh` - Train SentencePiece tokenizer
3. `bash apply_sp.sh` - Apply tokenization
4. `bash createconfig.sh` - Generate model config
5. `bash train.sh` - Train model

### Language-Specific Configurations

**For MBART/M2M100**: Always specify `--forced_bos_token <target_lang>` 
**For MT5/ByT5**: Use `--source_prefix "translate <source> to <target>: "`
**African language codes**: Use fake codes if language unsupported (e.g., 'sw' for Twi in M2M100)

## Critical File Locations

- **Datasets**: `lafand-mt/data/json_files/{lang-pair}/` (train.json, dev.json, test.json)
- **Modified datasets**: `lafand-mt/data/json_files_modified/` 
- **Model predictions**: `lafand-mt/model_predictions/{lang}/`
- **Pre-training configs**: `lafand-mt/mt5_byt5_pre_training/`

## External Dependencies & Models

### Pre-trained Models (HuggingFace Hub)
- **AfriMT5**: `masakhane/afri-mt5-base`
- **AfriByT5**: `masakhane/afri-byt5-base` 
- **AfriMBART**: `masakhane/afri-mbart50`

### Required Packages
```bash
pip install transformers==4.31.0 datasets sentencepiece sacrebleu accelerate torch
```

## Development Guidelines

### Working with Datasets
- **Always check data format**: JSON lines with nested "translation" objects
- **Language code consistency**: Use same codes across file paths, CLI args, and model configs
- **Data splits**: Standard train/dev/test splits; some languages only have dev/test
- **Evaluation**: Use SacreBLEU for BLEU scoring via `--predict_with_generate`

### Common Debugging Patterns
- **Tokenizer issues**: Check if language supported by model; use fake language codes if needed
- **Memory errors**: Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps`
- **Poor performance**: Ensure correct `source_prefix` format and language codes match data

### Performance Optimization
- Use `--fp16` for memory efficiency
- Set `--max_source_length` and `--max_target_length` based on language characteristics
- African languages benefit from character-level models (ByT5) vs. subword (MT5)

## Integration Points
- **JoeyNMT compatibility**: Data must be converted to separate source/target files
- **HuggingFace Hub**: Models automatically saved/loaded with proper tokenizer configs
- **Evaluation pipeline**: SacreBLEU integration for automatic metric computation