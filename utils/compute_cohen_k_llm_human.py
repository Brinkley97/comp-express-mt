import json
from collections import Counter
from math import isnan

from sklearn.metrics import cohen_kappa_score


# ============================================================
# Config
# ============================================================

# Human tag fields (many-to-one EN -> Akan)
M2O_FIELDS = [
    "AUD_SIZE",
    "STATUS",
    "AGE",
    "FORMALITY",
    "GENDER",
    "GENDER_2",
    "ANIMACY",
    "SPEECH_ACT",
]

# Human tag fields (one-to-many Akan -> EN)
# STATUS is not annotated in this direction
O2M_FIELDS = [
    "AUD_SIZE",
    "AGE",
    "FORMALITY",
    "GENDER",
    "GENDER_2",
    "ANIMACY",
    "SPEECH_ACT",
]

# Map human tag fields -> LLM predicted_tags keys
LLM_FIELD_MAP = {
    "AUD_SIZE": "Audience",
    "STATUS": "Status",
    "AGE": "Age",
    "FORMALITY": "Formality",
    "GENDER": "Gender_Subject",
    "GENDER_2": "Gender_Object",
    "ANIMACY": "Animacy",
    "SPEECH_ACT": "Speech_Act",
}

# Default model key expected inside Experiment B result JSON files
DEFAULT_MODEL_NAME = "llama-3.3-70b-instruct"
MODEL_NAME_ALIASES = {
    "gpt-oss-12b": "gpt-oss-120b",
}

MODELS_TO_EVAL = [
    "llama-3.3-70b-instruct",
    "gpt-oss-12b",
]

M2O_EVAL_RUNS = [
    {
        "key": "zero_shot",
        "label": "Zero-shot",
        "path": "./experiments/results/pure_selection_results/exp_b/many_to_1_experiment_b_zero_shot_results.json",
    },
    {
        "key": "few_shot",
        "label": "Few-shot",
        "path": "./experiments/results/pure_selection_results/exp_b/many_to_1_experiment_b_few_shot_results.json",
    },
    {
        "key": "chain_of_thought",
        "label": "Chain-of-thought",
        "path": "./experiments/results/pure_selection_results/exp_b/many_to_1_experiment_b_chain_of_thought_results.json",
    },
]

O2M_EVAL_RUNS = [
    {
        "key": "zero_shot",
        "label": "Zero-shot",
        "path": "./experiments/results/pure_selection_results/exp_b/1_to_many_experiment_b_zero_shot_results.json",
    },
    {
        "key": "few_shot",
        "label": "Few-shot",
        "path": "./experiments/results/pure_selection_results/exp_b/1_to_many_experiment_b_few_shot_results.json",
    },
    {
        "key": "chain_of_thought",
        "label": "Chain-of-thought",
        "path": "./experiments/results/pure_selection_results/exp_b/1_to_many_experiment_b_chain_of_thought_results.json",
    },
]

RESULT_JSON_PATH = "./experiments/results/pure_selection_results/exp_b/kappa_results.json"


# ============================================================
# Helpers
# ============================================================

def majority(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "NONE"
    return Counter(vals).most_common(1)[0][0]


def normalize_label(x):
    """Normalize LLM and human labels into a shared space."""
    if x is None:
        return "NONE"
    return str(x).strip().upper()


def aggregate_human_m2o(path, fields):
    """
    Human gold for many-to-one (EN -> Akan).

    File structure:
        data[eng][akan] = {tag_dict}

    We majority-vote over Akan variants per English sentence.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    agg = {}
    for eng, akan_dict in data.items():
        per_field = {f: [] for f in fields}
        for _akan, tags in akan_dict.items():
            for f in fields:
                per_field[f].append(normalize_label(tags.get(f, "NONE")))
        agg[eng] = {f: majority(per_field[f]) for f in fields}
    return agg


def aggregate_human_o2m(path, fields):
    """
    Human gold for one-to-many (Akan -> EN).

    File structure:
        data[akan][eng] = {tag_dict}

    We majority-vote over English variants per Akan sentence.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    agg = {}
    for akan, eng_dict in data.items():
        per_field = {f: [] for f in fields}
        for _eng, tags in eng_dict.items():
            for f in fields:
                per_field[f].append(normalize_label(tags.get(f, "NONE")))
        agg[akan] = {f: majority(per_field[f]) for f in fields}
    return agg


def aggregate_llm_results(path, fields, direction, model_name=DEFAULT_MODEL_NAME):
    """
    Aggregate LLM predicted_tags for a given Experiment B JSON.

    File structure (both directions):
        {
          "llama-3.3-70b-instruct": [
            {
              "src": "...",          # English (m2o) or Akan (o2m)
              "tgts": [
                {
                  "TARGET_1": {
                    "gold_selection": ...,
                    "llm_selection": ...,
                    "raw_output": "...",
                    "predicted_tags": {...}
                  }
                },
                ...
              ]
            },
            ...
          ]
        }

    We majority-vote over tgts for each src, per field.
    `direction` is just a string label ("m2o" or "o2m") for clarity.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        raise ValueError(f"{path} is empty; expected at least one model block")

    _ = direction  # Reserved for future logging/debugging

    chosen_model = model_name

    if chosen_model is None:
        if DEFAULT_MODEL_NAME in raw:
            chosen_model = DEFAULT_MODEL_NAME
        elif len(raw) == 1:
            chosen_model = next(iter(raw))
        else:
            # Multiple models available but no preference supplied; pick the first for backward compatibility
            chosen_model = next(iter(raw))

    if chosen_model not in raw and chosen_model in MODEL_NAME_ALIASES:
        alias = MODEL_NAME_ALIASES[chosen_model]
        if alias in raw:
            chosen_model = alias

    if chosen_model not in raw:
        available = ", ".join(raw.keys())
        raise KeyError(
            f"Model '{chosen_model}' not found in {path}. Available keys: [{available}]"
        )

    examples = raw[chosen_model]

    agg = {}
    for ex in examples:
        src = ex["src"]   # English for m2o, Akan for o2m
        per_field = {f: [] for f in fields}

        for tgt_entry in ex["tgts"]:
            # tgt_entry is a one-key dict: {target_string: {info}}
            ((_tgt_str, info),) = tgt_entry.items()
            tags = info.get("predicted_tags", {})

            for f in fields:
                llm_key = LLM_FIELD_MAP[f]
                val = tags.get(llm_key, None)
                per_field[f].append(normalize_label(val))

        agg[src] = {f: majority(per_field[f]) for f in fields}

    return agg


def compute_kappa(human_dict, llm_dict, fields):
    """
    Compute per-field and composite κ between human and LLM.

    We first intersect on item keys (English src for m2o,
    Akan src for o2m).
    """
    common_items = sorted(set(human_dict.keys()) & set(llm_dict.keys()))
    n_items = len(common_items)

    field_kappas = {}
    for f in fields:
        gold = [human_dict[i][f] for i in common_items]
        pred = [llm_dict[i][f] for i in common_items]
        field_kappas[f] = cohen_kappa_score(gold, pred)

    gold_bundle = ["|".join(human_dict[i][f] for f in fields)
                   for i in common_items]
    pred_bundle = ["|".join(llm_dict[i][f] for f in fields)
                   for i in common_items]
    composite_kappa = cohen_kappa_score(gold_bundle, pred_bundle)

    return field_kappas, composite_kappa, n_items


def pretty_print_result(name, fields, field_kappas, composite_kappa, n_items):
    print(f"[{name}]")
    print(f"  # items: {n_items}")
    print("  Per-field κ:")
    for f in fields:
        k = field_kappas[f]
        k_str = "NA" if isnan(k) else f"{k:.4f}"
        print(f"    {f:10s}: {k_str}")
    print(f"  Composite κ: {composite_kappa:.4f}\n")


def serialize_result(fields, field_kappas, composite_kappa, n_items):
    def safe(val):
        return None if isnan(val) else val

    return {
        "n_items": n_items,
        "composite_kappa": safe(composite_kappa),
        "field_kappas": {f: safe(field_kappas[f]) for f in fields},
    }


# ============================================================
# Main: Exp B, LLM vs Human
# ============================================================

def main():
    # ----------------------------
    # Human gold
    # ----------------------------
    human_m2o = aggregate_human_m2o(
        "./data/tagged_data/many_to_one_akan_eng_mappings_with_tags.json",
        M2O_FIELDS,
    )
    human_o2m = aggregate_human_o2m(
        "./data/tagged_data/one_to_many_akan_eng_mappings_with_tags.json",
        O2M_FIELDS,
    )

    summary = {}

    for model_name in MODELS_TO_EVAL:
        summary[model_name] = {
            "many_to_one": {},
            "one_to_many": {},
        }

        print("\n============================================================")
        print("LLM vs Human (many-to-one EN -> Akan, Experiment B)")
        print(f"Model: {model_name}")
        print("============================================================\n")

        for run in M2O_EVAL_RUNS:
            llm_preds = aggregate_llm_results(
                run["path"],
                M2O_FIELDS,
                direction="m2o",
                model_name=model_name,
            )
            field_kappas, composite_kappa, n_items = compute_kappa(human_m2o, llm_preds, M2O_FIELDS)
            pretty_print_result(run["label"], M2O_FIELDS, field_kappas, composite_kappa, n_items)
            summary[model_name]["many_to_one"][run["key"]] = serialize_result(
                M2O_FIELDS,
                field_kappas,
                composite_kappa,
                n_items,
            )

        print("\n============================================================")
        print("LLM vs Human (one-to-many Akan -> EN, Experiment B)")
        print(f"Model: {model_name}")
        print("============================================================\n")

        for run in O2M_EVAL_RUNS:
            llm_preds = aggregate_llm_results(
                run["path"],
                O2M_FIELDS,
                direction="o2m",
                model_name=model_name,
            )
            field_kappas, composite_kappa, n_items = compute_kappa(human_o2m, llm_preds, O2M_FIELDS)
            pretty_print_result(run["label"], O2M_FIELDS, field_kappas, composite_kappa, n_items)
            summary[model_name]["one_to_many"][run["key"]] = serialize_result(
                O2M_FIELDS,
                field_kappas,
                composite_kappa,
                n_items,
            )

    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()