import json
from collections import Counter
from math import isnan

from sklearn.metrics import cohen_kappa_score


# ============================================================
# Config
# ============================================================

# Tag set for many-to-one (EN -> Akan) direction
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

# Tag set for one-to-many (Akan -> EN) direction
# (STATUS is not annotated in this direction)
O2M_FIELDS = [
    "AUD_SIZE",
    "AGE",
    "FORMALITY",
    "GENDER",
    "GENDER_2",
    "ANIMACY",
    "SPEECH_ACT",
]


# ============================================================
# Helpers
# ============================================================

def majority_tag(values):
    """Return majority label from a list of categorical values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "NONE"
    return Counter(vals).most_common(1)[0][0]


def aggregate_m2o_per_english(annot_dict, fields):
    """
    Many-to-one (EN -> Akan) aggregation.

    Input structure:
        annot_dict[english][akan] = {field: value, ...}

    Output:
        agg[english][field] = majority_label over all Akan variants
    """
    agg = {}
    for eng, akan_dict in annot_dict.items():
        per_field_vals = {f: [] for f in fields}
        for akan, tags in akan_dict.items():
            for f in fields:
                per_field_vals[f].append(tags.get(f, "NONE"))
        agg[eng] = {f: majority_tag(per_field_vals[f]) for f in fields}
    return agg


def aggregate_o2m_per_akan(annot_dict, fields):
    """
    One-to-many (Akan -> EN) aggregation.

    Input structure:
        annot_dict[akan][english] = {field: value, ...}

    Output:
        agg[akan][field] = majority_label over all English variants
    """
    agg = {}
    for akan, eng_dict in annot_dict.items():
        per_field_vals = {f: [] for f in fields}
        for eng, tags in eng_dict.items():
            for f in fields:
                per_field_vals[f].append(tags.get(f, "NONE"))
        agg[akan] = {f: majority_tag(per_field_vals[f]) for f in fields}
    return agg


def pairwise_kappas(ann_a, ann_b, fields):
    """
    Compute per-field and composite Cohen's kappa for two annotators.

    ann_x[item][field] = label (after aggregation).
    """
    common_items = sorted(set(ann_a.keys()) & set(ann_b.keys()))
    n_items = len(common_items)

    field_kappas = {}
    for f in fields:
        a_labels = [ann_a[i][f] for i in common_items]
        b_labels = [ann_b[i][f] for i in common_items]
        field_kappas[f] = cohen_kappa_score(a_labels, b_labels)

    # Composite vector: treat full tag bundle as a single categorical label
    a_comp = ["|".join(ann_a[i][f] for f in fields) for i in common_items]
    b_comp = ["|".join(ann_b[i][f] for f in fields) for i in common_items]
    overall_kappa = cohen_kappa_score(a_comp, b_comp)

    return field_kappas, overall_kappa, n_items


def pretty_print_direction(name, fields, triples):
    """
    Print results for a given direction in a readable way.

    triples = list of (label, field_kappas, overall_kappa, n_items)
    """
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}\n")

    for label, field_k, overall_k, n_items in triples:
        print(f"[{label}]")
        print(f"  # items: {n_items}")
        print("  Per-field κ:")
        for f in fields:
            k = field_k[f]
            k_str = "NA" if isnan(k) else f"{k:.4f}"
            print(f"    {f:10s}: {k_str}")
        print(f"  Composite κ: {overall_k:.4f}\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # ----------------------------
    # Load many-to-one EN -> Akan
    # ----------------------------
    with open("./data/inter_annotator_data/Godfred_many_to_one_akan_eng_mappings_with_tags.json", "r", encoding="utf-8") as f:
        m2o_ann1_raw = json.load(f)

    with open("./data/inter_annotator_data/Kweku_many_to_one_akan_eng_mappings_with_tags.json", "r", encoding="utf-8") as f:
        m2o_ann2_raw = json.load(f)

    with open("./data/tagged_data/many_to_one_akan_eng_mappings_with_tags.json", "r", encoding="utf-8") as f:
        m2o_ann3_raw = json.load(f)

    # Aggregate majority tags per English sentence
    m2o_ann1 = aggregate_m2o_per_english(m2o_ann1_raw, M2O_FIELDS)
    m2o_ann2 = aggregate_m2o_per_english(m2o_ann2_raw, M2O_FIELDS)
    m2o_ann3 = aggregate_m2o_per_english(m2o_ann3_raw, M2O_FIELDS)

    # Pairwise kappas (many-to-one)
    m2o_12_fields, m2o_12_comp, m2o_12_n = pairwise_kappas(m2o_ann1, m2o_ann2, M2O_FIELDS)
    m2o_13_fields, m2o_13_comp, m2o_13_n = pairwise_kappas(m2o_ann1, m2o_ann3, M2O_FIELDS)
    m2o_23_fields, m2o_23_comp, m2o_23_n = pairwise_kappas(m2o_ann2, m2o_ann3, M2O_FIELDS)

    # ----------------------------
    # Load one-to-many Akan -> EN
    # ----------------------------
    with open("./data/inter_annotator_data/Godfred_one_to_many_akan_eng_mappings_with_tags.json", "r", encoding="utf-8") as f:
        o2m_ann1_raw = json.load(f)

    with open("./data/inter_annotator_data/Kweku_one_to_many_akan_eng_mappings_with_tags.json", "r", encoding="utf-8") as f:
        o2m_ann2_raw = json.load(f)

    with open("./data/tagged_data/one_to_many_akan_eng_mappings_with_tags.json", "r", encoding="utf-8") as f:
        o2m_ann3_raw = json.load(f)

    # Aggregate majority tags per Akan sentence
    o2m_ann1 = aggregate_o2m_per_akan(o2m_ann1_raw, O2M_FIELDS)
    o2m_ann2 = aggregate_o2m_per_akan(o2m_ann2_raw, O2M_FIELDS)
    o2m_ann3 = aggregate_o2m_per_akan(o2m_ann3_raw, O2M_FIELDS)

    # Pairwise kappas (one-to-many)
    o2m_12_fields, o2m_12_comp, o2m_12_n = pairwise_kappas(o2m_ann1, o2m_ann2, O2M_FIELDS)
    o2m_13_fields, o2m_13_comp, o2m_13_n = pairwise_kappas(o2m_ann1, o2m_ann3, O2M_FIELDS)
    o2m_23_fields, o2m_23_comp, o2m_23_n = pairwise_kappas(o2m_ann2, o2m_ann3, O2M_FIELDS)

    # Pretty print results
    pretty_print_direction(
        "Many-to-one EN -> Akan",
        M2O_FIELDS,
        [
            ("Annotator 1 vs 2", m2o_12_fields, m2o_12_comp, m2o_12_n),
            ("Annotator 1 vs 3", m2o_13_fields, m2o_13_comp, m2o_13_n),
            ("Annotator 2 vs 3", m2o_23_fields, m2o_23_comp, m2o_23_n),
        ],
    )

    pretty_print_direction(
        "One-to-many Akan -> EN",
        O2M_FIELDS,
        [
            ("Annotator 1 vs 2", o2m_12_fields, o2m_12_comp, o2m_12_n),
            ("Annotator 1 vs 3", o2m_13_fields, o2m_13_comp, o2m_13_n),
            ("Annotator 2 vs 3", o2m_23_fields, o2m_23_comp, o2m_23_n),
        ],
    )