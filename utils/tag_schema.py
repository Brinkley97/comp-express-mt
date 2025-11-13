"""Helpers for inferring pragmatic tag schemas from dataset values."""

from __future__ import annotations

from typing import Dict, Iterable, List


DIMENSION_DEFS: Dict[str, Dict[str, str]] = {
    "AUDIENCE": {
        "display": "AUDIENCE",
        "canonical": "Audience",
        "values": "Individual | Small_Group | Large_Group | Broadcast",
        "description": "Who is addressed?",
    },
    "STATUS": {
        "display": "STATUS",
        "canonical": "Status",
        "values": "Equal | Superior | Subordinate",
        "description": "Social relationship?",
    },
    "AGE": {
        "display": "AGE",
        "canonical": "Age",
        "values": "Peer | Elder | Younger",
        "description": "Age-based relationship?",
    },
    "FORMALITY": {
        "display": "FORMALITY",
        "canonical": "Formality",
        "values": "Formal | Casual",
        "description": "Register level?",
    },
    "GENDER_SUBJECT": {
        "display": "GENDER_SUBJECT",
        "canonical": "Gender_Subject",
        "values": "Masculine | Feminine | Neutral",
        "description": "Gender of the subject / speaker?",
    },
    "GENDER_OBJECT": {
        "display": "GENDER_OBJECT",
        "canonical": "Gender_Object",
        "values": "Masculine | Feminine | Neutral",
        "description": "Gender of the object / listener?",
    },
    "GENDER": {
        "display": "GENDER",
        "canonical": "Gender",
        "values": "Masculine | Feminine | Neutral",
        "description": "What gender is implied?",
    },
    "ANIMACY": {
        "display": "ANIMACY",
        "canonical": "Animacy",
        "values": "Animate | Inanimate",
        "description": "Living being or object?",
    },
    "SPEECH_ACT": {
        "display": "SPEECH_ACT",
        "canonical": "Speech_Act",
        "values": "Question | Answer | Statement | Command | Request | Greeting | Advice",
        "description": "Utterance function?",
    },
}

SCHEMA_PRIORITY = [
    "AUDIENCE",
    "STATUS",
    "AGE",
    "FORMALITY",
    "GENDER_SUBJECT",
    "GENDER_OBJECT",
    "GENDER",
    "ANIMACY",
    "SPEECH_ACT",
]


SCHEMA_PATTERNS = {
    "akan_to_english": {
        7: [
            "AUDIENCE",
            "AGE",
            "FORMALITY",
            "GENDER_SUBJECT",
            "GENDER_OBJECT",
            "ANIMACY",
            "SPEECH_ACT",
        ],
        8: [
            "AUDIENCE",
            "STATUS",
            "AGE",
            "FORMALITY",
            "GENDER_SUBJECT",
            "GENDER_OBJECT",
            "ANIMACY",
            "SPEECH_ACT",
        ],
    },
    "english_to_akan": {
        7: [
            "AUDIENCE",
            "AGE",
            "FORMALITY",
            "GENDER_SUBJECT",
            "GENDER_OBJECT",
            "ANIMACY",
            "SPEECH_ACT",
        ],
        8: [
            "AUDIENCE",
            "STATUS",
            "AGE",
            "FORMALITY",
            "GENDER_SUBJECT",
            "GENDER_OBJECT",
            "ANIMACY",
            "SPEECH_ACT",
        ],
    },
}

KEY_ALIASES = {
    "AUD_SIZE": "Audience",
    "AUDIENCE": "Audience",
    "STATUS": "Status",
    "AGE": "Age",
    "FORMALITY": "Formality",
    "GENDER": "Gender_Subject",
    "GENDER_1": "Gender_Subject",
    "GENDER_SUBJECT": "Gender_Subject",
    "GENDER2": "Gender_Object",
    "GENDER_2": "Gender_Object",
    "GENDER_OBJECT": "Gender_Object",
    "ANIMACY": "Animacy",
    "SPEECH_ACT": "Speech_Act",
    "SPEECHACT": "Speech_Act",
}

VALUE_ALIASES = {
    "individual": "INDIVIDUAL",
    "small_group": "SMALL_GROUP",
    "small group": "SMALL_GROUP",
    "large_group": "LARGE_GROUP",
    "large group": "LARGE_GROUP",
    "broadcast": "BROADCAST",
    "equal": "EQUAL",
    "superior": "SUPERIOR",
    "subordinate": "SUBORDINATE",
    "peer": "PEER",
    "elder": "ELDER",
    "younger": "YOUNGER",
    "formal": "FORMAL",
    "casual": "CASUAL",
    "masculine": "MASCULINE",
    "feminine": "FEMININE",
    "neutral": "NEUTRAL",
    "animate": "ANIMATE",
    "inanimate": "INANIMATE",
    "question": "QUESTION",
    "answer": "ANSWER",
    "statement": "STATEMENT",
    "command": "COMMAND",
    "request": "REQUEST",
    "greeting": "GREETING",
    "advice": "ADVICE",
}


def canonicalize_dataset_key(key: str) -> str:
    normalized = key.strip().upper().replace("-", "_")
    return KEY_ALIASES.get(normalized, key.strip())


def _canonicalize_value(value: str) -> str:
    normalized = value.strip().lower()
    return VALUE_ALIASES.get(normalized, value.strip().upper())


def build_schema(direction: str, tag_length: int) -> List[Dict[str, str]]:
    """Return the schema definitions for the given direction and tag count."""
    direction_key = direction.lower()
    if direction_key not in SCHEMA_PATTERNS:
        raise ValueError(f"Unsupported direction for schema inference: {direction}")

    pattern = SCHEMA_PATTERNS[direction_key].get(tag_length)
    if pattern is None:
        raise ValueError(
            f"No schema mapping for direction '{direction}' with {tag_length} tag entries. "
            "Please update SCHEMA_PATTERNS or ensure the dataset follows a supported format."
        )

    return [DIMENSION_DEFS[key] for key in pattern]


def map_values_to_schema(values: Iterable[str], schema: List[Dict[str, str]]) -> Dict[str, str]:
    values = list(values)
    if len(values) != len(schema):
        raise ValueError(
            f"Length mismatch when mapping tags: expected {len(schema)} values, got {len(values)}"
        )
    mapped = {}
    for schema_entry, value in zip(schema, values):
        mapped[schema_entry["canonical"]] = _canonicalize_value(value)
    return mapped


def map_dict_to_schema(value_dict: Dict[str, str], schema: List[Dict[str, str]]) -> Dict[str, str]:
    normalized = {}
    for raw_key, raw_value in value_dict.items():
        canonical_key = canonicalize_dataset_key(raw_key)
        normalized[canonical_key] = _canonicalize_value(raw_value)

    mapped = {}
    for entry in schema:
        canonical = entry["canonical"]
        if canonical not in normalized:
            mapped[canonical] = "UNKNOWN"
        else:
            mapped[canonical] = normalized[canonical]
    return mapped


def build_schema_from_keys(keys: Iterable[str]) -> List[Dict[str, str]]:
    canonical_keys = {canonicalize_dataset_key(k) for k in keys}
    ordered = [
        DIMENSION_DEFS[key]
        for key in SCHEMA_PRIORITY
        if key in canonical_keys
    ]
    if not ordered:
        raise ValueError("Unable to build schema: no recognized tag keys provided.")
    return ordered
