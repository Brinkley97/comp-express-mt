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
        "values": "Question | Answer | Statement | Command | Request | Greeting",
        "description": "Utterance function?",
    },
}


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
        mapped[schema_entry["canonical"]] = value
    return mapped
