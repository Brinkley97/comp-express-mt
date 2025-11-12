from __future__ import annotations

import re
from typing import Dict, Tuple

from pydantic import BaseModel, ValidationError
from typing_extensions import Literal


class PragmaticTagSet(BaseModel):
    Gender: Literal["Masculine", "Feminine", "Neutral"]
    Animacy: Literal["Animate", "Inanimate"]
    Status: Literal["Equal", "Superior", "Subordinate"]
    Age: Literal["Peer", "Elder", "Younger"]
    Formality: Literal["Formal", "Casual"]
    Audience: Literal["Individual", "Small_Group", "Large_Group", "Broadcast"]
    Speech_Act: Literal["Question", "Answer", "Statement", "Command", "Request", "Greeting"]


class TagParseError(ValueError):
    """Raised when TAGS/SELECTION cannot be parsed from model output."""


TAG_LINE_RE = re.compile(r"TAGS:\s*([^\n\r]+)", re.IGNORECASE)
SELECTION_RE = re.compile(r"SELECTION:\s*(\d+)", re.IGNORECASE)

KEY_CANONICALS = {
    "gender": "Gender",
    "animacy": "Animacy",
    "status": "Status",
    "age": "Age",
    "formality": "Formality",
    "audience": "Audience",
    "speech_act": "Speech_Act",
    "speechact": "Speech_Act",
}

VALUE_CANONICALS = {
    "masculine": "Masculine",
    "feminine": "Feminine",
    "neutral": "Neutral",
    "animate": "Animate",
    "inanimate": "Inanimate",
    "equal": "Equal",
    "superior": "Superior",
    "subordinate": "Subordinate",
    "peer": "Peer",
    "elder": "Elder",
    "younger": "Younger",
    "formal": "Formal",
    "casual": "Casual",
    "individual": "Individual",
    "small_group": "Small_Group",
    "smallgroup": "Small_Group",
    "large_group": "Large_Group",
    "largegroup": "Large_Group",
    "broadcast": "Broadcast",
    "question": "Question",
    "answer": "Answer",
    "statement": "Statement",
    "command": "Command",
    "request": "Request",
    "greeting": "Greeting",
}


def _canonicalize_key(raw_key: str) -> str:
    normalized = raw_key.strip().lower().replace("-", "_")
    return KEY_CANONICALS.get(normalized, raw_key.strip())


def _canonicalize_value(raw_value: str) -> str:
    normalized = raw_value.strip().lower().replace("-", "_")
    return VALUE_CANONICALS.get(normalized, raw_value.strip())


def _parse_tag_line(tag_line: str) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for chunk in tag_line.split(","):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        canonical_key = _canonicalize_key(key)
        canonical_value = _canonicalize_value(value)
        if canonical_key:
            tags[canonical_key] = canonical_value
    return tags


def parse_tags(response_text: str) -> PragmaticTagSet:
    """Extract and validate the pragmatic tags from a model response."""
    tag_match = TAG_LINE_RE.search(response_text)
    if not tag_match:
        raise TagParseError("Missing TAGS line in model response.")

    tags_dict = _parse_tag_line(tag_match.group(1))
    try:
        return PragmaticTagSet(**tags_dict)
    except ValidationError as exc:
        raise TagParseError(f"Invalid tag values: {exc}") from exc


def parse_tags_and_selection(response_text: str) -> Tuple[PragmaticTagSet, int]:
    """Extract validated tags and the numeric selection from a model response."""
    tags = parse_tags(response_text)

    selection_match = SELECTION_RE.search(response_text)
    if not selection_match:
        raise TagParseError("Missing SELECTION line in model response.")

    selection = int(selection_match.group(1))
    return tags, selection
